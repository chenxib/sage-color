from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage1_training.corr_state import cache_path_for_index, save_corr_npz
from scripts.stage1_training.feature_extractors import (
    CleanDIFTFeatureExtractor,
    DinoV2FeatureExtractor,
    SigLIP2FeatureExtractor,
    l2_normalize,
    resize_pil_square,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, default="datasets/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="datasets/corr_cache_stage1_1024")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--grid_size", type=int, default=None)
    parser.add_argument("--dino_model", type=str, default="model/dinov2-large")
    parser.add_argument("--vlm_model", type=str, default="model/siglip2-so400m-patch16-naflex")
    parser.add_argument("--cleandift_unet", type=str, default="model/cleandift/cleandift_sd21_unet.safetensors")
    parser.add_argument("--cleandift_vae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--cleandift_feature_key", type=str, default="us6")
    parser.add_argument("--cleandift_timestep", type=int, default=261)
    parser.add_argument("--cleandift_use_text_encoder", action="store_true")
    parser.add_argument("--disable_cleandift", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_regions", type=int, default=24)
    parser.add_argument("--top_m_regions", type=int, default=2)
    parser.add_argument("--top_k_sparse", type=int, default=16)
    parser.add_argument("--lambda_dino", type=float, default=0.5)
    parser.add_argument("--lambda_clean", type=float, default=0.5)
    parser.add_argument("--lambda_vlm_token", type=float, default=0.15)
    parser.add_argument("--lambda_vlm_region", type=float, default=0.25)
    parser.add_argument("--lambda_region_token", type=float, default=0.20)
    parser.add_argument("--tau_sparse", type=float, default=0.07)
    parser.add_argument("--tau_region", type=float, default=0.10)
    parser.add_argument("--lambda_xy", type=float, default=0.05)
    parser.add_argument("--kmeans_iters", type=int, default=10)
    parser.add_argument("--debug_dir", type=str, default=None)
    return parser.parse_args()


def load_rows(manifest: str | Path) -> list[dict]:
    with Path(manifest).open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {manifest}")
    return rows


def load_image(path: str | Path, resolution: int) -> Image.Image:
    return resize_pil_square(Image.open(path), resolution)


def torch_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def kmeans(features: torch.Tensor, num_clusters: int, iters: int = 10, seed: int = 0) -> torch.Tensor:
    features = features.float()
    num_tokens = features.shape[0]
    generator = torch.Generator(device=features.device).manual_seed(seed)
    if num_tokens >= num_clusters:
        init = torch.randperm(num_tokens, generator=generator, device=features.device)[:num_clusters]
        centers = features[init].clone()
    else:
        pad = num_clusters - num_tokens
        centers = torch.cat([features, features[:1].expand(pad, -1)], dim=0)
    labels = torch.zeros(num_tokens, device=features.device, dtype=torch.long)
    for _ in range(iters):
        dist = torch.cdist(features, centers)
        labels = dist.argmin(dim=1)
        new_centers = []
        for cluster in range(num_clusters):
            mask = labels == cluster
            if mask.any():
                new_centers.append(features[mask].mean(dim=0))
            else:
                farthest = dist.min(dim=1).values.argmax()
                new_centers.append(features[farthest])
        centers = torch.stack(new_centers, dim=0)
    return labels


def majority_filter(labels: torch.Tensor, grid_hw: tuple[int, int], num_regions: int, rounds: int = 1) -> torch.Tensor:
    height, width = grid_hw
    labels = labels.view(1, height, width)
    for _ in range(rounds):
        one_hot = F.one_hot(labels, num_classes=num_regions).permute(0, 3, 1, 2).float()
        counts = F.conv2d(one_hot, torch.ones(num_regions, 1, 3, 3, device=labels.device), padding=1, groups=num_regions)
        labels = counts.argmax(dim=1)
    return labels.reshape(-1)


def xy_features(grid_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
    height, width = grid_hw
    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(height * width, 2)


def region_centroids(features: torch.Tensor, labels: torch.Tensor, num_regions: int) -> torch.Tensor:
    channels = features.shape[-1]
    out = features.new_zeros(num_regions, channels)
    counts = features.new_zeros(num_regions, 1)
    index = labels[:, None].expand(-1, channels)
    out.scatter_add_(0, index, features)
    counts.scatter_add_(0, labels[:, None], torch.ones(labels.shape[0], 1, device=features.device, dtype=features.dtype))
    fallback = features.mean(dim=0, keepdim=True).expand(num_regions, -1)
    return torch.where(counts > 0, out / counts.clamp_min(1.0), fallback)


def compute_corr_state(
    f_c_dino: torch.Tensor,
    f_r_dino: torch.Tensor,
    f_c_vlm: torch.Tensor,
    f_r_vlm: torch.Tensor,
    *,
    grid_hw: tuple[int, int],
    num_regions: int,
    top_m_regions: int,
    top_k_sparse: int,
    lambda_vlm_token: float,
    lambda_vlm_region: float,
    lambda_region_token: float,
    tau_sparse: float,
    tau_region: float,
    lambda_xy: float,
    kmeans_iters: int,
    lambda_dino: float = 1.0,
    lambda_clean: float = 0.0,
    f_c_clean: torch.Tensor | None = None,
    f_r_clean: torch.Tensor | None = None,
    return_tensors: bool = False,
) -> dict[str, np.ndarray | torch.Tensor | tuple[int, int]]:
    device = f_c_dino.device
    f_c_dino = l2_normalize(f_c_dino)
    f_r_dino = l2_normalize(f_r_dino)
    if f_c_clean is not None and f_r_clean is not None and lambda_clean > 0:
        f_c_clean = l2_normalize(f_c_clean)
        f_r_clean = l2_normalize(f_r_clean)
    else:
        f_c_clean = None
        f_r_clean = None
        lambda_clean = 0.0
    if lambda_clean <= 0:
        lambda_dino = 1.0
    dense_weight_sum = max(float(lambda_dino + lambda_clean), 1e-6)
    w_dino = float(lambda_dino) / dense_weight_sum
    w_clean = float(lambda_clean) / dense_weight_sum
    f_c_vlm = l2_normalize(f_c_vlm)
    f_r_vlm = l2_normalize(f_r_vlm)

    cluster_parts_c = [(w_dino**0.5) * f_c_dino]
    cluster_parts_r = [(w_dino**0.5) * f_r_dino]
    if f_c_clean is not None and f_r_clean is not None and w_clean > 0:
        cluster_parts_c.append((w_clean**0.5) * f_c_clean)
        cluster_parts_r.append((w_clean**0.5) * f_r_clean)
    f_c_cluster = torch.cat(cluster_parts_c, dim=-1)
    f_r_cluster = torch.cat(cluster_parts_r, dim=-1)

    coords = xy_features(grid_hw, device=device)
    label_c = kmeans(torch.cat([f_c_cluster, lambda_xy * coords], dim=-1), num_regions, iters=kmeans_iters, seed=17)
    label_r = kmeans(torch.cat([f_r_cluster, lambda_xy * coords], dim=-1), num_regions, iters=kmeans_iters, seed=23)
    label_c = majority_filter(label_c, grid_hw, num_regions, rounds=1)
    label_r = majority_filter(label_r, grid_hw, num_regions, rounds=1)

    mu_c_dino = l2_normalize(region_centroids(f_c_dino, label_c, num_regions))
    mu_r_dino = l2_normalize(region_centroids(f_r_dino, label_r, num_regions))
    mu_c_vlm = l2_normalize(region_centroids(f_c_vlm, label_c, num_regions))
    mu_r_vlm = l2_normalize(region_centroids(f_r_vlm, label_r, num_regions))

    s_reg_dense = w_dino * (mu_c_dino @ mu_r_dino.T)
    if f_c_clean is not None and f_r_clean is not None and w_clean > 0:
        mu_c_clean = l2_normalize(region_centroids(f_c_clean, label_c, num_regions))
        mu_r_clean = l2_normalize(region_centroids(f_r_clean, label_r, num_regions))
        s_reg_dense = s_reg_dense + w_clean * (mu_c_clean @ mu_r_clean.T)
    s_reg_vlm = mu_c_vlm @ mu_r_vlm.T
    s_region = s_reg_dense + lambda_vlm_region * s_reg_vlm
    reg_val, reg_idx = torch.topk(s_region, k=min(top_m_regions, num_regions), dim=1)
    reg_weight = torch.softmax(reg_val / tau_region, dim=-1)

    region_topm_idx = reg_idx[label_c]
    region_topm_weight = reg_weight[label_c]

    s_dense = w_dino * (f_c_dino @ f_r_dino.T)
    if f_c_clean is not None and f_r_clean is not None and w_clean > 0:
        s_dense = s_dense + w_clean * (f_c_clean @ f_r_clean.T)
    s_vlm = f_c_vlm @ f_r_vlm.T
    region_token_bias = s_region[label_c][:, label_r]
    scores = s_dense + lambda_vlm_token * s_vlm + lambda_region_token * region_token_bias

    allowed_region = torch.zeros(num_regions, num_regions, device=device, dtype=torch.bool)
    allowed_region.scatter_(1, reg_idx, True)
    candidate_mask = allowed_region[label_c][:, label_r]
    too_few = candidate_mask.sum(dim=1) < top_k_sparse
    if too_few.any():
        candidate_mask[too_few] = True
    scores = scores.masked_fill(~candidate_mask, -1e4)
    topk_val, topk_idx = torch.topk(scores, k=top_k_sparse, dim=1)
    topk_weight = torch.softmax(topk_val / tau_sparse, dim=-1)

    entropy = -(topk_weight * torch.log(topk_weight.clamp_min(1e-6))).sum(dim=-1)
    conf_sparse = 1.0 - entropy / math.log(top_k_sparse)
    conf_region = torch.softmax(s_region / tau_region, dim=-1).max(dim=-1).values[label_c]
    conf_vlm = torch.softmax(s_reg_vlm / tau_region, dim=-1).max(dim=-1).values[label_c]
    corr_conf = torch.sqrt((conf_sparse * conf_region).clamp_min(0.0)) * (0.5 + 0.5 * conf_vlm)
    corr_conf = corr_conf.clamp(0.0, 1.0)

    tensor_state = {
        "grid_hw": grid_hw,
        "topk_idx": topk_idx.detach(),
        "topk_weight": topk_weight.detach(),
        "corr_conf": corr_conf.detach(),
        "region_topm_idx": region_topm_idx.detach(),
        "region_topm_weight": region_topm_weight.detach(),
        "label_c": label_c.detach(),
        "label_r": label_r.detach(),
        "region_sim": s_region.detach(),
    }
    if return_tensors:
        return tensor_state
    return {
        key: value.detach().cpu().numpy() if torch.is_tensor(value) else value
        for key, value in tensor_state.items()
    }


def save_debug(debug_dir: Path, index: int, state: dict[str, np.ndarray | tuple[int, int]]) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    grid_hw = state["grid_hw"]
    assert isinstance(grid_hw, tuple)
    height, width = grid_hw
    label_c = np.asarray(state["label_c"]).reshape(height, width)
    label_r = np.asarray(state["label_r"]).reshape(height, width)
    corr_conf = np.asarray(state["corr_conf"]).reshape(height, width)
    rng = np.random.default_rng(1234)
    palette = rng.integers(0, 255, size=(256, 3), dtype=np.uint8)
    Image.fromarray(palette[label_c]).resize((512, 512), Image.Resampling.NEAREST).save(
        debug_dir / f"{index:08d}_content_region_map.png"
    )
    Image.fromarray(palette[label_r]).resize((512, 512), Image.Resampling.NEAREST).save(
        debug_dir / f"{index:08d}_reference_region_map.png"
    )
    conf_img = (np.clip(corr_conf, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(conf_img).resize((512, 512), Image.Resampling.BICUBIC).save(debug_dir / f"{index:08d}_corr_conf.png")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.train_jsonl)
    end = len(rows) if args.max_items is None else min(len(rows), args.start_index + args.max_items)
    grid_size = args.grid_size or max(1, args.resolution // 16)
    grid_hw = (grid_size, grid_size)
    dtype = torch_dtype(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    dino = DinoV2FeatureExtractor(args.dino_model, device=device, dtype=dtype)
    vlm = SigLIP2FeatureExtractor(args.vlm_model, device=device, dtype=dtype)
    cleandift = None
    if not args.disable_cleandift:
        cleandift = CleanDIFTFeatureExtractor(
            args.cleandift_unet,
            vae_model_name_or_path=args.cleandift_vae,
            feature_key=args.cleandift_feature_key,
            timestep=args.cleandift_timestep,
            device=device,
            dtype=dtype,
            use_text_encoder=args.cleandift_use_text_encoder,
        )
    output_dir = Path(args.output_dir)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None

    for index in range(args.start_index, end):
        output_path = cache_path_for_index(output_dir, index)
        if output_path.exists() and not args.overwrite:
            print(f"skip existing {output_path}", flush=True)
            continue
        row = rows[index]
        content = load_image(row["content_image"], args.resolution)
        reference = load_image(row["reference_image"], args.resolution)
        with torch.no_grad():
            f_c_dino = dino([content], grid_hw)[0]
            f_r_dino = dino([reference], grid_hw)[0]
            f_c_vlm = vlm([content], grid_hw)[0]
            f_r_vlm = vlm([reference], grid_hw)[0]
            f_c_clean = None
            f_r_clean = None
            if cleandift is not None:
                f_c_clean = cleandift([content], grid_hw)[0]
                f_r_clean = cleandift([reference], grid_hw)[0]
            state = compute_corr_state(
                f_c_dino,
                f_r_dino,
                f_c_vlm,
                f_r_vlm,
                grid_hw=grid_hw,
                num_regions=args.num_regions,
                top_m_regions=args.top_m_regions,
                top_k_sparse=args.top_k_sparse,
                lambda_vlm_token=args.lambda_vlm_token,
                lambda_vlm_region=args.lambda_vlm_region,
                lambda_region_token=args.lambda_region_token,
                tau_sparse=args.tau_sparse,
                tau_region=args.tau_region,
                lambda_xy=args.lambda_xy,
                kmeans_iters=args.kmeans_iters,
                lambda_dino=args.lambda_dino,
                lambda_clean=args.lambda_clean,
                f_c_clean=f_c_clean,
                f_r_clean=f_r_clean,
            )
        save_corr_npz(output_path, **state)
        if debug_dir is not None:
            save_debug(debug_dir, index, state)
        print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
