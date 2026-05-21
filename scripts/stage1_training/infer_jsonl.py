"""
Batch JSONL inference for ColorEdit SD3 stage-1.

Example:

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node=2 \
  scripts/stage1_training/infer_jsonl.py \
  --pretrained_model_name_or_path model/stable-diffusion-3.5-medium \
  --checkpoint outputs/stage1/checkpoint-57000/color_edit_stage1.pt \
  --input_jsonl datasets/zs_1000/zs1000_clean.jsonl \
  --output_dir result/stage1_57000steps \
  --dino_model model/dinov2-large \
  --vlm_model model/siglip2-so400m-patch16-naflex \
  --disable_cleandift \
  --resolution 1024 \
  --num_inference_steps 28 \
  --reference_scale 1.0 \
  --corr_scale 1.0 \
  --seed 42 \
  --dtype bf16
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.jsonl_infer_utils import (
    DistributedContext,
    aggregate_rank_jsonl,
    barrier,
    check_image_exists,
    cleanup_distributed,
    ensure_2d_or_batched,
    get_dtype,
    init_distributed,
    make_corr_cache_path,
    make_merged_image,
    rank0_print,
    rank_print,
    read_jsonl,
    resolve_image_path,
    sanitize_filename,
    tensor_to_image,
)
from scripts.stage1_training.build_corr_cache import compute_corr_state
from scripts.stage1_training.corr_state import CorrState, load_corr_npz
from scripts.stage1_training.data import load_rgb, resize_to_tensor
from scripts.stage1_training.feature_extractors import (
    CleanDIFTFeatureExtractor,
    DinoV2FeatureExtractor,
    SigLIP2FeatureExtractor,
    resize_pil_square,
)
from scripts.stage1_training.model import (
    CorrespondenceGuidedReferenceAdapter,
    add_sd3_transformer_lora,
    decode_vae_latents,
    encode_vae_latents,
    expand_sd3_input_channels,
    get_corr_time_weight,
    inject_corr_reference_attention,
    zero_prompt_embeds,
)
from scripts.stage1_training.train import vision_hidden_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="model/stable-diffusion-3.5-medium")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dino_model", type=str, default="model/dinov2-large")
    parser.add_argument("--vlm_model", type=str, default="")
    parser.add_argument("--cleandift_unet", type=str, default="model/cleandift/cleandift_sd21_unet.safetensors")
    parser.add_argument("--cleandift_vae", type=str, default="")
    parser.add_argument("--cleandift_feature_key", type=str, default="")
    parser.add_argument("--cleandift_timestep", type=int, default=None)
    parser.add_argument("--cleandift_use_text_encoder", action="store_true")
    parser.add_argument("--disable_cleandift", action="store_true")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--reference_scale", type=float, default=1.0)
    parser.add_argument("--corr_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--same_seed_for_all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--corr_cache_dir", type=str, default="")
    parser.add_argument("--overwrite_corr_cache", action="store_true")
    parser.add_argument("--keep_rank_jsonl", action="store_true")
    return parser.parse_args()


def corr_state_from_raw(raw: dict[str, Any], device: torch.device, dtype: torch.dtype) -> CorrState:
    return CorrState(
        topk_idx=torch.from_numpy(raw["topk_idx"].astype(np.int64)) if isinstance(raw["topk_idx"], np.ndarray) else raw["topk_idx"],
        topk_weight=torch.from_numpy(raw["topk_weight"].astype(np.float32))
        if isinstance(raw["topk_weight"], np.ndarray)
        else raw["topk_weight"],
        corr_conf=torch.from_numpy(raw["corr_conf"].astype(np.float32))
        if isinstance(raw["corr_conf"], np.ndarray)
        else raw["corr_conf"],
        region_topm_idx=torch.from_numpy(raw["region_topm_idx"].astype(np.int64))
        if isinstance(raw["region_topm_idx"], np.ndarray)
        else raw["region_topm_idx"],
        region_topm_weight=torch.from_numpy(raw["region_topm_weight"].astype(np.float32))
        if isinstance(raw["region_topm_weight"], np.ndarray)
        else raw["region_topm_weight"],
        label_c=torch.from_numpy(raw["label_c"].astype(np.int64)) if isinstance(raw["label_c"], np.ndarray) else raw["label_c"],
        label_r=torch.from_numpy(raw["label_r"].astype(np.int64)) if isinstance(raw["label_r"], np.ndarray) else raw["label_r"],
        grid_hw=tuple(int(x) for x in raw["grid_hw"]),
    ).to(device=device, dtype=dtype)


class Stage1InferencePipeline:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        checkpoint_path: str,
        dino_model: str,
        vlm_model: str,
        resolution: int | None,
        dtype: torch.dtype,
        device: torch.device,
        args: argparse.Namespace,
    ) -> None:
        self.model_path = Path(pretrained_model_name_or_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.dtype = dtype
        self.checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.ckpt_args = self.checkpoint.get("args", {})
        self.resolution = int(resolution or self.ckpt_args.get("resolution", 1024))
        self.vlm_model = vlm_model or self.ckpt_args.get("vlm_model", "model/siglip2-so400m-patch16-naflex")

        self.vlm_extractor = SigLIP2FeatureExtractor(self.vlm_model, device=device, dtype=dtype)
        self.dino_extractor = DinoV2FeatureExtractor(dino_model, device=device, dtype=dtype)
        self.cleandift_extractor = None
        disable_cleandift = args.disable_cleandift or bool(self.ckpt_args.get("disable_cleandift", False))
        if not disable_cleandift:
            self.cleandift_extractor = CleanDIFTFeatureExtractor(
                args.cleandift_unet or self.ckpt_args.get("cleandift_unet", "model/cleandift/cleandift_sd21_unet.safetensors"),
                vae_model_name_or_path=args.cleandift_vae or self.ckpt_args.get("cleandift_vae", "stabilityai/sd-vae-ft-mse"),
                feature_key=args.cleandift_feature_key or self.ckpt_args.get("cleandift_feature_key", "us6"),
                timestep=int(args.cleandift_timestep or self.ckpt_args.get("cleandift_timestep", 261)),
                device=device,
                dtype=dtype,
                use_text_encoder=args.cleandift_use_text_encoder
                or bool(self.ckpt_args.get("cleandift_use_text_encoder", False)),
            )

        self.vae = AutoencoderKL.from_pretrained(self.model_path, subfolder="vae", torch_dtype=dtype).to(device)
        self.transformer = SD3Transformer2DModel.from_pretrained(
            self.model_path,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        expand_sd3_input_channels(self.transformer, new_in_channels=32)
        hidden_size = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        inject_corr_reference_attention(
            self.transformer,
            reference_hidden_size=hidden_size,
            local_start_layer=int(self.ckpt_args.get("local_start_layer", 4)),
            sparse_start_layer=int(self.ckpt_args.get("sparse_start_layer", 5)),
        )
        add_sd3_transformer_lora(
            self.transformer,
            rank=int(self.ckpt_args.get("lora_rank", 16)),
            alpha=self.ckpt_args.get("lora_alpha", None),
            dropout=float(self.ckpt_args.get("lora_dropout", 0.0)),
            layers=self.ckpt_args.get("lora_layers", None),
            blocks=self.ckpt_args.get("lora_blocks", None),
        )
        self.reference_adapter = CorrespondenceGuidedReferenceAdapter(
            siglip_dim=vision_hidden_size(self.vlm_extractor),
            hidden_dim=hidden_size,
            num_global_tokens=int(self.ckpt_args.get("num_global_tokens", 32)),
            num_regions=int(self.ckpt_args.get("num_regions", 24)),
            resampler_depth=int(self.ckpt_args.get("reference_resampler_depth", 4)),
            resampler_heads=int(self.ckpt_args.get("reference_resampler_heads", 16)),
            resampler_dim_head=int(self.ckpt_args.get("reference_resampler_dim_head", 64)),
        ).to(device=device, dtype=dtype)
        self.reference_adapter.load_state_dict(self.checkpoint["reference_adapter"])
        self.transformer.load_state_dict(self.checkpoint["transformer_trainable"], strict=False)
        self.transformer.to(device=device, dtype=dtype)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.model_path, subfolder="scheduler")
        self.vae.eval()
        self.transformer.eval()
        self.reference_adapter.eval()

    def load_or_build_corr_state(
        self,
        content_pil_square: Image.Image,
        reference_pil_square: Image.Image,
        corr_cache_path: Path | None,
        overwrite_corr_cache: bool,
    ) -> CorrState:
        expected_grid_hw = (max(1, self.resolution // 16), max(1, self.resolution // 16))
        if corr_cache_path is not None and corr_cache_path.exists() and not overwrite_corr_cache:
            raw = load_corr_npz(corr_cache_path)
            raw = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in raw.items()}
            cached_grid_hw = tuple(int(x) for x in raw["grid_hw"])
            if cached_grid_hw != expected_grid_hw:
                print(
                    "Warning: corr_cache grid "
                    f"{cached_grid_hw} does not match inference grid {expected_grid_hw}; "
                    "sparse correspondence will be resized to the SD3 token grid.",
                    flush=True,
                )
            return corr_state_from_raw(raw, self.device, self.dtype)

        grid_hw = expected_grid_hw
        with torch.no_grad():
            f_c_dino = self.dino_extractor([content_pil_square], grid_hw)[0]
            f_r_dino = self.dino_extractor([reference_pil_square], grid_hw)[0]
            f_c_vlm = self.vlm_extractor([content_pil_square], grid_hw)[0]
            f_r_vlm = self.vlm_extractor([reference_pil_square], grid_hw)[0]
            f_c_clean = None
            f_r_clean = None
            if self.cleandift_extractor is not None:
                f_c_clean = self.cleandift_extractor([content_pil_square], grid_hw)[0]
                f_r_clean = self.cleandift_extractor([reference_pil_square], grid_hw)[0]
            raw = compute_corr_state(
                f_c_dino,
                f_r_dino,
                f_c_vlm,
                f_r_vlm,
                grid_hw=grid_hw,
                num_regions=int(self.ckpt_args.get("num_regions", 24)),
                top_m_regions=2,
                top_k_sparse=16,
                lambda_vlm_token=0.15,
                lambda_vlm_region=0.25,
                lambda_region_token=0.20,
                tau_sparse=0.07,
                tau_region=0.10,
                lambda_xy=0.05,
                kmeans_iters=10,
                lambda_dino=float(self.ckpt_args.get("lambda_dino", 0.5)),
                lambda_clean=float(self.ckpt_args.get("lambda_clean", 0.5)),
                f_c_clean=f_c_clean,
                f_r_clean=f_r_clean,
            )

        if corr_cache_path is not None:
            from scripts.stage1_training.corr_state import save_corr_npz

            corr_cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_corr_npz(corr_cache_path, **raw)
        return corr_state_from_raw(raw, self.device, self.dtype)

    @torch.no_grad()
    def infer(
        self,
        content_original: Image.Image,
        reference_original: Image.Image,
        num_inference_steps: int,
        reference_scale: float,
        corr_scale: float,
        seed: int,
        corr_cache_path: Path | None,
        overwrite_corr_cache: bool,
    ) -> Image.Image:
        content_pil_square = resize_pil_square(content_original.convert("RGB"), self.resolution)
        reference_pil_square = resize_pil_square(reference_original.convert("RGB"), self.resolution)
        content = resize_to_tensor(content_pil_square, self.resolution).unsqueeze(0).to(
            device=self.device,
            dtype=self.dtype,
        )
        reference = resize_to_tensor(reference_pil_square, self.resolution).unsqueeze(0).to(
            device=self.device,
            dtype=self.dtype,
        )
        corr_state = self.load_or_build_corr_state(
            content_pil_square,
            reference_pil_square,
            corr_cache_path,
            overwrite_corr_cache,
        )
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        content_latents = encode_vae_latents(self.vae, content)
        siglip_ref_tokens = self.vlm_extractor([reference_pil_square], corr_state.grid_hw).to(
            device=self.device,
            dtype=self.dtype,
        )
        label_r = corr_state.label_r.unsqueeze(0) if corr_state.label_r.ndim == 1 else corr_state.label_r
        ref_global_tokens, ref_region_tokens, ref_local_tokens = self.reference_adapter(
            reference,
            siglip_ref_tokens,
            label_r,
            corr_state.grid_hw,
        )
        latents = torch.randn(content_latents.shape, generator=generator, device=self.device, dtype=self.dtype)
        prompt_embeds, pooled_prompt_embeds = zero_prompt_embeds(
            batch_size=1,
            sequence_length=333,
            joint_dim=self.transformer.config.joint_attention_dim,
            pooled_dim=self.transformer.config.pooled_projection_dim,
            device=self.device,
            dtype=self.dtype,
        )

        for step_index, timestep in enumerate(self.scheduler.timesteps):
            model_input = torch.cat([latents, content_latents], dim=1)
            sigma = self.scheduler.sigmas[step_index].to(device=self.device, dtype=self.dtype).view(1)
            model_pred = self.transformer(
                hidden_states=model_input,
                timestep=timestep.expand(1),
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs={
                    "ref_global_tokens": ref_global_tokens,
                    "ref_region_tokens": ref_region_tokens,
                    "ref_local_tokens": ref_local_tokens,
                    "region_topm_idx": ensure_2d_or_batched(corr_state.region_topm_idx, 2),
                    "region_topm_weight": ensure_2d_or_batched(corr_state.region_topm_weight, 2),
                    "topk_idx": ensure_2d_or_batched(corr_state.topk_idx, 2),
                    "topk_weight": ensure_2d_or_batched(corr_state.topk_weight, 2),
                    "corr_conf": ensure_2d_or_batched(corr_state.corr_conf, 1),
                    "ref_scale": reference_scale,
                    "corr_scale": corr_scale,
                    "corr_time_weight": get_corr_time_weight(sigma).to(device=self.device, dtype=self.dtype),
                },
                return_dict=False,
            )[0]
            latents = self.scheduler.step(model_pred, timestep, latents, return_dict=False)[0]

        return tensor_to_image(decode_vae_latents(self.vae, latents))


def run_records(args: argparse.Namespace, ctx: DistributedContext) -> None:
    input_jsonl = Path(args.input_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    result_dir = output_dir / "inference_results"
    merged_dir = output_dir / "merged"
    tmp_dir = output_dir / ".rank_jsonl_tmp"
    output_jsonl = output_dir / "results.jsonl"
    corr_cache_dir = Path(args.corr_cache_dir).resolve() if args.corr_cache_dir else None

    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if corr_cache_dir is not None:
        corr_cache_dir.mkdir(parents=True, exist_ok=True)
    barrier(ctx)

    records = read_jsonl(input_jsonl)
    jsonl_dir = input_jsonl.parent
    dtype = get_dtype(args.dtype, ctx.device)
    rank_print(ctx, f"device={ctx.device}, dtype={dtype}, total_records={len(records)}")
    pipeline = Stage1InferencePipeline(
        args.pretrained_model_name_or_path,
        args.checkpoint,
        args.dino_model,
        args.vlm_model,
        args.resolution,
        dtype,
        ctx.device,
        args,
    )
    rank_print(ctx, f"model_resolution={pipeline.resolution}")

    part_jsonl = tmp_dir / f"results.rank{ctx.rank:05d}.jsonl"
    assigned_indices = list(range(ctx.rank, len(records), ctx.world_size))
    rank_print(ctx, f"assigned_records={len(assigned_indices)}")
    with part_jsonl.open("w", encoding="utf-8") as writer:
        for local_i, idx in enumerate(assigned_indices):
            item = records[idx]
            source_path = resolve_image_path(item["source"], jsonl_dir)
            reference_path = resolve_image_path(item["reference"], jsonl_dir)
            target_path = resolve_image_path(item["target"], jsonl_dir)
            check_image_exists(source_path, "source", idx)
            check_image_exists(reference_path, "reference", idx)
            check_image_exists(target_path, "target", idx)
            source_image = load_rgb(source_path)
            reference_image = load_rgb(reference_path)
            target_image = load_rgb(target_path)
            content_size = source_image.size
            filename = f"{idx:06d}_{sanitize_filename(source_path.name)}.png"
            result_path = (result_dir / filename).resolve()
            merged_path = (merged_dir / filename).resolve()
            corr_cache_path = (
                make_corr_cache_path(corr_cache_dir, idx, source_path, reference_path)
                if corr_cache_dir is not None
                else None
            )

            if result_path.exists() and not args.overwrite:
                result_image = load_rgb(result_path)
                if result_image.size != content_size:
                    result_image = result_image.resize(content_size, Image.Resampling.BICUBIC)
                    result_image.save(result_path)
            else:
                sample_seed = args.seed if args.same_seed_for_all else args.seed + idx
                result_image = pipeline.infer(
                    source_image,
                    reference_image,
                    args.num_inference_steps,
                    args.reference_scale,
                    args.corr_scale,
                    sample_seed,
                    corr_cache_path,
                    args.overwrite_corr_cache,
                )
                if result_image.size != content_size:
                    result_image = result_image.resize(content_size, Image.Resampling.BICUBIC)
                result_image.save(result_path)

            if args.overwrite or not merged_path.exists():
                merged_image = make_merged_image(source_image, reference_image, target_image, result_image, content_size)
                merged_image.save(merged_path)

            output_item = dict(item)
            output_item["result"] = str(result_path)
            writer.write(json.dumps({"__idx": idx, "record": output_item}, ensure_ascii=False) + "\n")
            writer.flush()
            rank_print(
                ctx,
                f"[{local_i + 1}/{len(assigned_indices)}] global_idx={idx}, "
                f"result_size={result_image.size}, content_size={content_size}, result={result_path}",
            )

    barrier(ctx)
    if ctx.rank == 0:
        aggregate_rank_jsonl(tmp_dir, output_jsonl, ctx.world_size, len(records))
        if not args.keep_rank_jsonl:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        rank0_print(ctx, f"Saved final JSONL: {output_jsonl}")
        rank0_print(ctx, f"Saved inference results: {result_dir}")
        rank0_print(ctx, f"Saved merged images: {merged_dir}")
        if corr_cache_dir is not None:
            rank0_print(ctx, f"Saved correspondence cache: {corr_cache_dir}")
    barrier(ctx)


def main() -> None:
    args = parse_args()
    ctx = init_distributed()
    try:
        run_records(args, ctx)
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
