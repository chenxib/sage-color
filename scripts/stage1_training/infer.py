from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    parser.add_argument("--content_image", type=str, required=True)
    parser.add_argument("--reference_image", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/stage1/sample.png")
    parser.add_argument("--corr_cache", type=str, default="")
    parser.add_argument("--save_corr_cache", type=str, default="")
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
    return parser.parse_args()


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = (tensor.detach().float().clamp(-1, 1) + 1.0) / 2.0
    array = (tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def corr_state_from_raw(raw: dict, device: torch.device, dtype: torch.dtype) -> CorrState:
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


def load_or_build_corr_state(
    args,
    content_pil,
    reference_pil,
    vlm_extractor,
    cleandift_extractor,
    device,
    dtype,
    ckpt_args,
) -> CorrState:
    expected_grid_hw = (max(1, args.resolution // 16), max(1, args.resolution // 16))
    if args.corr_cache:
        raw = load_corr_npz(args.corr_cache)
        raw = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in raw.items()}
        raw["grid_hw"] = raw["grid_hw"]
        cached_grid_hw = tuple(int(x) for x in raw["grid_hw"])
        if cached_grid_hw != expected_grid_hw:
            print(
                "Warning: corr_cache grid "
                f"{cached_grid_hw} does not match inference grid {expected_grid_hw}; "
                "sparse correspondence will be resized to the SD3 token grid.",
                flush=True,
            )
        return corr_state_from_raw(raw, device, dtype)

    grid_hw = expected_grid_hw
    dino = DinoV2FeatureExtractor(args.dino_model, device=device, dtype=dtype)
    with torch.no_grad():
        f_c_dino = dino([content_pil], grid_hw)[0]
        f_r_dino = dino([reference_pil], grid_hw)[0]
        f_c_vlm = vlm_extractor([content_pil], grid_hw)[0]
        f_r_vlm = vlm_extractor([reference_pil], grid_hw)[0]
        f_c_clean = None
        f_r_clean = None
        if cleandift_extractor is not None:
            f_c_clean = cleandift_extractor([content_pil], grid_hw)[0]
            f_r_clean = cleandift_extractor([reference_pil], grid_hw)[0]
        raw = compute_corr_state(
            f_c_dino,
            f_r_dino,
            f_c_vlm,
            f_r_vlm,
            grid_hw=grid_hw,
            num_regions=int(ckpt_args.get("num_regions", 24)),
            top_m_regions=2,
            top_k_sparse=16,
            lambda_vlm_token=0.15,
            lambda_vlm_region=0.25,
            lambda_region_token=0.20,
            tau_sparse=0.07,
            tau_region=0.10,
            lambda_xy=0.05,
            kmeans_iters=10,
            lambda_dino=float(ckpt_args.get("lambda_dino", 0.5)),
            lambda_clean=float(ckpt_args.get("lambda_clean", 0.5)),
            f_c_clean=f_c_clean,
            f_r_clean=f_r_clean,
        )
    if args.save_corr_cache:
        from scripts.stage1_training.corr_state import save_corr_npz

        save_corr_npz(args.save_corr_cache, **raw)
    return corr_state_from_raw(raw, device, dtype)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    if args.resolution is None:
        args.resolution = int(ckpt_args.get("resolution", 1024))
    vlm_model = args.vlm_model or ckpt_args.get("vlm_model", "model/siglip2-so400m-patch16-naflex")
    vlm_extractor = SigLIP2FeatureExtractor(vlm_model, device=device, dtype=dtype)
    cleandift_extractor = None
    disable_cleandift = args.disable_cleandift or bool(ckpt_args.get("disable_cleandift", False))
    if not disable_cleandift:
        cleandift_extractor = CleanDIFTFeatureExtractor(
            args.cleandift_unet or ckpt_args.get("cleandift_unet", "model/cleandift/cleandift_sd21_unet.safetensors"),
            vae_model_name_or_path=args.cleandift_vae or ckpt_args.get("cleandift_vae", "stabilityai/sd-vae-ft-mse"),
            feature_key=args.cleandift_feature_key or ckpt_args.get("cleandift_feature_key", "us6"),
            timestep=int(args.cleandift_timestep or ckpt_args.get("cleandift_timestep", 261)),
            device=device,
            dtype=dtype,
            use_text_encoder=args.cleandift_use_text_encoder or bool(ckpt_args.get("cleandift_use_text_encoder", False)),
        )

    model_path = Path(args.pretrained_model_name_or_path)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(device)
    transformer = SD3Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
    expand_sd3_input_channels(transformer, new_in_channels=32)
    hidden_size = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    inject_corr_reference_attention(
        transformer,
        reference_hidden_size=hidden_size,
        local_start_layer=int(ckpt_args.get("local_start_layer", 4)),
        sparse_start_layer=int(ckpt_args.get("sparse_start_layer", 5)),
    )
    add_sd3_transformer_lora(
        transformer,
        rank=int(ckpt_args.get("lora_rank", 16)),
        alpha=ckpt_args.get("lora_alpha", None),
        dropout=float(ckpt_args.get("lora_dropout", 0.0)),
        layers=ckpt_args.get("lora_layers", None),
        blocks=ckpt_args.get("lora_blocks", None),
    )
    reference_adapter = CorrespondenceGuidedReferenceAdapter(
        siglip_dim=vision_hidden_size(vlm_extractor),
        hidden_dim=hidden_size,
        num_global_tokens=int(ckpt_args.get("num_global_tokens", 32)),
        num_regions=int(ckpt_args.get("num_regions", 24)),
        resampler_depth=int(ckpt_args.get("reference_resampler_depth", 4)),
        resampler_heads=int(ckpt_args.get("reference_resampler_heads", 16)),
        resampler_dim_head=int(ckpt_args.get("reference_resampler_dim_head", 64)),
    ).to(device=device, dtype=dtype)
    reference_adapter.load_state_dict(checkpoint["reference_adapter"])
    transformer.load_state_dict(checkpoint["transformer_trainable"], strict=False)
    transformer.to(device=device, dtype=dtype)

    vae.eval()
    transformer.eval()
    reference_adapter.eval()

    content_pil = resize_pil_square(load_rgb(args.content_image), args.resolution)
    reference_pil = resize_pil_square(load_rgb(args.reference_image), args.resolution)
    content = resize_to_tensor(content_pil, args.resolution).unsqueeze(0).to(device=device, dtype=dtype)
    reference = resize_to_tensor(reference_pil, args.resolution).unsqueeze(0).to(device=device, dtype=dtype)
    corr_state = load_or_build_corr_state(
        args,
        content_pil,
        reference_pil,
        vlm_extractor,
        cleandift_extractor,
        device,
        dtype,
        ckpt_args,
    )

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    scheduler.set_timesteps(args.num_inference_steps, device=device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad():
        content_latents = encode_vae_latents(vae, content)
        siglip_ref_tokens = vlm_extractor([reference_pil], corr_state.grid_hw).to(device=device, dtype=dtype)
        ref_global_tokens, ref_region_tokens, ref_local_tokens = reference_adapter(
            reference,
            siglip_ref_tokens,
            corr_state.label_r.unsqueeze(0) if corr_state.label_r.ndim == 1 else corr_state.label_r,
            corr_state.grid_hw,
        )
        latents = torch.randn(content_latents.shape, generator=generator, device=device, dtype=dtype)

        prompt_embeds, pooled_prompt_embeds = zero_prompt_embeds(
            batch_size=1,
            sequence_length=333,
            joint_dim=transformer.config.joint_attention_dim,
            pooled_dim=transformer.config.pooled_projection_dim,
            device=device,
            dtype=dtype,
        )

        for step_index, timestep in enumerate(scheduler.timesteps):
            model_input = torch.cat([latents, content_latents], dim=1)
            sigma = scheduler.sigmas[step_index].to(device=device, dtype=dtype).view(1)
            model_pred = transformer(
                hidden_states=model_input,
                timestep=timestep.expand(1),
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs={
                    "ref_global_tokens": ref_global_tokens,
                    "ref_region_tokens": ref_region_tokens,
                    "ref_local_tokens": ref_local_tokens,
                    "region_topm_idx": corr_state.region_topm_idx.unsqueeze(0)
                    if corr_state.region_topm_idx.ndim == 2
                    else corr_state.region_topm_idx,
                    "region_topm_weight": corr_state.region_topm_weight.unsqueeze(0)
                    if corr_state.region_topm_weight.ndim == 2
                    else corr_state.region_topm_weight,
                    "topk_idx": corr_state.topk_idx.unsqueeze(0) if corr_state.topk_idx.ndim == 2 else corr_state.topk_idx,
                    "topk_weight": corr_state.topk_weight.unsqueeze(0)
                    if corr_state.topk_weight.ndim == 2
                    else corr_state.topk_weight,
                    "corr_conf": corr_state.corr_conf.unsqueeze(0)
                    if corr_state.corr_conf.ndim == 1
                    else corr_state.corr_conf,
                    "ref_scale": args.reference_scale,
                    "corr_scale": args.corr_scale,
                    "corr_time_weight": get_corr_time_weight(sigma).to(device=device, dtype=dtype),
                },
                return_dict=False,
            )[0]
            latents = scheduler.step(model_pred, timestep, latents, return_dict=False)[0]

        image = tensor_to_image(decode_vae_latents(vae, latents))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
