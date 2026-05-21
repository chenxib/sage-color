from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from diffusers.training_utils import compute_loss_weighting_for_sd3
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.final_model.build_corr_cache import compute_corr_state
from scripts.final_model.corr_state import corr_state_from_batch, load_corr_npz
from scripts.final_model.data import ColorEditJsonlDataset, load_rgb, resize_to_tensor
from scripts.final_model.feature_extractors import (
    CleanDIFTFeatureExtractor,
    DinoV2FeatureExtractor,
    SigLIP2FeatureExtractor,
    resize_pil_square,
)
from scripts.final_model.content_intrinsic_extractor import (
    ContentStructureExtractor,
    DepthAnythingV2FeatureExtractor,
    Mask2FormerPanopticStructureExtractor,
    SegFormerSemanticStructureExtractor,
)
from scripts.final_model.intrinsic_prior_tokenizer import IntrinsicPriorTokenizer, xy_grid
from scripts.final_model.model import (
    CorrespondenceGuidedReferenceAdapter,
    add_sd3_transformer_lora,
    cig_processor_magnitudes,
    corr_reference_processor_state_dict,
    decode_vae_latents,
    encode_vae_latents,
    expand_sd3_input_channels,
    freeze_transformer_for_adapter_training,
    get_corr_time_weight,
    inject_corr_reference_attention,
    load_corr_reference_processor_state_dict,
    rgb_to_lab,
    set_corr_reference_processor_trainable,
    trainable_parameters,
    zero_prompt_embeds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="model/stable-diffusion-3.5-medium")
    parser.add_argument("--train_jsonl", type=str, default="datasets/train.jsonl")
    parser.add_argument("--corr_cache_dir", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", type=str, default="outputs/final-model")
    parser.add_argument("--init_from_stage1_checkpoint", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--adamw_foreach", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--dino_model", type=str, default="model/dinov2-large")
    parser.add_argument("--vlm_model", type=str, default="model/siglip2-so400m-patch16-naflex")
    parser.add_argument("--cleandift_unet", type=str, default="model/cleandift/cleandift_sd21_unet.safetensors")
    parser.add_argument("--cleandift_vae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--cleandift_feature_key", type=str, default="us6")
    parser.add_argument("--cleandift_timestep", type=int, default=261)
    parser.add_argument("--cleandift_use_text_encoder", action="store_true")
    parser.add_argument("--disable_cleandift", action="store_true")
    parser.add_argument("--depth_model", type=str, default="model/depth-anything-v2-base")
    parser.add_argument("--use_depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable_depth", action="store_true")
    parser.add_argument("--segmentation_model", type=str, default="model/segformer-b0-ade")
    parser.add_argument("--use_segmentation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable_segmentation", action="store_true")
    parser.add_argument("--panoptic_model", type=str, default="model/mask2former-swin-small-coco-panoptic")
    parser.add_argument("--use_panoptic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable_panoptic", action="store_true")
    parser.add_argument("--disable_dino_structure", action="store_true", help="Deprecated no-op: CIG never consumes DINO features.")
    parser.add_argument("--grid_size", type=int, default=None)
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
    parser.add_argument("--num_global_tokens", type=int, default=32)
    parser.add_argument("--num_regions", type=int, default=24)
    parser.add_argument("--reference_resampler_depth", type=int, default=4)
    parser.add_argument("--reference_resampler_heads", type=int, default=16)
    parser.add_argument("--reference_resampler_dim_head", type=int, default=64)
    parser.add_argument("--reference_dropout", type=float, default=0.0)
    parser.add_argument("--corr_dropout", type=float, default=0.0)
    parser.add_argument("--reference_scale", type=float, default=1.0)
    parser.add_argument("--corr_scale", type=float, default=1.0)
    parser.add_argument("--corr_warmup_steps", type=int, default=0)
    parser.add_argument("--local_start_layer", type=int, default=4)
    parser.add_argument("--sparse_start_layer", type=int, default=5)
    parser.add_argument("--use_cig", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable_cig", action="store_true")
    parser.add_argument("--cig_hidden_dim", type=int, default=0)
    parser.add_argument("--cig_num_masks", type=int, default=6)
    parser.add_argument("--cig_runtime_start_step", type=int, default=0)
    parser.add_argument("--cig_runtime_ramp_steps", type=int, default=0)
    parser.add_argument("--anchor_start_step", type=int, default=0)
    parser.add_argument("--anchor_ramp_steps", type=int, default=0)
    parser.add_argument("--cig_max_key_bias", type=float, default=0.8)
    parser.add_argument("--cig_max_anchor_scale", type=float, default=0.15)
    parser.add_argument("--cig_max_local_protect", type=float, default=0.30)
    parser.add_argument("--cig_max_region_protect", type=float, default=0.12)
    parser.add_argument("--cig_key_bias_scale", type=float, default=1.0)
    parser.add_argument("--cig_anchor_scale", type=float, default=1.0)
    parser.add_argument("--cig_ref_protect_scale", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_layers", type=str, default=None)
    parser.add_argument("--lora_blocks", type=str, default=None)
    parser.add_argument("--weighting_scheme", type=str, default="none")
    parser.add_argument("--use_color_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--color_loss_weight", type=float, default=0.05)
    parser.add_argument(
        "--color_decode_resolution",
        type=int,
        default=0,
        help="0 means decode full-size pred_x0 latents for the color loss, matching stage-1.",
    )
    parser.add_argument("--color_loss_sigma_power", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--validation_content_image", type=str, default="datasets/validation/content.png")
    parser.add_argument("--validation_reference_image", type=str, default="datasets/validation/reference.png")
    parser.add_argument("--validation_corr_cache", type=str, default="")
    parser.add_argument("--validation_num_inference_steps", type=int, default=4)
    parser.add_argument("--validation_seed", type=int, default=42)
    parser.add_argument("--disable_checkpoint_validation", action="store_true")
    return parser.parse_args()


def get_sigmas(noise_scheduler, timesteps, n_dim: int, dtype: torch.dtype, device: torch.device):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def tensor_to_image(tensor: torch.Tensor):
    tensor = (tensor.detach().float().clamp(-1, 1) + 1.0) / 2.0
    array = (tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    from PIL import Image

    return Image.fromarray(array)


def vision_hidden_size(extractor: SigLIP2FeatureExtractor) -> int:
    config = extractor.model.config
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "vision_config") and hasattr(config.vision_config, "hidden_size"):
        return int(config.vision_config.hidden_size)
    raise ValueError("Could not determine VLM hidden size.")


def correspondence_grid_hw(args: argparse.Namespace) -> tuple[int, int]:
    grid_size = args.grid_size or max(1, args.resolution // 16)
    return (grid_size, grid_size)


def ramp_value(step: int, start_step: int, ramp_steps: int) -> float:
    if step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (step - start_step) / ramp_steps)))


def cig_enabled(args: argparse.Namespace) -> bool:
    return bool(args.use_cig) and not bool(args.disable_cig)


def depth_enabled(args: argparse.Namespace) -> bool:
    return bool(args.use_depth) and not bool(args.disable_depth)


def segmentation_enabled(args: argparse.Namespace) -> bool:
    return bool(args.use_segmentation) and not bool(args.disable_segmentation)


def panoptic_enabled(args: argparse.Namespace) -> bool:
    return bool(args.use_panoptic) and not bool(args.disable_panoptic)


def vae_scale_factor(vae) -> int:
    block_out_channels = getattr(vae.config, "block_out_channels", None)
    if block_out_channels:
        return 2 ** (len(block_out_channels) - 1)
    return 8


def downsample_latents_for_color_decode(
    latents: torch.Tensor,
    decode_resolution: int,
    scale_factor: int,
) -> torch.Tensor:
    if decode_resolution <= 0:
        return latents
    target_h = max(1, decode_resolution // scale_factor)
    target_w = max(1, decode_resolution // scale_factor)
    target_h = min(target_h, latents.shape[-2])
    target_w = min(target_w, latents.shape[-1])
    if (target_h, target_w) == tuple(latents.shape[-2:]):
        return latents
    return F.interpolate(latents.float(), size=(target_h, target_w), mode="bilinear", align_corners=False).to(
        latents.dtype
    )


def compute_color_chroma_loss(
    vae,
    pred_x0_latents: torch.Tensor,
    target: torch.Tensor,
    sigmas: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    pred_x0_for_color = downsample_latents_for_color_decode(
        pred_x0_latents,
        args.color_decode_resolution,
        vae_scale_factor(vae),
    )
    vae_dtype = next(vae.parameters(), pred_x0_for_color).dtype
    pred_x0_for_color = pred_x0_for_color.to(dtype=vae_dtype)
    pred_rgb = decode_vae_latents(vae, pred_x0_for_color)
    target_rgb = F.interpolate(target.float(), size=pred_rgb.shape[-2:], mode="bilinear", align_corners=False).to(
        pred_rgb.dtype
    )

    pred_ab = rgb_to_lab(pred_rgb)[:, 1:3] / 128.0
    target_ab = rgb_to_lab(target_rgb)[:, 1:3] / 128.0
    per_sample = F.smooth_l1_loss(pred_ab, target_ab, reduction="none").mean(dim=(1, 2, 3))
    sigma_flat = sigmas.reshape(sigmas.shape[0], -1)[:, 0].float()
    sigma_weight = (1.0 - sigma_flat).clamp(0, 1).pow(args.color_loss_sigma_power).to(per_sample.device)
    return (per_sample * sigma_weight).mean()


def corr_state_from_raw_states(
    raw_states: list[dict[str, np.ndarray | torch.Tensor | tuple[int, int]]],
    device: torch.device,
    dtype: torch.dtype,
):
    int_keys = {"topk_idx", "region_topm_idx", "label_c", "label_r"}
    batch = {}
    for key in ("topk_idx", "topk_weight", "corr_conf", "region_topm_idx", "region_topm_weight", "label_c", "label_r"):
        values = []
        for state in raw_states:
            value = state[key]
            if torch.is_tensor(value):
                target_dtype = torch.long if key in int_keys else torch.float32
                values.append(value.to(device=device, dtype=target_dtype))
            elif key in int_keys:
                values.append(torch.from_numpy(np.asarray(value).astype(np.int64, copy=False)).to(device=device))
            else:
                values.append(torch.from_numpy(np.asarray(value).astype(np.float32, copy=False)).to(device=device))
        batch[key] = torch.stack(values, dim=0)
    batch["grid_hw"] = raw_states[0]["grid_hw"]
    return corr_state_from_batch(batch, device=device, dtype=dtype)


@torch.no_grad()
def compute_online_corr_state(
    dino_extractor: DinoV2FeatureExtractor,
    vlm_extractor: SigLIP2FeatureExtractor,
    cleandift_extractor: CleanDIFTFeatureExtractor | None,
    content_images: list,
    reference_images: list,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
):
    grid_hw = correspondence_grid_hw(args)
    f_c_dino = dino_extractor(content_images, grid_hw)
    f_r_dino = dino_extractor(reference_images, grid_hw)
    f_c_vlm = vlm_extractor(content_images, grid_hw)
    f_r_vlm = vlm_extractor(reference_images, grid_hw)
    f_c_clean = None
    f_r_clean = None
    if cleandift_extractor is not None:
        f_c_clean = cleandift_extractor(content_images, grid_hw)
        f_r_clean = cleandift_extractor(reference_images, grid_hw)
    raw_states = [
        compute_corr_state(
            f_c_dino[index],
            f_r_dino[index],
            f_c_vlm[index],
            f_r_vlm[index],
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
            f_c_clean=f_c_clean[index] if f_c_clean is not None else None,
            f_r_clean=f_r_clean[index] if f_r_clean is not None else None,
            return_tensors=True,
        )
        for index in range(len(content_images))
    ]
    return (
        corr_state_from_raw_states(raw_states, device=device, dtype=dtype),
        f_r_vlm.to(device=device, dtype=dtype),
    )


def validate_checkpoint_state(module: torch.nn.Module, state: dict[str, torch.Tensor], label: str, strict: bool) -> None:
    current = module.state_dict()
    unexpected = sorted(k for k in state if k not in current)
    mismatched = sorted(
        f"{k}: checkpoint {tuple(v.shape)} != current {tuple(current[k].shape)}"
        for k, v in state.items()
        if k in current and tuple(v.shape) != tuple(current[k].shape)
    )
    missing = sorted(k for k in current if k not in state) if strict else []
    if unexpected or mismatched or missing:
        lines = [f"Cannot load {label} from checkpoint cleanly."]
        if unexpected:
            lines.append(f"Unexpected keys: {unexpected[:10]}")
        if mismatched:
            lines.append(f"Shape mismatches: {mismatched[:10]}")
        if missing:
            lines.append(f"Missing keys: {missing[:10]}")
        raise RuntimeError("\n".join(lines))


_LEGACY_INACTIVE_PROCESSOR_SUFFIXES = (
    ".g_region",
    ".g_sparse",
    ".to_k_region.weight",
    ".to_v_region.weight",
    ".to_k_local.weight",
    ".to_v_local.weight",
)


def filter_legacy_inactive_processor_state(
    module: torch.nn.Module,
    state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    current = module.state_dict()
    filtered = {}
    ignored = []
    for key, value in state.items():
        if (
            key not in current
            and ".processor." in key
            and key.endswith(_LEGACY_INACTIVE_PROCESSOR_SUFFIXES)
        ):
            ignored.append(key)
            continue
        filtered[key] = value
    return filtered, sorted(ignored)


def load_stage1_checkpoint(
    transformer,
    reference_adapter,
    intrinsic_prior_tokenizer,
    checkpoint_path: str,
) -> dict:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"final model requires a stage-1 checkpoint, but it was not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "transformer_trainable" not in checkpoint:
        raise KeyError(f"Checkpoint is missing transformer_trainable: {ckpt_path}")
    if "reference_adapter" not in checkpoint:
        raise KeyError(f"Checkpoint is missing reference_adapter: {ckpt_path}")

    transformer_state = checkpoint["transformer_trainable"]
    adapter_state = checkpoint["reference_adapter"]
    transformer_state, ignored_legacy_keys = filter_legacy_inactive_processor_state(transformer, transformer_state)
    validate_checkpoint_state(transformer, transformer_state, "stage-1 transformer trainable weights", strict=False)
    validate_checkpoint_state(reference_adapter, adapter_state, "stage-1 reference adapter", strict=True)
    transformer.load_state_dict(transformer_state, strict=False)
    reference_adapter.load_state_dict(adapter_state, strict=True)

    if "intrinsic_prior_tokenizer" in checkpoint:
        intrinsic_prior_tokenizer.load_state_dict(checkpoint["intrinsic_prior_tokenizer"], strict=False)
    if "corr_reference_processors" in checkpoint:
        load_corr_reference_processor_state_dict(
            transformer,
            checkpoint.get("corr_reference_processors"),
            strict=False,
        )

    source_args = checkpoint.get("args", {})
    source_step = checkpoint.get("step", "?")
    source_output = source_args.get("output_dir", "")
    suffix = f" from {source_output}" if source_output else ""
    if "intrinsic_prior_tokenizer" in checkpoint or "corr_reference_processors" in checkpoint:
        print(f"initialized final model from CIG checkpoint step={source_step}{suffix}: {ckpt_path}", flush=True)
    else:
        print(
            f"initialized final model from stage-1 checkpoint step={source_step}{suffix}: {ckpt_path}; "
            "new CIG parameters start as a no-op",
            flush=True,
        )
    if ignored_legacy_keys:
        print(
            "ignored legacy inactive processor keys from checkpoint: "
            f"{len(ignored_legacy_keys)} keys; first={ignored_legacy_keys[:3]}",
            flush=True,
        )
    return source_args


def save_final_checkpoint(
    accelerator: Accelerator,
    transformer,
    reference_adapter,
    intrinsic_prior_tokenizer,
    args,
    step: int,
) -> Path | None:
    if not accelerator.is_main_process:
        return None
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_transformer = accelerator.unwrap_model(transformer)
    unwrapped_adapter = accelerator.unwrap_model(reference_adapter)
    unwrapped_intrinsic = accelerator.unwrap_model(intrinsic_prior_tokenizer)

    trainable_names = {name for name, param in unwrapped_transformer.named_parameters() if param.requires_grad}
    transformer_state = {
        name: tensor.detach().cpu()
        for name, tensor in unwrapped_transformer.state_dict().items()
        if any(name == train_name or name.startswith(train_name + ".") for train_name in trainable_names)
        or name.startswith("pos_embed.proj.")
    }
    torch.save(
        {
            "step": step,
            "args": vars(args),
            "reference_adapter": unwrapped_adapter.state_dict(),
            "intrinsic_prior_tokenizer": unwrapped_intrinsic.state_dict(),
            "corr_reference_processors": corr_reference_processor_state_dict(unwrapped_transformer),
            "transformer_trainable": transformer_state,
        },
        ckpt_dir / "color_edit_final.pt",
    )
    with (ckpt_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    return ckpt_dir


@torch.no_grad()
def save_checkpoint_validation_sample(
    accelerator: Accelerator,
    vae,
    transformer,
    reference_adapter,
    intrinsic_prior_tokenizer,
    content_intrinsic_extractor: ContentStructureExtractor,
    depth_extractor: DepthAnythingV2FeatureExtractor | None,
    segmentation_extractor: SegFormerSemanticStructureExtractor | None,
    panoptic_extractor: Mask2FormerPanopticStructureExtractor | None,
    vlm_extractor: SigLIP2FeatureExtractor,
    dino_extractor: DinoV2FeatureExtractor,
    cleandift_extractor: CleanDIFTFeatureExtractor | None,
    args,
    weight_dtype: torch.dtype,
    ckpt_dir: Path | None,
) -> None:
    if ckpt_dir is None or args.disable_checkpoint_validation:
        return
    if not accelerator.is_main_process:
        return

    unwrapped_transformer = accelerator.unwrap_model(transformer)
    unwrapped_adapter = accelerator.unwrap_model(reference_adapter)
    unwrapped_intrinsic = accelerator.unwrap_model(intrinsic_prior_tokenizer)
    transformer_was_training = unwrapped_transformer.training
    adapter_was_training = unwrapped_adapter.training
    intrinsic_was_training = unwrapped_intrinsic.training
    unwrapped_transformer.eval()
    unwrapped_adapter.eval()
    unwrapped_intrinsic.eval()

    device = accelerator.device
    content_pil = resize_pil_square(load_rgb(args.validation_content_image), args.resolution)
    reference_pil = resize_pil_square(load_rgb(args.validation_reference_image), args.resolution)
    content = resize_to_tensor(content_pil, args.resolution).unsqueeze(0).to(device=device, dtype=weight_dtype)
    reference = resize_to_tensor(reference_pil, args.resolution).unsqueeze(0).to(device=device, dtype=weight_dtype)

    cache_path = Path(args.validation_corr_cache) if args.validation_corr_cache else None
    if cache_path is not None and cache_path.exists():
        corr_raw = load_corr_npz(cache_path)
        batch_corr = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in corr_raw.items()}
        corr_state = corr_state_from_batch(batch_corr, device=device, dtype=weight_dtype)
        siglip_ref_tokens = vlm_extractor([reference_pil], corr_state.grid_hw).to(device=device, dtype=weight_dtype)
    else:
        if cache_path is not None:
            print(f"validation_corr_cache not found; computing online: {cache_path}", flush=True)
        corr_state, siglip_ref_tokens = compute_online_corr_state(
            dino_extractor,
            vlm_extractor,
            cleandift_extractor,
            [content_pil],
            [reference_pil],
            args,
            device=device,
            dtype=weight_dtype,
        )

    content_latents = encode_vae_latents(vae, content)
    ref_global_tokens, ref_region_tokens, ref_local_tokens = unwrapped_adapter(
        reference,
        siglip_ref_tokens,
        corr_state.label_r,
        corr_state.grid_hw,
    )
    intrinsic_masks = protect_mask = additive_anchor = None
    if cig_enabled(args):
        depth_map = depth_extractor([content_pil], content.shape[-2:], dtype=weight_dtype) if depth_extractor is not None else None
        semantic_state = (
            segmentation_extractor([content_pil], content.shape[-2:], dtype=weight_dtype)
            if segmentation_extractor is not None
            else None
        )
        panoptic_state = (
            panoptic_extractor([content_pil], content.shape[-2:], dtype=weight_dtype)
            if panoptic_extractor is not None
            else None
        )
        intrinsic_raw = content_intrinsic_extractor(
            content,
            label_c=None,
            grid_hw=corr_state.grid_hw,
            depth_map=depth_map,
            semantic_state=semantic_state,
            panoptic_state=panoptic_state,
            content_dino=None,
        )
        xy = xy_grid(corr_state.grid_hw, device=device, dtype=weight_dtype).unsqueeze(0)
        intrinsic_masks, protect_mask, additive_anchor = unwrapped_intrinsic(
            intrinsic_raw.patch_stats,
            intrinsic_raw.heuristic_priors,
            xy=xy,
        )

    generator = torch.Generator(device=device).manual_seed(args.validation_seed)
    latents = torch.randn(content_latents.shape, generator=generator, device=device, dtype=weight_dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    scheduler.set_timesteps(args.validation_num_inference_steps, device=device)

    prompt_embeds, pooled_prompt_embeds = zero_prompt_embeds(
        batch_size=1,
        sequence_length=333,
        joint_dim=unwrapped_transformer.config.joint_attention_dim,
        pooled_dim=unwrapped_transformer.config.pooled_projection_dim,
        device=device,
        dtype=weight_dtype,
    )

    for step_index, timestep in enumerate(scheduler.timesteps):
        model_input = torch.cat([latents, content_latents], dim=1)
        sigma = scheduler.sigmas[step_index].to(device=device, dtype=weight_dtype).view(1)
        model_pred = unwrapped_transformer(
            hidden_states=model_input,
            timestep=timestep.expand(1),
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs={
                "ref_global_tokens": ref_global_tokens,
                "ref_region_tokens": ref_region_tokens,
                "ref_local_tokens": ref_local_tokens,
                "region_topm_idx": corr_state.region_topm_idx,
                "region_topm_weight": corr_state.region_topm_weight,
                "topk_idx": corr_state.topk_idx,
                "topk_weight": corr_state.topk_weight,
                "corr_conf": corr_state.corr_conf,
                "ref_scale": args.reference_scale,
                "corr_scale": args.corr_scale,
                "corr_time_weight": get_corr_time_weight(sigma).to(device=device, dtype=weight_dtype),
                "intrinsic_masks": intrinsic_masks,
                "protect_mask": protect_mask,
                "additive_anchor": additive_anchor,
                "sigma": sigma,
                "cig_enabled": cig_enabled(args),
                "cig_key_bias_scale": args.cig_key_bias_scale,
                "cig_anchor_scale": args.cig_anchor_scale,
                "cig_ref_protect_scale": args.cig_ref_protect_scale,
                "cig_train_ramp": 1.0,
                "anchor_train_ramp": 1.0,
            },
            return_dict=False,
        )[0]
        latents = scheduler.step(model_pred, timestep, latents, return_dict=False)[0].to(weight_dtype)

    image = tensor_to_image(decode_vae_latents(vae, latents.to(weight_dtype)))
    image.save(ckpt_dir / "validation_sample.png")
    cig_metrics = cig_processor_magnitudes(unwrapped_transformer)
    metrics = {
        "cig_enabled": cig_enabled(args),
        "cig_key_bias_gate": cig_metrics["key_bias"],
        "cig_anchor_gate": cig_metrics["anchor"],
        "cig_ref_protect_gate": cig_metrics["ref_protect"],
        "protect_mask_mean": float(protect_mask.detach().float().mean().cpu()) if protect_mask is not None else 0.0,
        "additive_anchor_norm": float(additive_anchor.detach().float().norm(dim=-1).mean().cpu())
        if additive_anchor is not None
        else 0.0,
    }
    with (ckpt_dir / "validation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if transformer_was_training:
        unwrapped_transformer.train()
    if adapter_was_training:
        unwrapped_adapter.train()
    if intrinsic_was_training:
        unwrapped_intrinsic.train()


def main() -> None:
    args = parse_args()
    logging_dir = Path(args.output_dir) / "logs"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_dir)),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model_path = Path(args.pretrained_model_name_or_path)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    noise_scheduler_copy.set_timesteps(1000, device=accelerator.device)

    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=weight_dtype)
    transformer = SD3Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=weight_dtype)
    expand_sd3_input_channels(transformer, new_in_channels=32)

    dino_extractor = DinoV2FeatureExtractor(args.dino_model, device=accelerator.device, dtype=weight_dtype)
    vlm_extractor = SigLIP2FeatureExtractor(args.vlm_model, device=accelerator.device, dtype=weight_dtype)
    cleandift_extractor = None
    if not args.disable_cleandift:
        cleandift_extractor = CleanDIFTFeatureExtractor(
            args.cleandift_unet,
            vae_model_name_or_path=args.cleandift_vae,
            feature_key=args.cleandift_feature_key,
            timestep=args.cleandift_timestep,
            device=accelerator.device,
            dtype=weight_dtype,
            use_text_encoder=args.cleandift_use_text_encoder,
        )
    depth_extractor = None
    if cig_enabled(args) and depth_enabled(args):
        depth_extractor = DepthAnythingV2FeatureExtractor(
            args.depth_model,
            device=accelerator.device,
            dtype=weight_dtype,
        )
    segmentation_extractor = None
    if cig_enabled(args) and segmentation_enabled(args):
        segmentation_extractor = SegFormerSemanticStructureExtractor(
            args.segmentation_model,
            device=accelerator.device,
            dtype=weight_dtype,
        )
    panoptic_extractor = None
    if cig_enabled(args) and panoptic_enabled(args):
        panoptic_extractor = Mask2FormerPanopticStructureExtractor(
            args.panoptic_model,
            device=accelerator.device,
            dtype=weight_dtype,
        )
    hidden_size = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    reference_adapter = CorrespondenceGuidedReferenceAdapter(
        siglip_dim=vision_hidden_size(vlm_extractor),
        hidden_dim=hidden_size,
        num_global_tokens=args.num_global_tokens,
        num_regions=args.num_regions,
        resampler_depth=args.reference_resampler_depth,
        resampler_heads=args.reference_resampler_heads,
        resampler_dim_head=args.reference_resampler_dim_head,
    )
    content_intrinsic_extractor = ContentStructureExtractor()
    intrinsic_prior_tokenizer = IntrinsicPriorTokenizer(
        patch_stat_dim=ContentStructureExtractor.patch_stat_dim,
        sd3_hidden_dim=hidden_size,
        hidden_dim=args.cig_hidden_dim or hidden_size,
        num_mask_types=args.cig_num_masks,
    )
    if not cig_enabled(args):
        intrinsic_prior_tokenizer.requires_grad_(False)

    inject_corr_reference_attention(
        transformer,
        reference_hidden_size=hidden_size,
        local_start_layer=args.local_start_layer,
        sparse_start_layer=args.sparse_start_layer,
        num_intrinsic_masks=args.cig_num_masks,
        max_key_bias=args.cig_max_key_bias,
        max_anchor_scale=args.cig_max_anchor_scale,
        max_local_protect=args.cig_max_local_protect,
        max_region_protect=args.cig_max_region_protect,
    )
    add_sd3_transformer_lora(
        transformer,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        layers=args.lora_layers,
        blocks=args.lora_blocks,
    )
    load_stage1_checkpoint(
        transformer,
        reference_adapter,
        intrinsic_prior_tokenizer,
        args.init_from_stage1_checkpoint,
    )
    freeze_transformer_for_adapter_training(transformer)
    set_corr_reference_processor_trainable(transformer, enable_cig=cig_enabled(args))

    if args.gradient_checkpointing and hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()

    vae.requires_grad_(False)
    vae.eval()
    transformer.train()
    reference_adapter.train()
    intrinsic_prior_tokenizer.train()

    optimizer = torch.optim.AdamW(
        trainable_parameters(transformer, reference_adapter, intrinsic_prior_tokenizer),
        lr=args.learning_rate,
        foreach=args.adamw_foreach,
    )
    dataset = ColorEditJsonlDataset(args.train_jsonl, args.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    transformer, reference_adapter, intrinsic_prior_tokenizer, optimizer, dataloader = accelerator.prepare(
        transformer, reference_adapter, intrinsic_prior_tokenizer, optimizer, dataloader
    )
    vae.to(accelerator.device, dtype=weight_dtype)

    global_step = 0
    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(transformer, reference_adapter, intrinsic_prior_tokenizer):
                content = batch["content"].to(device=accelerator.device, dtype=weight_dtype)
                target = batch["target"].to(device=accelerator.device, dtype=weight_dtype)
                reference = batch["reference"].to(device=accelerator.device, dtype=weight_dtype)
                content_images = [
                    resize_pil_square(load_rgb(path), args.resolution) for path in batch["content_image_path"]
                ]
                reference_images = [
                    resize_pil_square(load_rgb(path), args.resolution) for path in batch["reference_image_path"]
                ]
                corr_state, siglip_ref_tokens = compute_online_corr_state(
                    dino_extractor,
                    vlm_extractor,
                    cleandift_extractor,
                    content_images,
                    reference_images,
                    args,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )

                with torch.no_grad():
                    target_latents = encode_vae_latents(vae, target)
                    content_latents = encode_vae_latents(vae, content)

                noise = torch.randn_like(target_latents)
                batch_size = target_latents.shape[0]
                u = torch.rand(batch_size, device=target_latents.device)
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=target_latents.device)
                sigmas = get_sigmas(
                    noise_scheduler_copy,
                    timesteps,
                    n_dim=target_latents.ndim,
                    dtype=target_latents.dtype,
                    device=target_latents.device,
                )
                noisy_target_latents = (1.0 - sigmas) * target_latents + sigmas * noise
                model_input = torch.cat([noisy_target_latents, content_latents], dim=1)

                ref_global_tokens, ref_region_tokens, ref_local_tokens = reference_adapter(
                    reference,
                    siglip_ref_tokens,
                    corr_state.label_r,
                    corr_state.grid_hw,
                )

                if args.reference_dropout > 0:
                    keep = (torch.rand(batch_size, device=ref_global_tokens.device) >= args.reference_dropout).view(
                        batch_size, 1, 1
                    )
                    ref_global_tokens = ref_global_tokens * keep
                    ref_region_tokens = ref_region_tokens * keep
                    ref_local_tokens = ref_local_tokens * keep

                intrinsic_masks = protect_mask = additive_anchor = None
                if cig_enabled(args):
                    depth_map = (
                        depth_extractor(content_images, content.shape[-2:], dtype=weight_dtype)
                        if depth_extractor is not None
                        else None
                    )
                    semantic_state = (
                        segmentation_extractor(content_images, content.shape[-2:], dtype=weight_dtype)
                        if segmentation_extractor is not None
                        else None
                    )
                    panoptic_state = (
                        panoptic_extractor(content_images, content.shape[-2:], dtype=weight_dtype)
                        if panoptic_extractor is not None
                        else None
                    )
                    intrinsic_raw = content_intrinsic_extractor(
                        content,
                        label_c=None,
                        grid_hw=corr_state.grid_hw,
                        depth_map=depth_map,
                        semantic_state=semantic_state,
                        panoptic_state=panoptic_state,
                        content_dino=None,
                    )
                    xy = xy_grid(corr_state.grid_hw, device=target_latents.device, dtype=weight_dtype)
                    xy = xy.unsqueeze(0).expand(batch_size, -1, -1)
                    intrinsic_masks, protect_mask, additive_anchor = intrinsic_prior_tokenizer(
                        intrinsic_raw.patch_stats,
                        intrinsic_raw.heuristic_priors,
                        xy=xy,
                    )

                corr_time_weight = get_corr_time_weight(sigmas[:, 0, 0, 0]).to(
                    device=target_latents.device, dtype=weight_dtype
                )
                if args.corr_dropout > 0:
                    corr_keep = (torch.rand(batch_size, device=target_latents.device) >= args.corr_dropout).to(
                        dtype=weight_dtype
                    )
                    corr_time_weight = corr_time_weight * corr_keep
                warmup = 1.0 if args.corr_warmup_steps <= 0 else min(1.0, global_step / args.corr_warmup_steps)
                corr_scale = args.corr_scale * warmup
                cig_train_ramp = ramp_value(global_step, args.cig_runtime_start_step, args.cig_runtime_ramp_steps)
                anchor_train_ramp = ramp_value(global_step, args.anchor_start_step, args.anchor_ramp_steps)

                prompt_embeds, pooled_prompt_embeds = zero_prompt_embeds(
                    batch_size=batch_size,
                    sequence_length=333,
                    joint_dim=transformer.module.config.joint_attention_dim
                    if hasattr(transformer, "module")
                    else transformer.config.joint_attention_dim,
                    pooled_dim=transformer.module.config.pooled_projection_dim
                    if hasattr(transformer, "module")
                    else transformer.config.pooled_projection_dim,
                    device=target_latents.device,
                    dtype=weight_dtype,
                )

                model_pred = transformer(
                    hidden_states=model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs={
                        "ref_global_tokens": ref_global_tokens,
                        "ref_region_tokens": ref_region_tokens,
                        "ref_local_tokens": ref_local_tokens,
                        "region_topm_idx": corr_state.region_topm_idx,
                        "region_topm_weight": corr_state.region_topm_weight,
                        "topk_idx": corr_state.topk_idx,
                        "topk_weight": corr_state.topk_weight,
                        "corr_conf": corr_state.corr_conf,
                        "ref_scale": args.reference_scale,
                        "corr_scale": corr_scale,
                        "corr_time_weight": corr_time_weight,
                        "intrinsic_masks": intrinsic_masks,
                        "protect_mask": protect_mask,
                        "additive_anchor": additive_anchor,
                        "sigma": sigmas[:, 0, 0, 0],
                        "cig_enabled": cig_enabled(args),
                        "cig_key_bias_scale": args.cig_key_bias_scale,
                        "cig_anchor_scale": args.cig_anchor_scale,
                        "cig_ref_protect_scale": args.cig_ref_protect_scale,
                        "cig_train_ramp": cig_train_ramp,
                        "anchor_train_ramp": anchor_train_ramp,
                    },
                    return_dict=False,
                )[0]

                target_flow = noise - target_latents
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                flow_loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target_flow.float()) ** 2).reshape(batch_size, -1),
                    dim=1,
                ).mean()
                loss = flow_loss
                color_loss = flow_loss.new_zeros(())
                if args.use_color_loss and args.color_loss_weight > 0:
                    pred_x0_latents = noisy_target_latents - sigmas * model_pred
                    color_loss = compute_color_chroma_loss(vae, pred_x0_latents, target, sigmas, args)
                    loss = loss + args.color_loss_weight * color_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params = trainable_parameters(transformer, reference_adapter, intrinsic_prior_tokenizer)
                    accelerator.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    cig_metrics = cig_processor_magnitudes(accelerator.unwrap_model(transformer))
                    print(
                        f"step={global_step} "
                        f"loss={loss.detach().float().item():.6f} "
                        f"flow={flow_loss.detach().float().item():.6f} "
                        f"color={color_loss.detach().float().item():.6f} "
                        f"cig_ramp={cig_train_ramp:.3f} "
                        f"anchor_ramp={anchor_train_ramp:.3f} "
                        f"cig_key={cig_metrics['key_bias']:.6f} "
                        f"cig_protect={cig_metrics['ref_protect']:.6f} "
                        f"cig_anchor={cig_metrics['anchor']:.6f}",
                        flush=True,
                    )
                if global_step % args.checkpointing_steps == 0:
                    ckpt_dir = save_final_checkpoint(
                        accelerator,
                        transformer,
                        reference_adapter,
                        intrinsic_prior_tokenizer,
                        args,
                        global_step,
                    )
                    save_checkpoint_validation_sample(
                        accelerator,
                        vae,
                        transformer,
                        reference_adapter,
                        intrinsic_prior_tokenizer,
                        content_intrinsic_extractor,
                        depth_extractor,
                        segmentation_extractor,
                        panoptic_extractor,
                        vlm_extractor,
                        dino_extractor,
                        cleandift_extractor,
                        args,
                        weight_dtype,
                        ckpt_dir,
                    )
                    accelerator.wait_for_everyone()
                if global_step >= args.max_train_steps:
                    break

    ckpt_dir = save_final_checkpoint(
        accelerator,
        transformer,
        reference_adapter,
        intrinsic_prior_tokenizer,
        args,
        global_step,
    )
    save_checkpoint_validation_sample(
        accelerator,
        vae,
        transformer,
        reference_adapter,
        intrinsic_prior_tokenizer,
        content_intrinsic_extractor,
        depth_extractor,
        segmentation_extractor,
        panoptic_extractor,
        vlm_extractor,
        dino_extractor,
        cleandift_extractor,
        args,
        weight_dtype,
        ckpt_dir,
    )
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
