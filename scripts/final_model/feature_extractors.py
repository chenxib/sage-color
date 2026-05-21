from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL as DiffusersAutoencoderKL
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoImageProcessor, AutoModel

from scripts.final_model.cleandift_min_sd21 import SD21UNetModel


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x.float(), dim=dim, eps=1e-6)


def resize_pil_square(image: Image.Image, resolution: int) -> Image.Image:
    return image.convert("RGB").resize((resolution, resolution), Image.Resampling.BICUBIC)


def tokens_to_grid(tokens: torch.Tensor, target_grid: tuple[int, int]) -> torch.Tensor:
    bsz, num_tokens, channels = tokens.shape
    target_h, target_w = target_grid
    target_n = target_h * target_w
    if num_tokens == target_n:
        return tokens
    if num_tokens == target_n + 1:
        return tokens[:, 1:]
    if num_tokens > 1:
        side_minus_cls = int(math.sqrt(num_tokens - 1))
        if side_minus_cls * side_minus_cls == num_tokens - 1:
            tokens = tokens[:, 1:]
            num_tokens -= 1
    side = int(math.sqrt(num_tokens))
    if side * side == num_tokens:
        grid = tokens.transpose(1, 2).reshape(bsz, channels, side, side)
        grid = F.interpolate(grid.float(), size=target_grid, mode="bilinear", align_corners=False)
        return grid.flatten(2).transpose(1, 2).to(tokens.dtype)
    if num_tokens >= target_n:
        return tokens[:, :target_n]
    return F.pad(tokens, (0, 0, 0, target_n - num_tokens))


def _processor_call(processor, images: list[Image.Image]):
    try:
        return processor(
            images=images,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
    except TypeError:
        return processor(images=images, return_tensors="pt")


def _image_hw(image: Image.Image | np.ndarray | torch.Tensor) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        width, height = image.size
        return height, width
    if torch.is_tensor(image):
        shape = tuple(image.shape)
    else:
        shape = np.asarray(image).shape
    if len(shape) < 2:
        raise ValueError(f"Cannot infer image size from shape {shape}")
    if len(shape) == 3 and shape[0] in (1, 3, 4):
        return int(shape[1]), int(shape[2])
    return int(shape[0]), int(shape[1])


def _siglip2_max_num_patches(processor, images: list[Image.Image]) -> int:
    patch_size = int(getattr(processor, "patch_size", 16))
    max_num_patches = int(getattr(processor, "max_num_patches", 0) or 0)
    for image in images:
        height, width = _image_hw(image)
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"SigLIP2 NaFlex input size {(height, width)} must be divisible by patch_size={patch_size} "
                "when do_resize=False."
            )
        max_num_patches = max(max_num_patches, (height // patch_size) * (width // patch_size))
    return max_num_patches


def _ensure_siglip2_patch_metadata(inputs):
    if "pixel_values" not in inputs or "spatial_shapes" not in inputs:
        return inputs
    pixel_values = inputs["pixel_values"]
    spatial_shapes = inputs["spatial_shapes"]
    if not torch.is_tensor(pixel_values) or not torch.is_tensor(spatial_shapes):
        return inputs

    batch_size, num_patches = pixel_values.shape[:2]
    mask = inputs.get("pixel_attention_mask")
    if torch.is_tensor(mask) and tuple(mask.shape) == (batch_size, num_patches):
        return inputs

    new_mask = torch.zeros(batch_size, num_patches, dtype=torch.int32)
    spatial_shapes_cpu = spatial_shapes.detach().cpu()
    for index in range(batch_size):
        valid = int(spatial_shapes_cpu[index, 0].item() * spatial_shapes_cpu[index, 1].item())
        new_mask[index, : min(valid, num_patches)] = 1
    inputs["pixel_attention_mask"] = new_mask
    return inputs


def _siglip2_processor_call(processor, images: list[Image.Image]):
    max_num_patches = _siglip2_max_num_patches(processor, images)
    try:
        inputs = processor(
            images=images,
            return_tensors="pt",
            do_resize=False,
            max_num_patches=max_num_patches,
        )
    except TypeError:
        inputs = processor(images=images, return_tensors="pt")
    return _ensure_siglip2_patch_metadata(inputs)


def _move_inputs(inputs, device: torch.device, dtype: torch.dtype | None = None):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if value.is_floating_point() and dtype is not None:
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


class DinoV2FeatureExtractor:
    def __init__(
        self,
        model_name_or_path: str | Path = "model/dinov2-large",
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name_or_path, use_fast=False)
        except TypeError:
            self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        try:
            self.model = AutoModel.from_pretrained(model_name_or_path, dtype=dtype).to(self.device)
        except TypeError:
            self.model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=dtype).to(self.device)
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images: list[Image.Image], grid_hw: tuple[int, int]) -> torch.Tensor:
        inputs = _move_inputs(_processor_call(self.processor, images), self.device, self.dtype)
        try:
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                interpolate_pos_encoding=True,
                return_dict=True,
            )
        except TypeError:
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states[-4:]
        patch_tokens = []
        for hidden in hidden_states:
            patch_tokens.append(tokens_to_grid(hidden, grid_hw))
        tokens = torch.stack(patch_tokens, dim=0).mean(dim=0)
        return l2_normalize(tokens, dim=-1)


class SigLIP2FeatureExtractor:
    def __init__(
        self,
        model_name_or_path: str | Path = "model/siglip2-so400m-patch16-naflex",
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name_or_path, use_fast=False)
        except TypeError:
            self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        try:
            from transformers import Siglip2VisionModel

            try:
                self.model = Siglip2VisionModel.from_pretrained(model_name_or_path, dtype=dtype)
            except TypeError:
                self.model = Siglip2VisionModel.from_pretrained(model_name_or_path, torch_dtype=dtype)
        except Exception:
            try:
                self.model = AutoModel.from_pretrained(model_name_or_path, dtype=dtype)
            except TypeError:
                self.model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images: list[Image.Image], grid_hw: tuple[int, int]) -> torch.Tensor:
        inputs = _move_inputs(_siglip2_processor_call(self.processor, images), self.device, self.dtype)
        try:
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        except TypeError:
            pixel_values = inputs["pixel_values"]
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None and hasattr(outputs, "vision_model_output"):
            hidden_states = outputs.vision_model_output.hidden_states
        if hidden_states is None:
            raise RuntimeError("Could not find SigLIP2 hidden states in model output.")
        tokens = hidden_states[-2]
        return l2_normalize(tokens_to_grid(tokens, grid_hw), dim=-1)


class SD21UNetFeatureExtractor(SD21UNetModel):
    """CleanDIFT SD2.1 UNet forward that returns intermediate feature maps.

    This mirrors the official CompVis/CleanDIFT `SD21UNetFeatureExtractor`
    notebook helper while keeping final model self-contained.
    """

    def forward(self, sample, timesteps, encoder_hidden_states, added_cond_kwargs=None, **kwargs):
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)
        s0 = sample
        sample, [s1, s2, s3] = self.down_blocks[0](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )
        sample, [s4, s5, s6] = self.down_blocks[1](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )
        sample, [s7, s8, s9] = self.down_blocks[2](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )
        sample, [s10, s11] = self.down_blocks[3](
            sample,
            temb=emb,
        )

        sample_mid = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        _, [us1, us2, us3] = self.up_blocks[0](
            hidden_states=sample_mid,
            temb=emb,
            res_hidden_states_tuple=[s9, s10, s11],
        )
        _, [us4, us5, us6] = self.up_blocks[1](
            hidden_states=us3,
            temb=emb,
            res_hidden_states_tuple=[s6, s7, s8],
            encoder_hidden_states=encoder_hidden_states,
        )
        _, [us7, us8, us9] = self.up_blocks[2](
            hidden_states=us6,
            temb=emb,
            res_hidden_states_tuple=[s3, s4, s5],
            encoder_hidden_states=encoder_hidden_states,
        )
        _, [us10, us11, _] = self.up_blocks[3](
            hidden_states=us9,
            temb=emb,
            res_hidden_states_tuple=[s0, s1, s2],
            encoder_hidden_states=encoder_hidden_states,
        )

        return {
            "mid": sample_mid,
            "us1": us1,
            "us2": us2,
            "us3": us3,
            "us4": us4,
            "us5": us5,
            "us6": us6,
            "us7": us7,
            "us8": us8,
            "us9": us9,
            "us10": us10,
        }


def _images_to_tensor(images: list[Image.Image], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    arrays = []
    for image in images:
        rgb = image.convert("RGB")
        tensor = torch.from_numpy(np.asarray(rgb, dtype="float32") / 127.5 - 1.0)
        arrays.append(tensor.permute(2, 0, 1))
    return torch.stack(arrays, dim=0).to(device=device, dtype=dtype)


class CleanDIFTFeatureExtractor:
    """CleanDIFT SD2.1 feature extractor for final model dense correspondence.

    The default `feature_key=us6` follows the official CleanDIFT semantic
    correspondence notebook. Images are encoded with an SD2.1 VAE, then passed
    through the CleanDIFT UNet checkpoint on clean latents.
    """

    def __init__(
        self,
        unet_path: str | Path = "model/cleandift/cleandift_sd21_unet.safetensors",
        vae_model_name_or_path: str | Path = "stabilityai/sd-vae-ft-mse",
        feature_key: str = "us6",
        timestep: int = 261,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_text_encoder: bool = False,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.feature_key = feature_key
        self.timestep = int(timestep)

        self.vae = self._load_vae(vae_model_name_or_path, dtype).to(self.device)
        self.vae.eval().requires_grad_(False)

        self.model = SD21UNetFeatureExtractor()
        state_dict = load_file(str(unet_path), device="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device=self.device, dtype=dtype)
        self.model.eval().requires_grad_(False)

        self.empty_prompt_embeds = None
        if use_text_encoder:
            self.empty_prompt_embeds = self._load_empty_prompt_embeds(vae_model_name_or_path)

    @staticmethod
    def _vae_load_order(model_name_or_path: str | Path) -> list[dict[str, str]]:
        model_path = Path(model_name_or_path)
        model_id = str(model_name_or_path)
        has_vae_subdir = model_path.exists() and (model_path / "vae").is_dir()
        looks_like_standalone_vae = model_id.rstrip("/") in {
            "stabilityai/sd-vae-ft-mse",
            "stabilityai/sd-vae-ft-ema",
        }
        if has_vae_subdir:
            return [{"subfolder": "vae"}, {}]
        if looks_like_standalone_vae or (model_path.exists() and not has_vae_subdir):
            return [{}, {"subfolder": "vae"}]
        return [{"subfolder": "vae"}, {}]

    @classmethod
    def _load_vae(cls, model_name_or_path: str | Path, dtype: torch.dtype) -> DiffusersAutoencoderKL:
        errors: list[str] = []
        for local_files_only in (True, False):
            for kwargs in cls._vae_load_order(model_name_or_path):
                try:
                    return DiffusersAutoencoderKL.from_pretrained(
                        model_name_or_path,
                        torch_dtype=dtype,
                        local_files_only=local_files_only,
                        **kwargs,
                    )
                except Exception as exc:  # noqa: BLE001 - keep fallback robust for HF/network metadata errors.
                    label = kwargs.get("subfolder", "<root>")
                    mode = "local" if local_files_only else "online"
                    errors.append(f"{mode}/{label}: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            f"Failed to load CleanDIFT VAE from {model_name_or_path}. Tried: " + " | ".join(errors)
        )

    @torch.no_grad()
    def _load_empty_prompt_embeds(self, model_name_or_path: str | Path) -> torch.Tensor:
        from transformers import CLIPTextModel, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_name_or_path, subfolder="text_encoder", torch_dtype=self.dtype)
        text_encoder.to(self.device)
        text_inputs = tokenizer(
            [""],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        embeds = text_encoder(input_ids, return_dict=True).last_hidden_state.to(dtype=self.dtype)
        text_encoder.to("cpu")
        del text_encoder, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return embeds

    def _prompt_embeds(self, batch_size: int) -> torch.Tensor:
        if self.empty_prompt_embeds is not None:
            return self.empty_prompt_embeds.repeat(batch_size, 1, 1).to(device=self.device, dtype=self.dtype)
        return torch.zeros(batch_size, 77, 1024, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def __call__(self, images: list[Image.Image], grid_hw: tuple[int, int]) -> torch.Tensor:
        image_tensor = _images_to_tensor(images, self.device, self.dtype)
        encoded = self.vae.encode(image_tensor, return_dict=False)[0]
        if hasattr(encoded, "mode"):
            latents = encoded.mode()
        else:
            latents = encoded.sample()
        latents = latents * 0.18215

        batch_size = latents.shape[0]
        timesteps = torch.full((batch_size,), self.timestep, device=self.device, dtype=torch.long)
        feats = self.model(
            latents,
            timesteps,
            encoder_hidden_states=self._prompt_embeds(batch_size),
            added_cond_kwargs={},
        )[self.feature_key]
        feats = F.interpolate(feats.float(), size=grid_hw, mode="bilinear", align_corners=False)
        tokens = feats.flatten(2).transpose(1, 2).to(self.dtype)
        return l2_normalize(tokens, dim=-1)
