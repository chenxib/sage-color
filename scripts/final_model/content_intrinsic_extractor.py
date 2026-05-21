from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    AutoModelForSemanticSegmentation,
    Mask2FormerForUniversalSegmentation,
)

from scripts.final_model.model import gaussian_blur


@dataclass
class ContentIntrinsicRawState:
    patch_stats: torch.Tensor
    heuristic_priors: torch.Tensor
    debug_maps: dict[str, torch.Tensor] | None = None


@dataclass
class SemanticStructureState:
    labels: torch.Tensor
    boundary: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor


def _kernel(device: torch.device, dtype: torch.dtype, values: list[list[float]]) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=dtype).view(1, 1, 3, 3)


def to_achromatic_pil(image: Image.Image) -> Image.Image:
    """Drop chroma before a frozen structure model sees the content image."""
    gray = image.convert("L")
    return Image.merge("RGB", (gray, gray, gray))


def achromatic_luminance(images: torch.Tensor) -> torch.Tensor:
    """Return a single achromatic intensity channel in [0, 1].

    CIG is allowed to use intensity edges and local contrast, but not RGB
    channels or Lab ab chroma. Downstream code also avoids raw/low-frequency
    intensity means so this tensor is used only to derive structure responses.
    """
    rgb = (images.float().clamp(-1, 1) + 1.0) * 0.5
    weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True).clamp(0.0, 1.0)


def sobel_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = x.dtype
    kx = _kernel(x.device, dtype, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
    ky = _kernel(x.device, dtype, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0
    channels = x.shape[1]
    x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(x_pad, kx.expand(channels, 1, 3, 3), groups=channels)
    gy = F.conv2d(x_pad, ky.expand(channels, 1, 3, 3), groups=channels)
    return gx, gy


def laplacian(x: torch.Tensor) -> torch.Tensor:
    kernel = _kernel(x.device, x.dtype, [[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    channels = x.shape[1]
    x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(x_pad, kernel.expand(channels, 1, 3, 3), groups=channels)


def local_std(x: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    pad = kernel_size // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    mean = F.avg_pool2d(x_pad, kernel_size=kernel_size, stride=1)
    mean_sq = F.avg_pool2d(x_pad * x_pad, kernel_size=kernel_size, stride=1)
    return (mean_sq - mean * mean).clamp_min(1e-6).sqrt()


def normalize_quantile(x: torch.Tensor, q: float = 0.95, eps: float = 1e-6) -> torch.Tensor:
    x_float = x.float()
    flat = x_float.flatten(2).abs()
    scale = torch.quantile(flat, q, dim=-1, keepdim=True).clamp_min(eps)
    while scale.ndim < x_float.ndim:
        scale = scale.unsqueeze(-1)
    return (x_float / scale.to(x_float.dtype)).clamp(0.0, 1.0)


def normalize_percentile_range(x: torch.Tensor, low: float = 0.02, high: float = 0.98, eps: float = 1e-6) -> torch.Tensor:
    x_float = x.float()
    flat = x_float.flatten(2)
    lo = torch.quantile(flat, low, dim=-1, keepdim=True)
    hi = torch.quantile(flat, high, dim=-1, keepdim=True)
    while lo.ndim < x_float.ndim:
        lo = lo.unsqueeze(-1)
        hi = hi.unsqueeze(-1)
    return ((x_float - lo) / (hi - lo).clamp_min(eps)).clamp(0.0, 1.0)


def safe_gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    min_hw = max(2, min(x.shape[-2:]))
    max_sigma = max(0.25, (min_hw - 1) / 4.0)
    return gaussian_blur(x, sigma=min(float(sigma), max_sigma))


def patch_mean(x: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    return F.adaptive_avg_pool2d(x.float(), grid_hw).flatten(2).transpose(1, 2)


def patch_max(x: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    return F.adaptive_max_pool2d(x.float(), grid_hw).flatten(2).transpose(1, 2)


def patch_std(x: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    mean = F.adaptive_avg_pool2d(x.float(), grid_hw)
    mean_sq = F.adaptive_avg_pool2d(x.float() * x.float(), grid_hw)
    return (mean_sq - mean * mean).clamp_min(1e-6).sqrt().flatten(2).transpose(1, 2)


def label_boundary(label_c: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    batch_size = label_c.shape[0]
    height, width = grid_hw
    labels = label_c.to(dtype=torch.long).reshape(batch_size, height, width)
    boundary = torch.zeros(batch_size, height, width, device=labels.device, dtype=torch.float32)
    boundary[:, 1:, :] = torch.maximum(boundary[:, 1:, :], (labels[:, 1:, :] != labels[:, :-1, :]).float())
    boundary[:, :-1, :] = torch.maximum(boundary[:, :-1, :], (labels[:, 1:, :] != labels[:, :-1, :]).float())
    boundary[:, :, 1:] = torch.maximum(boundary[:, :, 1:], (labels[:, :, 1:] != labels[:, :, :-1]).float())
    boundary[:, :, :-1] = torch.maximum(boundary[:, :, :-1], (labels[:, :, 1:] != labels[:, :, :-1]).float())
    boundary = F.max_pool2d(boundary.unsqueeze(1), kernel_size=3, stride=1, padding=1)
    return boundary.flatten(2).transpose(1, 2)


def dense_label_boundary(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(dtype=torch.long)
    batch_size, height, width = labels.shape
    boundary = torch.zeros(batch_size, height, width, device=labels.device, dtype=torch.float32)
    boundary[:, 1:, :] = torch.maximum(boundary[:, 1:, :], (labels[:, 1:, :] != labels[:, :-1, :]).float())
    boundary[:, :-1, :] = torch.maximum(boundary[:, :-1, :], (labels[:, 1:, :] != labels[:, :-1, :]).float())
    boundary[:, :, 1:] = torch.maximum(boundary[:, :, 1:], (labels[:, :, 1:] != labels[:, :, :-1]).float())
    boundary[:, :, :-1] = torch.maximum(boundary[:, :, :-1], (labels[:, :, 1:] != labels[:, :, :-1]).float())
    return F.max_pool2d(boundary.unsqueeze(1), kernel_size=3, stride=1, padding=1)


def region_area(label_c: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    batch_size, num_tokens = label_c.shape
    values = torch.zeros(batch_size, num_tokens, 1, device=label_c.device, dtype=torch.float32)
    for batch_idx in range(batch_size):
        labels = label_c[batch_idx].to(torch.long)
        counts = torch.bincount(labels, minlength=int(labels.max().item()) + 1).float().to(label_c.device)
        values[batch_idx, :, 0] = counts[labels] / float(num_tokens)
    return values


def region_local_purity(label_c: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    batch_size = label_c.shape[0]
    height, width = grid_hw
    labels = label_c.to(dtype=torch.float32).reshape(batch_size, 1, height, width)
    patches = F.unfold(labels, kernel_size=3, padding=1).reshape(batch_size, 9, height * width)
    center = label_c.to(dtype=torch.float32).unsqueeze(1)
    return (patches == center).float().mean(dim=1, keepdim=False).unsqueeze(-1)


def approximate_boundary_distance(boundary_tokens: torch.Tensor, grid_hw: tuple[int, int], rounds: int = 16) -> torch.Tensor:
    batch_size = boundary_tokens.shape[0]
    height, width = grid_hw
    boundary = boundary_tokens.transpose(1, 2).reshape(batch_size, 1, height, width).float().clamp(0, 1)
    reached = boundary.clone()
    distance = torch.zeros_like(boundary)
    for step in range(1, rounds + 1):
        reached_next = F.max_pool2d(reached, kernel_size=3, stride=1, padding=1)
        newly_reached = (reached_next > 0.5) & (reached <= 0.5)
        distance = torch.where(newly_reached, torch.full_like(distance, float(step)), distance)
        reached = reached_next
    distance = torch.where(reached > 0.5, distance / float(rounds), torch.ones_like(distance))
    return distance.flatten(2).transpose(1, 2).clamp(0.0, 1.0)


class DepthAnythingV2FeatureExtractor:
    """Frozen Depth Anything V2 depth predictor used only for final model CIG priors."""

    def __init__(
        self,
        model_name_or_path: str | Path = "model/depth-anything-v2-base",
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name_or_path, use_fast=False)
        except TypeError:
            self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        try:
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name_or_path, dtype=dtype)
        except TypeError:
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name_or_path, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images: list[Image.Image], size: tuple[int, int], dtype: torch.dtype | None = None) -> torch.Tensor:
        images = [to_achromatic_pil(image) for image in images]
        inputs = self.processor(images=images, return_tensors="pt")
        moved = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device=self.device, dtype=self.dtype) if value.is_floating_point() else value.to(self.device)
            else:
                moved[key] = value
        inputs = moved
        outputs = self.model(**inputs)
        depth = outputs.predicted_depth.unsqueeze(1).float()
        depth = F.interpolate(depth, size=size, mode="bicubic", align_corners=False)
        depth = normalize_percentile_range(depth, 0.02, 0.98)
        return depth.to(device=self.device, dtype=dtype or self.dtype)


class SegFormerSemanticStructureExtractor:
    """Frozen semantic structure prior used by final model CIG.

    This extractor is intentionally separate from the stage-1 DINO/SigLIP/CleanDIFT
    correspondence stack. It contributes object/scene region boundaries,
    confidence, and uncertainty for CIG, while the existing stage-1 stack remains
    responsible for reference color correspondence.
    """

    def __init__(
        self,
        model_name_or_path: str | Path = "model/segformer-b0-ade",
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = torch.device(device)
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name_or_path, use_fast=False)
        except TypeError:
            self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        try:
            self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name_or_path, dtype=self.dtype)
        except TypeError:
            self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name_or_path, torch_dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images: list[Image.Image], size: tuple[int, int], dtype: torch.dtype | None = None) -> SemanticStructureState:
        images = [to_achromatic_pil(image) for image in images]
        inputs = self.processor(images=images, return_tensors="pt")
        moved = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device=self.device, dtype=self.dtype) if value.is_floating_point() else value.to(self.device)
            else:
                moved[key] = value
        outputs = self.model(**moved)
        logits = outputs.logits.float()
        probs = torch.softmax(logits, dim=1)
        confidence, labels = probs.max(dim=1)
        num_classes = max(int(probs.shape[1]), 2)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        entropy = entropy / float(np.log(num_classes))
        boundary = dense_label_boundary(labels)
        boundary = F.interpolate(boundary, size=size, mode="nearest")
        confidence = F.interpolate(confidence.unsqueeze(1), size=size, mode="bilinear", align_corners=False)
        entropy = F.interpolate(entropy, size=size, mode="bilinear", align_corners=False)
        return SemanticStructureState(
            labels=labels.to(self.device),
            boundary=boundary.to(device=self.device, dtype=dtype or self.dtype),
            confidence=confidence.to(device=self.device, dtype=dtype or self.dtype),
            entropy=entropy.to(device=self.device, dtype=dtype or self.dtype),
        )


class Mask2FormerPanopticStructureExtractor:
    """Frozen panoptic object-boundary prior for CIG.

    The extractor uses only query masks, per-query confidence, and assignment
    entropy. It never exposes class names, RGB values, or feature vectors to the
    trainable CIG modules.
    """

    def __init__(
        self,
        model_name_or_path: str | Path = "model/mask2former-swin-small-coco-panoptic",
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = torch.device(device)
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name_or_path, use_fast=False)
        except TypeError:
            self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        try:
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name_or_path, dtype=self.dtype)
        except TypeError:
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name_or_path, torch_dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images: list[Image.Image], size: tuple[int, int], dtype: torch.dtype | None = None) -> SemanticStructureState:
        images = [to_achromatic_pil(image) for image in images]
        inputs = self.processor(images=images, return_tensors="pt")
        moved = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device=self.device, dtype=self.dtype) if value.is_floating_point() else value.to(self.device)
            else:
                moved[key] = value
        outputs = self.model(**moved)
        class_probs = torch.softmax(outputs.class_queries_logits.float(), dim=-1)
        query_scores = class_probs[..., :-1].amax(dim=-1).clamp(0, 1)
        mask_probs = torch.sigmoid(outputs.masks_queries_logits.float())
        mask_probs = F.interpolate(mask_probs, size=size, mode="bilinear", align_corners=False)
        weighted_masks = mask_probs * query_scores[:, :, None, None]
        confidence, labels = weighted_masks.max(dim=1)
        active = confidence > 0.05
        labels = torch.where(active, labels + 1, torch.zeros_like(labels))
        boundary = dense_label_boundary(labels) * active.unsqueeze(1).float()
        denom = weighted_masks.sum(dim=1, keepdim=True).clamp_min(1e-6)
        assignment = weighted_masks / denom
        entropy = -(assignment * assignment.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        entropy = (entropy / float(np.log(max(int(weighted_masks.shape[1]), 2)))).clamp(0, 1)
        entropy = entropy * active.unsqueeze(1).float()
        return SemanticStructureState(
            labels=labels.to(self.device),
            boundary=boundary.to(device=self.device, dtype=dtype or self.dtype),
            confidence=confidence.unsqueeze(1).to(device=self.device, dtype=dtype or self.dtype),
            entropy=entropy.to(device=self.device, dtype=dtype or self.dtype),
        )


class ContentStructureExtractor(nn.Module):
    """Online non-trainable CIG prior extractor.

    The extractor intentionally avoids RGB/Lab chroma features and does not
    consume stage-1 correspondence labels or DINO tokens. The default structure
    side uses CIG-focused priors: achromatic local structure responses, Depth
    Anything V2 relative geometry, and optional grayscale-fed SegFormer /
    Mask2Former object boundaries.
    """

    patch_stat_dim = 44
    num_priors = 6

    @staticmethod
    def _analytic_structure_stats(
        soft_edge_token: torch.Tensor,
        texture_energy: torch.Tensor,
        combined_boundary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        analytic_contrast = texture_energy.clamp(0, 1)
        analytic_boundary = soft_edge_token.clamp(0, 1)
        analytic_purity = (1.0 - soft_edge_token).clamp(0, 1)
        analytic_edge_alignment = (torch.maximum(combined_boundary, analytic_boundary) * soft_edge_token).clamp(0, 1)
        return analytic_contrast, analytic_boundary, analytic_purity, analytic_edge_alignment

    @torch.no_grad()
    def forward(
        self,
        content_image: torch.Tensor,
        label_c: torch.Tensor | None,
        grid_hw: tuple[int, int],
        depth_map: torch.Tensor | None = None,
        semantic_state: SemanticStructureState | None = None,
        panoptic_state: SemanticStructureState | None = None,
        content_dino: torch.Tensor | None = None,
        return_debug: bool = False,
    ) -> ContentIntrinsicRawState:
        # Kept in the signature for call-site compatibility only. These values
        # come from the stage-1 correspondence branch and may encode color, so CIG
        # must not consume them.
        _ = label_c, content_dino
        work_dtype = content_image.dtype
        image = content_image.float().clamp(-1, 1)
        luminance = achromatic_luminance(image)

        std = local_std(luminance, kernel_size=15)
        std_norm = normalize_quantile(std, 0.95)
        hp1 = ((luminance - safe_gaussian_blur(luminance, 1.5)) / (std + 1e-4)).clamp(-3, 3) / 3.0
        hp2 = ((luminance - safe_gaussian_blur(luminance, 3.0)) / (std + 1e-4)).clamp(-3, 3) / 3.0
        hp3 = ((luminance - safe_gaussian_blur(luminance, 6.0)) / (std + 1e-4)).clamp(-3, 3) / 3.0

        gx, gy = sobel_xy(luminance)
        grad_mag = normalize_quantile((gx * gx + gy * gy).clamp_min(1e-8).sqrt(), 0.95)
        lap_abs = normalize_quantile(laplacian(luminance).abs(), 0.95)
        soft_edge = torch.sigmoid((grad_mag - 0.25) / 0.05)

        jxx = safe_gaussian_blur(gx * gx, 2.0)
        jyy = safe_gaussian_blur(gy * gy, 2.0)
        jxy = safe_gaussian_blur(gx * gy, 2.0)
        trace = jxx + jyy
        delta = ((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy).clamp_min(1e-8).sqrt()
        lambda1 = 0.5 * (trace + delta)
        lambda2 = 0.5 * (trace - delta)
        coherence = ((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)).clamp(0, 1)
        orientation = 0.5 * torch.atan2(2.0 * jxy, jxx - jyy + 1e-8)
        orient_sin = torch.sin(2.0 * orientation)
        orient_cos = torch.cos(2.0 * orientation)

        if depth_map is None:
            depth = torch.zeros_like(luminance)
        else:
            depth = F.interpolate(depth_map.float(), size=luminance.shape[-2:], mode="bilinear", align_corners=False)
            depth = normalize_percentile_range(depth, 0.02, 0.98)
        depth_dx, depth_dy = sobel_xy(depth)
        depth_grad = normalize_quantile((depth_dx * depth_dx + depth_dy * depth_dy).clamp_min(1e-8).sqrt(), 0.95)
        normal = torch.cat([-2.0 * depth_dx, -2.0 * depth_dy, torch.ones_like(depth)], dim=1)
        normal = F.normalize(normal, dim=1, eps=1e-6)
        normal_gx, normal_gy = sobel_xy(normal)
        normal_grad = normalize_quantile(
            (normal_gx * normal_gx + normal_gy * normal_gy).sum(dim=1, keepdim=True).clamp_min(1e-8).sqrt(),
            0.95,
        )
        geom_edge_soft = torch.sigmoid((depth_grad + normal_grad - 0.25) / 0.05)

        batch_size = content_image.shape[0]
        num_tokens = grid_hw[0] * grid_hw[1]
        if semantic_state is not None:
            region_boundary = patch_mean(semantic_state.boundary.float(), grid_hw).clamp(0, 1)
            semantic_confidence = patch_mean(semantic_state.confidence.float(), grid_hw).clamp(0, 1)
            semantic_entropy = patch_mean(semantic_state.entropy.float(), grid_hw).clamp(0, 1)
            semantic_labels = F.interpolate(
                semantic_state.labels.unsqueeze(1).float(),
                size=grid_hw,
                mode="nearest",
            ).flatten(2).squeeze(1).to(device=content_image.device, dtype=torch.long)
        else:
            semantic_labels = torch.zeros(batch_size, num_tokens, device=content_image.device, dtype=torch.long)
            region_boundary = torch.zeros(batch_size, num_tokens, 1, device=content_image.device, dtype=torch.float32)
            semantic_confidence = torch.zeros_like(region_boundary)
            semantic_entropy = torch.zeros_like(region_boundary)
        region_boundary_map = region_boundary.transpose(1, 2).reshape(batch_size, 1, *grid_hw)
        region_boundary_image = F.interpolate(region_boundary_map, size=luminance.shape[-2:], mode="nearest")
        region_area_token = region_area(semantic_labels, grid_hw)
        purity_token = region_local_purity(semantic_labels, grid_hw)
        distance_token = approximate_boundary_distance(region_boundary, grid_hw)
        if panoptic_state is not None:
            panoptic_boundary = patch_mean(panoptic_state.boundary.float(), grid_hw).clamp(0, 1)
            panoptic_confidence = patch_mean(panoptic_state.confidence.float(), grid_hw).clamp(0, 1)
            panoptic_entropy = patch_mean(panoptic_state.entropy.float(), grid_hw).clamp(0, 1)
        else:
            panoptic_boundary = torch.zeros_like(region_boundary)
            panoptic_confidence = torch.zeros_like(region_boundary)
            panoptic_entropy = torch.zeros_like(region_boundary)

        band16 = (safe_gaussian_blur(luminance, 1.0) - safe_gaussian_blur(luminance, 4.0)).abs()
        band32 = (safe_gaussian_blur(luminance, 2.0) - safe_gaussian_blur(luminance, 8.0)).abs()
        band_energy_map = normalize_quantile(band16 + 0.5 * band32, 0.95)
        band_gx, band_gy = sobel_xy(band_energy_map)
        band_grad_map = normalize_quantile((band_gx * band_gx + band_gy * band_gy).clamp_min(1e-8).sqrt(), 0.95)

        hp1_max = patch_max(hp1.abs(), grid_hw)
        hp1_abs = patch_mean(hp1.abs(), grid_hw)
        hp2_max = patch_max(hp2.abs(), grid_hw)
        hp2_abs = patch_mean(hp2.abs(), grid_hw)
        hp3_abs = patch_mean(hp3.abs(), grid_hw)
        grad_mean = patch_mean(grad_mag, grid_hw)
        grad_max = patch_max(grad_mag, grid_hw)
        lap_absmean = patch_mean(lap_abs, grid_hw)
        local_std_mean = patch_mean(std_norm, grid_hw)
        coherence_mean = patch_mean(coherence, grid_hw)
        orient_sin_mean = patch_mean(orient_sin, grid_hw)
        orient_cos_mean = patch_mean(orient_cos, grid_hw)

        depth_mean = patch_mean(depth, grid_hw)
        depth_std = patch_std(depth, grid_hw)
        depth_grad_mean = patch_mean(depth_grad, grid_hw)
        depth_grad_max = patch_max(depth_grad, grid_hw)
        normal_mean = patch_mean(normal, grid_hw)
        normal_grad_mean = patch_mean(normal_grad, grid_hw)
        geom_edge = patch_mean(geom_edge_soft, grid_hw)

        soft_edge_token = patch_mean(soft_edge, grid_hw)
        edge_boundary_alignment = region_boundary * soft_edge_token
        panoptic_edge_alignment = panoptic_boundary * soft_edge_token

        texture_energy = (
            0.4 * hp1_abs
            + 0.3 * hp2_abs
            + 0.2 * grad_mean
            + 0.1 * lap_absmean
        ).clamp(0, 1)
        analytic_contrast, analytic_boundary, analytic_purity, analytic_edge_alignment = self._analytic_structure_stats(
            soft_edge_token,
            texture_energy,
            torch.maximum(region_boundary, panoptic_boundary),
        )
        combined_boundary = torch.maximum(torch.maximum(region_boundary, panoptic_boundary), analytic_boundary)
        band16_abs = patch_mean(band16, grid_hw)
        band32_abs = patch_mean(band32, grid_hw)
        band_energy = patch_mean(band_energy_map, grid_hw)
        band_grad = patch_mean(band_grad_map, grid_hw)
        flatness = (1.0 - torch.maximum(torch.maximum(texture_energy, combined_boundary), geom_edge)).clamp(0, 1)
        structure_confidence = torch.maximum(torch.maximum(texture_energy, combined_boundary), geom_edge).clamp(0, 1)
        structure_uncertainty = torch.maximum(semantic_entropy, panoptic_entropy).clamp(0, 1)

        patch_stats = torch.cat(
            [
                hp1_max,
                hp1_abs,
                hp2_max,
                hp2_abs,
                hp3_abs,
                grad_mean,
                grad_max,
                lap_absmean,
                local_std_mean,
                coherence_mean,
                orient_sin_mean,
                orient_cos_mean,
                depth_mean,
                depth_std,
                depth_grad_mean,
                depth_grad_max,
                normal_mean,
                normal_grad_mean,
                geom_edge,
                region_boundary,
                region_area_token,
                purity_token,
                semantic_confidence,
                semantic_entropy,
                edge_boundary_alignment,
                distance_token,
                panoptic_boundary,
                panoptic_confidence,
                panoptic_entropy,
                panoptic_edge_alignment,
                analytic_contrast,
                analytic_boundary,
                analytic_purity,
                analytic_edge_alignment,
                band16_abs,
                band32_abs,
                band_energy,
                band_grad,
                texture_energy,
                flatness,
                structure_confidence,
                structure_uncertainty,
            ],
            dim=-1,
        )
        if patch_stats.shape[-1] != self.patch_stat_dim:
            raise RuntimeError(f"Expected {self.patch_stat_dim} patch stats, got {patch_stats.shape[-1]}")

        unreliable = (0.70 * structure_uncertainty + 0.30 * flatness).clamp(0, 1)
        p_texture = (0.35 * hp1_abs + 0.25 * hp2_abs + 0.20 * grad_mean + 0.10 * lap_absmean + 0.10 * coherence_mean)
        p_texture = p_texture.clamp(0, 1)
        p_boundary = (
            0.30 * region_boundary
            + 0.25 * panoptic_boundary
            + 0.20 * analytic_boundary
            + 0.15 * soft_edge_token
            + 0.10 * torch.maximum(edge_boundary_alignment, torch.maximum(panoptic_edge_alignment, analytic_edge_alignment))
        ).clamp(0, 1)
        boundary_confidence = torch.maximum(semantic_confidence, panoptic_confidence)
        p_boundary = (p_boundary * (0.75 + 0.25 * boundary_confidence)).clamp(0, 1)
        p_geometry = (0.45 * depth_grad_mean + 0.35 * normal_grad_mean + 0.20 * geom_edge).clamp(0, 1)
        p_material = (0.45 * band_energy + 0.25 * band_grad + 0.20 * texture_energy + 0.10 * coherence_mean)
        p_material = (p_material * (1.0 - 0.5 * unreliable)).clamp(0, 1)
        p_texture = (0.85 * p_texture + 0.15 * analytic_contrast).clamp(0, 1)
        p_flat = (1.0 - torch.maximum(torch.maximum(p_texture, p_boundary), p_geometry)).clamp(0, 1)
        heuristic_priors = torch.cat([p_texture, p_boundary, p_geometry, p_material, p_flat, unreliable], dim=-1)

        debug_maps = None
        if return_debug:
            debug_maps = {
                "achromatic_luminance": luminance.detach(),
                "soft_edge": soft_edge.detach(),
                "depth": depth.detach(),
                "semantic_boundary": region_boundary_image.detach(),
                "panoptic_boundary": F.interpolate(
                    panoptic_boundary.transpose(1, 2).reshape(batch_size, 1, *grid_hw),
                    size=luminance.shape[-2:],
                    mode="nearest",
                ).detach(),
                "band_energy": band_energy_map.detach(),
            }
        return ContentIntrinsicRawState(
            patch_stats=patch_stats.to(dtype=work_dtype),
            heuristic_priors=heuristic_priors.to(dtype=work_dtype),
            debug_maps=debug_maps,
        )
