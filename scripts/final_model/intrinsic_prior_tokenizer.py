from __future__ import annotations

import torch
import torch.nn as nn


def zero_init_last(module: nn.Module) -> None:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            nn.init.zeros_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
            return


def xy_grid(grid_hw: tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    height, width = grid_hw
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(height * width, 2)


class IntrinsicPriorTokenizer(nn.Module):
    """Convert CIG structure priors into attention masks and anchors.

    final model keeps DINO/SigLIP2/CleanDIFT in the stage-1 reference/correspondence
    branch, but the CIG structure branch should not depend on the same feature
    stack. This tokenizer consumes only structure-side patch stats, heuristic
    priors, and xy position.
    """

    def __init__(
        self,
        patch_stat_dim: int,
        sd3_hidden_dim: int,
        hidden_dim: int | None = None,
        num_mask_types: int = 6,
    ):
        super().__init__()
        hidden_dim = hidden_dim or sd3_hidden_dim
        self.num_mask_types = num_mask_types

        self.stat_proj = nn.Sequential(
            nn.LayerNorm(patch_stat_dim),
            nn.Linear(patch_stat_dim, 512),
            nn.SiLU(),
            nn.Linear(512, hidden_dim),
        )
        self.xy_proj = nn.Sequential(
            nn.Linear(2, 128),
            nn.SiLU(),
            nn.Linear(128, hidden_dim),
        )
        self.base_norm = nn.LayerNorm(hidden_dim)
        self.type_embed = nn.Parameter(torch.randn(num_mask_types, hidden_dim) * 0.02)
        self.mask_delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.protect_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.anchor_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, sd3_hidden_dim),
        )
        zero_init_last(self.mask_delta_head)
        zero_init_last(self.protect_head)
        zero_init_last(self.anchor_head)

    def forward(
        self,
        patch_stats: torch.Tensor,
        heuristic_priors: torch.Tensor,
        xy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = next(self.parameters()).dtype
        patch_stats = patch_stats.to(dtype=dtype)
        base = self.stat_proj(patch_stats)
        if xy is not None:
            base = base + self.xy_proj(xy.to(device=patch_stats.device, dtype=dtype))
        base = self.base_norm(base)

        type_hidden = base[:, :, None, :] + self.type_embed.to(dtype=dtype)[None, None, :, :]
        delta = self.mask_delta_head(type_hidden).squeeze(-1)
        prior = heuristic_priors.to(device=patch_stats.device, dtype=dtype).clamp(1e-4, 1.0 - 1e-4)
        intrinsic_masks = torch.sigmoid(torch.logit(prior) + delta)

        texture, boundary, geometry, material, flat, unreliable = intrinsic_masks.split(1, dim=-1)
        protect_prior = (
            0.45 * texture
            + 0.30 * boundary
            + 0.25 * geometry
            + 0.25 * material
            - 0.15 * flat
        ).clamp(0.0, 1.0)
        protect_delta = 0.25 * torch.tanh(self.protect_head(base))
        protect_mask = (protect_prior + protect_delta).clamp(0.0, 1.0)
        protect_mask = protect_mask * (1.0 - 0.3 * unreliable)
        additive_anchor = self.anchor_head(base)
        return intrinsic_masks, protect_mask, additive_anchor
