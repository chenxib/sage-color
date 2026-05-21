from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from peft import LoraConfig


def expand_sd3_input_channels(transformer: nn.Module, new_in_channels: int = 32) -> None:
    """Expand SD3 patch input from 16 target latent channels to target+content channels."""

    proj = transformer.pos_embed.proj
    old_in_channels = proj.in_channels
    if old_in_channels == new_in_channels:
        return
    if new_in_channels < old_in_channels:
        raise ValueError(f"new_in_channels={new_in_channels} must be >= {old_in_channels}")

    new_proj = nn.Conv2d(
        new_in_channels,
        proj.out_channels,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
        dilation=proj.dilation,
        groups=proj.groups,
        bias=proj.bias is not None,
        padding_mode=proj.padding_mode,
        device=proj.weight.device,
        dtype=proj.weight.dtype,
    )
    with torch.no_grad():
        new_proj.weight.zero_()
        new_proj.weight[:, :old_in_channels].copy_(proj.weight)
        if proj.bias is not None:
            new_proj.bias.copy_(proj.bias)

    transformer.pos_embed.proj = new_proj
    if hasattr(transformer, "register_to_config"):
        transformer.register_to_config(in_channels=new_in_channels)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class PerceiverAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 16, dim_head: int = 64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.norm_image = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, image_hidden_states: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        batch_size = image_hidden_states.shape[0]
        image_hidden_states = self.norm_image(image_hidden_states)
        latents = self.norm_latents(latents)
        key_value_states = torch.cat([image_hidden_states, latents], dim=1)

        query = self.to_q(latents)
        key, value = self.to_kv(key_value_states).chunk(2, dim=-1)

        query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * self.dim_head)
        return self.to_out(hidden_states)


class GlobalReferenceResampler(nn.Module):
    """IP-Adapter-style resampler for frozen VLM patch tokens."""

    def __init__(
        self,
        image_hidden_size: int,
        output_hidden_size: int,
        num_tokens: int = 32,
        num_heads: int = 16,
        depth: int = 4,
        dim_head: int = 64,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_tokens, output_hidden_size) / output_hidden_size**0.5)
        self.image_proj = nn.Sequential(
            nn.LayerNorm(image_hidden_size),
            nn.Linear(image_hidden_size, output_hidden_size),
            nn.LayerNorm(output_hidden_size),
        )
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": PerceiverAttention(output_hidden_size, heads=num_heads, dim_head=dim_head),
                        "ff": FeedForward(output_hidden_size),
                    }
                )
                for _ in range(depth)
            ]
        )
        self.proj_out = nn.Linear(output_hidden_size, output_hidden_size)
        self.norm_out = nn.LayerNorm(output_hidden_size)

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        image_hidden_states = self.image_proj(image_hidden_states)
        latents = self.latents.unsqueeze(0).expand(image_hidden_states.shape[0], -1, -1)
        for layer in self.layers:
            latents = latents + layer["attn"](image_hidden_states, latents)
            latents = latents + layer["ff"](latents)
        return self.norm_out(self.proj_out(latents))


def rgb_to_lab(images: torch.Tensor) -> torch.Tensor:
    """Approximate RGB[-1,1] to CIE Lab. Returns L in [0,100], a/b roughly [-128,127]."""

    rgb = (images.float().clamp(-1, 1) + 1.0) * 0.5
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))
    r, g, b = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    x = x / 0.95047
    z = z / 1.08883

    eps = 216 / 24389
    kappa = 24389 / 27

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > eps, t.clamp_min(1e-8).pow(1.0 / 3.0), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    lab_l = 116.0 * fy - 16.0
    lab_a = 500.0 * (fx - fy)
    lab_b = 200.0 * (fy - fz)
    return torch.cat([lab_l, lab_a, lab_b], dim=1)


def patch_lab_stats(images: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    lab = rgb_to_lab(images)
    mean = F.adaptive_avg_pool2d(lab, grid_hw)
    mean_sq = F.adaptive_avg_pool2d(lab * lab, grid_hw)
    std = (mean_sq - mean * mean).clamp_min(1e-6).sqrt()
    stats = torch.cat([mean, std], dim=1).flatten(2).transpose(1, 2)
    stats = stats.clone()
    stats[..., 0] = stats[..., 0] / 50.0 - 1.0
    stats[..., 1] = stats[..., 1] / 128.0
    stats[..., 2] = stats[..., 2] / 128.0
    stats[..., 3] = stats[..., 3] / 50.0
    stats[..., 4] = stats[..., 4] / 128.0
    stats[..., 5] = stats[..., 5] / 128.0
    return stats


class LocalReferenceProjector(nn.Module):
    def __init__(self, vlm_dim: int, hidden_dim: int, color_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(vlm_dim + color_dim),
            nn.Linear(vlm_dim + color_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, ref_vlm_tokens: torch.Tensor, ref_lab_stats: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([ref_vlm_tokens, ref_lab_stats.to(ref_vlm_tokens.dtype)], dim=-1))


def resize_token_grid(tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    bsz, num_tokens, channels = tokens.shape
    target_h, target_w = grid_hw
    if num_tokens == target_h * target_w:
        return tokens
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        if num_tokens > target_h * target_w:
            tokens = tokens[:, : target_h * target_w]
            return tokens
        pad = target_h * target_w - num_tokens
        return F.pad(tokens, (0, 0, 0, pad))
    grid = tokens.transpose(1, 2).reshape(bsz, channels, side, side)
    grid = F.interpolate(grid.float(), size=grid_hw, mode="bilinear", align_corners=False).to(tokens.dtype)
    return grid.flatten(2).transpose(1, 2)


def pool_tokens_by_region(tokens: torch.Tensor, labels: torch.Tensor, num_regions: int) -> torch.Tensor:
    bsz, num_tokens, channels = tokens.shape
    labels = labels.to(device=tokens.device, dtype=torch.long).clamp(0, num_regions - 1)
    output = tokens.new_zeros(bsz, num_regions, channels)
    counts = tokens.new_zeros(bsz, num_regions, 1)
    index = labels.unsqueeze(-1).expand(-1, -1, channels)
    output.scatter_add_(1, index, tokens)
    counts.scatter_add_(1, labels.unsqueeze(-1), torch.ones(bsz, num_tokens, 1, device=tokens.device, dtype=tokens.dtype))
    mean_token = tokens.mean(dim=1, keepdim=True)
    output = output / counts.clamp_min(1.0)
    empty = counts <= 0
    output = torch.where(empty, mean_token.expand(-1, num_regions, -1), output)
    return output


class CorrespondenceGuidedReferenceAdapter(nn.Module):
    def __init__(
        self,
        siglip_dim: int,
        hidden_dim: int,
        num_global_tokens: int = 32,
        num_regions: int = 24,
        resampler_depth: int = 4,
        resampler_heads: int = 16,
        resampler_dim_head: int = 64,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.global_resampler = GlobalReferenceResampler(
            image_hidden_size=siglip_dim,
            output_hidden_size=hidden_dim,
            num_tokens=num_global_tokens,
            num_heads=resampler_heads,
            depth=resampler_depth,
            dim_head=resampler_dim_head,
        )
        self.local_projector = LocalReferenceProjector(siglip_dim, hidden_dim, color_dim=6)
        self.region_projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        reference_image: torch.Tensor,
        siglip_ref_tokens: torch.Tensor,
        label_r: torch.Tensor,
        grid_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ref_global_tokens = self.global_resampler(siglip_ref_tokens)
        ref_vlm_tokens = resize_token_grid(siglip_ref_tokens, grid_hw)
        ref_lab_stats = patch_lab_stats(reference_image, grid_hw).to(ref_vlm_tokens.device)
        ref_local_tokens = self.local_projector(ref_vlm_tokens, ref_lab_stats)
        ref_region_tokens = pool_tokens_by_region(ref_local_tokens, label_r, self.num_regions)
        ref_region_tokens = self.region_projector(ref_region_tokens)
        return ref_global_tokens, ref_region_tokens, ref_local_tokens


@dataclass
class ReferenceAttentionConfig:
    hidden_size: int
    heads: int
    dim_head: int
    reference_hidden_size: int
    layer_id: int = 0
    local_start_layer: int = 4
    sparse_start_layer: int = 5


def _reshape_heads(x: torch.Tensor, heads: int, dim_head: int) -> torch.Tensor:
    bsz = x.shape[0]
    return x.view(bsz, -1, heads, dim_head).transpose(1, 2)


def _as_hidden_scale(scale, hidden_states: torch.Tensor) -> torch.Tensor | float:
    if not torch.is_tensor(scale):
        return float(scale)
    while scale.ndim < hidden_states.ndim:
        scale = scale.unsqueeze(-1)
    return scale.to(device=hidden_states.device, dtype=hidden_states.dtype)


def _square_grid(num_tokens: int) -> tuple[int, int] | None:
    side = int(math.sqrt(num_tokens))
    if side * side == num_tokens:
        return (side, side)
    return None


def _resize_token_condition(
    condition: torch.Tensor,
    target_tokens: int,
    *,
    mode: str,
) -> torch.Tensor:
    source_tokens = condition.shape[1]
    if source_tokens == target_tokens:
        return condition

    source_grid = _square_grid(source_tokens)
    target_grid = _square_grid(target_tokens)
    if source_grid is None or target_grid is None:
        if source_tokens > target_tokens:
            return condition[:, :target_tokens]
        repeats = math.ceil(target_tokens / source_tokens)
        return condition.repeat_interleave(repeats, dim=1)[:, :target_tokens]

    batch_size = condition.shape[0]
    trailing_shape = condition.shape[2:]
    grid = condition.reshape(batch_size, source_grid[0], source_grid[1], -1).permute(0, 3, 1, 2)
    if condition.is_floating_point() and mode != "nearest":
        grid = F.interpolate(grid.float(), size=target_grid, mode=mode, align_corners=False)
        grid = grid.to(dtype=condition.dtype)
    else:
        grid = F.interpolate(grid.float(), size=target_grid, mode="nearest")
        if not condition.is_floating_point():
            grid = grid.round().to(dtype=condition.dtype)
        else:
            grid = grid.to(dtype=condition.dtype)
    return grid.permute(0, 2, 3, 1).reshape(batch_size, target_tokens, *trailing_shape)


def _prepare_token_condition(
    condition: torch.Tensor,
    batch_size: int,
    target_tokens: int,
    *,
    mode: str = "nearest",
) -> torch.Tensor:
    if condition.ndim == 1:
        condition = condition.unsqueeze(0)
    if condition.shape[0] == 1 and batch_size > 1:
        condition = condition.expand(batch_size, *condition.shape[1:])
    elif condition.shape[0] != batch_size:
        raise ValueError(f"condition batch={condition.shape[0]} does not match query batch={batch_size}")
    return _resize_token_condition(condition, target_tokens, mode=mode)


def sparse_topk_cross_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor | None = None,
    prior_scale: float | torch.Tensor = 1.0,
    head_chunk_size: int = 4,
) -> torch.Tensor:
    bsz, heads, num_tokens, dim_head = q.shape
    topk_idx = _prepare_token_condition(topk_idx, bsz, num_tokens, mode="nearest")
    if topk_weight is not None:
        topk_weight = _prepare_token_condition(topk_weight, bsz, num_tokens, mode="nearest")
    topk_idx = topk_idx.to(device=q.device, dtype=torch.long).clamp(0, k.shape[2] - 1)
    outputs = []
    prior = None
    if topk_weight is not None:
        prior = torch.log(topk_weight.to(device=q.device, dtype=q.dtype).clamp_min(1e-6))

    # Avoid materializing a huge expanded int64 gather index of shape
    # [B, heads, tokens, topk, dim_head]. At 1024 resolution this temporary is
    # ~805 MB per sparse branch. Chunking heads keeps the same math while
    # bounding peak memory to the selected K/V tensors for a few heads.
    for batch_idx in range(bsz):
        idx = topk_idx[batch_idx]
        batch_chunks = []
        for head_start in range(0, heads, head_chunk_size):
            head_end = min(head_start + head_chunk_size, heads)
            q_chunk = q[batch_idx, head_start:head_end]
            k_chunk = k[batch_idx, head_start:head_end]
            v_chunk = v[batch_idx, head_start:head_end]
            k_top = k_chunk[:, idx, :]
            v_top = v_chunk[:, idx, :]
            logits = (q_chunk.unsqueeze(-2) * k_top).sum(dim=-1) / math.sqrt(dim_head)
            if prior is not None:
                logits = logits + prior_scale * prior[batch_idx].unsqueeze(0)
            probs = torch.softmax(logits, dim=-1)
            batch_chunks.append((probs.unsqueeze(-1) * v_top).sum(dim=-2))
        outputs.append(torch.cat(batch_chunks, dim=0))
    return torch.stack(outputs, dim=0)


class CorrReferenceAttentionProcessor(nn.Module):
    """SD3 joint attention plus global, region, and sparse correspondence-guided reference branches."""

    def __init__(self, config: ReferenceAttentionConfig):
        super().__init__()
        self.base_processor = JointAttnProcessor2_0()
        self.layer_id = config.layer_id
        self.local_enabled = config.layer_id >= config.local_start_layer
        self.sparse_enabled = config.layer_id >= config.sparse_start_layer
        inner_dim = config.heads * config.dim_head
        self.heads = config.heads
        self.dim_head = config.dim_head
        self.to_k_global = nn.Linear(config.reference_hidden_size, inner_dim, bias=False)
        self.to_v_global = nn.Linear(config.reference_hidden_size, inner_dim, bias=False)
        self.to_k_region = (
            nn.Linear(config.reference_hidden_size, inner_dim, bias=False) if self.local_enabled else None
        )
        self.to_v_region = (
            nn.Linear(config.reference_hidden_size, inner_dim, bias=False) if self.local_enabled else None
        )
        self.to_k_local = (
            nn.Linear(config.reference_hidden_size, inner_dim, bias=False) if self.sparse_enabled else None
        )
        self.to_v_local = (
            nn.Linear(config.reference_hidden_size, inner_dim, bias=False) if self.sparse_enabled else None
        )
        self.to_out_ref = nn.Linear(inner_dim, config.hidden_size, bias=False)
        self.g_global = nn.Parameter(torch.zeros(()))
        self.g_region = nn.Parameter(torch.zeros(())) if self.local_enabled else None
        self.g_sparse = nn.Parameter(torch.zeros(())) if self.sparse_enabled else None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        ref_global_tokens: torch.FloatTensor | None = None,
        ref_region_tokens: torch.FloatTensor | None = None,
        ref_local_tokens: torch.FloatTensor | None = None,
        region_topm_idx: torch.Tensor | None = None,
        region_topm_weight: torch.Tensor | None = None,
        topk_idx: torch.Tensor | None = None,
        topk_weight: torch.Tensor | None = None,
        corr_conf: torch.Tensor | None = None,
        ref_scale: float = 1.0,
        corr_scale: float = 1.0,
        corr_time_weight: float | torch.Tensor = 1.0,
        *args,
        **kwargs,
    ):
        output = self.base_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            *args,
            **kwargs,
        )

        if isinstance(output, tuple):
            image_hidden_states, context_hidden_states = output
        else:
            image_hidden_states, context_hidden_states = output, None

        if ref_global_tokens is None and ref_region_tokens is None and ref_local_tokens is None:
            return output

        batch_size = image_hidden_states.shape[0]
        query = attn.to_q(hidden_states)
        query = _reshape_heads(query, self.heads, self.dim_head)
        if attn.norm_q is not None:
            query = attn.norm_q(query)

        ref_out = image_hidden_states.new_zeros(image_hidden_states.shape)

        if ref_global_tokens is not None:
            key = _reshape_heads(self.to_k_global(ref_global_tokens), self.heads, self.dim_head)
            value = _reshape_heads(self.to_v_global(ref_global_tokens), self.heads, self.dim_head)
            out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            out = out.transpose(1, 2).reshape(batch_size, -1, self.heads * self.dim_head)
            ref_out = ref_out + torch.tanh(self.g_global) * out

        time_scale = _as_hidden_scale(corr_time_weight, image_hidden_states)
        if (
            self.local_enabled
            and self.to_k_region is not None
            and self.to_v_region is not None
            and self.g_region is not None
            and ref_region_tokens is not None
            and region_topm_idx is not None
        ):
            key = _reshape_heads(self.to_k_region(ref_region_tokens), self.heads, self.dim_head)
            value = _reshape_heads(self.to_v_region(ref_region_tokens), self.heads, self.dim_head)
            out = sparse_topk_cross_attention(
                query,
                key,
                value,
                topk_idx=region_topm_idx,
                topk_weight=region_topm_weight,
                prior_scale=corr_scale,
            )
            out = out.transpose(1, 2).reshape(batch_size, -1, self.heads * self.dim_head)
            ref_out = ref_out + torch.tanh(self.g_region) * time_scale * out

        if (
            self.sparse_enabled
            and self.to_k_local is not None
            and self.to_v_local is not None
            and self.g_sparse is not None
            and ref_local_tokens is not None
            and topk_idx is not None
        ):
            key = _reshape_heads(self.to_k_local(ref_local_tokens), self.heads, self.dim_head)
            value = _reshape_heads(self.to_v_local(ref_local_tokens), self.heads, self.dim_head)
            out = sparse_topk_cross_attention(
                query,
                key,
                value,
                topk_idx=topk_idx,
                topk_weight=topk_weight,
                prior_scale=corr_scale,
            )
            out = out.transpose(1, 2).reshape(batch_size, -1, self.heads * self.dim_head)
            if corr_conf is not None:
                corr_conf = _prepare_token_condition(corr_conf, batch_size, out.shape[1], mode="bilinear")
                out = out * corr_conf.to(device=out.device, dtype=out.dtype).unsqueeze(-1)
            ref_out = ref_out + torch.tanh(self.g_sparse) * time_scale * out

        ref_out = self.to_out_ref(ref_out).to(image_hidden_states.dtype)
        image_hidden_states = image_hidden_states + ref_scale * ref_out

        if context_hidden_states is not None:
            return image_hidden_states, context_hidden_states
        return image_hidden_states


def _layer_id_from_processor_name(name: str) -> int:
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part == "transformer_blocks" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return 0
    return 0


def inject_corr_reference_attention(
    transformer: nn.Module,
    reference_hidden_size: int,
    local_start_layer: int = 4,
    sparse_start_layer: int = 5,
) -> None:
    processors = {}
    hidden_size = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    for name in transformer.attn_processors:
        config = ReferenceAttentionConfig(
            hidden_size=hidden_size,
            heads=transformer.config.num_attention_heads,
            dim_head=transformer.config.attention_head_dim,
            reference_hidden_size=reference_hidden_size,
            layer_id=_layer_id_from_processor_name(name),
            local_start_layer=local_start_layer,
            sparse_start_layer=sparse_start_layer,
        )
        processors[name] = CorrReferenceAttentionProcessor(config)
    transformer.set_attn_processor(processors)


def add_sd3_transformer_lora(
    transformer: nn.Module,
    rank: int = 16,
    alpha: int | None = None,
    dropout: float = 0.0,
    layers: str | None = None,
    blocks: str | None = None,
) -> None:
    if rank <= 0:
        return
    if alpha is None:
        alpha = rank
    if layers:
        target_modules = [layer.strip() for layer in layers.split(",") if layer.strip()]
    else:
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
    if blocks:
        target_blocks = [int(block.strip()) for block in blocks.split(",") if block.strip()]
        target_modules = [f"transformer_blocks.{block}.{module}" for block in target_blocks for module in target_modules]

    transformer.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
    )


def freeze_transformer_for_adapter_training(transformer: nn.Module, train_content_projection: bool = True) -> None:
    for name, param in transformer.named_parameters():
        param.requires_grad = (
            (train_content_projection and name.startswith("pos_embed.proj."))
            or ".processor." in name
            or ".lora_A." in name
            or ".lora_B." in name
        )


def trainable_parameters(transformer: nn.Module, reference_adapter: nn.Module) -> list[nn.Parameter]:
    params = list(reference_adapter.parameters())
    params.extend(p for _, p in transformer.named_parameters() if p.requires_grad)
    return params


def encode_vae_latents(vae: nn.Module, images: torch.Tensor) -> torch.Tensor:
    latents = vae.encode(images).latent_dist.sample()
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


def decode_vae_latents(vae: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    latents = latents / vae.config.scaling_factor + vae.config.shift_factor
    return vae.decode(latents, return_dict=False)[0]


def zero_prompt_embeds(batch_size: int, sequence_length: int, joint_dim: int, pooled_dim: int, device, dtype):
    prompt_embeds = torch.zeros(batch_size, sequence_length, joint_dim, device=device, dtype=dtype)
    pooled_prompt_embeds = torch.zeros(batch_size, pooled_dim, device=device, dtype=dtype)
    return prompt_embeds, pooled_prompt_embeds


def get_corr_time_weight(sigmas: torch.Tensor) -> torch.Tensor:
    sigma = sigmas.flatten().float()
    weight = torch.where(
        sigma > 0.75,
        torch.full_like(sigma, 0.30),
        torch.where(sigma > 0.25, torch.ones_like(sigma), torch.full_like(sigma, 0.80)),
    )
    return weight
