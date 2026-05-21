from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw, ImageFont, ImageOps


@dataclass
class DistributedContext:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def init_distributed() -> DistributedContext:
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed inference with NCCL requires CUDA.")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=device)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return DistributedContext(True, rank, world_size, local_rank, device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistributedContext(False, 0, 1, 0, device)


def barrier(ctx: DistributedContext) -> None:
    if ctx.distributed:
        dist.barrier()


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.distributed and dist.is_initialized():
        dist.destroy_process_group()


def get_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float32


def rank_print(ctx: DistributedContext, message: str) -> None:
    print(f"[rank {ctx.rank}/{ctx.world_size}] {message}", flush=True)


def rank0_print(ctx: DistributedContext, message: str) -> None:
    if ctx.rank == 0:
        print(message, flush=True)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = (tensor.detach().float().clamp(-1, 1) + 1.0) / 2.0
    array = (tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def sanitize_filename(text: str, max_len: int = 80) -> str:
    text = Path(text).stem
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    text = text.strip("._-")
    if not text:
        text = "sample"
    return text[:max_len]


def resolve_image_path(path_value: str, jsonl_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (jsonl_dir / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_idx}: {exc}") from exc
            for key in ("source", "reference", "target"):
                if key not in item:
                    raise KeyError(f'Line {line_idx} is missing required key "{key}".')
            records.append(item)
    return records


def check_image_exists(path: Path, key: str, idx: int) -> None:
    if not path.exists():
        raise FileNotFoundError(f'Image file does not exist. line_index={idx}, key="{key}", path="{path}"')


def letterbox_image(
    image: Image.Image,
    cell_size: tuple[int, int],
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    image = image.convert("RGB")
    cell_w, cell_h = cell_size
    if cell_w <= 0 or cell_h <= 0:
        raise ValueError(f"Invalid cell_size: {cell_size}")
    fitted = ImageOps.contain(image, (cell_w, cell_h), method=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (cell_w, cell_h), background)
    canvas.paste(fitted, ((cell_w - fitted.width) // 2, (cell_h - fitted.height) // 2))
    return canvas


def make_merged_image(
    source_image: Image.Image,
    reference_image: Image.Image,
    target_image: Image.Image,
    result_image: Image.Image,
    cell_size: tuple[int, int],
) -> Image.Image:
    labels = ["source", "reference", "target", "result"]
    panels = [
        letterbox_image(source_image, cell_size),
        letterbox_image(reference_image, cell_size),
        letterbox_image(target_image, cell_size),
        letterbox_image(result_image, cell_size),
    ]
    cell_w, cell_h = cell_size
    label_h = max(32, cell_h // 32)
    merged = Image.new("RGB", (cell_w * len(panels), cell_h + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(merged)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, (label, panel) in enumerate(zip(labels, panels)):
        x = idx * cell_w
        merged.paste(panel, (x, label_h))
        if font is not None:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else:
            text_w = len(label) * 8
            text_h = 12
        draw.text((x + (cell_w - text_w) // 2, (label_h - text_h) // 2), label, fill=(0, 0, 0), font=font)
    return merged


def make_corr_cache_path(corr_cache_dir: Path, idx: int, source_path: Path, reference_path: Path) -> Path:
    source_name = sanitize_filename(source_path.name, max_len=48)
    reference_name = sanitize_filename(reference_path.name, max_len=48)
    return corr_cache_dir / f"{idx:06d}_{source_name}__{reference_name}.npz"


def ensure_2d_or_batched(tensor: torch.Tensor, expected_ndim_without_batch: int) -> torch.Tensor:
    if tensor.ndim == expected_ndim_without_batch:
        return tensor.unsqueeze(0)
    return tensor


def aggregate_rank_jsonl(tmp_dir: Path, output_jsonl: Path, world_size: int, num_records: int) -> None:
    indexed_records: dict[int, dict[str, Any]] = {}
    for rank in range(world_size):
        part_path = tmp_dir / f"results.rank{rank:05d}.jsonl"
        if not part_path.exists():
            raise FileNotFoundError(f"Missing rank output JSONL: {part_path}")
        with part_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                package = json.loads(line)
                idx = int(package["__idx"])
                if idx in indexed_records:
                    raise RuntimeError(f"Duplicate record index during aggregation: {idx}")
                indexed_records[idx] = package["record"]

    if len(indexed_records) != num_records:
        missing = sorted(set(range(num_records)) - set(indexed_records.keys()))
        raise RuntimeError(
            f"Aggregation failed. Expected {num_records} records, got {len(indexed_records)}. "
            f"Missing indices preview: {missing[:20]}"
        )

    with output_jsonl.open("w", encoding="utf-8") as writer:
        for idx in range(num_records):
            writer.write(json.dumps(indexed_records[idx], ensure_ascii=False) + "\n")
