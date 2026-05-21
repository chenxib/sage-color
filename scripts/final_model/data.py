from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from scripts.final_model.corr_state import cache_path_for_index, load_corr_npz


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_to_tensor(image: Image.Image, resolution: int) -> torch.Tensor:
    image = image.resize((resolution, resolution), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


class ColorEditJsonlDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        resolution: int,
        corr_cache_dir: str | Path | None = None,
        load_corr_cache: bool = False,
    ):
        self.manifest = Path(manifest)
        self.resolution = resolution
        self.corr_cache_dir = Path(corr_cache_dir) if corr_cache_dir else None
        self.load_corr_cache = load_corr_cache
        with self.manifest.open("r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        if not self.rows:
            raise ValueError(f"No rows found in {self.manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | tuple[int, int]]:
        row = self.rows[index]
        sample = {
            "content": resize_to_tensor(load_rgb(row["content_image"]), self.resolution),
            "reference": resize_to_tensor(load_rgb(row["reference_image"]), self.resolution),
            "target": resize_to_tensor(load_rgb(row["target_image"]), self.resolution),
            "content_image_path": row["content_image"],
            "reference_image_path": row["reference_image"],
            "target_image_path": row["target_image"],
            "caption": row.get("caption", ""),
        }
        if self.load_corr_cache:
            if self.corr_cache_dir is None and not row.get("corr_cache"):
                raise ValueError("load_corr_cache=True requires corr_cache_dir or per-row corr_cache.")
            cache_path = row.get("corr_cache") or cache_path_for_index(self.corr_cache_dir, index)
            cache_path = Path(cache_path)
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Missing final model correspondence cache for row {index}: {cache_path}. "
                    "Run scripts/final_model/build_corr_cache.py first."
                )
            sample.update(load_corr_npz(cache_path))
        return sample
