from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class CorrState:
    topk_idx: torch.Tensor
    topk_weight: torch.Tensor
    corr_conf: torch.Tensor
    region_topm_idx: torch.Tensor
    region_topm_weight: torch.Tensor
    label_c: torch.Tensor
    label_r: torch.Tensor
    grid_hw: tuple[int, int]

    def to(self, device: torch.device, dtype: torch.dtype | None = None) -> "CorrState":
        float_dtype = dtype or torch.float32
        return CorrState(
            topk_idx=self.topk_idx.to(device=device, dtype=torch.long),
            topk_weight=self.topk_weight.to(device=device, dtype=float_dtype),
            corr_conf=self.corr_conf.to(device=device, dtype=float_dtype),
            region_topm_idx=self.region_topm_idx.to(device=device, dtype=torch.long),
            region_topm_weight=self.region_topm_weight.to(device=device, dtype=float_dtype),
            label_c=self.label_c.to(device=device, dtype=torch.long),
            label_r=self.label_r.to(device=device, dtype=torch.long),
            grid_hw=self.grid_hw,
        )


def cache_path_for_index(cache_dir: str | Path, index: int) -> Path:
    return Path(cache_dir) / f"{index:08d}.npz"


def load_corr_npz(path: str | Path) -> dict[str, torch.Tensor | tuple[int, int]]:
    data = np.load(path)
    grid = tuple(int(x) for x in data["grid_hw"].tolist())
    return {
        "topk_idx": torch.from_numpy(data["topk_idx"].astype(np.int64)),
        "topk_weight": torch.from_numpy(data["topk_weight"].astype(np.float32)),
        "corr_conf": torch.from_numpy(data["corr_conf"].astype(np.float32)),
        "region_topm_idx": torch.from_numpy(data["region_topm_idx"].astype(np.int64)),
        "region_topm_weight": torch.from_numpy(data["region_topm_weight"].astype(np.float32)),
        "label_c": torch.from_numpy(data["label_c"].astype(np.int64)),
        "label_r": torch.from_numpy(data["label_r"].astype(np.int64)),
        "grid_hw": grid,
    }


def corr_state_from_batch(batch: dict, device: torch.device, dtype: torch.dtype) -> CorrState:
    grid_hw = batch["grid_hw"]
    if isinstance(grid_hw, torch.Tensor):
        grid_hw = tuple(int(x) for x in grid_hw[0].tolist())
    elif isinstance(grid_hw, (list, tuple)) and grid_hw and isinstance(grid_hw[0], torch.Tensor):
        grid_hw = (int(grid_hw[0][0]), int(grid_hw[1][0]))
    else:
        grid_hw = tuple(int(x) for x in grid_hw)
    state = CorrState(
        topk_idx=batch["topk_idx"],
        topk_weight=batch["topk_weight"],
        corr_conf=batch["corr_conf"],
        region_topm_idx=batch["region_topm_idx"],
        region_topm_weight=batch["region_topm_weight"],
        label_c=batch["label_c"],
        label_r=batch["label_r"],
        grid_hw=grid_hw,
    )
    return state.to(device=device, dtype=dtype)


def save_corr_npz(
    path: str | Path,
    *,
    grid_hw: tuple[int, int],
    topk_idx: np.ndarray,
    topk_weight: np.ndarray,
    corr_conf: np.ndarray,
    region_topm_idx: np.ndarray,
    region_topm_weight: np.ndarray,
    label_c: np.ndarray,
    label_r: np.ndarray,
    region_sim: np.ndarray | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "grid_hw": np.asarray(grid_hw, dtype=np.int16),
        "topk_idx": topk_idx.astype(np.uint16),
        "topk_weight": topk_weight.astype(np.float16),
        "corr_conf": corr_conf.astype(np.float16),
        "region_topm_idx": region_topm_idx.astype(np.uint8),
        "region_topm_weight": region_topm_weight.astype(np.float16),
        "label_c": label_c.astype(np.uint8),
        "label_r": label_r.astype(np.uint8),
    }
    if region_sim is not None:
        payload["region_sim"] = region_sim.astype(np.float16)
    np.savez_compressed(path, **payload)
