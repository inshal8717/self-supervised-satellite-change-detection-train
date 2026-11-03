from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .io import discover_images, normalize_reflectance, read_image


def augment(x: torch.Tensor) -> torch.Tensor:
    """Satellite-safe spatial and spectral augmentation for one C x H x W patch."""
    if random.random() < 0.5:
        x = x.flip(-1)
    if random.random() < 0.5:
        x = x.flip(-2)
    x = torch.rot90(x, random.randrange(4), (-2, -1))
    gain = torch.empty(x.shape[0], 1, 1).uniform_(0.85, 1.15)
    bias = torch.empty(x.shape[0], 1, 1).uniform_(-0.05, 0.05)
    x = x * gain + bias
    if random.random() < 0.5:
        x = x + torch.randn_like(x) * 0.015
    if random.random() < 0.25:
        x = F.avg_pool2d(x[None], 3, 1, 1)[0]
    return x.clamp(0, 1)


class ContrastivePatchDataset(Dataset):
    """Yields two augmented views of a random patch from a random image.

    Masks are excluded from discovery, so only imagery is sampled.
    """

    def __init__(self, root: str | Path, patch_size: int = 64, samples_per_epoch: int = 4096, bands: int = 4):
        self.paths = discover_images(root)
        if not self.paths:
            raise ValueError(f"No .npy/.tif imagery found in {root} (masks are excluded)")
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.bands = bands

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[index % len(self.paths)]
        arr, _ = read_image(path)
        if arr.shape[0] < self.bands:
            raise ValueError(f"{path} has {arr.shape[0]} band(s); the model expects {self.bands}")
        arr = normalize_reflectance(arr[: self.bands])
        _, h, w = arr.shape
        if min(h, w) < self.patch_size:
            pad_h, pad_w = max(0, self.patch_size - h), max(0, self.patch_size - w)
            arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            _, h, w = arr.shape
        y = random.randint(0, h - self.patch_size)
        x = random.randint(0, w - self.patch_size)
        patch = torch.from_numpy(arr[:, y:y + self.patch_size, x:x + self.patch_size].copy())
        return augment(patch.clone()), augment(patch.clone())
