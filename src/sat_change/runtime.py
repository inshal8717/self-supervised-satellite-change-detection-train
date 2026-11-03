from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .io import normalize_reflectance, read_image
from .model import SatelliteEncoder, feature_change


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint: str | Path, device: str = "cpu") -> SatelliteEncoder:
    """Load an encoder from a checkpoint written by ``sat_change.train``.

    Note: checkpoints are pickled; only load files you trust.
    """
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = payload.get("config", {})
    model = SatelliteEncoder(config.get("bands", 4), config.get("width", 32), config.get("projection_dim", 128))
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


def tile_starts(length: int, tile: int, step: int) -> list[int]:
    """Start offsets that cover ``[0, length)`` with windows of ``tile``, stepping by ``step``.

    The final window is always flush with the far edge, so every pixel is covered.
    """
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def score_pair(
    model: SatelliteEncoder,
    before_path: str | Path,
    after_path: str | Path,
    device: str = "cpu",
    tile_size: int = 512,
    overlap: int = 32,
) -> tuple[np.ndarray, dict]:
    """Dense change score (H x W) for a registered before/after pair.

    Both scenes are read and normalized in full; only the network forward pass
    runs tile by tile, with overlapping tiles averaged together.
    """
    if tile_size < 16 or tile_size <= overlap:
        raise ValueError("tile_size must be at least 16 and greater than overlap")
    before, profile = read_image(before_path)
    after, _ = read_image(after_path)
    if before.shape != after.shape:
        raise ValueError(f"Before/after shapes differ: {before.shape} vs {after.shape}; coregister first")
    bands = model.stem[0].in_channels
    a = torch.from_numpy(normalize_reflectance(before[:bands]))
    b = torch.from_numpy(normalize_reflectance(after[:bands]))
    h, w = a.shape[-2:]
    step = tile_size - overlap
    total = np.zeros((h, w), np.float32)
    weights = np.zeros((h, w), np.float32)
    with torch.inference_mode():
        for y in tile_starts(h, tile_size, step):
            for x in tile_starts(w, tile_size, step):
                window = (slice(None), slice(y, min(y + tile_size, h)), slice(x, min(x + tile_size, w)))
                score = feature_change(model, a[window][None].to(device), b[window][None].to(device))[0].cpu().numpy()
                th, tw = score.shape
                total[y:y + th, x:x + tw] += score
                weights[y:y + th, x:x + tw] += 1
    return total / np.maximum(weights, 1), profile
