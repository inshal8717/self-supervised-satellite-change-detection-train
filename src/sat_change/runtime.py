from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .io import normalize_reflectance, read_image
from .model import SatelliteEncoder, feature_change


def load_model(checkpoint: str | Path, device: str = "cpu") -> SatelliteEncoder:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = payload.get("config", {})
    model = SatelliteEncoder(config.get("bands", 4), config.get("width", 32), config.get("projection_dim", 128))
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


def score_pair(model: SatelliteEncoder, before_path: str | Path, after_path: str | Path, device: str = "cpu", tile_size: int = 512, overlap: int = 32) -> tuple[np.ndarray, dict]:
    before, profile = read_image(before_path); after, _ = read_image(after_path)
    if before.shape != after.shape:
        raise ValueError(f"Before/after shapes differ: {before.shape} vs {after.shape}; coregister first")
    bands = model.stem[0].in_channels
    a = torch.from_numpy(normalize_reflectance(before[:bands]))
    b = torch.from_numpy(normalize_reflectance(after[:bands]))
    h, w = a.shape[-2:]
    if tile_size <= overlap or tile_size < 16:
        raise ValueError("tile_size must be at least 16 and greater than overlap")
    total = np.zeros((h, w), np.float32); weights = np.zeros((h, w), np.float32)
    ys = list(range(0, max(h - tile_size, 0) + 1, tile_size - overlap)); xs = list(range(0, max(w - tile_size, 0) + 1, tile_size - overlap))
    if not ys or ys[-1] != max(0, h - tile_size): ys.append(max(0, h - tile_size))
    if not xs or xs[-1] != max(0, w - tile_size): xs.append(max(0, w - tile_size))
    with torch.inference_mode():
        for y in ys:
            for x in xs:
                aa = a[:, y:min(y + tile_size, h), x:min(x + tile_size, w)][None].to(device)
                bb = b[:, y:min(y + tile_size, h), x:min(x + tile_size, w)][None].to(device)
                score = feature_change(model, aa, bb)[0].cpu().numpy()
                th, tw = score.shape; total[y:y + th, x:x + tw] += score; weights[y:y + th, x:x + tw] += 1
    return total / np.maximum(weights, 1), profile
