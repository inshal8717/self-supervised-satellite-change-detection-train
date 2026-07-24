from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


def read_image(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read CxHxW imagery and return array plus geospatial profile."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        return arr, {}
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("GeoTIFF support requires: pip install -e '.[geo]'") from exc
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()
        if src.nodata is not None:
            arr[:, np.any(arr == src.nodata, axis=0)] = np.nan
    return arr, profile


def write_image(path: str | Path, arr: np.ndarray, profile: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        np.save(path, arr.astype(np.float32))
        return
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("GeoTIFF support requires: pip install -e '.[geo]'") from exc
    data = arr[None] if arr.ndim == 2 else arr
    out_profile = dict(profile or {})
    out_profile.update(driver="GTiff", height=data.shape[-2], width=data.shape[-1], count=data.shape[0], dtype="float32", compress="deflate")
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data.astype(np.float32))


def normalize_reflectance(arr: np.ndarray) -> np.ndarray:
    """Robust per-band normalization preserving nodata as zeros."""
    x = arr.astype(np.float32, copy=True)
    finite = np.isfinite(x)
    if finite.any() and np.nanpercentile(x, 99) > 2:
        x /= 10000.0
    for band in range(x.shape[0]):
        valid = np.isfinite(x[band])
        if not valid.any():
            x[band] = 0
            continue
        lo, hi = np.percentile(x[band, valid], [2, 98])
        x[band] = np.clip((x[band] - lo) / max(float(hi - lo), 1e-6), 0, 1)
    return np.nan_to_num(x)


def scene_key(path: str | Path) -> str:
    return re.sub(r"[-_]?\d{4}[-]?\d{2}[-]?\d{2}$", "", Path(path).stem)


def discover_images(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in {".npy", ".tif", ".tiff"})


def pair_images(root: str | Path) -> list[tuple[Path, Path]]:
    groups: dict[str, list[Path]] = {}
    for path in discover_images(root):
        groups.setdefault(scene_key(path), []).append(path)
    pairs = []
    for paths in groups.values():
        paths.sort()
        if len(paths) >= 2:
            pairs.append((paths[0], paths[-1]))
    return pairs
