"""Reading, writing and discovering satellite imagery.

Imagery is C x H x W, stored as ``.npy`` or ``.tif``/``.tiff``. Ground-truth
masks live in directories named ``masks`` and are never treated as imagery.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

IMAGE_SUFFIXES = {".npy", ".tif", ".tiff"}
MASK_DIR_NAME = "masks"


def read_image(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a C x H x W array and its geospatial profile (empty for ``.npy``)."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        return arr, {}
    rasterio = _import_rasterio()
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()
        if src.nodata is not None:
            arr[:, np.any(arr == src.nodata, axis=0)] = np.nan
    return arr, profile


def write_image(path: str | Path, arr: np.ndarray, profile: dict[str, Any] | None = None) -> None:
    """Write a 2-D or C x H x W array as ``.npy`` or GeoTIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        np.save(path, arr.astype(np.float32))
        return
    rasterio = _import_rasterio()
    data = arr[None] if arr.ndim == 2 else arr
    out_profile = dict(profile or {})
    out_profile.update(driver="GTiff", height=data.shape[-2], width=data.shape[-1], count=data.shape[0], dtype="float32", compress="deflate")
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data.astype(np.float32))


def normalize_reflectance(arr: np.ndarray) -> np.ndarray:
    """Scale each band to [0, 1] using its 2nd-98th percentiles.

    Sentinel-2 integer reflectance (0..10000) is detected and rescaled first.
    NaN (nodata) pixels become 0.
    """
    x = arr.astype(np.float32, copy=True)
    if np.isfinite(x).any() and np.nanpercentile(x, 99) > 2:
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
    """Scene name with a trailing YYYY-MM-DD / YYYYMMDD date removed."""
    return re.sub(r"[-_]?\d{4}[-]?\d{2}[-]?\d{2}$", "", Path(path).stem)


def discover_images(root: str | Path) -> list[Path]:
    """All imagery under ``root``, excluding anything inside a ``masks`` directory."""
    return sorted(
        p for p in Path(root).rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES and MASK_DIR_NAME not in p.parent.parts
    )


def discover_masks(root: str | Path) -> list[Path]:
    """Mask files under ``root``: any imagery inside a ``masks`` directory."""
    return sorted(
        p for p in Path(root).rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES and MASK_DIR_NAME in p.parent.parts
    )


def pair_images(root: str | Path) -> list[tuple[Path, Path]]:
    """Pair the earliest and latest acquisition of each scene under ``root``."""
    groups: dict[str, list[Path]] = {}
    for path in discover_images(root):
        groups.setdefault(scene_key(path), []).append(path)
    pairs = []
    for paths in groups.values():
        paths.sort()
        if len(paths) >= 2:
            pairs.append((paths[0], paths[-1]))
    return pairs


def _import_rasterio():
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("GeoTIFF support requires: pip install -e '.[geo]'") from exc
    return rasterio
