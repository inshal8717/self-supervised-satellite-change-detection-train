"""Render one held-out scene as images for the project page.

Writes six PNGs and a small JSON file with the numbers shown in their captions:

    before, after      the two photos (red, green, blue bands)
    truth              where the scene really changed
    model              what the trained encoder flags as changed (shared stretch, the current method)
    model_old          the same network with each photo stretched on its own (the old, flawed scaling)
    raw                colour difference on the raw values, with no stretching

Usage (after scripts/reproduce_results.py has trained a checkpoint):
    python scripts/make_example_images.py
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sat_change.io import normalize_pair, normalize_reflectance, pair_images, read_image, scene_key  # noqa: E402
from sat_change.metrics import segmentation_metrics  # noqa: E402
from sat_change.postprocess import binary_mask  # noqa: E402
from sat_change.runtime import load_model, score_pair  # noqa: E402

SCALE = 3  # enlarge the 96 x 96 scenes so they are readable on the page
CHANGED = (255, 140, 0)  # orange
UNCHANGED = (38, 44, 53)  # dark slate


def per_photo_difference(before: Path, after: Path) -> np.ndarray:
    """Colour distance after stretching each photo on its own (the flawed scaling)."""
    xb = normalize_reflectance(read_image(before)[0])
    xa = normalize_reflectance(read_image(after)[0])
    return np.sqrt(((xb - xa) ** 2).sum(0))


def shared_difference(before: Path, after: Path) -> np.ndarray:
    """Colour distance after stretching both photos with one shared stretch."""
    xb, xa = normalize_pair(read_image(before)[0], read_image(after)[0])
    return np.sqrt(((xb - xa) ** 2).sum(0))


def raw_difference(before: Path, after: Path) -> np.ndarray:
    """Per-pixel Euclidean distance between the raw before and after values, with no normalisation."""
    return np.sqrt(((read_image(before)[0] - read_image(after)[0]) ** 2).sum(0))


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Write an H x W x 3 uint8 array as a PNG, with no imaging library."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def upscale(rgb: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(rgb, SCALE, axis=0), SCALE, axis=1)


def photo(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Bands 2, 1, 0 (red, green, blue), stretched with a shared range."""
    img = np.moveaxis(arr[[2, 1, 0]], 0, -1)
    return (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def mask_image(mask: np.ndarray) -> np.ndarray:
    out = np.empty((*mask.shape, 3), np.uint8)
    out[:] = UNCHANGED
    out[mask.astype(bool)] = CHANGED
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heldout", default=str(ROOT / "data" / "reproduce" / "heldout"))
    parser.add_argument("--checkpoint", default=str(ROOT / "data" / "reproduce" / "outputs" / "best.pt"))
    parser.add_argument("--output", default=str(ROOT / "docs" / "img"))
    parser.add_argument("--data", default=str(ROOT / "docs" / "data"))
    args = parser.parse_args()

    heldout = Path(args.heldout)
    before_path, after_path = pair_images(heldout)[0]  # first held-out scene
    key = scene_key(before_path)
    truth = read_image(heldout / "masks" / f"{key}.npy")[0][0] > 0.5

    model = load_model(args.checkpoint)
    model_score, _ = score_pair(model, before_path, after_path, tile_size=96, overlap=16)
    model_mask, _ = binary_mask(model_score)
    old_score, _ = score_pair(model, before_path, after_path, tile_size=96, overlap=16, stretch="per_photo")
    old_mask, _ = binary_mask(old_score)
    raw_mask, _ = binary_mask(raw_difference(before_path, after_path))

    before, _ = read_image(before_path)
    after, _ = read_image(after_path)
    both = np.stack([before[[2, 1, 0]], after[[2, 1, 0]]])
    lo, hi = np.percentile(both, [1, 99])

    out = Path(args.output)
    panels = {
        "before": photo(before, lo, hi),
        "after": photo(after, lo, hi),
        "truth": mask_image(truth),
        "model": mask_image(model_mask),
        "model_old": mask_image(old_mask),
        "raw": mask_image(raw_mask),
    }
    for name, rgb in panels.items():
        write_png(out / f"example_{name}.png", upscale(rgb))

    summary = {
        "scene": key,
        "truth_changed_pct": float(truth.mean() * 100),
        "model_changed_pct": float(model_mask.mean() * 100),
        "model_old_changed_pct": float(old_mask.mean() * 100),
        "raw_changed_pct": float(raw_mask.mean() * 100),
        "model_f1": segmentation_metrics(model_mask, truth)["f1"],
        "model_old_f1": segmentation_metrics(old_mask, truth)["f1"],
        "raw_f1": segmentation_metrics(raw_mask, truth)["f1"],
    }
    data = Path(args.data)
    data.mkdir(parents=True, exist_ok=True)
    (data / "example.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
