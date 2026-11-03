"""Generate paired synthetic scenes with ground-truth change masks.

Each scene is a smooth random background with one elliptical region that
changes between the "before" and "after" dates. Output layout:

    <output>/images/sceneNNN_20230101.npy   (before)
    <output>/images/sceneNNN_20240101.npy   (after)
    <output>/masks/sceneNNN.npy             (binary change mask)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Spectral signature of the changed region for each change type (4 bands).
CHANGE_SIGNATURES = (
    np.array([0.32, 0.28, 0.22, 0.25]),  # deforestation: vegetation to bare soil
    np.array([0.50, 0.48, 0.50, 0.25]),  # urbanisation: bright built surface
    np.array([0.08, 0.09, 0.07, 0.03]),  # flooding: dark in visible and NIR
)


def smooth(x: np.ndarray, rounds: int) -> np.ndarray:
    """Repeated 4-neighbour averaging with wrap-around."""
    for _ in range(rounds):
        x = (x + np.roll(x, 1, -1) + np.roll(x, -1, -1) + np.roll(x, 1, -2) + np.roll(x, -1, -2)) / 5
    return x


def make_scene(rng: np.random.Generator, size: int, kind: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (before, after, mask) for one scene; ``kind`` selects the change type."""
    field = smooth(rng.random((size, size), dtype=np.float32), rounds=10)
    before = np.stack([
        0.12 + 0.15 * field,
        0.20 + 0.30 * field,
        0.12 + 0.12 * field,
        0.35 + 0.50 * field,
    ]).astype(np.float32)

    yy, xx = np.ogrid[:size, :size]
    cy, cx = rng.integers(size // 4, 3 * size // 4, 2)
    ry, rx = rng.integers(size // 10, size // 4, 2)
    mask = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1).astype(np.uint8)

    after = before.copy()
    after[:, mask > 0] = CHANGE_SIGNATURES[kind % len(CHANGE_SIGNATURES)][:, None]

    before += rng.normal(0, 0.01, before.shape).astype(np.float32)
    after += rng.normal(0, 0.01, after.shape).astype(np.float32)
    return np.clip(before, 0, 1), np.clip(after, 0, 1), mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out = Path(args.output)
    images = out / "images"
    masks = out / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    for i in range(args.scenes):
        before, after, mask = make_scene(rng, args.size, i)
        key = f"scene{i:03d}"
        np.save(images / f"{key}_20230101.npy", before)
        np.save(images / f"{key}_20240101.npy", after)
        np.save(masks / f"{key}.npy", mask)
    print(f"wrote {args.scenes} paired scenes to {out}")


if __name__ == "__main__":
    main()
