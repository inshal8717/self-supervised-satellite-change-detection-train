from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def smooth(x: np.ndarray, rounds: int = 5) -> np.ndarray:
    for _ in range(rounds):
        x = (x + np.roll(x, 1, -1) + np.roll(x, -1, -1) + np.roll(x, 1, -2) + np.roll(x, -1, -2)) / 5
    return x


def make_scene(rng: np.random.Generator, size: int, kind: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    field = smooth(rng.random((size, size), dtype=np.float32), 10)
    before = np.stack([.12 + .15 * field, .2 + .3 * field, .12 + .12 * field, .35 + .5 * field]).astype(np.float32)
    after = before.copy(); mask = np.zeros((size, size), np.uint8)
    yy, xx = np.ogrid[:size, :size]
    cy, cx = rng.integers(size // 4, 3 * size // 4, 2); ry, rx = rng.integers(size // 10, size // 4, 2)
    mask[((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1] = 1
    if kind % 3 == 0:  # deforestation: vegetation to soil
        after[:, mask > 0] = np.array([.32, .28, .22, .25])[:, None]
    elif kind % 3 == 1:  # urbanization: bright built surface
        after[:, mask > 0] = np.array([.5, .48, .5, .25])[:, None]
    else:  # flooding: dark visible/NIR
        after[:, mask > 0] = np.array([.08, .09, .07, .03])[:, None]
    before += rng.normal(0, .01, before.shape).astype(np.float32); after += rng.normal(0, .01, after.shape).astype(np.float32)
    return np.clip(before, 0, 1), np.clip(after, 0, 1), mask


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--output", default="data/synthetic"); p.add_argument("--scenes", type=int, default=12); p.add_argument("--size", type=int, default=128); p.add_argument("--seed", type=int, default=7); a = p.parse_args()
    out = Path(a.output); images = out / "images"; masks = out / "masks"; images.mkdir(parents=True, exist_ok=True); masks.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for i in range(a.scenes):
        before, after, mask = make_scene(rng, a.size, i); key = f"scene{i:03d}"
        np.save(images / f"{key}_20230101.npy", before); np.save(images / f"{key}_20240101.npy", after); np.save(masks / f"{key}.npy", mask)
    print(f"wrote {a.scenes} paired scenes to {out}")


if __name__ == "__main__": main()

