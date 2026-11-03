"""Evaluate change maps against ground-truth masks.

Pairs are matched to masks by scene key. Masks may sit in one flat directory
or in per-scene ``masks/`` folders (the layout written by fetch_sentinel2.py).

Example:
    python -m sat_change.evaluate --images data/synthetic/images \\
        --masks data/synthetic/masks --checkpoint outputs/pretrain/best.pt --output outputs/metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .io import discover_masks, pair_images, read_image, scene_key
from .metrics import rank_auc, segmentation_metrics
from .postprocess import binary_mask
from .runtime import default_device, load_model, score_pair

log = logging.getLogger("sat_change.evaluate")
METRIC_KEYS = ("iou", "f1", "precision", "recall", "accuracy", "auroc")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate change maps against masks")
    p.add_argument("--images", required=True, help="root containing before/after imagery")
    p.add_argument("--masks", required=True, help="root containing ground-truth masks")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True, help="path for the JSON report")
    p.add_argument("--threshold", type=float, help="fixed threshold; default is per-scene Otsu")
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--device", default=default_device())
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    model = load_model(args.checkpoint, args.device)

    masks = {scene_key(path): path for path in discover_masks(args.masks)}
    rows = []
    skipped = []
    for before, after in pair_images(args.images):
        key = scene_key(before)
        if key not in masks:
            skipped.append(key)
            continue
        score, _ = score_pair(model, before, after, args.device, args.tile_size, args.overlap)
        pred, threshold = binary_mask(score, args.threshold)
        target, _ = read_image(masks[key])
        target = target[0] > 0.5
        rows.append({
            "scene": key,
            "threshold": threshold,
            **segmentation_metrics(pred, target),
            "auroc": rank_auc(score, target),
        })
        log.info("scored %s", key)

    if not rows:
        raise SystemExit(f"No image pairs matched masks by scene key (masks found: {sorted(masks)})")
    if skipped:
        log.warning("no mask for scene(s): %s", ", ".join(skipped))

    summary = {k: float(np.nanmean([r[k] for r in rows])) for k in METRIC_KEYS}
    report = {"summary": summary, "scenes": rows, "skipped_scenes": skipped}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
