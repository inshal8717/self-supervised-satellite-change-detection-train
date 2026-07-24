from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .io import pair_images, read_image, scene_key
from .metrics import rank_auc, segmentation_metrics
from .postprocess import binary_mask
from .runtime import load_model, score_pair


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--images", required=True); p.add_argument("--masks", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True)
    p.add_argument("--threshold", type=float); p.add_argument("--tile-size", type=int, default=512); p.add_argument("--overlap", type=int, default=32); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); a = p.parse_args()
    model = load_model(a.checkpoint, a.device); rows = []
    masks = {scene_key(x): x for x in Path(a.masks).glob("*") if x.suffix.lower() in {".npy", ".tif", ".tiff"}}
    for before, after in pair_images(a.images):
        key = scene_key(before)
        if key not in masks: continue
        score, _ = score_pair(model, before, after, a.device, a.tile_size, a.overlap); pred, threshold = binary_mask(score, a.threshold)
        target, _ = read_image(masks[key]); target = target[0] > 0.5
        row = {"scene": key, "threshold": threshold, **segmentation_metrics(pred, target), "auroc": rank_auc(score, target)}; rows.append(row)
    if not rows: raise ValueError("No image pairs matched masks by scene key")
    summary = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k not in {"scene", "threshold"}}
    result = {"summary": summary, "scenes": rows}; Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(json.dumps(result, indent=2)); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
