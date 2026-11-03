"""Run change detection on one registered before/after pair.

Example:
    python -m sat_change.predict --before a.tif --after b.tif \\
        --checkpoint outputs/pretrain/best.pt --output outputs/change.tif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .io import write_image
from .postprocess import binary_mask
from .runtime import default_device, load_model, score_pair


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a dense self-supervised change map")
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True, help="path for the score map; the mask and JSON are written beside it")
    p.add_argument("--threshold", type=float, help="fixed threshold; default is Otsu on this scene")
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--device", default=default_device())
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    model = load_model(args.checkpoint, args.device)
    score, profile = score_pair(model, args.before, args.after, args.device, args.tile_size, args.overlap)
    mask, threshold = binary_mask(score, args.threshold)

    output = Path(args.output)
    mask_path = output.with_name(f"{output.stem}_mask{output.suffix}")
    write_image(output, score, profile)
    write_image(mask_path, mask.astype(np.float32), profile)
    summary = {"threshold": threshold, "changed_fraction": float(mask.mean())}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"score={output} mask={mask_path} threshold={threshold:.6f}")


if __name__ == "__main__":
    main()
