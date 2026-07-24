from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .io import write_image
from .postprocess import binary_mask
from .runtime import load_model, score_pair


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a dense self-supervised change map")
    p.add_argument("--before", required=True); p.add_argument("--after", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True)
    p.add_argument("--threshold", type=float); p.add_argument("--tile-size", type=int, default=512); p.add_argument("--overlap", type=int, default=32); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); model = load_model(a.checkpoint, a.device); score, profile = score_pair(model, a.before, a.after, a.device, a.tile_size, a.overlap)
    mask, threshold = binary_mask(score, a.threshold); output = Path(a.output)
    write_image(output, score, profile)
    mask_path = output.with_name(output.stem + "_mask" + output.suffix); write_image(mask_path, mask.astype(np.float32), profile)
    output.with_suffix(".json").write_text(json.dumps({"threshold": threshold, "changed_fraction": float(mask.mean())}, indent=2))
    print(f"score={output} mask={mask_path} threshold={threshold:.6f}")


if __name__ == "__main__": main()
