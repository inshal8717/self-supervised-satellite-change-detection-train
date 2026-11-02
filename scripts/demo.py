"""One-command end-to-end demo: generate or use data, pretrain briefly, evaluate.

Uses real data from ``data/real`` when it contains image pairs and masks;
otherwise generates synthetic scenes. Runs from any working directory.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REAL_ROOT = ROOT / "data" / "real"
SYNTH_ROOT = ROOT / "data" / "synthetic"
DEMO_OUT = ROOT / "outputs" / "demo"


def run(*args: str) -> None:
    """Run a Python command from the repo root, with ``src/`` importable even if the package isn't installed."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")]))}
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT, env=env)


def has_real_data() -> bool:
    """True when ``data/real`` has at least one image pair and at least one mask."""
    sys.path.insert(0, str(ROOT / "src"))
    from sat_change.io import discover_masks, pair_images

    return REAL_ROOT.is_dir() and bool(pair_images(REAL_ROOT)) and bool(discover_masks(REAL_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true", help="use data/real (requires image pairs and masks)")
    mode.add_argument("--synthetic", action="store_true", help="always generate fresh synthetic scenes")
    args = parser.parse_args()

    use_real = args.real or (not args.synthetic and has_real_data())
    if use_real:
        data_root = REAL_ROOT
        print(f"Using real satellite data from {data_root}")
    else:
        data_root = SYNTH_ROOT
        print("Generating synthetic data...")
        run("scripts/make_synthetic.py", "--output", str(SYNTH_ROOT), "--scenes", "8", "--size", "96")

    # Synthetic data keeps imagery in images/; real data is already per-scene.
    images_dir = data_root if use_real else data_root / "images"
    print(f"Pretraining on {images_dir}...")
    run(
        "-m", "sat_change.train",
        "--data", str(images_dir),
        "--output", str(DEMO_OUT),
        "--epochs", "2",
        "--samples-per-epoch", "128",
        "--batch-size", "16",
        "--patch-size", "48",
    )

    # Masks are found recursively, so both the flat synthetic layout and the
    # per-scene real layout are covered.
    print(f"Evaluating on {data_root} with masks found under it...")
    run(
        "-m", "sat_change.evaluate",
        "--images", str(data_root),
        "--masks", str(data_root),
        "--checkpoint", str(DEMO_OUT / "best.pt"),
        "--output", str(DEMO_OUT / "metrics.json"),
    )


if __name__ == "__main__":
    main()
