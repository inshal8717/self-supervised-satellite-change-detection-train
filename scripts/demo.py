from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True)


def has_real_data(root: str = "data/real") -> bool:
    """Check if real satellite data exists with at least one pair and masks."""
    r = Path(root)
    if not r.is_dir():
        return False
    from sat_change.io import pair_images, discover_images
    pairs = pair_images(r)
    if not pairs:
        return False
    masks = any(p.suffix.lower() in {".tif", ".tiff", ".npy"} for p in r.rglob("masks/*"))
    return masks


def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end demo: auto-selects real satellite data if available, otherwise generates synthetic data.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true", help="Force real satellite data (requires data/real/ with images + masks)")
    mode.add_argument("--synthetic", action="store_true", help="Force synthetic data (generates fresh scenes)")
    a = p.parse_args()

    use_real = a.real or (not a.synthetic and has_real_data())
    data_root = "data/real" if use_real else "data/synthetic"
    images_dir = data_root
    masks_dir = str(Path(data_root) / "masks") if use_real else str(Path(data_root) / "masks")

    if use_real:
        print(f"Using real satellite data from {data_root}")
    else:
        print("Generating synthetic data...")
        run("scripts/make_synthetic.py", "--output", "data/synthetic", "--scenes", "8", "--size", "96")
        images_dir = "data/synthetic/images"
        masks_dir = "data/synthetic/masks"

    print(f"Pretraining on {images_dir}...")
    run("-m", "sat_change.train", "--data", images_dir, "--output", "outputs/demo", "--epochs", "2", "--samples-per-epoch", "128", "--batch-size", "16", "--patch-size", "48")

    has_masks = Path(masks_dir).is_dir() and any(Path(masks_dir).iterdir())
    if has_masks:
        print(f"Evaluating on {images_dir} with masks from {masks_dir}...")
        run("-m", "sat_change.evaluate", "--images", images_dir, "--masks", masks_dir, "--checkpoint", "outputs/demo/best.pt", "--output", "outputs/demo/metrics.json")
    else:
        print("No ground-truth masks found — skipping evaluation (IoU/F1/AUROC unavailable).")
        print("Provide masks via --mask during fetch_sentinel2.py for evaluation.")


if __name__ == "__main__":
    main()

