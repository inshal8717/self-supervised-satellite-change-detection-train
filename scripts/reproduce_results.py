"""Reproduce the published results on synthetic data with a held-out split.

Generates 12 synthetic scenes, pretrains on the first 8 (imagery only, no
masks), and evaluates on the last 4, which the model never sees during
training. The training curve and held-out metrics are copied into docs/data/
for the project page.

Usage (from the repo root, after installing the package):
    python scripts/reproduce_results.py
    python scripts/build_site.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import make_synthetic  # noqa: E402

from sat_change import evaluate, train  # noqa: E402
from sat_change.io import discover_masks, normalize_reflectance, pair_images, read_image, scene_key  # noqa: E402
from sat_change.metrics import rank_auc, segmentation_metrics  # noqa: E402
from sat_change.model import SatelliteEncoder  # noqa: E402
from sat_change.postprocess import binary_mask  # noqa: E402
from sat_change.runtime import load_model, score_pair  # noqa: E402

TRAIN_SCENES = range(0, 8)
HELDOUT_SCENES = range(8, 12)


def spectral_difference(before: Path, after: Path) -> np.ndarray:
    """Per-pixel Euclidean distance between normalised before and after reflectance."""
    xb = normalize_reflectance(read_image(before)[0])
    xa = normalize_reflectance(read_image(after)[0])
    return np.sqrt(((xb - xa) ** 2).sum(0))


def compare_baselines(heldout: Path, checkpoint: Path) -> dict[str, dict[str, float]]:
    """Held-out metrics for the trained encoder and two baselines, averaged over scenes.

    The random encoder uses the same architecture with untrained weights, which
    shows how much of the signal comes from the architecture alone.
    """
    masks = {scene_key(p): p for p in discover_masks(heldout)}
    trained = load_model(checkpoint)
    torch.manual_seed(0)
    random_encoder = SatelliteEncoder(4, 32, 128).eval()
    scorers = {
        "trained encoder": lambda b, a: score_pair(trained, b, a, tile_size=96, overlap=16)[0],
        "random encoder": lambda b, a: score_pair(random_encoder, b, a, tile_size=96, overlap=16)[0],
        "spectral difference": spectral_difference,
    }
    per_scene: dict[str, list[dict[str, float]]] = {name: [] for name in scorers}
    for before, after in pair_images(heldout):
        key = scene_key(before)
        target = read_image(masks[key])[0][0] > 0.5
        for name, score_fn in scorers.items():
            score = score_fn(before, after)
            pred, _ = binary_mask(score)
            per_scene[name].append({**segmentation_metrics(pred, target), "auroc": rank_auc(score, target)})
    return {
        name: {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
        for name, rows in per_scene.items()
    }


def scene_files(scene: int, kind: str) -> list[str]:
    key = f"scene{scene:03d}"
    if kind == "images":
        return [f"{key}_20230101.npy", f"{key}_20240101.npy"]
    return [f"{key}.npy"]


def copy_scenes(src: Path, dst: Path, scenes, include_masks: bool) -> None:
    (dst / "images").mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        for name in scene_files(scene, "images"):
            shutil.copy2(src / "images" / name, dst / "images" / name)
        if include_masks:
            (dst / "masks").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / "masks" / scene_files(scene, "masks")[0], dst / "masks" / scene_files(scene, "masks")[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--work", default=str(ROOT / "data" / "reproduce"), help="scratch directory for data and checkpoints")
    parser.add_argument("--docs", default=str(ROOT / "docs" / "data"), help="where to write history.json and metrics.json")
    args = parser.parse_args()

    work = Path(args.work)
    synth = work / "synthetic"
    train_dir = work / "train"
    heldout_dir = work / "heldout"
    outputs = work / "outputs"
    for d in (train_dir, heldout_dir):
        shutil.rmtree(d, ignore_errors=True)

    print("1/5 generating 12 synthetic scenes")
    sys.argv = ["make_synthetic", "--output", str(synth), "--scenes", "12", "--size", "96", "--seed", "7"]
    make_synthetic.main()

    print("2/5 splitting: scenes 0-7 for pretraining, 8-11 held out")
    copy_scenes(synth, train_dir, TRAIN_SCENES, include_masks=False)
    copy_scenes(synth, heldout_dir, HELDOUT_SCENES, include_masks=True)

    print(f"3/5 pretraining for {args.epochs} epochs on CPU")
    train.main([
        "--data", str(train_dir), "--output", str(outputs), "--epochs", str(args.epochs),
        "--samples-per-epoch", "1024", "--batch-size", "32", "--patch-size", "48",
        "--seed", "42", "--device", "cpu",
    ])

    print("4/5 evaluating on the held-out scenes")
    evaluate.main([
        "--images", str(heldout_dir), "--masks", str(heldout_dir), "--checkpoint", str(outputs / "best.pt"),
        "--output", str(outputs / "metrics.json"), "--device", "cpu", "--tile-size", "96", "--overlap", "16",
    ])

    print("5/5 comparing against baselines on the same held-out scenes")
    baselines = compare_baselines(heldout_dir, outputs / "best.pt")
    for name, summary in baselines.items():
        print(f"  {name:20s} AUROC={summary['auroc']:.3f}  F1={summary['f1']:.3f}  IoU={summary['iou']:.3f}")

    docs = Path(args.docs)
    docs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(outputs / "history.json", docs / "history.json")
    shutil.copy2(outputs / "metrics.json", docs / "metrics.json")
    (docs / "baselines.json").write_text(json.dumps(baselines, indent=2))
    print(f"done. wrote results to {docs}")


if __name__ == "__main__":
    main()
