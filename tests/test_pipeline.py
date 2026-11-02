"""End-to-end smoke test: synthetic data -> train -> predict -> evaluate.

Uses tiny sizes so it runs on CPU in seconds.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import make_synthetic  # noqa: E402
from sat_change import evaluate, predict, train  # noqa: E402


def test_full_pipeline_on_synthetic_data(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    outputs = tmp_path / "outputs"

    monkeypatch.setattr(sys, "argv", ["make_synthetic", "--output", str(data), "--scenes", "3", "--size", "40"])
    make_synthetic.main()
    assert len(list((data / "masks").glob("*.npy"))) == 3

    # Train on the dataset root, which contains masks/: they must be ignored.
    train.main(["--data", str(data), "--output", str(outputs), "--epochs", "1", "--samples-per-epoch", "8",
                "--batch-size", "4", "--patch-size", "16", "--width", "8", "--projection-dim", "16", "--device", "cpu"])
    checkpoint = outputs / "best.pt"
    assert checkpoint.exists()
    history = json.loads((outputs / "history.json").read_text())
    assert len(history) == 1 and np.isfinite(history[0]["loss"])

    pair = data / "images"
    before = next(pair.glob("scene000_20230101.npy"))
    after = next(pair.glob("scene000_20240101.npy"))
    predict.main(["--before", str(before), "--after", str(after), "--checkpoint", str(checkpoint),
                  "--output", str(tmp_path / "pred" / "scene000.npy"), "--device", "cpu", "--tile-size", "32", "--overlap", "8"])
    assert (tmp_path / "pred" / "scene000_mask.npy").exists()

    evaluate.main(["--images", str(data), "--masks", str(data), "--checkpoint", str(checkpoint),
                   "--output", str(tmp_path / "metrics.json"), "--device", "cpu", "--tile-size", "32", "--overlap", "8"])
    report = json.loads((tmp_path / "metrics.json").read_text())
    assert len(report["scenes"]) == 3
    assert report["skipped_scenes"] == []
    assert set(report["summary"]) == {"iou", "f1", "precision", "recall", "accuracy", "auroc"}
    assert all(np.isfinite(v) for v in report["summary"].values())


def test_training_rejects_too_few_bands(tmp_path: Path):
    np.save(tmp_path / "scene_20230101.npy", np.zeros((2, 32, 32), np.float32))
    np.save(tmp_path / "scene_20240101.npy", np.zeros((2, 32, 32), np.float32))
    with pytest.raises(ValueError, match="band"):
        train.main(["--data", str(tmp_path), "--output", str(tmp_path / "out"), "--epochs", "1",
                    "--samples-per-epoch", "4", "--batch-size", "2", "--patch-size", "16", "--device", "cpu"])
