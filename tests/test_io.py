from pathlib import Path

import numpy as np

from sat_change.io import normalize_reflectance, pair_images, scene_key


def test_scene_key_and_pairs(tmp_path: Path):
    np.save(tmp_path / "tile_a_20230101.npy", np.zeros((4, 8, 8)))
    np.save(tmp_path / "tile_a_20240101.npy", np.zeros((4, 8, 8)))
    np.save(tmp_path / "unpaired_20230101.npy", np.zeros((4, 8, 8)))
    assert scene_key("tile_a_2023-01-01.tif") == "tile_a"
    assert len(pair_images(tmp_path)) == 1


def test_normalization_handles_scaled_and_nan():
    x = np.linspace(0, 10000, 400, dtype=np.float32).reshape(4, 10, 10)
    x[0, 0, 0] = np.nan
    y = normalize_reflectance(x)
    assert y.shape == x.shape and np.isfinite(y).all()
    assert 0 <= y.min() <= y.max() <= 1

