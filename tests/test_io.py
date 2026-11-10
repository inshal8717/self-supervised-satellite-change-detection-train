from pathlib import Path

import numpy as np

from sat_change.io import (
    discover_images,
    discover_masks,
    normalize_pair,
    normalize_reflectance,
    pair_images,
    scene_key,
)


def test_scene_key_strips_date_suffix():
    assert scene_key("tile_a_2023-01-01.tif") == "tile_a"
    assert scene_key("tile_a_20230101.npy") == "tile_a"


def test_scene_key_and_pairs(tmp_path: Path):
    np.save(tmp_path / "tile_a_20230101.npy", np.zeros((4, 8, 8)))
    np.save(tmp_path / "tile_a_20240101.npy", np.zeros((4, 8, 8)))
    np.save(tmp_path / "unpaired_20230101.npy", np.zeros((4, 8, 8)))
    assert len(pair_images(tmp_path)) == 1


def test_masks_are_never_imagery(tmp_path: Path):
    """Per-scene masks/ folders must not be sampled as images or paired with them."""
    images = tmp_path / "vegas" / "images"
    masks = tmp_path / "vegas" / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    np.save(images / "vegas_2023-03-31.npy", np.zeros((4, 8, 8)))
    np.save(images / "vegas_2023-09-07.npy", np.zeros((4, 8, 8)))
    np.save(masks / "vegas.npy", np.zeros((1, 8, 8)))

    assert all("masks" not in p.parts for p in discover_images(tmp_path))
    pairs = pair_images(tmp_path)
    assert len(pairs) == 1
    before, after = pairs[0]
    assert before.name == "vegas_2023-03-31.npy" and after.name == "vegas_2023-09-07.npy"
    assert [p.name for p in discover_masks(tmp_path)] == ["vegas.npy"]


def test_flat_mask_folder_is_discovered(tmp_path: Path):
    (tmp_path / "masks").mkdir()
    np.save(tmp_path / "masks" / "scene000.npy", np.zeros((8, 8)))
    assert [scene_key(p) for p in discover_masks(tmp_path)] == ["scene000"]


def test_normalization_handles_scaled_and_nan():
    x = np.linspace(0, 10000, 400, dtype=np.float32).reshape(4, 10, 10)
    x[0, 0, 0] = np.nan
    y = normalize_reflectance(x)
    assert y.shape == x.shape and np.isfinite(y).all()
    assert 0 <= y.min() <= y.max() <= 1


def test_normalize_pair_keeps_unchanged_pixels_identical():
    """Unchanged pixels must get the same value in both dates, even with a big change elsewhere."""
    rng = np.random.default_rng(0)
    before = (0.2 + 0.5 * rng.random((4, 40, 40))).astype(np.float32)
    after = before.copy()
    after[:, :15, :15] = 0.01  # large change
    nb, na = normalize_pair(before, after)
    assert np.allclose(nb[:, 20:, 20:], na[:, 20:, 20:])
    assert not np.allclose(nb[:, :15, :15], na[:, :15, :15])

    # Per-photo stretching moves the unchanged background, which is the bug this avoids.
    assert not np.allclose(normalize_reflectance(before)[:, 20:, 20:], normalize_reflectance(after)[:, 20:, 20:])


def test_normalize_pair_scales_integer_reflectance_and_handles_nan():
    rng = np.random.default_rng(1)
    before = rng.random((4, 16, 16)).astype(np.float32) * 10000
    after = before.copy()
    before[0, 0, 0] = np.nan
    nb, na = normalize_pair(before, after)
    assert np.isfinite(nb).all() and np.isfinite(na).all()
    assert 0 <= nb.min() <= nb.max() <= 1 and 0 <= na.min() <= na.max() <= 1
    assert nb[0, 0, 0] == 0  # nodata becomes 0
