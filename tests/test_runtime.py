from pathlib import Path

import numpy as np
import pytest

from sat_change.io import normalize_reflectance
from sat_change.runtime import tile_starts

torch = pytest.importorskip("torch")

from sat_change.model import SatelliteEncoder, feature_change  # noqa: E402
from sat_change.runtime import score_pair  # noqa: E402


@pytest.mark.parametrize("length,tile,step", [(10, 4, 3), (64, 16, 8), (50, 32, 24), (33, 32, 31), (7, 32, 16)])
def test_tile_starts_cover_every_pixel(length, tile, step):
    starts = tile_starts(length, tile, step)
    covered = np.zeros(length, dtype=bool)
    for s in starts:
        covered[s:min(s + tile, length)] = True
    assert covered.all()
    assert starts[0] == 0
    assert all(0 <= s <= max(length - tile, 0) for s in starts)


def test_single_tile_matches_direct_model_output(tmp_path: Path):
    """When the image fits in one tile, tiled inference equals a plain forward pass."""
    rng = np.random.default_rng(1)
    before = rng.random((4, 40, 40), dtype=np.float32) * 10000
    after = rng.random((4, 40, 40), dtype=np.float32) * 10000
    np.save(tmp_path / "before.npy", before)
    np.save(tmp_path / "after.npy", after)

    torch.manual_seed(0)
    model = SatelliteEncoder(4, width=8, projection_dim=16).eval()
    tiled, _ = score_pair(model, tmp_path / "before.npy", tmp_path / "after.npy", tile_size=64, overlap=8)

    a = torch.from_numpy(normalize_reflectance(before))[None]
    b = torch.from_numpy(normalize_reflectance(after))[None]
    with torch.no_grad():
        direct = feature_change(model, a, b)[0].numpy()
    assert tiled.shape == (40, 40)
    assert np.allclose(tiled, direct, atol=1e-5)


def test_multi_tile_output_is_full_size_and_finite(tmp_path: Path):
    rng = np.random.default_rng(2)
    np.save(tmp_path / "before.npy", rng.random((4, 70, 50), dtype=np.float32))
    np.save(tmp_path / "after.npy", rng.random((4, 70, 50), dtype=np.float32))
    torch.manual_seed(0)
    model = SatelliteEncoder(4, width=8, projection_dim=16).eval()
    score, _ = score_pair(model, tmp_path / "before.npy", tmp_path / "after.npy", tile_size=32, overlap=8)
    assert score.shape == (70, 50)
    assert np.isfinite(score).all()


def test_tile_must_exceed_overlap(tmp_path: Path):
    np.save(tmp_path / "before.npy", np.zeros((4, 20, 20), np.float32))
    np.save(tmp_path / "after.npy", np.zeros((4, 20, 20), np.float32))
    model = SatelliteEncoder(4, width=8, projection_dim=16).eval()
    with pytest.raises(ValueError):
        score_pair(model, tmp_path / "before.npy", tmp_path / "after.npy", tile_size=16, overlap=16)
