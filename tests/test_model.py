import pytest

torch = pytest.importorskip("torch")

from sat_change.model import SatelliteEncoder, feature_change, nt_xent


def test_model_shapes_and_loss():
    model = SatelliteEncoder(4, width=8, projection_dim=16)
    a = torch.rand(4, 4, 32, 32); b = a + torch.randn_like(a) * .01
    loss = nt_xent(model(a), model(b))
    assert loss.isfinite() and loss.item() > 0
    score = feature_change(model.eval(), a[:1], b[:1])
    assert score.shape == (1, 32, 32) and score.min() >= -1e-5

