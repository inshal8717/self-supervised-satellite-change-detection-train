import numpy as np

from sat_change.metrics import rank_auc, segmentation_metrics
from sat_change.postprocess import binary_mask, otsu_threshold


def test_metrics_perfect():
    y = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    metrics = segmentation_metrics(y, y)
    assert metrics["f1"] == 1 and metrics["iou"] == 1
    assert rank_auc(y.astype(float), y) == 1


def test_auc_with_ties_matches_pairwise_definition():
    """AUROC equals P(score_pos > score_neg) + 0.5 * P(tie), regardless of pixel order."""
    rng = np.random.default_rng(0)
    score = rng.integers(0, 4, size=200).astype(float)  # many ties
    target = rng.random(200) < 0.4
    pos, neg = score[target], score[~target]
    expected = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (pos.size * neg.size)
    assert np.isclose(rank_auc(score, target), expected)


def test_auc_is_nan_when_one_class_missing():
    assert np.isnan(rank_auc(np.array([0.1, 0.2]), np.array([0, 0])))


def test_otsu_separates_bimodal_values():
    score = np.r_[np.zeros(100), np.ones(100)].reshape(10, 20)
    threshold = otsu_threshold(score)
    mask, _ = binary_mask(score, threshold)
    assert mask.sum() == 100
