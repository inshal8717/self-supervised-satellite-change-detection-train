import numpy as np

from sat_change.metrics import rank_auc, segmentation_metrics
from sat_change.postprocess import binary_mask, otsu_threshold


def test_metrics_perfect():
    y = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    assert segmentation_metrics(y, y)["f1"] == 1
    assert rank_auc(y.astype(float), y) == 1


def test_otsu_separates_bimodal_values():
    score = np.r_[np.zeros(100), np.ones(100)].reshape(10, 20)
    threshold = otsu_threshold(score)
    mask, _ = binary_mask(score, threshold)
    assert mask.sum() == 100
