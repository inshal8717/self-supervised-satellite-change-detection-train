from __future__ import annotations

import numpy as np


def segmentation_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """IoU, F1, precision, recall and accuracy for a binary prediction."""
    p = pred.astype(bool).ravel()
    y = target.astype(bool).ravel()
    tp = int(np.sum(p & y))
    fp = int(np.sum(p & ~y))
    fn = int(np.sum(~p & y))
    tn = int(np.sum(~p & ~y))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "iou": tp / max(tp + fp + fn, 1),
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "precision": precision,
        "recall": recall,
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def rank_auc(score: np.ndarray, target: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic with average ranks for ties.

    Returns NaN when only one class is present.
    """
    s = score.ravel().astype(np.float64)
    y = target.astype(bool).ravel()
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks where tied values share the mean of their positions."""
    order = np.argsort(values, kind="stable")
    sorted_vals = values[order]
    # Start index of each run of equal values in the sorted array.
    run_starts = np.flatnonzero(np.r_[True, sorted_vals[1:] != sorted_vals[:-1]])
    run_ends = np.r_[run_starts[1:], values.size]
    mean_ranks = (run_starts + run_ends + 1) / 2.0  # mean of 1-based positions in each run
    run_index = np.repeat(np.arange(run_starts.size), run_ends - run_starts)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = mean_ranks[run_index]
    return ranks
