from __future__ import annotations

import numpy as np


def segmentation_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    p, y = pred.astype(bool).ravel(), target.astype(bool).ravel()
    tp = int(np.sum(p & y)); fp = int(np.sum(p & ~y)); fn = int(np.sum(~p & y)); tn = int(np.sum(~p & ~y))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "iou": tp / max(tp + fp + fn, 1),
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "precision": precision, "recall": recall,
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def rank_auc(score: np.ndarray, target: np.ndarray) -> float:
    """Dependency-free AUROC using pairwise rank statistics."""
    s, y = score.ravel(), target.astype(bool).ravel()
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

