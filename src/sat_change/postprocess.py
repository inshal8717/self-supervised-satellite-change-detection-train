from __future__ import annotations

import numpy as np


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    x = values[np.isfinite(values)].astype(np.float64)
    if x.size == 0 or np.ptp(x) == 0:
        return float(x[0]) if x.size else 0.0
    hist, edges = np.histogram(x, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(weight2[::-1], 1))[::-1]
    variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    return float(centers[int(np.argmax(variance))])


def binary_mask(score: np.ndarray, threshold: float | None = None) -> tuple[np.ndarray, float]:
    threshold = otsu_threshold(score) if threshold is None else threshold
    return (score >= threshold).astype(np.uint8), float(threshold)

