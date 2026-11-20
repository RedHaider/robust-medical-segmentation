from __future__ import annotations

import numpy as np


def segmentation_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = target.astype(bool)
    prediction = prediction.astype(bool)
    tp = np.logical_and(target, prediction).sum()
    fp = np.logical_and(~target, prediction).sum()
    fn = np.logical_and(target, ~prediction).sum()
    tn = np.logical_and(~target, ~prediction).sum()
    eps = 1e-9
    return {
        "dice": float((2 * tp + eps) / (2 * tp + fp + fn + eps)),
        "iou": float((tp + eps) / (tp + fp + fn + eps)),
        "sensitivity": float((tp + eps) / (tp + fn + eps)),
        "specificity": float((tn + eps) / (tn + fp + eps)),
    }
