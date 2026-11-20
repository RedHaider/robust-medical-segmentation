from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .features import pixel_features


class PixelSegmenter:
    def __init__(self, include_frequency: bool, seed: int = 0):
        self.include_frequency = include_frequency
        self.estimator = LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed)

    def fit(self, images: np.ndarray, masks: np.ndarray, max_pixels: int = 60000) -> "PixelSegmenter":
        x = np.concatenate([pixel_features(img, self.include_frequency) for img in images])
        y = masks.reshape(-1)
        if len(y) > max_pixels:
            rng = np.random.default_rng(19)
            idx = rng.choice(len(y), max_pixels, replace=False)
            x, y = x[idx], y[idx]
        self.estimator.fit(x, y)
        return self

    def predict(self, images: np.ndarray) -> np.ndarray:
        results = []
        for image in images:
            pred = self.estimator.predict(pixel_features(image, self.include_frequency))
            results.append(pred.reshape(image.shape))
        return np.stack(results).astype(np.uint8)
