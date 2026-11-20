from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from .corruptions import corrupt
from .features import fft_residual
from .metrics import segmentation_metrics


def _box_mean(image: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(image, radius, mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)
    width = radius * 2 + 1
    for y in range(width):
        for x in range(width):
            out += padded[y:y + image.shape[0], x:x + image.shape[1]]
    return out / float(width * width)


def multiscale_features(image: np.ndarray, include_frequency: bool = True) -> np.ndarray:
    gy, gx = np.gradient(image)
    mean3, mean5 = _box_mean(image, 1), _box_mean(image, 2)
    columns = [image, mean3, mean5, image - mean3, image - mean5, gx, gy, np.hypot(gx, gy)]
    if include_frequency:
        columns.extend([fft_residual(image, 0.10), fft_residual(image, 0.20)])
    return np.stack(columns, axis=-1).reshape(-1, len(columns)).astype(np.float32)


class RobustExtraTreesSegmenter:
    def __init__(self, seed: int, augment: bool = False):
        self.seed = seed
        self.augment = augment
        self.threshold = 0.5
        self.estimator = ExtraTreesClassifier(
            n_estimators=90, max_depth=16, min_samples_leaf=3,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )

    def fit(self, images: np.ndarray, masks: np.ndarray, max_pixels: int = 90000):
        train_images, train_masks = list(images), list(masks)
        if self.augment:
            rng = np.random.default_rng(self.seed + 100)
            for image, mask in zip(images, masks):
                kind = ("noise", "blur", "contrast")[rng.integers(0, 3)]
                train_images.append(corrupt(image, kind, float(rng.uniform(.08, .22)), rng))
                train_masks.append(mask)
        x = np.concatenate([multiscale_features(im) for im in train_images])
        y = np.asarray(train_masks).reshape(-1)
        if len(y) > max_pixels:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(y), max_pixels, replace=False)
            x, y = x[idx], y[idx]
        self.estimator.fit(x, y)
        return self

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        return np.stack([
            self.estimator.predict_proba(multiscale_features(im))[:, 1].reshape(im.shape)
            for im in images
        ])

    def tune_threshold(self, images: np.ndarray, masks: np.ndarray):
        probabilities = self.predict_proba(images)
        candidates = np.linspace(.25, .75, 21)
        scores = [np.mean([segmentation_metrics(t, p >= threshold)["dice"] for t, p in zip(masks, probabilities)]) for threshold in candidates]
        self.threshold = float(candidates[int(np.argmax(scores))])
        return self

    def predict(self, images: np.ndarray) -> np.ndarray:
        return (self.predict_proba(images) >= self.threshold).astype(np.uint8)
