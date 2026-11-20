from __future__ import annotations

import numpy as np


def make_sample(size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Create one grayscale image with a soft lesion-like ellipse and binary mask."""
    y, x = np.mgrid[:size, :size]
    cx, cy = rng.uniform(0.3, 0.7, 2) * size
    rx, ry = rng.uniform(0.10, 0.24, 2) * size
    angle = rng.uniform(0, np.pi)
    xr = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
    yr = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
    distance = (xr / rx) ** 2 + (yr / ry) ** 2
    mask = distance <= 1.0

    background = 0.25 + 0.10 * (x / size) + 0.07 * np.sin(y / 5.0)
    texture = rng.normal(0, 0.035, (size, size))
    lesion = 0.48 * np.exp(-1.6 * distance)
    image = np.clip(background + texture + lesion, 0, 1).astype(np.float32)
    return image, mask.astype(np.uint8)


def make_dataset(count: int, size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pairs = [make_sample(size, rng) for _ in range(count)]
    return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
