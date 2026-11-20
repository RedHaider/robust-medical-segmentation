from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def corrupt(image: np.ndarray, kind: str, severity: float, rng: np.random.Generator) -> np.ndarray:
    if kind == "noise":
        result = image + rng.normal(0, severity, image.shape)
    elif kind == "contrast":
        result = (image - image.mean()) * (1.0 - severity) + image.mean()
    elif kind == "blur":
        source = Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255))
        result = np.asarray(source.filter(ImageFilter.GaussianBlur(radius=severity * 4)), dtype=np.float32) / 255.0
    else:
        raise ValueError(f"Unknown corruption: {kind}")
    return np.clip(result, 0, 1).astype(np.float32)
