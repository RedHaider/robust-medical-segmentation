from __future__ import annotations

import numpy as np


def _mean3(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, 1, mode="reflect")
    return sum(padded[i:i + image.shape[0], j:j + image.shape[1]] for i in range(3) for j in range(3)) / 9.0


def fft_residual(image: np.ndarray, radius_fraction: float = 0.15) -> np.ndarray:
    """Return a normalized high-frequency reconstruction residual."""
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    h, w = image.shape
    yy, xx = np.ogrid[:h, :w]
    radius = min(h, w) * radius_fraction
    low = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 <= radius ** 2
    high_spectrum = spectrum.copy()
    high_spectrum[low] = 0
    residual = np.abs(np.fft.ifft2(np.fft.ifftshift(high_spectrum)))
    scale = residual.max() or 1.0
    return (residual / scale).astype(np.float32)


def pixel_features(image: np.ndarray, include_frequency: bool) -> np.ndarray:
    gy, gx = np.gradient(image)
    columns = [image, _mean3(image), np.abs(gx), np.abs(gy)]
    if include_frequency:
        columns.append(fft_residual(image))
    return np.stack(columns, axis=-1).reshape(-1, len(columns)).astype(np.float32)
