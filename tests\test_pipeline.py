import numpy as np

from robust_medseg.corruptions import corrupt
from robust_medseg.features import fft_residual, pixel_features
from robust_medseg.metrics import segmentation_metrics
from robust_medseg.synthetic import make_dataset


def test_dataset_is_reproducible():
    a = make_dataset(2, 24, 7)
    b = make_dataset(2, 24, 7)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_feature_shapes_and_range():
    images, _ = make_dataset(1, 24, 2)
    assert pixel_features(images[0], False).shape == (576, 4)
    assert pixel_features(images[0], True).shape == (576, 5)
    residual = fft_residual(images[0])
    assert residual.min() >= 0 and residual.max() <= 1


def test_metrics_are_correct_for_perfect_prediction():
    target = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    values = segmentation_metrics(target, target)
    assert all(abs(value - 1.0) < 1e-8 for value in values.values())


def test_corruptions_preserve_shape_and_range():
    images, _ = make_dataset(1, 24, 2)
    rng = np.random.default_rng(4)
    for kind in ("noise", "blur", "contrast"):
        result = corrupt(images[0], kind, 0.2, rng)
        assert result.shape == images[0].shape
        assert result.min() >= 0 and result.max() <= 1
