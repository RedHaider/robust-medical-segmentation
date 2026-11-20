from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .corruptions import corrupt
from .metrics import segmentation_metrics
from .model import PixelSegmenter
from .synthetic import make_dataset


def _evaluate(model: PixelSegmenter, images: np.ndarray, masks: np.ndarray) -> dict[str, float]:
    predictions = model.predict(images)
    per_image = [segmentation_metrics(t, p) for t, p in zip(masks, predictions)]
    return {key: float(np.mean([row[key] for row in per_image])) for key in per_image[0]}


def run(output: Path, seed: int = 27, size: int = 64) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    train_x, train_y = make_dataset(36, size, seed)
    test_x, test_y = make_dataset(12, size, seed + 1)
    rng = np.random.default_rng(seed + 2)
    report: dict[str, dict] = {"metadata": {"seed": seed, "image_size": size, "synthetic_data": True}, "results": {}}
    rows = []
    for name, frequency in (("spatial", False), ("frequency", True)):
        model = PixelSegmenter(frequency, seed).fit(train_x, train_y)
        conditions = {"clean": test_x}
        for kind in ("noise", "blur", "contrast"):
            conditions[f"{kind}_mild"] = np.stack([corrupt(x, kind, 0.12, rng) for x in test_x])
            conditions[f"{kind}_strong"] = np.stack([corrupt(x, kind, 0.28, rng) for x in test_x])
        report["results"][name] = {}
        for condition, images in conditions.items():
            metrics = _evaluate(model, images, test_y)
            report["results"][name][condition] = metrics
            rows.append({"model": name, "condition": condition, **metrics})
        if name == "frequency":
            pred = model.predict(test_x[:1])[0]
            panel = np.concatenate([test_x[0], test_y[0], pred], axis=1)
            Image.fromarray(np.uint8(panel * 255)).save(output / "example_image-mask-prediction.png")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Robust MedSeg Lite benchmark")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    report = run(args.output, args.seed, args.size)
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
