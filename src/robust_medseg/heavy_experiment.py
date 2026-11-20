from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from .advanced_model import RobustExtraTreesSegmenter
from .corruptions import corrupt
from .metrics import segmentation_metrics
from .model import PixelSegmenter
from .synthetic import make_dataset

SEEDS = (27, 41, 73)


def evaluate(model, images, masks):
    predictions = model.predict(images)
    rows = [segmentation_metrics(t, p) for t, p in zip(masks, predictions)]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, predictions


def conditions(images, seed):
    output = {"clean": images}
    for index, kind in enumerate(("noise", "blur", "contrast")):
        rng = np.random.default_rng(seed + 900 + index)
        output[f"{kind}_strong"] = np.stack([corrupt(im, kind, .28, rng) for im in images])
    return output


def confidence(values):
    values = np.asarray(values, float)
    mean, std = float(values.mean()), float(values.std(ddof=1))
    half = 1.96 * std / np.sqrt(len(values))
    return {"mean": mean, "std": std, "ci95_low": mean-half, "ci95_high": mean+half}


def draw_chart(summary, path):
    names = ["spatial_logistic", "frequency_logistic", "extra_trees", "robust_extra_trees"]
    canvas = Image.new("RGB", (900, 480), "white"); draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), "Mean Dice across three seeds", fill="black")
    colors = ["#8da0cb", "#66c2a5", "#fc8d62", "#e78ac3"]
    for group, condition in enumerate(("clean", "noise_strong", "blur_strong", "contrast_strong")):
        draw.text((70 + group*205, 430), condition, fill="black")
        for j, name in enumerate(names):
            value = summary[name][condition]["dice"]["mean"]
            x0 = 55 + group*205 + j*36; y0 = 400-int(value*330)
            draw.rectangle((x0, y0, x0+28, 400), fill=colors[j])
    canvas.save(path)


def run(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    raw_rows, example = [], None
    for seed in SEEDS:
        train_x, train_y = make_dataset(60, 64, seed)
        val_x, val_y = make_dataset(16, 64, seed+1)
        test_x, test_y = make_dataset(24, 64, seed+2)
        models = {
            "spatial_logistic": PixelSegmenter(False, seed).fit(train_x, train_y),
            "frequency_logistic": PixelSegmenter(True, seed).fit(train_x, train_y),
            "extra_trees": RobustExtraTreesSegmenter(seed, False).fit(train_x, train_y).tune_threshold(val_x, val_y),
            "robust_extra_trees": RobustExtraTreesSegmenter(seed, True).fit(train_x, train_y).tune_threshold(val_x, val_y),
        }
        for model_name, model in models.items():
            for condition, images in conditions(test_x, seed).items():
                metrics, predictions = evaluate(model, images, test_y)
                raw_rows.append({"seed":seed,"model":model_name,"condition":condition,"threshold":getattr(model,"threshold",.5),**metrics})
                if seed == SEEDS[0] and model_name == "robust_extra_trees" and condition == "clean":
                    example = np.concatenate([test_x[0], test_y[0], predictions[0]], axis=1)
    summary = {}
    for model in sorted({r["model"] for r in raw_rows}):
        summary[model] = {}
        for condition in sorted({r["condition"] for r in raw_rows}):
            selected = [r for r in raw_rows if r["model"]==model and r["condition"]==condition]
            summary[model][condition] = {metric: confidence([r[metric] for r in selected]) for metric in ("dice","iou","sensitivity","specificity")}
    baseline = summary["spatial_logistic"]["noise_strong"]["dice"]["mean"]
    proposed = summary["robust_extra_trees"]["noise_strong"]["dice"]["mean"]
    report = {"protocol":{"seeds":SEEDS,"train":60,"validation":16,"test":24,"patient_level_analogue_split":True},"summary":summary,"primary_comparison":{"metric":"strong-noise Dice","baseline":baseline,"proposed":proposed,"absolute_gain":proposed-baseline}}
    (output/"heavy_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (output/"per_seed_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=raw_rows[0].keys());writer.writeheader();writer.writerows(raw_rows)
    Image.fromarray(np.uint8(example*255)).save(output/"qualitative_example.png")
    draw_chart(summary, output/"dice_comparison.png")
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=Path("outputs/heavy"));args=parser.parse_args();report=run(args.output);print(json.dumps(report["primary_comparison"],indent=2))
if __name__=="__main__":main()
