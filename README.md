# Robust MedSeg: Heavy Robustness Benchmark

An original, lightweight medical-image segmentation robustness benchmark inspired by research on efficient frequency-aware medical segmentation. It is designed as a portfolio bridge from CNN/computer-vision classification to trustworthy healthcare AI.

This repository does **not** reproduce FFTMed and is not a clinical tool. It implements a distinct classical baseline: frequency-enhanced pixel features, logistic segmentation, and controlled corruption testing on procedurally generated medical-style images.

## Research question

Can simple frequency-domain features improve the robustness of a lightweight segmentation model when images are degraded by noise, blur, or contrast changes?

## What is included

- Reproducible synthetic lesion-like image and mask generation
- Spatial intensity and gradient features
- FFT high-frequency residual features
- Lightweight logistic pixel classifier
- Dice, IoU, sensitivity, and specificity metrics
- Gaussian-noise, blur, and contrast stress tests
- JSON and CSV experiment reports
- Automated unit tests

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .[dev]
python -m robust_medseg.experiment --output outputs/demo --seed 27
pytest
```

## Experiment design

Two models are compared:

1. `spatial`: intensity, neighborhood mean, and image gradients.
2. `frequency`: the spatial features plus an FFT-derived high-frequency residual.

Both are evaluated on the same clean test set and three corruption families. The generated report is intended to demonstrate experimental design, reproducibility, and honest robustness analysis—not state-of-the-art accuracy.

## Verified baseline result

The included seed-27 experiment produced the following Dice scores:

| Condition | Spatial | Frequency |
|---|---:|---:|
| Clean | 0.665 | 0.664 |
| Mild Gaussian noise | 0.414 | 0.408 |
| Strong Gaussian noise | 0.249 | 0.248 |
| Strong blur | 0.776 | 0.774 |
| Strong contrast reduction | 0.838 | 0.839 |

The frequency residual did not materially improve this logistic baseline. This rejects the initial improvement hypothesis for the present feature design and synthetic dataset. The most useful next experiment is a learned fusion or compact U-Net baseline on a licensed real dataset, with patient-level splits and repeated seeds.

## Heavy experiment

The upgraded experiment adds two nonlinear multiscale Extra Trees models, corruption-based training augmentation, validation-only threshold tuning, three independent seeds, 24 held-out test images per seed, confidence intervals, ablations, and saved quantitative/qualitative figures.

```bash
python -m robust_medseg.heavy_experiment --output outputs/heavy
```

The primary claim is limited to this documented synthetic benchmark: the corruption-augmented multiscale model is compared with the original spatial logistic baseline using strong-noise Dice. It is **not** claimed to improve a professor's published architecture or clinical performance.

### Verified heavy result

Across three independent seeds, the corruption-augmented multiscale Extra Trees model achieved **0.670 mean strong-noise Dice** (95% CI 0.649-0.690), compared with **0.271** (95% CI 0.252-0.289) for the original spatial logistic baseline. That is a **+0.399 absolute Dice gain** under this synthetic protocol.

| Condition | Spatial logistic | Extra Trees | Robust Extra Trees |
|---|---:|---:|---:|
| Clean | 0.686 | 0.855 | 0.853 |
| Strong noise | 0.271 | 0.565 | **0.670** |
| Strong blur | 0.796 | 0.858 | **0.859** |
| Strong contrast reduction | **0.841** | 0.802 | 0.813 |

The augmentation improves strong-noise robustness over the same multiscale model by +0.104 Dice, while slightly reducing clean Dice by 0.003. The contrast result is a documented limitation: the logistic baseline remains stronger there. See `results/heavy_metrics.csv` for the reproducible summary.

## Real-data extension

The next version can replace the synthetic generator with a public dataset such as Kvasir-SEG or BUSI. Keep patient-level train/test separation, document the license, and avoid reporting clinical claims without external validation.

## Connection to Syed Ali Redwanul Haider's background

The project builds on prior deep-learning classification, computer vision, IoT sensing, Python analytics, and model-evaluation experience. It adds segmentation, frequency-domain image processing, robustness testing, and healthcare-oriented experimental reporting.

## Ethical scope

- Synthetic data only in the included experiment
- No patient data or protected health information
- No diagnosis or treatment recommendation
- Results are research demonstrations, not clinical validation

## License

MIT
