# Project brief

## Portfolio title

Robust MedSeg Lite: Frequency-Aware Stress Testing for Lightweight Medical Image Segmentation

## Professor connection

The project was selected because Truong-Son Hy's current work includes efficient and adversarially resilient medical image segmentation. This implementation explores the broader research question using a different method and synthetic dataset. It does not reproduce his model, code, data, or experimental claims.

## Applicant connection

Syed Ali Redwanul Haider has experience in Python, machine learning, deep learning, computer vision, CNN/YOLO classification, IoT sensing, analytics, and model evaluation. The project demonstrates a credible transition from classification to segmentation and trustworthy healthcare AI.

## Original contributions

1. Procedural lesion-like image and mask generator requiring no patient data.
2. Controlled comparison of spatial and FFT-residual pixel features.
3. Reproducible corruption suite for noise, blur, and contrast shift.
4. Multi-metric evaluation and machine-readable reports.
5. Explicit negative-result analysis and next-step hypotheses.

## Limitations

- Synthetic images do not represent clinical complexity.
- Logistic pixel classification ignores spatial context beyond engineered features.
- One seed is included as a fast demonstration, not a statistical conclusion.
- No clinical validity or diagnostic claim is made.

## Next milestone

Implement a compact learned segmentation model, run repeated seeds, and validate on a properly licensed public dataset such as Kvasir-SEG or BUSI with leakage-resistant splits.
