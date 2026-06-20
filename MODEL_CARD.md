# Model Card — ECG Heartbeat Classifier (ecgnet)

## Overview
A binary classifier (normal vs. abnormal single heartbeat) trained on 140-sample
single-lead ECG traces. Primary model: an MLP neural network; gradient boosting
serves as a baseline. Intended as a research/education demonstration of a
calibrated, interpretable ML pipeline for physiological signals.

## Intended use
- Educational and research demonstration of responsible ML on biosignals.
- A template for reproducible clinical-signal modeling (evaluation + calibration
  + interpretability + tests/CI).

## Out-of-scope / non-use
- **Not** a medical device and **not** for clinical decision-making.
- Single-beat, single-lead input only; not validated on multi-lead or continuous
  recordings, pediatric populations, or specific arrhythmia subtypes.

## Data
- **Primary:** ECG5000 (PhysioNet) via the public TensorFlow mirror, 5,000 beats,
  downloaded at runtime.
- **Offline benchmark:** realistic synthetic PQRST waveforms with morphology
  jitter, baseline wander, noise, and injected overlapping pathologies.
- Label convention: `1 = abnormal` (positive class).

## Metrics (held-out test, synthetic benchmark, seed=42)
AUROC 0.970 · AUPRC 0.975 · F1 0.901 · Accuracy 0.903 · Brier 0.068 · ECE 0.035.
Model selection by validation AUROC (MLP 0.960 > gradient boosting 0.934).

## Calibration
Reliability is reported via Brier score and Expected Calibration Error, with a
reliability diagram in `results/calibration.png`. Probabilities should be
re-validated (and ideally re-calibrated) on any new population before use.

## Limitations & ethical considerations
- Performance is dataset-specific; real-world ECG has device, demographic, and
  acquisition shifts not represented here.
- Subgroup performance is not characterized (the public benchmark lacks
  demographic metadata); a production system would require subgroup fairness
  evaluation before deployment.
- Errors in a clinical context carry asymmetric risk; threshold selection should
  be driven by the clinical cost of false negatives vs. false positives.

## Reproducibility
Fixed seeds, train-only standardization, deterministic splits, unit tests, and
CI. Run `python scripts/run.py` to regenerate all metrics and figures.
