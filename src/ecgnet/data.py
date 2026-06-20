"""Data loading and preprocessing for ECG heartbeat classification.

Primary source: ECG5000 (PhysioNet) via the public TensorFlow mirror
(http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv) - 5,000
single-heartbeat traces (length 140), labeled normal/abnormal. This downloads
automatically when run on a networked machine.

Offline fallback: a realistic synthetic generator that builds PQRST waveforms
with per-sample morphology jitter, baseline wander, and noise, and injects
overlapping pathologies (T-wave inversion, ST shift, conduction changes,
ectopic/premature beats). The overlap keeps the task non-trivial so reported
metrics are meaningful even without network access.
"""
from __future__ import annotations
import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

L = 140


def _download(url: str, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def _beat(rng: np.random.Generator, abnormal: bool) -> np.ndarray:
    t = np.linspace(0, 1, L)
    g = lambda c, a, w: a * np.exp(-((t - c) ** 2) / (2 * w * w))
    j = lambda s: rng.normal(0, s)
    sig = (
        g(0.20 + j(0.010), rng.normal(0.12, 0.03), 0.012)   # P
        + g(0.33 + j(0.008), -abs(rng.normal(0.10, 0.03)), 0.008)  # Q
        + g(0.37 + j(0.008), rng.normal(1.00, 0.12), 0.010)  # R
        + g(0.42 + j(0.008), -abs(rng.normal(0.18, 0.05)), 0.010)  # S
        + g(0.62 + j(0.015), rng.normal(0.30, 0.06), 0.030)  # T
    )
    if abnormal:
        kind = int(rng.integers(0, 4))
        if kind == 0:        # partial / full T-wave inversion
            sig = sig - g(0.62, rng.uniform(0.25, 0.55), 0.030)
        elif kind == 1:      # ST-segment shift
            seg = (t > 0.44) & (t < 0.58)
            sig = sig + (rng.choice([-1.0, 1.0]) * rng.uniform(0.06, 0.16)) * seg
        elif kind == 2:      # reduced + widened R (conduction change)
            sig = sig - g(0.37, rng.uniform(0.30, 0.55), 0.010) \
                      + g(0.37, rng.uniform(0.20, 0.40), 0.024)
        else:                # premature timing shift + ectopic bump
            sig = np.roll(sig, int(rng.uniform(4, 12)))
            sig = sig + g(rng.uniform(0.80, 0.95), rng.uniform(0.12, 0.28), 0.012)
    sig = sig + 0.05 * np.sin(2 * np.pi * rng.uniform(0.5, 1.5) * t + rng.uniform(0, 6.28))
    sig = sig + rng.normal(0, 0.075, L)
    return sig.astype("float32")


def generate_synthetic(n: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        ab = rng.random() < 0.5
        X.append(_beat(rng, ab)); y.append(1.0 if ab else 0.0)
    return np.asarray(X, "float32"), np.asarray(y, "float32")


def load_raw(cfg: dict):
    """Return (X, y, source). y=1 denotes an abnormal heartbeat (positive class)."""
    try:
        path = _download(cfg["data"]["url"], cfg["data"]["cache"])
        arr = pd.read_csv(path, header=None).values.astype("float32")
        X = arr[:, :-1]
        labels = arr[:, -1].astype(int)
        y = (labels == 0).astype("float32")  # mirror encodes 1=normal, 0=abnormal
        return X, y, "ECG5000 (PhysioNet, TensorFlow mirror)"
    except Exception as exc:  # network unavailable -> reproducible synthetic benchmark
        X, y = generate_synthetic(5000, cfg["seed"])
        return X, y, f"synthetic ECG benchmark (no network: {type(exc).__name__})"


def make_splits(X: np.ndarray, y: np.ndarray, cfg: dict):
    """Stratified train/val/test split with train-fit standardization (no leakage)."""
    seed = cfg["seed"]
    holdout = cfg["data"]["test_size"] + cfg["data"]["val_size"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=holdout, stratify=y, random_state=seed)
    rel = cfg["data"]["test_size"] / holdout
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=rel, stratify=y_tmp, random_state=seed)
    mu, sd = float(X_tr.mean()), float(X_tr.std()) + 1e-8
    norm = lambda a: ((a - mu) / sd).astype("float32")
    return (norm(X_tr), y_tr), (norm(X_val), y_val), (norm(X_te), y_te)
