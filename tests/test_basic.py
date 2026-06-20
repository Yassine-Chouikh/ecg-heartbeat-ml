import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from ecgnet.data import generate_synthetic, make_splits
from ecgnet.model import build_models

CFG = {"seed": 0, "data": {"test_size": 0.2, "val_size": 0.2},
       "model": {"hidden_layers": [16], "alpha": 1e-4, "batch_size": 16,
                 "lr": 0.01, "max_iter": 30, "patience": 5, "gb_max_depth": 3}}


def test_synthetic_shapes():
    X, y = generate_synthetic(200, 0)
    assert X.shape == (200, 140)
    assert set(np.unique(y)).issubset({0.0, 1.0})


def test_splits_partition():
    X, y = generate_synthetic(200, 0)
    (Xtr, _), (Xval, _), (Xte, _) = make_splits(X, y, CFG)
    assert len(Xtr) + len(Xval) + len(Xte) == 200


def test_build_and_fit():
    X, y = generate_synthetic(150, 0)
    (Xtr, ytr), _, (Xte, yte) = make_splits(X, y, CFG)
    models = build_models(CFG)
    assert len(models) == 2
    m = models["gradient_boosting"].fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    assert len(p) == len(yte) and p.min() >= 0 and p.max() <= 1
