"""Model builders: a neural-network classifier (MLP) and a gradient-boosted baseline.

Both consume a fixed-length single-heartbeat ECG vector and output P(abnormal).
The MLP is the primary model; gradient boosting is a strong, fast baseline for
comparison and ablation.
"""
from __future__ import annotations
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


def build_models(cfg: dict) -> dict:
    m = cfg["model"]
    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(m["hidden_layers"]),
        alpha=m["alpha"],
        batch_size=m["batch_size"],
        learning_rate_init=m["lr"],
        max_iter=m["max_iter"],
        early_stopping=True,
        n_iter_no_change=m["patience"],
        validation_fraction=0.15,
        random_state=cfg["seed"],
    )
    gb = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=300,
        max_depth=m.get("gb_max_depth"),
        l2_regularization=1.0,
        random_state=cfg["seed"],
    )
    return {"neural_net (MLP)": mlp, "gradient_boosting": gb}
