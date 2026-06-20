"""End-to-end pipeline: train, model-select, evaluate, calibrate, interpret."""
from __future__ import annotations
import json, os, random
import numpy as np
import joblib
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, brier_score_loss, confusion_matrix,
                             roc_curve, precision_recall_curve)
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import load_raw, make_splits
from .model import build_models


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for i in range(n_bins):
        m = (p > bins[i]) & (p <= bins[i + 1])
        if m.sum():
            ece += abs(p[m].mean() - y[m].mean()) * m.mean()
    return float(ece)


def _metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }


def run(cfg, outdir="results"):
    set_seed(cfg["seed"]); os.makedirs(outdir, exist_ok=True)
    X, y, source = load_raw(cfg)
    (Xtr, ytr), (Xval, yval), (Xte, yte) = make_splits(X, y, cfg)

    models = build_models(cfg); leaderboard = {}
    best_name, best_model, best_val = None, None, -1.0
    for name, m in models.items():
        m.fit(Xtr, ytr)
        auc = float(roc_auc_score(yval, m.predict_proba(Xval)[:, 1]))
        leaderboard[name] = {"val_auroc": round(auc, 4)}
        if auc > best_val:
            best_val, best_name, best_model = auc, name, m

    p = best_model.predict_proba(Xte)[:, 1]
    metrics = _metrics(yte, p)
    metrics.update({"best_model": best_name, "val_auroc_best": round(best_val, 4),
                    "leaderboard": leaderboard, "data_source": source,
                    "n_samples": int(len(y)), "n_test": int(len(yte)),
                    "prevalence_abnormal": round(float(y.mean()), 4)})

    fpr, tpr, _ = roc_curve(yte, p); prec, rec, _ = precision_recall_curve(yte, p)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "--", c="gray")
    ax[0].set_title("ROC (AUROC=%.3f)" % metrics["auroc"]); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
    ax[1].plot(rec, prec); ax[1].set_title("PR (AUPRC=%.3f)" % metrics["auprc"])
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "roc_pr.png"), dpi=130); plt.close(fig)

    bins = np.linspace(0, 1, 11); idx = np.digitize(p, bins) - 1; xs, ys = [], []
    for b in range(10):
        m = idx == b
        if m.sum():
            xs.append(p[m].mean()); ys.append(yte[m].mean())
    fig, ax = plt.subplots(figsize=(5, 4)); ax.plot([0, 1], [0, 1], "--", c="gray")
    ax.plot(xs, ys, "o-"); ax.set_title("Calibration (ECE=%.3f)" % metrics["ece"])
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "calibration.png"), dpi=130); plt.close(fig)

    cm = confusion_matrix(yte, (p >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["normal", "abnormal"]); ax.set_yticklabels(["normal", "abnormal"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "confusion.png"), dpi=130); plt.close(fig)

    r = permutation_importance(best_model, Xte, yte, n_repeats=5,
                               random_state=cfg["seed"], scoring="roc_auc")
    imp = r.importances_mean
    mean_abn = X[y == 1].mean(0)
    impn = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(mean_abn, c="k", lw=1, label="mean abnormal beat")
    sc = ax.scatter(range(len(mean_abn)), mean_abn, c=impn, cmap="Reds", s=14)
    fig.colorbar(sc, ax=ax, label="permutation importance (norm.)")
    ax.set_title("Which parts of the heartbeat drive the prediction")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "importance.png"), dpi=130); plt.close(fig)

    nn = models.get("neural_net (MLP)")
    if nn is not None and hasattr(nn, "loss_curve_"):
        fig, ax = plt.subplots(figsize=(5, 4)); ax.plot(nn.loss_curve_)
        ax.set_title("MLP training loss"); ax.set_xlabel("iteration"); ax.set_ylabel("loss")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "training_curve.png"), dpi=130); plt.close(fig)

    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(best_model, os.path.join(outdir, "model.joblib"))
    return metrics
