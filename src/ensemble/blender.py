"""Blending and threshold optimisation.

* ``optimize_blend_weights`` – find OOF-AUC-maximising convex combination
  of model predictions via ``scipy.optimize.minimize``.
* ``find_optimal_threshold``  – sweep thresholds for best churn=1 F1.
* ``blend`` – apply pre-determined weights to prediction arrays.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score, roc_auc_score


def optimize_blend_weights(
    oof_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> dict[str, float]:
    """Find weights that maximise OOF ROC-AUC.

    Returns a dict mapping model name -> optimal weight (sums to 1).
    """
    names = sorted(oof_preds.keys())
    n_models = len(names)
    if n_models == 1:
        return {names[0]: 1.0}

    preds = np.column_stack([oof_preds[n] for n in names])

    def _neg_auc(raw_weights: np.ndarray) -> float:
        w = np.abs(raw_weights)
        w = w / w.sum()
        blended = preds @ w
        return -roc_auc_score(y_true, blended)

    x0 = np.ones(n_models) / n_models
    result = minimize(
        _neg_auc, x0,
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8},
    )
    w = np.abs(result.x)
    w = w / w.sum()
    return {name: float(weight) for name, weight in zip(names, w)}


def blend(
    preds: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """Apply *weights* to *preds* and return the weighted average."""
    total_w = sum(weights[n] for n in preds if n in weights)
    if total_w == 0:
        names = sorted(preds.keys())
        return np.mean([preds[n] for n in names], axis=0)

    result = np.zeros_like(next(iter(preds.values())), dtype="float64")
    for name, arr in preds.items():
        if name in weights:
            result += arr * (weights[name] / total_w)
    return result.astype("float32")


def blend_from_cfg(
    preds: dict[str, np.ndarray],
    cfg,
) -> np.ndarray:
    """Blend using the weights stored in *cfg*."""
    weight_map = {
        "xgb": cfg.blend_weight_xgb,
        "lgbm": cfg.blend_weight_lgbm,
        "catboost": cfg.blend_weight_catboost,
        "xgb_pseudo": cfg.blend_weight_xgb_pseudo,
        "stacking": cfg.blend_weight_stacking,
    }
    active = {k: v for k, v in weight_map.items() if k in preds and v > 0}
    return blend(preds, active)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Sweep thresholds and return (best_threshold, best_f1) for class 1."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.90, 0.005)

    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, pos_label=1)
        if score > best_f1:
            best_f1, best_t = score, float(t)
    return best_t, best_f1
