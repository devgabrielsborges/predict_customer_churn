"""Level-1 stacking meta-learner.

Trains a simple model on the OOF predictions from Level-0 models to produce
a blended prediction that captures complementary strengths.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from ensemble.config import CFG


def _get_meta_learner(cfg: CFG):
    if cfg.meta_learner == "logistic_regression":
        return LogisticRegression(
            C=1.0, max_iter=1000, random_state=cfg.RANDOM_SEED,
        )
    if cfg.meta_learner == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=cfg.RANDOM_SEED,
            verbosity=0,
            eval_metric="auc",
        )
    # Default: Ridge
    return Ridge(alpha=1.0, random_state=cfg.RANDOM_SEED)


def _predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Unified predict that returns P(class=1)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.clip(model.predict(X), 0, 1)


def stack(
    oof_preds: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    y_train: np.ndarray,
    cfg: CFG,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a meta-learner on Level-0 OOF predictions.

    Parameters
    ----------
    oof_preds : dict mapping model name -> OOF prediction array (len = n_train)
    test_preds : dict mapping model name -> mean test prediction array (len = n_test)
    y_train : true training labels
    cfg : pipeline configuration

    Returns
    -------
    stacked_oof : meta-learner OOF predictions (len = n_train)
    stacked_test : meta-learner test predictions (len = n_test)
    """
    names = sorted(oof_preds.keys())
    X_oof = np.column_stack([oof_preds[n] for n in names])
    X_test = np.column_stack([test_preds[n] for n in names])

    skf = StratifiedKFold(
        n_splits=cfg.INNER_FOLDS, shuffle=True, random_state=cfg.RANDOM_SEED,
    )

    stacked_oof = np.zeros(len(y_train), dtype="float32")
    stacked_test = np.zeros(X_test.shape[0], dtype="float32")

    for tr_idx, va_idx in skf.split(X_oof, y_train):
        model = _get_meta_learner(cfg)
        model.fit(X_oof[tr_idx], y_train[tr_idx])
        stacked_oof[va_idx] = _predict_proba(model, X_oof[va_idx])
        stacked_test += _predict_proba(model, X_test) / cfg.INNER_FOLDS

    return stacked_oof, stacked_test.astype("float32")
