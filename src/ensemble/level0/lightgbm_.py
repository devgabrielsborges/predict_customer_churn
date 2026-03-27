"""LightGBM Level-0 model with ridge_pred feature and class-imbalance handling."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from ensemble.config import CFG


def train_and_predict(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_te: pd.DataFrame,
    cfg: CFG,
) -> tuple[np.ndarray, np.ndarray]:
    """Train LightGBM and return (val_pred, test_pred)."""
    params = dict(cfg.LGB_PARAMS)
    params["is_unbalance"] = True
    cols = X_tr.columns

    model = LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(500, verbose=False), log_evaluation(0)],
    )

    val_pred = model.predict_proba(X_val)[:, 1].astype("float32")
    test_pred = model.predict_proba(X_te[cols])[:, 1].astype("float32")
    return val_pred, test_pred
