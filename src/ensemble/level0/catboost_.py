"""CatBoost Level-0 model with balanced class weights and native categoricals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from ensemble.config import CFG


def train_and_predict(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_te: pd.DataFrame,
    cfg: CFG,
) -> tuple[np.ndarray, np.ndarray]:
    """Train CatBoost and return (val_pred, test_pred)."""
    params = dict(cfg.CB_PARAMS)
    params["auto_class_weights"] = "Balanced"
    cols = X_tr.columns.tolist()

    cat_features = [
        c for c in cols
        if X_tr[c].dtype.name == "category" or X_tr[c].dtype == object
    ]

    for c in cat_features:
        for df in (X_tr, X_val, X_te):
            df[c] = df[c].astype(str)

    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_pred = model.predict_proba(X_val)[:, 1].astype("float32")
    test_pred = model.predict_proba(X_te[cols])[:, 1].astype("float32")
    return val_pred, test_pred
