"""Stage-1 Ridge linear model.

StandardScaler on numerics + OneHotEncoder on categoricals -> sparse matrix ->
Ridge regression.  Predictions clipped to [0, 1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ensemble.config import CATS, CFG


def train_and_predict(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    X_te: pd.DataFrame,
    numeric_features: list[str],
    cfg: CFG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train Ridge and return (train_pred, val_pred, test_pred)."""
    scaler = StandardScaler()
    X_tr_num = scaler.fit_transform(X_tr[numeric_features].fillna(0))
    X_val_num = scaler.transform(X_val[numeric_features].fillna(0))
    X_te_num = scaler.transform(X_te[numeric_features].fillna(0))

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    cat_cols = [c for c in CATS if c in X_tr.columns]
    X_tr_cat = ohe.fit_transform(X_tr[cat_cols].astype(str))
    X_val_cat = ohe.transform(X_val[cat_cols].astype(str))
    X_te_cat = ohe.transform(X_te[cat_cols].astype(str))

    X_tr_sparse = sparse.hstack([X_tr_num, X_tr_cat]).tocsr()
    X_val_sparse = sparse.hstack([X_val_num, X_val_cat]).tocsr()
    X_te_sparse = sparse.hstack([X_te_num, X_te_cat]).tocsr()

    model = Ridge(alpha=cfg.RIDGE_ALPHA, random_state=cfg.RANDOM_SEED)
    model.fit(X_tr_sparse, y_tr)

    tr_pred = np.clip(model.predict(X_tr_sparse), 0, 1).astype("float32")
    val_pred = np.clip(model.predict(X_val_sparse), 0, 1).astype("float32")
    te_pred = np.clip(model.predict(X_te_sparse), 0, 1).astype("float32")

    return tr_pred, val_pred, te_pred
