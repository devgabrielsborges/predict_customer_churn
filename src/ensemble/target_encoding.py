"""Leak-free target encoding via inner K-Fold.

Three encoding methods:
1. ``inner_kfold_te``      – std / min / max statistics per categorical group
2. ``inner_kfold_te_ngram`` – mean TE for bi-gram / tri-gram composites
3. ``sklearn_te_mean``     – sklearn ``TargetEncoder`` (mean, internal CV)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from ensemble.config import CFG

STATS = ["std", "min", "max"]


def inner_kfold_te(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    X_te: pd.DataFrame,
    te_columns: list[str],
    target: str,
    cfg: CFG,
) -> list[str]:
    """Compute std / min / max target-encoding columns via inner K-Fold.

    Modifies *X_tr*, *X_val*, *X_te* **in-place**.
    Returns list of new column names.
    """
    skf = StratifiedKFold(
        n_splits=cfg.INNER_FOLDS, shuffle=True, random_state=cfg.RANDOM_SEED,
    )
    created: list[str] = []

    # Inner fold encoding (leak-free for X_tr)
    for _, (in_tr, in_va) in enumerate(skf.split(X_tr, y_tr)):
        X_tr2 = X_tr.loc[in_tr, te_columns + [target]].copy()
        X_va2 = X_tr.loc[in_va, te_columns].copy()
        for col in te_columns:
            tmp = X_tr2.groupby(col, observed=False)[target].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            X_va2 = X_va2.merge(tmp, on=col, how="left")
            for c in tmp.columns:
                X_tr.loc[in_va, c] = X_va2[c].values.astype("float32")

    # Full-fold encoding for val / test
    for col in te_columns:
        tmp = X_tr.groupby(col, observed=False)[target].agg(STATS)
        tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
        tmp = tmp.astype("float32")
        X_val_tmp = X_val[[col]].merge(tmp, on=col, how="left")
        X_te_tmp = X_te[[col]].merge(tmp, on=col, how="left")
        for c in tmp.columns:
            X_val[c] = X_val_tmp[c].values
            X_te[c] = X_te_tmp[c].values
            for df in (X_tr, X_val, X_te):
                df[c] = df[c].fillna(0).astype("float32")
            if c not in created:
                created.append(c)

    return created


def inner_kfold_te_ngram(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    X_te: pd.DataFrame,
    ngram_cols: list[str],
    target: str,
    cfg: CFG,
) -> list[str]:
    """Mean target encoding for n-gram composite columns via inner K-Fold.

    Modifies *X_tr*, *X_val*, *X_te* **in-place**.
    Returns list of new column names.
    """
    if not ngram_cols:
        return []

    skf = StratifiedKFold(
        n_splits=cfg.INNER_FOLDS, shuffle=True, random_state=cfg.RANDOM_SEED,
    )
    created: list[str] = []

    for _, (in_tr, in_va) in enumerate(skf.split(X_tr, y_tr)):
        X_tr2 = X_tr.loc[in_tr].copy()
        X_va2 = X_tr.loc[in_va].copy()
        for col in ngram_cols:
            ng_te = X_tr2.groupby(col, observed=False)[target].mean()
            ng_name = f"TE_ng_{col}"
            mapped = X_va2[col].astype(str).map(ng_te)
            X_tr.loc[in_va, ng_name] = (
                pd.to_numeric(mapped, errors="coerce").fillna(0.5).astype("float32").values
            )

    for col in ngram_cols:
        ng_te = X_tr.groupby(col, observed=False)[target].mean()
        ng_name = f"TE_ng_{col}"
        X_val[ng_name] = (
            pd.to_numeric(X_val[col].astype(str).map(ng_te), errors="coerce")
            .fillna(0.5).astype("float32")
        )
        X_te[ng_name] = (
            pd.to_numeric(X_te[col].astype(str).map(ng_te), errors="coerce")
            .fillna(0.5).astype("float32")
        )
        if ng_name in X_tr.columns:
            X_tr[ng_name] = (
                pd.to_numeric(X_tr[ng_name], errors="coerce")
                .fillna(0.5).astype("float32")
            )
        else:
            X_tr[ng_name] = 0.5
        created.append(ng_name)

    return created


def sklearn_te_mean(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    X_te: pd.DataFrame,
    te_columns: list[str],
    cfg: CFG,
) -> list[str]:
    """Apply sklearn ``TargetEncoder`` (mean encoding with internal CV).

    Modifies *X_tr*, *X_val*, *X_te* **in-place**.
    Returns list of new column names.
    """
    te_mean_cols = [f"TE_{col}" for col in te_columns]
    te = TargetEncoder(
        cv=cfg.INNER_FOLDS,
        shuffle=True,
        smooth="auto",
        target_type="binary",
        random_state=cfg.RANDOM_SEED,
    )
    X_tr[te_mean_cols] = te.fit_transform(X_tr[te_columns], y_tr)
    X_val[te_mean_cols] = te.transform(X_val[te_columns])
    X_te[te_mean_cols] = te.transform(X_te[te_columns])
    return te_mean_cols
