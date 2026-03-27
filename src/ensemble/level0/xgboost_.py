"""XGBoost Level-0 model with ridge_pred feature and optional pseudo labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from ensemble.config import CFG


def _build_model(cfg: CFG) -> XGBClassifier:
    params = dict(cfg.XGB_PARAMS)
    if cfg.SCALE_POS_WEIGHT is not None:
        params["scale_pos_weight"] = cfg.SCALE_POS_WEIGHT
    return XGBClassifier(**params)


def train_and_predict(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_te: pd.DataFrame,
    cfg: CFG,
    use_pseudo_labels: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Train XGBoost and return (val_pred, test_pred).

    When *use_pseudo_labels* is True, high-confidence test predictions are
    added to training and the model is retrained — kept only if OOF AUC
    improves.
    """
    cols = X_tr.columns
    model = _build_model(cfg)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)

    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_te[cols])[:, 1]

    if use_pseudo_labels:
        base_auc = roc_auc_score(y_val, val_pred)
        threshold = cfg.PSEUDO_LABEL_THRESHOLD
        mask = (test_pred > threshold) | (test_pred < 1 - threshold)
        if mask.sum() > 0:
            X_tr_pl = pd.concat([X_tr, X_te[cols][mask]], axis=0)
            y_tr_pl = np.concatenate([y_tr, (test_pred[mask] > 0.5).astype(int)])

            model2 = _build_model(cfg)
            model2.fit(X_tr_pl, y_tr_pl, eval_set=[(X_val, y_val)], verbose=0)
            val_pred2 = model2.predict_proba(X_val)[:, 1]
            pl_auc = roc_auc_score(y_val, val_pred2)

            if pl_auc > base_auc:
                val_pred = val_pred2
                test_pred = model2.predict_proba(X_te[cols])[:, 1]

    return val_pred.astype("float32"), test_pred.astype("float32")
