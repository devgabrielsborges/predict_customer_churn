"""Optuna-driven joint optimisation of features, hyperparams, and ensemble.

The single ``OptunaOptimizer.run()`` method launches a study that explores
feature-group toggles, per-model hyperparameters, model inclusion flags,
meta-learner choice, and blending weights — all in one trial.  Each trial
runs a fast *OPTUNA_FOLDS*-fold (default 5) ensemble pipeline and returns
the blended OOF ROC-AUC.
"""
from __future__ import annotations

import gc
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ensemble.blender import blend, find_optimal_threshold
from ensemble.config import CATS, NUMS, CFG
from ensemble.features import FeatureGroups, engineer_features
from ensemble.level0 import catboost_ as cb_model
from ensemble.level0 import lightgbm_ as lgb_model
from ensemble.level0 import ridge as ridge_model
from ensemble.level0 import xgboost_ as xgb_model
from ensemble.stacking import stack
from ensemble.target_encoding import (
    inner_kfold_te,
    inner_kfold_te_ngram,
    sklearn_te_mean,
)


# ─── Search space definition ────────────────────────────────────────────

def _suggest(trial: optuna.Trial) -> dict:
    """Build a flat dict of all Optuna-suggested values for one trial."""
    p: dict = {}

    # A) Feature selection
    p["use_distribution_features"] = trial.suggest_categorical(
        "use_distribution_features", [True, False],
    )
    p["use_quantile_features"] = trial.suggest_categorical(
        "use_quantile_features", [True, False],
    )
    p["use_digit_features"] = trial.suggest_categorical(
        "use_digit_features", [True, False],
    )
    p["use_ngram_features"] = trial.suggest_categorical(
        "use_ngram_features", [True, False],
    )
    p["use_orig_proba"] = trial.suggest_categorical(
        "use_orig_proba", [True, False],
    )
    p["ngram_top_k"] = trial.suggest_int("ngram_top_k", 4, 6)

    # B) Per-model hyperparams
    # Ridge
    p["RIDGE_ALPHA"] = trial.suggest_float("RIDGE_ALPHA", 0.1, 100.0, log=True)

    # XGBoost
    p["xgb_learning_rate"] = trial.suggest_float(
        "xgb_learning_rate", 1e-3, 0.1, log=True,
    )
    p["xgb_max_depth"] = trial.suggest_int("xgb_max_depth", 3, 8)
    p["xgb_subsample"] = trial.suggest_float("xgb_subsample", 0.5, 1.0)
    p["xgb_colsample_bytree"] = trial.suggest_float(
        "xgb_colsample_bytree", 0.2, 0.8,
    )
    p["xgb_min_child_weight"] = trial.suggest_int("xgb_min_child_weight", 1, 10)
    p["xgb_reg_alpha"] = trial.suggest_float("xgb_reg_alpha", 1e-3, 10.0, log=True)
    p["xgb_reg_lambda"] = trial.suggest_float("xgb_reg_lambda", 1e-3, 10.0, log=True)
    p["xgb_gamma"] = trial.suggest_float("xgb_gamma", 1e-3, 5.0, log=True)

    # LightGBM
    p["lgb_learning_rate"] = trial.suggest_float(
        "lgb_learning_rate", 1e-3, 0.1, log=True,
    )
    p["lgb_max_depth"] = trial.suggest_int("lgb_max_depth", 3, 10)
    p["lgb_num_leaves"] = trial.suggest_int("lgb_num_leaves", 15, 127)
    p["lgb_subsample"] = trial.suggest_float("lgb_subsample", 0.5, 1.0)
    p["lgb_colsample_bytree"] = trial.suggest_float(
        "lgb_colsample_bytree", 0.3, 0.8,
    )
    p["lgb_min_child_samples"] = trial.suggest_int("lgb_min_child_samples", 5, 100)
    p["lgb_reg_alpha"] = trial.suggest_float("lgb_reg_alpha", 1e-3, 10.0, log=True)
    p["lgb_reg_lambda"] = trial.suggest_float("lgb_reg_lambda", 1e-3, 10.0, log=True)

    # CatBoost
    p["cb_learning_rate"] = trial.suggest_float(
        "cb_learning_rate", 1e-3, 0.1, log=True,
    )
    p["cb_depth"] = trial.suggest_int("cb_depth", 4, 8)
    p["cb_l2_leaf_reg"] = trial.suggest_float("cb_l2_leaf_reg", 0.1, 10.0, log=True)
    p["cb_bagging_temperature"] = trial.suggest_float(
        "cb_bagging_temperature", 0.0, 10.0,
    )

    # Pseudo-label threshold
    p["PSEUDO_LABEL_THRESHOLD"] = trial.suggest_float(
        "PSEUDO_LABEL_THRESHOLD", 0.98, 0.999,
    )

    # C) Model selection & ensemble
    p["include_xgboost"] = True  # always on
    p["include_lightgbm"] = trial.suggest_categorical(
        "include_lightgbm", [True, False],
    )
    p["include_catboost"] = trial.suggest_categorical(
        "include_catboost", [True, False],
    )
    p["include_xgb_pseudo"] = trial.suggest_categorical(
        "include_xgb_pseudo", [True, False],
    )
    p["meta_learner"] = trial.suggest_categorical(
        "meta_learner", ["ridge", "logistic_regression", "xgboost"],
    )

    # Blend weights (raw; normalised at blend time)
    p["blend_weight_xgb"] = trial.suggest_float("blend_weight_xgb", 0.05, 1.0)
    if p["include_lightgbm"]:
        p["blend_weight_lgbm"] = trial.suggest_float("blend_weight_lgbm", 0.05, 1.0)
    else:
        p["blend_weight_lgbm"] = 0.0
    if p["include_catboost"]:
        p["blend_weight_catboost"] = trial.suggest_float(
            "blend_weight_catboost", 0.05, 1.0,
        )
    else:
        p["blend_weight_catboost"] = 0.0
    if p["include_xgb_pseudo"]:
        p["blend_weight_xgb_pseudo"] = trial.suggest_float(
            "blend_weight_xgb_pseudo", 0.05, 1.0,
        )
    else:
        p["blend_weight_xgb_pseudo"] = 0.0
    p["blend_weight_stacking"] = trial.suggest_float(
        "blend_weight_stacking", 0.05, 1.0,
    )

    return p


# ─── Fast pipeline for a single trial ───────────────────────────────────

def _run_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    orig_df: pd.DataFrame,
    cfg: CFG,
    n_folds: int,
) -> float:
    """Run the full ensemble pipeline with *n_folds* and return OOF AUC."""
    train = train_df.copy()
    test = test_df.copy()
    orig = orig_df.copy()

    fg: FeatureGroups = engineer_features(train, test, orig, cfg)

    y_all = train[cfg.TARGET].values
    cfg.compute_scale_pos_weight(y_all)

    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=cfg.RANDOM_SEED,
    )

    oof: dict[str, np.ndarray] = {}
    test_avg: dict[str, np.ndarray] = {}
    model_names: list[str] = []

    if cfg.include_xgboost:
        model_names.append("xgb")
        oof["xgb"] = np.zeros(len(train))
        test_avg["xgb"] = np.zeros(len(test))
    if cfg.include_lightgbm:
        model_names.append("lgbm")
        oof["lgbm"] = np.zeros(len(train))
        test_avg["lgbm"] = np.zeros(len(test))
    if cfg.include_catboost:
        model_names.append("catboost")
        oof["catboost"] = np.zeros(len(train))
        test_avg["catboost"] = np.zeros(len(test))
    if cfg.include_xgb_pseudo:
        model_names.append("xgb_pseudo")
        oof["xgb_pseudo"] = np.zeros(len(train))
        test_avg["xgb_pseudo"] = np.zeros(len(test))

    ridge_oof = np.zeros(len(train))
    ridge_test = np.zeros(len(test))

    features = fg.all_features
    te_cols = fg.te_columns
    te_ng_cols = fg.te_ngram_columns
    to_remove = fg.to_remove

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train, y_all)):
        X_tr = train.loc[tr_idx, features + [cfg.TARGET]].reset_index(drop=True).copy()
        y_tr = y_all[tr_idx]
        X_val = train.loc[va_idx, features].reset_index(drop=True).copy()
        y_val = y_all[va_idx]
        X_te = test[features].reset_index(drop=True).copy()

        # Target encoding
        te1_cols = inner_kfold_te(X_tr, y_tr, X_val, X_te, te_cols, cfg.TARGET, cfg)
        te_ng_new = inner_kfold_te_ngram(
            X_tr, y_tr, X_val, X_te, te_ng_cols, cfg.TARGET, cfg,
        )
        te_mean_cols = sklearn_te_mean(X_tr, y_tr, X_val, X_te, te_cols, cfg)

        # Ridge numeric features
        ridge_num = (
            list(NUMS) + fg.new_nums
            + [c for c in X_tr.columns if c.startswith("TE1_")]
            + [c for c in X_tr.columns if c.startswith("TE_ng_")]
            + te_mean_cols
        )

        # Stage 1: Ridge
        ridge_tr, ridge_va, ridge_te = ridge_model.train_and_predict(
            X_tr, y_tr, X_val, X_te, ridge_num, cfg,
        )
        ridge_oof[va_idx] = ridge_va
        ridge_test += ridge_te / n_folds

        # Add ridge_pred to features
        X_tr["ridge_pred"] = ridge_tr
        X_val["ridge_pred"] = ridge_va
        X_te["ridge_pred"] = ridge_te

        # Prepare for tree models: cast cats, drop raw categoricals
        for df in (X_tr, X_val, X_te):
            for c in CATS + fg.num_as_cat:
                if c in df.columns:
                    df[c] = df[c].astype(str).astype("category")
            df.drop(
                columns=[c for c in to_remove if c in df.columns],
                inplace=True, errors="ignore",
            )
        X_tr.drop(columns=[cfg.TARGET], inplace=True, errors="ignore")
        tree_cols = X_tr.columns

        # Stage 2: Tree models
        if cfg.include_xgboost:
            xgb_va, xgb_te = xgb_model.train_and_predict(
                X_tr, y_tr, X_val, y_val, X_te, cfg, use_pseudo_labels=False,
            )
            oof["xgb"][va_idx] = xgb_va
            test_avg["xgb"] += xgb_te / n_folds

        if cfg.include_lightgbm:
            lgb_va, lgb_te = lgb_model.train_and_predict(
                X_tr, y_tr, X_val, y_val, X_te, cfg,
            )
            oof["lgbm"][va_idx] = lgb_va
            test_avg["lgbm"] += lgb_te / n_folds

        if cfg.include_catboost:
            cb_va, cb_te = cb_model.train_and_predict(
                X_tr, y_tr, X_val, y_val, X_te, cfg,
            )
            oof["catboost"][va_idx] = cb_va
            test_avg["catboost"] += cb_te / n_folds

        if cfg.include_xgb_pseudo:
            xgbp_va, xgbp_te = xgb_model.train_and_predict(
                X_tr, y_tr, X_val, y_val, X_te, cfg, use_pseudo_labels=True,
            )
            oof["xgb_pseudo"][va_idx] = xgbp_va
            test_avg["xgb_pseudo"] += xgbp_te / n_folds

        del X_tr, X_val, X_te, y_tr, y_val
        gc.collect()

    # Stacking
    stacked_oof, stacked_test = stack(oof, test_avg, y_all, cfg)
    oof["stacking"] = stacked_oof
    test_avg["stacking"] = stacked_test

    # Blend
    from ensemble.blender import blend_from_cfg
    blended_oof = blend_from_cfg(oof, cfg)
    return float(roc_auc_score(y_all, blended_oof))


# ─── Optimizer ───────────────────────────────────────────────────────────

class OptunaOptimizer:
    """Joint Optuna optimiser for the full ensemble pipeline."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        orig_df: pd.DataFrame,
        base_cfg: CFG | None = None,
    ):
        self.train_df = train_df
        self.test_df = test_df
        self.orig_df = orig_df
        self.base_cfg = base_cfg or CFG()

    def _objective(self, trial: optuna.Trial) -> float:
        params = _suggest(trial)
        cfg = deepcopy(self.base_cfg)
        cfg.apply_overrides(params)

        try:
            auc = _run_pipeline(
                self.train_df, self.test_df, self.orig_df,
                cfg, n_folds=cfg.OPTUNA_FOLDS,
            )
        except Exception as exc:
            print(f"  Trial {trial.number} failed: {exc}")
            return 0.5

        print(f"  Trial {trial.number}: AUC={auc:.5f}")
        return auc

    def run(self, n_trials: int | None = None) -> dict:
        """Launch the Optuna study and return the best config dict."""
        n_trials = n_trials or self.base_cfg.OPTUNA_N_TRIALS

        study = optuna.create_study(
            direction="maximize",
            study_name="ensemble_optimizer",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        )
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)

        print("\n" + "=" * 70)
        print("OPTUNA OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"  Best trial : {study.best_trial.number}")
        print(f"  Best AUC   : {study.best_value:.5f}")
        print(f"  Best params:")
        for k, v in study.best_params.items():
            print(f"    {k}: {v}")

        self.study = study
        return dict(study.best_params)

    def best_config(self) -> dict:
        """Return the best trial's full parameter dict."""
        return dict(self.study.best_params)

    def save_best(self, path: str | Path = "best_config.json"):
        """Write best params to a JSON file."""
        Path(path).write_text(
            json.dumps(self.study.best_params, indent=2, default=str),
        )
        print(f"Best config saved to {path}")
