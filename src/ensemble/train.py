#!/usr/bin/env python3
"""Ensemble training orchestrator.

Phase 1 (optional): Optuna optimisation — fast 5-fold pipeline search.
Phase 2:            Final production run — 20-fold CV with best config.

Usage
-----
    # Full two-phase run
    uv run --python 3.11 src/ensemble/train.py

    # Skip Optuna, use saved config
    uv run --python 3.11 src/ensemble/train.py --config best_config.json

    # Only run Optuna optimisation (no final run)
    uv run --python 3.11 src/ensemble/train.py --optimize-only

    # Custom trial count
    uv run --python 3.11 src/ensemble/train.py --n-trials 50
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

# Ensure src/ is on the path for local imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ensemble.blender import (
    blend,
    blend_from_cfg,
    find_optimal_threshold,
    optimize_blend_weights,
)
from ensemble.config import CATS, NUMS, CFG
from ensemble.features import FeatureGroups, engineer_features
from ensemble.level0 import catboost_ as cb_model
from ensemble.level0 import lightgbm_ as lgb_model
from ensemble.level0 import ridge as ridge_model
from ensemble.level0 import xgboost_ as xgb_model
from ensemble.optimizer import OptunaOptimizer
from ensemble.stacking import stack
from ensemble.target_encoding import (
    inner_kfold_te,
    inner_kfold_te_ngram,
    sklearn_te_mean,
)


# ── Data loading ─────────────────────────────────────────────────────────

def load_data(cfg: CFG) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[1/3] Loading competition train set …")
    train = pd.read_csv(cfg.train_path)
    train[cfg.TARGET] = train[cfg.TARGET].map({"No": 0, "Yes": 1}).astype(int)

    print("[2/3] Loading competition test set …")
    test = pd.read_csv(cfg.test_path)

    print("[3/3] Loading original IBM Telco dataset …")
    orig = pd.read_csv(cfg.original_path)
    orig[cfg.TARGET] = orig[cfg.TARGET].map({"No": 0, "Yes": 1}).astype(int)
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce")
    orig["TotalCharges"].fillna(orig["TotalCharges"].median(), inplace=True)
    if "customerID" in orig.columns:
        orig.drop(columns=["customerID"], inplace=True)

    print(f"  Train : {len(train):,} rows  |  Churn rate : {train[cfg.TARGET].mean()*100:.2f}%")
    print(f"  Orig  : {len(orig):,}   rows  |  Churn rate : {orig[cfg.TARGET].mean()*100:.2f}%")
    print(f"  Test  : {len(test):,} rows")
    return train, test, orig


# ── Final production run ─────────────────────────────────────────────────

def run_final_pipeline(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    orig_raw: pd.DataFrame,
    cfg: CFG,
) -> None:
    train = train_raw.copy()
    test = test_raw.copy()
    orig = orig_raw.copy()

    train_ids = train["id"].copy()
    test_ids = test["id"].copy()

    print("\n" + "=" * 70)
    print("PHASE 2 — FINAL ENSEMBLE PIPELINE")
    print(f"  Outer folds : {cfg.N_FOLDS}")
    print(f"  Inner folds : {cfg.INNER_FOLDS}")
    print("=" * 70)

    # Feature engineering
    print("\n[Feature Engineering] …")
    fg: FeatureGroups = engineer_features(train, test, orig, cfg)
    print(f"  Total features: {len(fg.all_features)}")

    y_all = train[cfg.TARGET].values
    cfg.compute_scale_pos_weight(y_all)

    skf = StratifiedKFold(
        n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.RANDOM_SEED,
    )

    # OOF / test accumulators
    oof: dict[str, np.ndarray] = {}
    test_avg: dict[str, np.ndarray] = {}
    fold_scores: dict[str, list[float]] = {}

    active_models: list[str] = []
    if cfg.include_xgboost:
        active_models.append("xgb")
    if cfg.include_lightgbm:
        active_models.append("lgbm")
    if cfg.include_catboost:
        active_models.append("catboost")
    if cfg.include_xgb_pseudo:
        active_models.append("xgb_pseudo")

    for m in active_models:
        oof[m] = np.zeros(len(train))
        test_avg[m] = np.zeros(len(test))
        fold_scores[m] = []

    ridge_oof = np.zeros(len(train))
    ridge_test = np.zeros(len(test))
    ridge_fold_scores: list[float] = []

    features = fg.all_features
    te_cols = fg.te_columns
    te_ng_cols = fg.te_ngram_columns
    to_remove = fg.to_remove

    t0 = time.time()

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train, y_all)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{cfg.N_FOLDS}")
        print(f"{'='*60}")

        X_tr = train.loc[tr_idx, features + [cfg.TARGET]].reset_index(drop=True).copy()
        y_tr = y_all[tr_idx]
        X_val = train.loc[va_idx, features].reset_index(drop=True).copy()
        y_val = y_all[va_idx]
        X_te = test[features].reset_index(drop=True).copy()

        # ── Target Encoding ──────────────────────────────────────────
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

        # ── Stage 1: Ridge ───────────────────────────────────────────
        ridge_tr, ridge_va, ridge_te = ridge_model.train_and_predict(
            X_tr, y_tr, X_val, X_te, ridge_num, cfg,
        )
        ridge_oof[va_idx] = ridge_va
        ridge_test += ridge_te / cfg.N_FOLDS
        ridge_auc = roc_auc_score(y_val, ridge_va)
        ridge_fold_scores.append(ridge_auc)
        print(f"  Ridge AUC : {ridge_auc:.5f}")

        # Add ridge_pred to features
        X_tr["ridge_pred"] = ridge_tr
        X_val["ridge_pred"] = ridge_va
        X_te["ridge_pred"] = ridge_te

        # Prepare for tree models
        for df in (X_tr, X_val, X_te):
            for c in CATS + fg.num_as_cat:
                if c in df.columns:
                    df[c] = df[c].astype(str).astype("category")
            df.drop(
                columns=[c for c in to_remove if c in df.columns],
                inplace=True, errors="ignore",
            )
        X_tr.drop(columns=[cfg.TARGET], inplace=True, errors="ignore")

        # ── Stage 2: Tree models ─────────────────────────────────────
        if cfg.include_xgboost:
            va_p, te_p = xgb_model.train_and_predict(
                X_tr, y_tr, X_val, y_val, X_te, cfg, use_pseudo_labels=False,
            )
            oof["xgb"][va_idx] = va_p
            test_avg["xgb"] += te_p / cfg.N_FOLDS
            auc = roc_auc_score(y_val, va_p)
            fold_scores["xgb"].append(auc)
            print(f"  XGB AUC   : {auc:.5f}")

        if cfg.include_lightgbm:
            va_p, te_p = lgb_model.train_and_predict(
                X_tr, y_tr, X_val, y_val, X_te, cfg,
            )
            oof["lgbm"][va_idx] = va_p
            test_avg["lgbm"] += te_p / cfg.N_FOLDS
            auc = roc_auc_score(y_val, va_p)
            fold_scores["lgbm"].append(auc)
            print(f"  LGBM AUC  : {auc:.5f}")

        if cfg.include_catboost:
            va_p, te_p = cb_model.train_and_predict(
                X_tr, y_tr, X_val, y_val, X_te, cfg,
            )
            oof["catboost"][va_idx] = va_p
            test_avg["catboost"] += te_p / cfg.N_FOLDS
            auc = roc_auc_score(y_val, va_p)
            fold_scores["catboost"].append(auc)
            print(f"  CB AUC    : {auc:.5f}")

        if cfg.include_xgb_pseudo:
            va_p, te_p = xgb_model.train_and_predict(
                X_tr, y_tr, X_val, y_val, X_te, cfg, use_pseudo_labels=True,
            )
            oof["xgb_pseudo"][va_idx] = va_p
            test_avg["xgb_pseudo"] += te_p / cfg.N_FOLDS
            auc = roc_auc_score(y_val, va_p)
            fold_scores["xgb_pseudo"].append(auc)
            print(f"  XGB+PL AUC: {auc:.5f}")

        del X_tr, X_val, X_te, y_tr, y_val
        gc.collect()

    elapsed = (time.time() - t0) / 60

    # ── Stacking ─────────────────────────────────────────────────────
    print("\n[Stacking] Training meta-learner …")
    stacked_oof, stacked_test = stack(oof, test_avg, y_all, cfg)
    oof["stacking"] = stacked_oof
    test_avg["stacking"] = stacked_test
    stacking_auc = roc_auc_score(y_all, stacked_oof)
    print(f"  Stacking OOF AUC : {stacking_auc:.5f}")

    # ── Blending ─────────────────────────────────────────────────────
    print("\n[Blending] Optimising weights …")
    opt_weights = optimize_blend_weights(oof, y_all)
    blended_oof = blend(oof, opt_weights)
    blended_test = blend(test_avg, opt_weights)
    blend_auc = roc_auc_score(y_all, blended_oof)
    print(f"  Optimised blend OOF AUC : {blend_auc:.5f}")
    for name, w in opt_weights.items():
        print(f"    {name:15s}: {w:.4f}")

    # Also compute cfg-weighted blend for comparison
    cfg_blended_oof = blend_from_cfg(oof, cfg)
    cfg_blend_auc = roc_auc_score(y_all, cfg_blended_oof)
    print(f"  CFG-weighted blend OOF AUC : {cfg_blend_auc:.5f}")

    # Use whichever blend is better
    if blend_auc >= cfg_blend_auc:
        final_oof = blended_oof
        final_test = blended_test
        final_auc = blend_auc
        blend_method = "optimized"
    else:
        final_oof = cfg_blended_oof
        final_test = blend_from_cfg(test_avg, cfg)
        final_auc = cfg_blend_auc
        blend_method = "cfg-weighted"

    # ── Threshold tuning for churn=1 ─────────────────────────────────
    print("\n[Threshold Tuning] Optimising for churn=1 F1 …")
    best_t, best_f1 = find_optimal_threshold(y_all, final_oof)
    y_pred_opt = (final_oof >= best_t).astype(int)
    recall_1 = recall_score(y_all, y_pred_opt, pos_label=1)
    precision_1 = precision_score(y_all, y_pred_opt, pos_label=1)
    print(f"  Optimal threshold : {best_t:.3f}")
    print(f"  Churn=1 F1        : {best_f1:.4f}")
    print(f"  Churn=1 Recall    : {recall_1:.4f}")
    print(f"  Churn=1 Precision : {precision_1:.4f}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Ridge OOF AUC    : {roc_auc_score(y_all, ridge_oof):.5f}")
    for m in active_models:
        m_auc = roc_auc_score(y_all, oof[m])
        m_mean = np.mean(fold_scores[m])
        m_std = np.std(fold_scores[m])
        print(f"  {m:15s} OOF AUC : {m_auc:.5f}  (fold mean {m_mean:.5f} +/- {m_std:.5f})")
    print(f"  Stacking OOF AUC : {stacking_auc:.5f}")
    print(f"  Blend OOF AUC    : {final_auc:.5f} ({blend_method})")
    print(f"  Wall time        : {elapsed:.1f} min")

    print("\n  Classification report (threshold={:.3f}):".format(best_t))
    print(classification_report(y_all, y_pred_opt, target_names=["No Churn", "Churn"]))

    # ── Save outputs ─────────────────────────────────────────────────
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    pd.DataFrame({"id": train_ids, cfg.TARGET: ridge_oof}).to_csv(
        output_dir / "oof_ridge.csv", index=False,
    )
    for m in active_models:
        pd.DataFrame({"id": train_ids, cfg.TARGET: oof[m]}).to_csv(
            output_dir / f"oof_{m}.csv", index=False,
        )
    pd.DataFrame({"id": train_ids, cfg.TARGET: stacked_oof}).to_csv(
        output_dir / "oof_stacked.csv", index=False,
    )
    pd.DataFrame({"id": train_ids, cfg.TARGET: final_oof}).to_csv(
        output_dir / "oof_blend.csv", index=False,
    )

    # Submission
    sub = pd.DataFrame({"id": test_ids, cfg.TARGET: final_test})
    sub.to_csv(output_dir / "submission_blend.csv", index=False)
    print(f"\n  Submission saved to {output_dir / 'submission_blend.csv'}")
    print(f"  Prediction range : [{final_test.min():.5f}, {final_test.max():.5f}]")
    print(f"  Mean churn prob  : {final_test.mean():.5f}")

    # ── MLflow logging (best-effort) ─────────────────────────────────
    try:
        import mlflow
        from config.mlflow_init import init_mlflow

        init_mlflow()
        with mlflow.start_run(run_name="ensemble_blend"):
            mlflow.set_tag("model_type", "ensemble_blend")
            mlflow.set_tag("blend_method", blend_method)
            mlflow.log_metric("blend_oof_auc", final_auc)
            mlflow.log_metric("stacking_oof_auc", stacking_auc)
            mlflow.log_metric("ridge_oof_auc", roc_auc_score(y_all, ridge_oof))
            mlflow.log_metric("churn1_f1", best_f1)
            mlflow.log_metric("churn1_recall", recall_1)
            mlflow.log_metric("churn1_precision", precision_1)
            mlflow.log_metric("optimal_threshold", best_t)
            for m in active_models:
                mlflow.log_metric(f"{m}_oof_auc", roc_auc_score(y_all, oof[m]))
            for name, w in opt_weights.items():
                mlflow.log_metric(f"blend_weight_{name}", w)
            mlflow.log_artifacts(str(output_dir), artifact_path="outputs")
            print("  MLflow run logged successfully")
    except Exception as exc:
        print(f"  MLflow logging skipped: {exc}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ensemble training pipeline")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to best_config.json (skip Optuna, go straight to final run)",
    )
    parser.add_argument(
        "--optimize-only", action="store_true",
        help="Only run Optuna optimisation; do not run final pipeline",
    )
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="Number of Optuna trials (default: from CFG)",
    )
    parser.add_argument(
        "--n-folds", type=int, default=None,
        help="Override number of outer folds for final run",
    )
    args = parser.parse_args()

    cfg = CFG()
    if args.n_folds:
        cfg.N_FOLDS = args.n_folds

    print("=" * 70)
    print("ENSEMBLE PIPELINE — Customer Churn Prediction")
    print("=" * 70)

    train_raw, test_raw, orig_raw = load_data(cfg)

    # Phase 1: Optuna
    if args.config is None:
        print("\n" + "=" * 70)
        print("PHASE 1 — OPTUNA OPTIMISATION")
        print("=" * 70)
        optimizer = OptunaOptimizer(train_raw, test_raw, orig_raw, cfg)
        best_params = optimizer.run(n_trials=args.n_trials)
        optimizer.save_best("best_config.json")
        cfg.apply_overrides(best_params)
    else:
        print(f"\n  Loading config from {args.config}")
        best_params = json.loads(Path(args.config).read_text())
        cfg.apply_overrides(best_params)

    if args.optimize_only:
        print("\n  --optimize-only: stopping after Phase 1")
        cfg.to_json("best_config_full.json")
        return

    # Phase 2: Final run
    run_final_pipeline(train_raw, test_raw, orig_raw, cfg)
    cfg.to_json("outputs/final_config.json")


if __name__ == "__main__":
    main()
