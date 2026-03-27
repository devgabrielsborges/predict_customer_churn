from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)


CATS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]

TOP_CATS_FOR_NGRAM = [
    "Contract", "InternetService", "PaymentMethod",
    "OnlineSecurity", "TechSupport", "PaperlessBilling",
]

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


@dataclass
class CFG:
    # --- CV ---
    N_FOLDS: int = 20
    OPTUNA_FOLDS: int = 5
    INNER_FOLDS: int = 5
    RANDOM_SEED: int = 42

    # --- Optuna ---
    OPTUNA_N_TRIALS: int = 100

    # --- Paths ---
    DATA_RAW_DIR: str = field(default_factory=lambda: os.getenv("DATA_RAW_DIR", "data/raw"))
    TARGET: str = "Churn"

    # --- Feature group toggles (Optuna-tunable) ---
    use_distribution_features: bool = True
    use_quantile_features: bool = True
    use_digit_features: bool = True
    use_ngram_features: bool = True
    use_orig_proba: bool = True
    ngram_top_k: int = 6

    # --- Pseudo labels ---
    PSEUDO_LABEL_THRESHOLD: float = 0.995
    use_pseudo_labels: bool = True

    # --- Model inclusion (Optuna-tunable) ---
    include_xgboost: bool = True
    include_lightgbm: bool = True
    include_catboost: bool = True
    include_xgb_pseudo: bool = True

    # --- Meta-learner ---
    meta_learner: str = "ridge"  # "ridge" | "logistic_regression" | "xgboost"

    # --- Blend weights (Optuna-tunable, normalized at blend time) ---
    blend_weight_xgb: float = 0.35
    blend_weight_lgbm: float = 0.25
    blend_weight_catboost: float = 0.20
    blend_weight_xgb_pseudo: float = 0.10
    blend_weight_stacking: float = 0.10

    # --- Device ---
    DEVICE: str = field(default_factory=lambda: os.getenv("DEVICE", "cpu").lower())

    # --- Class balance ---
    SCALE_POS_WEIGHT: float | None = None

    # --- Ridge ---
    RIDGE_ALPHA: float = 10.0

    # --- XGBoost ---
    XGB_PARAMS: dict = field(default_factory=lambda: {
        "n_estimators": 50_000,
        "learning_rate": 0.0063,
        "max_depth": 5,
        "subsample": 0.81,
        "colsample_bytree": 0.32,
        "min_child_weight": 6,
        "reg_alpha": 3.5017,
        "reg_lambda": 1.2925,
        "gamma": 0.790,
        "early_stopping_rounds": 500,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "enable_categorical": True,
        "verbosity": 0,
    })

    # --- LightGBM ---
    LGB_PARAMS: dict = field(default_factory=lambda: {
        "n_estimators": 50_000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_samples": 20,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
    })

    # --- CatBoost ---
    CB_PARAMS: dict = field(default_factory=lambda: {
        "iterations": 50_000,
        "learning_rate": 0.01,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 1.0,
        "random_strength": 1.0,
        "border_count": 128,
        "verbose": 0,
        "eval_metric": "AUC",
        "od_type": "Iter",
        "od_wait": 500,
    })

    # ------------------------------------------------------------------
    def __post_init__(self):
        self._apply_device()

    def _apply_device(self):
        if self.DEVICE == "cuda":
            self.XGB_PARAMS["device"] = "cuda"
            self.LGB_PARAMS["device"] = "gpu"
            self.CB_PARAMS["task_type"] = "GPU"
        else:
            self.XGB_PARAMS.pop("device", None)
            self.LGB_PARAMS.pop("device", None)
            self.CB_PARAMS.pop("task_type", None)

        self.XGB_PARAMS["random_state"] = self.RANDOM_SEED
        self.LGB_PARAMS["random_state"] = self.RANDOM_SEED
        self.CB_PARAMS["random_seed"] = self.RANDOM_SEED

    # ------------------------------------------------------------------
    @property
    def train_path(self) -> Path:
        return Path(self.DATA_RAW_DIR) / "train.csv"

    @property
    def test_path(self) -> Path:
        return Path(self.DATA_RAW_DIR) / "test.csv"

    @property
    def original_path(self) -> Path:
        return Path(self.DATA_RAW_DIR) / "original.csv"

    # ------------------------------------------------------------------
    def compute_scale_pos_weight(self, y):
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        self.SCALE_POS_WEIGHT = n_neg / max(n_pos, 1)

    # ------------------------------------------------------------------
    def apply_overrides(self, params: dict):
        """Merge Optuna trial suggestions into this config."""
        simple_fields = {
            "use_distribution_features", "use_quantile_features",
            "use_digit_features", "use_ngram_features", "use_orig_proba",
            "ngram_top_k", "PSEUDO_LABEL_THRESHOLD", "use_pseudo_labels",
            "include_xgboost", "include_lightgbm", "include_catboost",
            "include_xgb_pseudo", "meta_learner",
            "blend_weight_xgb", "blend_weight_lgbm",
            "blend_weight_catboost", "blend_weight_xgb_pseudo",
            "blend_weight_stacking", "RIDGE_ALPHA",
        }
        xgb_map = {
            "xgb_learning_rate": "learning_rate",
            "xgb_max_depth": "max_depth",
            "xgb_subsample": "subsample",
            "xgb_colsample_bytree": "colsample_bytree",
            "xgb_min_child_weight": "min_child_weight",
            "xgb_reg_alpha": "reg_alpha",
            "xgb_reg_lambda": "reg_lambda",
            "xgb_gamma": "gamma",
        }
        lgb_map = {
            "lgb_learning_rate": "learning_rate",
            "lgb_max_depth": "max_depth",
            "lgb_num_leaves": "num_leaves",
            "lgb_subsample": "subsample",
            "lgb_colsample_bytree": "colsample_bytree",
            "lgb_min_child_samples": "min_child_samples",
            "lgb_reg_alpha": "reg_alpha",
            "lgb_reg_lambda": "reg_lambda",
        }
        cb_map = {
            "cb_learning_rate": "learning_rate",
            "cb_depth": "depth",
            "cb_l2_leaf_reg": "l2_leaf_reg",
            "cb_bagging_temperature": "bagging_temperature",
        }

        for key, value in params.items():
            if key in simple_fields:
                setattr(self, key, value)
            elif key in xgb_map:
                self.XGB_PARAMS[xgb_map[key]] = value
            elif key in lgb_map:
                self.LGB_PARAMS[lgb_map[key]] = value
            elif key in cb_map:
                self.CB_PARAMS[cb_map[key]] = value

        self._apply_device()

    # ------------------------------------------------------------------
    def to_json(self, path: str | Path):
        """Serialize the full config to JSON for reproducibility."""
        data = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def from_json(cls, path: str | Path) -> "CFG":
        raw = json.loads(Path(path).read_text())
        xgb = raw.pop("XGB_PARAMS", {})
        lgb = raw.pop("LGB_PARAMS", {})
        cb = raw.pop("CB_PARAMS", {})
        simple_keys = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in raw.items() if k in simple_keys}
        cfg = cls(**kwargs)
        cfg.XGB_PARAMS.update(xgb)
        cfg.LGB_PARAMS.update(lgb)
        cfg.CB_PARAMS.update(cb)
        cfg._apply_device()
        return cfg
