import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import optuna
from xgboost import XGBRegressor

from models.model import BaseModel


class XGBoostRegressorModel(BaseModel):
    model_name = "xgboost_regressor"

    def build_model(self, params: dict | None = None):
        params = params or {}
        if self.device == "cuda":
            params.setdefault("device", "cuda")
        return XGBRegressor(verbosity=0, n_jobs=-1, **params)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        }


if __name__ == "__main__":
    XGBoostRegressorModel().run()
