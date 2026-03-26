import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import optuna
from sklearn.linear_model import LogisticRegression

from models.model import BaseModel


class LogisticRegressionModel(BaseModel):
    model_name = "logistic_regression"

    def build_model(self, params: dict | None = None):
        params = params or {}
        return LogisticRegression(solver="saga", **params)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "max_iter": trial.suggest_int("max_iter", 100, 1000, step=100),
        }


if __name__ == "__main__":
    LogisticRegressionModel().run()
