import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import optuna
from sklearn.linear_model import Lasso

from models.model import BaseModel


class LassoRegressionModel(BaseModel):
    model_name = "lasso"

    def build_model(self, params: dict | None = None):
        params = params or {}
        return Lasso(**params)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
            "fit_intercept": trial.suggest_categorical(
                "fit_intercept", [True, False]
            ),
            "max_iter": trial.suggest_int("max_iter", 500, 5000, step=500),
            "tol": trial.suggest_float("tol", 1e-5, 1e-2, log=True),
        }


if __name__ == "__main__":
    LassoRegressionModel().run()
