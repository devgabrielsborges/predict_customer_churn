import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import optuna
from sklearn.linear_model import Ridge

from models.model import BaseModel


class RidgeRegressionModel(BaseModel):
    model_name = "ridge"

    def build_model(self, params: dict | None = None):
        params = params or {}
        return Ridge(**params)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
            "fit_intercept": trial.suggest_categorical(
                "fit_intercept", [True, False]
            ),
            "solver": trial.suggest_categorical(
                "solver", ["auto", "svd", "cholesky", "lsqr", "sag", "saga"]
            ),
        }


if __name__ == "__main__":
    RidgeRegressionModel().run()
