import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import optuna
from sklearn.linear_model import LinearRegression

from models.model import BaseModel


class LinearRegressionModel(BaseModel):
    model_name = "linear_regression"

    def build_model(self, params: dict | None = None):
        params = params or {}
        return LinearRegression(n_jobs=-1, **params)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "fit_intercept": trial.suggest_categorical(
                "fit_intercept", [True, False]
            ),
        }


if __name__ == "__main__":
    LinearRegressionModel().run()
