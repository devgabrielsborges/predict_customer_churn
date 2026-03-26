import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import optuna
from sklearn.ensemble import AdaBoostRegressor

from models.model import BaseModel


class AdaBoostRegressorModel(BaseModel):
    model_name = "adaboost_regressor"

    def build_model(self, params: dict | None = None):
        params = params or {}
        return AdaBoostRegressor(**params)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 2.0, log=True),
            "loss": trial.suggest_categorical(
                "loss", ["linear", "square", "exponential"]
            ),
        }


if __name__ == "__main__":
    AdaBoostRegressorModel().run()
