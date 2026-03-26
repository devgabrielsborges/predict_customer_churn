import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import optuna
from sklearn.ensemble import GradientBoostingRegressor

from models.model import BaseModel


class GradientBoostingRegressorModel(BaseModel):
    model_name = "gradient_boosting_regressor"

    def build_model(self, params: dict | None = None):
        params = params or {}
        return GradientBoostingRegressor(**params)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "loss": trial.suggest_categorical(
                "loss", ["squared_error", "absolute_error", "huber"]
            ),
        }


if __name__ == "__main__":
    GradientBoostingRegressorModel().run()
