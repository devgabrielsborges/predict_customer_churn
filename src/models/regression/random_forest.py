import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import optuna
from sklearn.ensemble import RandomForestRegressor

from models.model import BaseModel


class RandomForestRegressorModel(BaseModel):
    model_name = "random_forest_regressor"

    def build_model(self, params: dict | None = None):
        params = params or {}
        return RandomForestRegressor(n_jobs=-1, **params)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "criterion": trial.suggest_categorical(
                "criterion", ["squared_error", "absolute_error", "friedman_mse"]
            ),
        }


if __name__ == "__main__":
    RandomForestRegressorModel().run()
