import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import optuna
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from models.model import BaseModel


class RandomForestModel(BaseModel):
    model_name = "random_forest"

    def build_model(self, params: dict | None = None):
        params = params or {}
        if self.task_type == "regression":
            return RandomForestRegressor(n_jobs=-1, **params)
        return RandomForestClassifier(n_jobs=-1, **params)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
        }


if __name__ == "__main__":
    RandomForestModel().run()
