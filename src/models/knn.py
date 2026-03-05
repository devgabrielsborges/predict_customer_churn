import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import optuna
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from models.model import BaseModel


class KNNModel(BaseModel):
    model_name = "knn"

    def build_model(self, params: dict | None = None):
        params = params or {}
        if self.task_type == "regression":
            return KNeighborsRegressor(n_jobs=-1, **params)
        return KNeighborsClassifier(n_jobs=-1, **params)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical(
                "metric", ["euclidean", "manhattan", "minkowski"]
            ),
            "p": trial.suggest_int("p", 1, 5),
        }


if __name__ == "__main__":
    KNNModel().run()
