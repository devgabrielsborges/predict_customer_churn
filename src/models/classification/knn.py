import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import optuna
from sklearn.neighbors import KNeighborsClassifier

from models.model import BaseModel


class KNNClassifierModel(BaseModel):
    model_name = "knn_classifier"

    def build_model(self, params: dict | None = None):
        params = params or {}
        return KNeighborsClassifier(n_jobs=-1, **params)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical(
                "distance_metric", ["euclidean", "manhattan", "minkowski"]
            ),
            "p": trial.suggest_int("p", 1, 5),
        }


if __name__ == "__main__":
    KNNClassifierModel().run()
