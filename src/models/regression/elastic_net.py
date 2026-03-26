import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import optuna
from sklearn.linear_model import ElasticNet

from models.model import BaseModel


class ElasticNetModel(BaseModel):
    model_name = "elastic_net"

    def build_model(self, params: dict | None = None):
        params = params or {}
        return ElasticNet(**params)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "fit_intercept": trial.suggest_categorical(
                "fit_intercept", [True, False]
            ),
            "max_iter": trial.suggest_int("max_iter", 500, 5000, step=500),
            "tol": trial.suggest_float("tol", 1e-5, 1e-2, log=True),
        }


if __name__ == "__main__":
    ElasticNetModel().run()
