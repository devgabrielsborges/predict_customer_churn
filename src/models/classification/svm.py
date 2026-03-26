import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import optuna
from sklearn.svm import SVC

from models.model import BaseModel


class SVCModel(BaseModel):
    model_name = "svc"

    def build_model(self, params: dict | None = None):
        params = params or {}
        if "max_iter" not in params:
            params["max_iter"] = 10000
        if "cache_size" not in params:
            params["cache_size"] = 1000
        return SVC(**params)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        params = {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "kernel": kernel,
            "max_iter": trial.suggest_int("max_iter", 1000, 10000),
            "tol": trial.suggest_float("tol", 1e-4, 1e-2, log=True),
        }
        if kernel in ("rbf", "poly"):
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)
        return params


if __name__ == "__main__":
    SVCModel().run()
