import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import optuna
from catboost import CatBoostClassifier

from models.model import BaseModel


class CatBoostClassifierModel(BaseModel):
    model_name = "catboost_classifier"

    def build_model(self, params: dict | None = None):
        params = params or {}
        if self.device == "cuda":
            params.setdefault("task_type", "GPU")
        return CatBoostClassifier(verbose=0, **params)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "iterations": trial.suggest_int("iterations", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 0.0, 10.0
            ),
            "random_strength": trial.suggest_float(
                "random_strength", 1e-2, 10.0, log=True
            ),
            "border_count": trial.suggest_int("border_count", 32, 255),
        }


if __name__ == "__main__":
    CatBoostClassifierModel().run()
