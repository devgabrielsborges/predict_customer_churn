import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import optuna
from sklearn.tree import DecisionTreeRegressor

from models.model import BaseModel


class DecisionTreeRegressorModel(BaseModel):
    model_name = "decision_tree_regressor"

    def build_model(self, params: dict | None = None):
        params = params or {}
        return DecisionTreeRegressor(**params)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "criterion": trial.suggest_categorical(
                "criterion", ["squared_error", "absolute_error", "friedman_mse"]
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
        }


if __name__ == "__main__":
    DecisionTreeRegressorModel().run()
