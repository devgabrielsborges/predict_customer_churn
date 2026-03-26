import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import optuna
from sklearn.naive_bayes import GaussianNB

from models.model import BaseModel


class NaiveBayesModel(BaseModel):
    model_name = "naive_bayes"

    def build_model(self, params: dict | None = None):
        params = params or {}
        return GaussianNB(**params)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "var_smoothing": trial.suggest_float(
                "var_smoothing", 1e-12, 1e-2, log=True
            ),
        }


if __name__ == "__main__":
    NaiveBayesModel().run()
