import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
from dotenv import load_dotenv
from scipy import stats
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, accuracy_score,
                             classification_report, f1_score, get_scorer,
                             log_loss, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.mlflow_init import init_mlflow
from utils.generate_submission_file import generate_submission_file

load_dotenv(override=True)

METRIC = os.getenv("METRIC", "accuracy")
LOG_ALL_METRICS = os.getenv("LOG_ALL_METRICS", "False").lower() in ("true", "1", "yes")
DEVICE = os.getenv("DEVICE", "cpu").lower()

SKLEARN_SCORING = {
    "accuracy": "accuracy",
    "f1": "f1_weighted",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "roc_auc": "roc_auc",
    "r2": "r2",
    "neg_mean_squared_error": "neg_mean_squared_error",
    "neg_mean_absolute_error": "neg_mean_absolute_error",
    "neg_log_loss": "neg_log_loss",
}

METRIC_DIRECTION = {
    "accuracy": "maximize",
    "f1": "maximize",
    "precision": "maximize",
    "recall": "maximize",
    "roc_auc": "maximize",
    "r2": "maximize",
    "neg_mean_squared_error": "minimize",
    "neg_mean_absolute_error": "minimize",
    "neg_log_loss": "minimize",
}

CLASSIFICATION_METRICS = {
    "accuracy": lambda y, p, **_: accuracy_score(y, p),
    "f1": lambda y, p, **_: f1_score(y, p, average="weighted"),
    "precision": lambda y, p, **_: precision_score(
        y, p, average="weighted", zero_division=0
    ),
    "recall": lambda y, p, **_: recall_score(y, p, average="weighted", zero_division=0),
    "roc_auc": lambda y, p, proba=None, **_: (
        roc_auc_score(y, proba) if proba is not None else None
    ),
    "log_loss": lambda y, p, proba=None, **_: (
        log_loss(y, proba) if proba is not None else None
    ),
}

REGRESSION_METRICS = {
    "r2": lambda y, p, **_: r2_score(y, p),
    "mse": lambda y, p, **_: mean_squared_error(y, p),
    "mae": lambda y, p, **_: mean_absolute_error(y, p),
}


class BaseModel(ABC):
    model_name: str = "base_model"

    def __init__(self, data_dir: str = "data/processed", n_trials: int = 50):
        self.data_dir = Path(data_dir)
        self.model = None
        self.best_params: dict | None = None
        self.n_trials = n_trials
        self.task_type = os.getenv("TASK_TYPE", "classification")
        self.device = DEVICE
        self.metric = METRIC
        self.scoring = SKLEARN_SCORING.get(self.metric, self.metric)
        self.direction = METRIC_DIRECTION.get(self.metric, "maximize")
        init_mlflow()

    def load_data(self):
        X_train = np.load(self.data_dir / "X_train_preprocessed.npy", allow_pickle=True)
        X_test = np.load(self.data_dir / "X_test_preprocessed.npy", allow_pickle=True)
        y_train = np.load(self.data_dir / "y_train.npy", allow_pickle=True)
        y_test = np.load(self.data_dir / "y_test.npy", allow_pickle=True)
        return X_train, X_test, y_train, y_test

    @abstractmethod
    def build_model(self, params: dict | None = None):
        """Return an unfitted estimator, optionally configured with hyperparams."""
        ...

    @abstractmethod
    def suggest_params(self, trial: optuna.Trial) -> dict:
        """Define the Optuna search space and return a dict of suggested hyperparams."""
        ...

    def _objective(self, trial: optuna.Trial, X_train, y_train):
        params = self.suggest_params(trial)
        model = self.build_model(params)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=self.scoring)
        return scores.mean()

    def optimize(self, X_train, y_train):
        study = optuna.create_study(direction=self.direction)
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train),
            n_trials=self.n_trials,
        )
        self.best_params = study.best_params
        print(f"Best {self.metric} (CV): {study.best_value:.4f}")
        print(f"Best params: {self.best_params}")
        return study

    def train(self, X_train, y_train):
        self.model = self.build_model(self.best_params)
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.model.predict(X)

    def _predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:, 1]
            return proba
        return None

    def evaluate(self, y_true, y_pred, y_proba=None) -> dict:
        if LOG_ALL_METRICS:
            suite = (
                CLASSIFICATION_METRICS
                if self.task_type == "classification"
                else REGRESSION_METRICS
            )
            results = {}
            for name, fn in suite.items():
                value = fn(y_true, y_pred, proba=y_proba)
                if value is not None:
                    results[name] = value
            return results

        scorer = get_scorer(self.scoring)
        score = scorer._score_func(y_true, y_pred, **scorer._kwargs)
        return {self.metric: score}

    def _log_classification_plots(self, y_true, y_pred, y_proba, plots_dir: Path):
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
        ax.set_title(f"{self.model_name} — Confusion Matrix")
        fig.savefig(plots_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        unique_labels = np.unique(y_true)

        if y_proba is not None:
            pos_label = unique_labels[-1] if len(unique_labels) == 2 else None

            fig, ax = plt.subplots(figsize=(8, 6))
            RocCurveDisplay.from_predictions(
                y_true, y_proba, ax=ax, pos_label=pos_label
            )
            ax.set_title(f"{self.model_name} — ROC Curve")
            fig.savefig(plots_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 6))
            PrecisionRecallDisplay.from_predictions(
                y_true, y_proba, ax=ax, pos_label=pos_label
            )
            ax.set_title(f"{self.model_name} — Precision-Recall Curve")
            fig.savefig(
                plots_dir / "precision_recall_curve.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

            if len(unique_labels) == 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                CalibrationDisplay.from_predictions(
                    y_true, y_proba, n_bins=10, ax=ax, pos_label=pos_label,
                    name=self.model_name,
                )
                ax.set_title(f"{self.model_name} — Calibration Curve")
                fig.savefig(
                    plots_dir / "calibration_curve.png", dpi=150, bbox_inches="tight"
                )
                plt.close(fig)

        labels, actual_counts = np.unique(y_true, return_counts=True)
        pred_labels, pred_counts = np.unique(y_pred, return_counts=True)
        pred_count_map = dict(zip(pred_labels, pred_counts))
        pred_counts_aligned = np.array([pred_count_map.get(l, 0) for l in labels])

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(x - width / 2, actual_counts, width, label="Actual")
        ax.bar(x + width / 2, pred_counts_aligned, width, label="Predicted")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_title(f"{self.model_name} — Class Distribution")
        fig.savefig(
            plots_dir / "class_distribution.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    def _log_extended_classification_plots(
        self, y_true, y_pred, y_proba, plots_dir: Path
    ):
        report = classification_report(y_true, y_pred, output_dict=True)
        class_labels = [k for k in report if k not in (
            "accuracy", "macro avg", "weighted avg",
        )]
        precisions = [report[c]["precision"] for c in class_labels]
        recalls = [report[c]["recall"] for c in class_labels]
        f1s = [report[c]["f1-score"] for c in class_labels]

        x = np.arange(len(class_labels))
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(8, len(class_labels) * 1.2), 6))
        ax.bar(x - width, precisions, width, label="Precision")
        ax.bar(x, recalls, width, label="Recall")
        ax.bar(x + width, f1s, width, label="F1-Score")
        ax.set_xlabel("Class")
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title(f"{self.model_name} — Per-Class Metrics")
        fig.savefig(
            plots_dir / "per_class_metrics.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        if y_proba is not None:
            unique_labels = np.unique(y_true)

            if y_proba.ndim == 1 and len(unique_labels) == 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                for label in unique_labels:
                    mask = y_true == label
                    ax.hist(
                        y_proba[mask], bins=30, alpha=0.6,
                        label=f"Class {label}", edgecolor="k",
                    )
                ax.set_xlabel("Predicted Probability")
                ax.set_ylabel("Count")
                ax.legend()
                ax.set_title(
                    f"{self.model_name} — Predicted Probability Distribution"
                )
                fig.savefig(
                    plots_dir / "probability_histogram.png",
                    dpi=150, bbox_inches="tight",
                )
                plt.close(fig)

                pos = unique_labels[-1]
                y_binary = (y_true == pos).astype(int)
                sorted_indices = np.argsort(-y_proba)
                y_sorted = y_binary[sorted_indices]
                cumulative_gains = np.cumsum(y_sorted) / y_sorted.sum()
                fractions = np.arange(1, len(y_sorted) + 1) / len(y_sorted)

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fractions, cumulative_gains, label="Model", linewidth=2)
                ax.plot([0, 1], [0, 1], "r--", label="Random", linewidth=1)
                ax.set_xlabel("Fraction of Population")
                ax.set_ylabel("Fraction of Positives Captured")
                ax.legend()
                ax.set_title(f"{self.model_name} — Cumulative Gains")
                fig.savefig(
                    plots_dir / "cumulative_gains.png",
                    dpi=150, bbox_inches="tight",
                )
                plt.close(fig)

    def _log_regression_plots(self, y_true, y_pred, plots_dir: Path):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidths=0.3)
        limits = [
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max()),
        ]
        ax.plot(limits, limits, "r--", linewidth=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{self.model_name} — Predicted vs Actual")
        fig.savefig(plots_dir / "predicted_vs_actual.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_pred, residuals, alpha=0.5, edgecolors="k", linewidths=0.3)
        ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        ax.set_title(f"{self.model_name} — Residuals")
        fig.savefig(plots_dir / "residuals.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(residuals, bins=30, edgecolor="k", alpha=0.7, density=True)
        kde_x = np.linspace(residuals.min(), residuals.max(), 200)
        ax.plot(kde_x, stats.gaussian_kde(residuals)(kde_x), color="r", linewidth=2)
        ax.set_xlabel("Residual")
        ax.set_ylabel("Density")
        ax.set_title(f"{self.model_name} — Residual Distribution")
        fig.savefig(
            plots_dir / "residual_histogram.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f"{self.model_name} — Q-Q Plot of Residuals")
        fig.savefig(plots_dir / "qq_plot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _log_extended_regression_plots(self, y_true, y_pred, plots_dir: Path):
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            n_features = len(importances)
            top_n = min(20, n_features)
            indices = np.argsort(importances)[-top_n:]
            fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.35)))
            ax.barh(
                range(top_n), importances[indices],
                align="center", edgecolor="k", alpha=0.7,
            )
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([f"Feature {i}" for i in indices])
            ax.set_xlabel("Importance")
            ax.set_title(f"{self.model_name} — Feature Importance (Top {top_n})")
            fig.savefig(
                plots_dir / "feature_importance.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

        abs_error = np.abs(y_true - y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            y_true, abs_error, alpha=0.5, edgecolors="k", linewidths=0.3,
            c=abs_error, cmap="YlOrRd",
        )
        fig.colorbar(scatter, ax=ax, label="Absolute Error")
        ax.set_xlabel("Actual Value")
        ax.set_ylabel("Absolute Error")
        ax.set_title(f"{self.model_name} — Prediction Error by Magnitude")
        fig.savefig(
            plots_dir / "prediction_error_magnitude.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

        residuals = y_true - y_pred
        std_residuals = residuals / (np.std(residuals) + 1e-12)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            y_pred, np.sqrt(np.abs(std_residuals)),
            alpha=0.5, edgecolors="k", linewidths=0.3,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("√|Standardized Residuals|")
        ax.set_title(f"{self.model_name} — Scale-Location")
        fig.savefig(
            plots_dir / "scale_location.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    def _log_optuna_plots(self, study, plots_dir: Path):
        try:
            from optuna.visualization.matplotlib import (
                plot_optimization_history, plot_param_importances)

            ax = plot_optimization_history(study)
            ax.figure.savefig(
                plots_dir / "optimization_history.png", dpi=150, bbox_inches="tight"
            )
            plt.close(ax.figure)

            if len(study.trials) > 1:
                ax = plot_param_importances(study)
                ax.figure.savefig(
                    plots_dir / "param_importances.png", dpi=150, bbox_inches="tight"
                )
                plt.close(ax.figure)
        except Exception:
            pass

        if LOG_ALL_METRICS and len(study.trials) > 1:
            try:
                from optuna.visualization.matplotlib import (
                    plot_contour, plot_parallel_coordinate, plot_slice)

                ax = plot_parallel_coordinate(study)
                ax.figure.savefig(
                    plots_dir / "parallel_coordinate.png",
                    dpi=150, bbox_inches="tight",
                )
                plt.close(ax.figure)

                ax = plot_slice(study)
                if isinstance(ax, np.ndarray):
                    fig = ax.flat[0].figure
                else:
                    fig = ax.figure
                fig.savefig(
                    plots_dir / "slice_plot.png", dpi=150, bbox_inches="tight"
                )
                plt.close(fig)

                if len(study.best_params) >= 2:
                    ax = plot_contour(study)
                    if isinstance(ax, np.ndarray):
                        fig = ax.flat[0].figure
                    else:
                        fig = ax.figure
                    fig.savefig(
                        plots_dir / "contour_plot.png",
                        dpi=150, bbox_inches="tight",
                    )
                    plt.close(fig)
            except Exception:
                pass

    def _generate_and_log_submission(self):
        submission_path = self.data_dir / "X_submission_preprocessed.npy"
        if not submission_path.exists():
            print("Skipping submission: X_submission_preprocessed.npy not found")
            return

        raw_dir = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
        test_csv = raw_dir / "test.csv"
        if not test_csv.exists():
            print("Skipping submission: test.csv not found")
            return

        X_sub = np.load(submission_path, allow_pickle=True)
        predictions = self.predict(X_sub)

        target_column = os.getenv("TARGET_COLUMN", "target")
        output_path = generate_submission_file(
            predictions=predictions,
            model_name=self.model_name,
            test_csv_path=test_csv,
            target_column=target_column,
        )

        mlflow.log_artifact(str(output_path), artifact_path="submissions")
        print(f"Submission logged: {output_path.name}")

    def _log_plots(self, y_true, y_pred, y_proba, study):
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = Path(tmpdir)

            if self.task_type == "classification":
                self._log_classification_plots(y_true, y_pred, y_proba, plots_dir)
                if LOG_ALL_METRICS:
                    self._log_extended_classification_plots(
                        y_true, y_pred, y_proba, plots_dir
                    )
            else:
                self._log_regression_plots(y_true, y_pred, plots_dir)
                if LOG_ALL_METRICS:
                    self._log_extended_regression_plots(
                        y_true, y_pred, plots_dir
                    )

            self._log_optuna_plots(study, plots_dir)

            mlflow.log_artifacts(str(plots_dir), artifact_path="plots")

    def run(self):
        X_train, X_test, y_train, y_test = self.load_data()

        with mlflow.start_run(run_name=self.model_name):
            mlflow.set_tag("model_type", self.model_name)
            mlflow.set_tag("task_type", self.task_type)
            mlflow.set_tag("device", self.device)
            mlflow.log_param("metric", self.metric)
            mlflow.log_param("n_trials", self.n_trials)

            study = self.optimize(X_train, y_train)

            mlflow.log_params(self.best_params)
            mlflow.log_metric(f"best_cv_{self.metric}", study.best_value)

            self.train(X_train, y_train)
            y_pred = self.predict(X_test)
            y_proba = self._predict_proba(X_test)
            metrics = self.evaluate(y_test, y_pred, y_proba)

            for name, value in metrics.items():
                mlflow.log_metric(f"test_{name}", value)
                print(f"Test {name}: {value:.4f}")

            mlflow.sklearn.log_model(self.model, artifact_path="model")
            self._log_plots(y_test, y_pred, y_proba, study)
            self._generate_and_log_submission()

        return metrics
