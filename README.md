# Base Kaggle Competition Template

A ready-to-use ML experiment pipeline for Kaggle competitions. Download data, preprocess, train multiple models with automated hyperparameter tuning (Optuna), and track everything in MLflow — all orchestrated through a simple Makefile.

## Stack

- **Python 3.13+** managed with [uv](https://github.com/astral-sh/uv)
- **MLflow** for experiment tracking and artifact storage
- **Optuna** for Bayesian hyperparameter optimization (50 trials, 5-fold CV)
- **Docker Compose** stack: PostgreSQL (backend store) + MinIO (S3-compatible artifact store) + MLflow server
- **scikit-learn**, **XGBoost**, **pandas**, **NumPy**, **matplotlib**

## Project Structure

```
├── data/
│   ├── raw/                # Downloaded CSV files
│   └── processed/          # Preprocessed NumPy arrays
├── notebooks/
│   └── eda.ipynb           # Exploratory data analysis
├── src/
│   ├── config/
│   │   └── mlflow_init.py  # MLflow experiment setup
│   ├── models/
│   │   ├── model.py        # BaseModel + training loop
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   ├── svm.py
│   │   ├── xgboost_.py
│   │   ├── gradient_boosting.py
│   │   └── knn.py
│   ├── preprocessing/
│   │   └── preprocess.py   # Feature engineering pipeline
│   └── utils/
│       └── download_dataset.py
├── docker-compose.yml
├── Dockerfile.mlflow
├── Makefile
├── pyproject.toml
└── .env.example
```

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env to match your competition
```

| Variable | Description | Example |
|---|---|---|
| `KAGGLE_COMPETITION_NAME` | Kaggle competition slug | `playground-series-s6e2` |
| `TARGET_COLUMN` | Name of the target column | `Heart Disease` |
| `METRIC` | Optimization metric | `accuracy` |
| `TASK_TYPE` | `classification` or `regression` | `classification` |
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5000` |

### 2. Install dependencies

```bash
uv sync
```

### 3. Start infrastructure

```bash
make up
```

This spins up PostgreSQL, MinIO, and the MLflow server.

| Service | URL | Credentials |
|---|---|---|
| MLflow UI | http://localhost:5000 | — |
| MinIO Console | http://localhost:9001 | `minioadmin` / `minioadmin` |

### 4. Download and preprocess data

```bash
make init
```

> **Note:** Requires Kaggle credentials configured at `~/.kaggle/kaggle.json`.

### 5. Train models

```bash
make train-random_forest    # single model
make train-all              # all models sequentially
```

Available models: `logistic_regression`, `random_forest`, `svm`, `xgboost_`, `gradient_boosting`, `knn`.

### 6. Tear down

```bash
make down     # stop services
make clean    # stop services and delete volumes
```

## Available Models

Each model inherits from `BaseModel` and defines its own Optuna search space. The training loop automatically handles cross-validation, metric logging, and artifact storage.

| Model | Classification | Regression |
|---|---|---|
| Logistic Regression | LogisticRegression | — |
| Random Forest | RandomForestClassifier | RandomForestRegressor |
| SVM | SVC | SVR |
| XGBoost | XGBClassifier | XGBRegressor |
| Gradient Boosting | GradientBoostingClassifier | GradientBoostingRegressor |
| KNN | KNeighborsClassifier | KNeighborsRegressor |

## Preprocessing

The pipeline in `src/preprocessing/preprocess.py` applies:

- **Numerical features:** KNN imputation + standard scaling
- **Categorical features:** most-frequent imputation + one-hot encoding

Outputs are saved as `.npy` arrays under `data/processed/`.

## MLflow Tracking

Every training run logs:

- Model type, task type, optimization metric, and number of trials
- Best hyperparameters from Optuna
- Cross-validation score and full test-set metrics
- Trained model artifact (sklearn format)
- Plots: confusion matrix, ROC curve, precision-recall curve (classification) or predicted-vs-actual, residuals (regression), plus Optuna optimization history and parameter importances

## Adding a New Model

1. Create `src/models/my_model.py`
2. Subclass `BaseModel` and implement `build_model(params)` and `suggest_params(trial)`
3. Add a `__main__` block that instantiates and calls `.run()`
4. Add the model name to the `MODELS` list in the `Makefile`

```python
from src.models.model import BaseModel

class MyModel(BaseModel):
    name = "my_model"

    def build_model(self, params):
        ...

    def suggest_params(self, trial):
        ...

if __name__ == "__main__":
    MyModel().run()
```

## Makefile Reference

```
make help           Show all available commands
make up             Start Postgres + MinIO + MLflow
make down           Stop all services
make restart        Restart all services
make logs           Tail service logs
make clean          Stop services and delete volumes
make init           Download dataset + run preprocessing
make train-<model>  Train a single model
make train-all      Train all models sequentially
```
