import os

import mlflow
from dotenv import load_dotenv

load_dotenv(override=True)

for token_var in ("AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN"):
    os.environ.pop(token_var, None)


def init_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("EXPERIMENT_NAME", "default")
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is not None and experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)

    mlflow.set_experiment(experiment_name)
    mlflow.enable_system_metrics_logging()
