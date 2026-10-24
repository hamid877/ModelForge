import mlflow

from backend.app.config import settings

EXPERIMENT_NAME = "ModelForge Churn Prediction"


def setup_mlflow() -> None:
    """Configure MLflow tracking and the experiment."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
