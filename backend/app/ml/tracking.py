import mlflow

EXPERIMENT_NAME = "ModelForge Churn Prediction"


def setup_mlflow() -> None:
    """Configure the MLflow experiment."""
    mlflow.set_experiment(EXPERIMENT_NAME)
