import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline


def load_registered_model(
    model_name: str,
    version: int,
) -> Pipeline:
    """Load a specific registered model version from MLflow."""
    model_uri = f"models:/{model_name}/{version}"
    return mlflow.sklearn.load_model(model_uri)


def set_model_alias(
    model_name: str,
    alias: str,
    version: int,
) -> None:
    """Assign an alias to a registered model version."""
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        model_name,
        alias,
        version,
    )


def load_registered_model_by_alias(
    model_name: str,
    alias: str,
) -> Pipeline:
    """Load a registered model using an MLflow alias."""
    model_uri = f"models:/{model_name}@{alias}"
    return mlflow.sklearn.load_model(model_uri)
