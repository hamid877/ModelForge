import mlflow.sklearn
from sklearn.pipeline import Pipeline


def load_registered_model(
    model_name: str,
    version: int,
) -> Pipeline:
    """Load a specific registered model version from MLflow."""
    model_uri = f"models:/{model_name}/{version}"
    return mlflow.sklearn.load_model(model_uri)
