from backend.app.ml.tracking import EXPERIMENT_NAME


def test_mlflow_experiment_name():
    assert EXPERIMENT_NAME == "ModelForge Churn Prediction"
