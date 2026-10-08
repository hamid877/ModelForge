from unittest.mock import patch

import mlflow.sklearn

from backend.app.ml.registry import (
    load_registered_model,
    load_registered_model_by_alias,
    set_model_alias,
)


def test_load_registered_model():
    with patch.object(mlflow.sklearn, "load_model") as mock_load_model:
        load_registered_model("churn-classifier", 5)

        mock_load_model.assert_called_once_with("models:/churn-classifier/5")


def test_set_model_alias():
    with patch("backend.app.ml.registry.mlflow.MlflowClient") as mock_client:
        client = mock_client.return_value

        set_model_alias("churn-classifier", "champion", 5)

        client.set_registered_model_alias.assert_called_once_with(
            "churn-classifier",
            "champion",
            5,
        )


def test_load_registered_model_by_alias():
    with patch.object(mlflow.sklearn, "load_model") as mock_load_model:
        load_registered_model_by_alias(
            "churn-classifier",
            "champion",
        )

        mock_load_model.assert_called_once_with("models:/churn-classifier@champion")
