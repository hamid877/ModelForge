from unittest.mock import patch

import mlflow.sklearn

from backend.app.ml.registry import load_registered_model


def test_load_registered_model():
    with patch.object(mlflow.sklearn, "load_model") as mock_load_model:
        load_registered_model("churn-classifier", 5)

        mock_load_model.assert_called_once_with("models:/churn-classifier/5")
