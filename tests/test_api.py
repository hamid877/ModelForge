import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.api.api import app


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_health_endpoint(mock_load_model):
    mock_load_model.return_value.predict.return_value = [0]

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    mock_load_model.assert_called_once_with(
        "churn-classifier",
        "champion",
    )


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_predict_endpoint(mock_load_model):
    mock_load_model.return_value.predict.return_value = [1]

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "age": 35,
                "tenure": 24,
                "monthly_charges": 79.99,
                "contract_type": "month-to-month",
                "support_tickets": 2,
            },
        )

    assert response.status_code == 200
    assert response.json() == {"churn_prediction": 1}
    mock_load_model.assert_called_once_with(
        "churn-classifier",
        "champion",
    )


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_predict_rejects_invalid_age(mock_load_model):
    mock_load_model.return_value.predict.return_value = [0]

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "age": 17,
                "tenure": 24,
                "monthly_charges": 79.99,
                "contract_type": "month-to-month",
                "support_tickets": 2,
            },
        )

    assert response.status_code == 422
    mock_load_model.assert_called_once_with(
        "churn-classifier",
        "champion",
    )


def test_ready_endpoint():
    """Return ready when the cached model is loaded."""
    with patch("backend.app.api.api.load_registered_model_by_alias") as mock_load:
        mock_load.return_value.predict.return_value = [0]

        with TestClient(app) as client:
            response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_endpoint_without_model():
    """Return not ready when no model is cached."""
    with patch("backend.app.api.api.load_registered_model_by_alias") as mock_load:
        mock_load.return_value.predict.return_value = [0]

        with TestClient(app) as client:
            app.state.churn_model = None
            response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_startup_fails_when_model_cannot_load():
    """Propagate model-loading failures during application startup."""
    with patch(
        "backend.app.api.api.load_registered_model_by_alias",
        side_effect=RuntimeError("Model loading failed"),
    ):
        with pytest.raises(RuntimeError, match="Model loading failed"):
            with TestClient(app):
                pass


def test_predict_returns_503_when_model_prediction_fails():
    """Return 503 when the cached model cannot make a prediction."""
    with patch("backend.app.api.api.load_registered_model_by_alias") as mock_load:
        mock_load.return_value.predict.side_effect = RuntimeError("Prediction failed")

        with TestClient(app) as client:
            response = client.post(
                "/predict",
                json={
                    "age": 35,
                    "tenure": 12,
                    "monthly_charges": 75.0,
                    "contract_type": "monthly",
                    "support_tickets": 2,
                },
            )

    assert response.status_code == 503
    assert response.json() == {"detail": "The churn prediction service is unavailable."}


def test_startup_logs_model_loading_success(caplog):
    """Log successful model loading during startup."""
    with patch("backend.app.api.api.load_registered_model_by_alias") as mock_load:
        mock_load.return_value.predict.return_value = [0]

        with caplog.at_level(
            logging.INFO,
            logger="backend.app.api.api",
        ):
            with TestClient(app):
                pass

    assert "model_loading_started" in caplog.text
    assert "model_loading_succeeded" in caplog.text


def test_startup_logs_model_loading_failure(caplog):
    """Log model-loading failures before propagating them."""
    with patch(
        "backend.app.api.api.load_registered_model_by_alias",
        side_effect=RuntimeError("Model loading failed"),
    ):
        with caplog.at_level(
            logging.ERROR,
            logger="backend.app.api.api",
        ):
            with pytest.raises(
                RuntimeError,
                match="Model loading failed",
            ):
                with TestClient(app):
                    pass

    assert "model_loading_failed" in caplog.text
    assert "RuntimeError: Model loading failed" in caplog.text


def test_prediction_failure_is_logged(caplog):
    """Log prediction failures without exposing details to clients."""
    with patch("backend.app.api.api.load_registered_model_by_alias") as mock_load:
        mock_load.return_value.predict.side_effect = RuntimeError(
            "Internal prediction failure"
        )

        with TestClient(app) as client:
            with caplog.at_level(
                logging.ERROR,
                logger="backend.app.api.api",
            ):
                response = client.post(
                    "/predict",
                    json={
                        "age": 35,
                        "tenure": 12,
                        "monthly_charges": 75.0,
                        "contract_type": "monthly",
                        "support_tickets": 2,
                    },
                )

    assert response.status_code == 503
    assert response.json() == {"detail": "The churn prediction service is unavailable."}
    assert "prediction_failed" in caplog.text
    assert "Internal prediction failure" in caplog.text


def test_predict_returns_503_when_model_is_missing(caplog):
    """Log a missing model and return a safe error response."""
    with patch("backend.app.api.api.load_registered_model_by_alias") as mock_load:
        mock_load.return_value.predict.return_value = [0]

        with TestClient(app) as client:
            app.state.churn_model = None

            with caplog.at_level(
                logging.ERROR,
                logger="backend.app.api.api",
            ):
                response = client.post(
                    "/predict",
                    json={
                        "age": 35,
                        "tenure": 12,
                        "monthly_charges": 75.0,
                        "contract_type": "monthly",
                        "support_tickets": 2,
                    },
                )

    assert response.status_code == 503
    assert response.json() == {"detail": "The churn prediction service is unavailable."}
    assert "prediction_failed reason=model_not_loaded" in caplog.text
    assert "NoneType" not in caplog.text
