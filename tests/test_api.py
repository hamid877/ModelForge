from unittest.mock import patch

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


def test_predict_rejects_invalid_age():
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
