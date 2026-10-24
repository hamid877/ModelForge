import logging
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.api import api as api_module

app = api_module.app


@pytest.fixture(autouse=True)
def mock_mlflow_setup(monkeypatch):
    monkeypatch.setattr(api_module, "setup_mlflow", lambda: None)


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


@pytest.mark.parametrize(
    "model_output",
    [
        [2],
        [],
        [0, 1],
    ],
)
def test_predict_returns_503_for_invalid_model_output(model_output):
    """Reject predictions that violate the binary single-prediction contract."""
    with patch("backend.app.api.api.load_registered_model_by_alias") as mock_load:
        mock_load.return_value.predict.return_value = model_output

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


def test_metrics_endpoint_starts_with_zero_counters():
    """Expose zeroed prediction metrics before any requests."""
    with patch("backend.app.api.api.load_registered_model_by_alias") as mock_load:
        mock_load.return_value.predict.return_value = [0]

        with TestClient(app) as client:
            response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json() == {
        "prediction_requests_total": 0,
        "predictions_succeeded_total": 0,
        "predictions_failed_total": 0,
        "prediction_latency_total_seconds": 0.0,
        "prediction_latency_max_seconds": 0.0,
        "prediction_latency_average_seconds": 0.0,
        "prediction_latency_p50_seconds": 0.0,
        "prediction_latency_p95_seconds": 0.0,
        "prediction_latency_p99_seconds": 0.0,
    }


def test_metrics_endpoint_tracks_successful_prediction():
    """Increment request and success counters after a prediction."""
    with patch("backend.app.api.api.load_registered_model_by_alias") as mock_load:
        mock_load.return_value.predict.return_value = [1]

        with TestClient(app) as client:
            prediction_response = client.post(
                "/predict",
                json={
                    "age": 35,
                    "tenure": 24,
                    "monthly_charges": 79.99,
                    "contract_type": "monthly",
                    "support_tickets": 2,
                },
            )
            metrics_response = client.get("/metrics")

        metrics = metrics_response.json()

        assert prediction_response.status_code == 200
        assert metrics["prediction_requests_total"] == 1
        assert metrics["predictions_succeeded_total"] == 1
        assert metrics["predictions_failed_total"] == 0
        assert metrics["prediction_latency_total_seconds"] > 0
        assert metrics["prediction_latency_max_seconds"] > 0


def test_metrics_endpoint_tracks_failed_prediction():
    """Increment request and failure counters when prediction fails."""
    with patch("backend.app.api.api.load_registered_model_by_alias") as mock_load:
        mock_load.return_value.predict.side_effect = RuntimeError("Prediction failed")

        with TestClient(app) as client:
            prediction_response = client.post(
                "/predict",
                json={
                    "age": 35,
                    "tenure": 24,
                    "monthly_charges": 79.99,
                    "contract_type": "monthly",
                    "support_tickets": 2,
                },
            )
            metrics_response = client.get("/metrics")

        metrics = metrics_response.json()

        assert prediction_response.status_code == 503
        assert metrics["prediction_requests_total"] == 1
        assert metrics["predictions_succeeded_total"] == 0
        assert metrics["predictions_failed_total"] == 1
        assert metrics["prediction_latency_total_seconds"] > 0
        assert metrics["prediction_latency_max_seconds"] > 0


def test_metrics_endpoint_tracks_prediction_latency():
    """Expose total and maximum latency after a successful prediction."""
    with patch("backend.app.api.api.load_registered_model_by_alias") as mock_load:
        mock_load.return_value.predict.return_value = [1]

        with TestClient(app) as client:
            prediction_response = client.post(
                "/predict",
                json={
                    "age": 35,
                    "tenure": 24,
                    "monthly_charges": 79.99,
                    "contract_type": "monthly",
                    "support_tickets": 2,
                },
            )
            metrics_response = client.get("/metrics")

    metrics = metrics_response.json()

    assert prediction_response.status_code == 200
    assert metrics["prediction_latency_total_seconds"] > 0
    assert metrics["prediction_latency_max_seconds"] > 0
    assert (
        metrics["prediction_latency_total_seconds"]
        >= metrics["prediction_latency_max_seconds"]
    )


def test_metrics_endpoint_tracks_latency_when_prediction_fails():
    """Record latency even when the model raises an exception."""
    with patch("backend.app.api.api.load_registered_model_by_alias") as mock_load:
        mock_load.return_value.predict.side_effect = RuntimeError("Prediction failed")

        with TestClient(app) as client:
            prediction_response = client.post(
                "/predict",
                json={
                    "age": 35,
                    "tenure": 24,
                    "monthly_charges": 79.99,
                    "contract_type": "monthly",
                    "support_tickets": 2,
                },
            )
            metrics_response = client.get("/metrics")

    metrics = metrics_response.json()

    assert prediction_response.status_code == 503
    assert metrics["prediction_latency_total_seconds"] > 0
    assert metrics["prediction_latency_max_seconds"] > 0


def test_metrics_endpoint_tracks_latency_when_model_is_missing():
    """Record latency when the model is unavailable."""
    with patch("backend.app.api.api.load_registered_model_by_alias") as mock_load:
        mock_load.return_value.predict.return_value = [0]

        with TestClient(app) as client:
            app.state.churn_model = None
            prediction_response = client.post(
                "/predict",
                json={
                    "age": 35,
                    "tenure": 24,
                    "monthly_charges": 79.99,
                    "contract_type": "monthly",
                    "support_tickets": 2,
                },
            )
            metrics_response = client.get("/metrics")

    metrics = metrics_response.json()

    assert prediction_response.status_code == 503
    assert metrics["prediction_requests_total"] == 1
    assert metrics["predictions_failed_total"] == 1
    assert metrics["prediction_latency_total_seconds"] > 0
    assert metrics["prediction_latency_max_seconds"] > 0


def test_metrics_endpoint_reports_average_prediction_latency():
    """Calculate average latency from total latency and request count."""
    with patch("backend.app.api.api.load_registered_model_by_alias") as mock_load:
        mock_load.return_value.predict.return_value = [1]

        with TestClient(app) as client:
            for _ in range(2):
                response = client.post(
                    "/predict",
                    json={
                        "age": 35,
                        "tenure": 24,
                        "monthly_charges": 79.99,
                        "contract_type": "monthly",
                        "support_tickets": 2,
                    },
                )
                assert response.status_code == 200

            metrics_response = client.get("/metrics")

    metrics = metrics_response.json()

    assert metrics["prediction_requests_total"] == 2
    assert metrics["prediction_latency_total_seconds"] > 0
    assert metrics["prediction_latency_average_seconds"] > 0
    assert metrics["prediction_latency_average_seconds"] == pytest.approx(
        metrics["prediction_latency_total_seconds"] / 2
    )


def test_metrics_endpoint_reports_latency_percentiles():
    """Report p50, p95, and p99 from observed prediction latencies."""
    with patch("backend.app.api.api.load_registered_model_by_alias") as mock_load:
        mock_load.return_value.predict.return_value = [1]

        with TestClient(app) as client:
            for _ in range(3):
                response = client.post(
                    "/predict",
                    json={
                        "age": 35,
                        "tenure": 24,
                        "monthly_charges": 79.99,
                        "contract_type": "monthly",
                        "support_tickets": 2,
                    },
                )
                assert response.status_code == 200

            metrics_response = client.get("/metrics")

    metrics = metrics_response.json()

    assert metrics["prediction_latency_p50_seconds"] > 0
    assert metrics["prediction_latency_p95_seconds"] > 0
    assert metrics["prediction_latency_p99_seconds"] > 0
    assert (
        metrics["prediction_latency_p50_seconds"]
        <= metrics["prediction_latency_p95_seconds"]
        <= metrics["prediction_latency_p99_seconds"]
    )


def test_metrics_endpoint_reports_zero_latency_percentiles_without_requests():
    """Report zero percentiles before any predictions."""
    with patch("backend.app.api.api.load_registered_model_by_alias") as mock_load:
        mock_load.return_value.predict.return_value = [0]

        with TestClient(app) as client:
            response = client.get("/metrics")

    assert response.status_code == 200
    metrics = response.json()

    assert metrics["prediction_latency_p50_seconds"] == 0.0
    assert metrics["prediction_latency_p95_seconds"] == 0.0
    assert metrics["prediction_latency_p99_seconds"] == 0.0


def test_prometheus_metrics_endpoint_exposes_prediction_metrics():
    """Expose prediction metrics in Prometheus text format."""
    with patch("backend.app.api.api.load_registered_model_by_alias") as mock_load:
        mock_load.return_value.predict.return_value = [1]

        with TestClient(app) as client:
            response = client.get("/metrics/prometheus")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "# TYPE modelforge_prediction_requests_total counter" in response.text
    assert "# TYPE modelforge_prediction_latency_p95_seconds gauge" in response.text
    assert "modelforge_prediction_requests_total 0" in response.text
    assert "modelforge_predictions_succeeded_total 0" in response.text
    assert "modelforge_predictions_failed_total 0" in response.text


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_drift_endpoint_returns_report(mock_load_model, tmp_path):
    """Return a drift report when valid reference data is available."""
    mock_load_model.return_value.predict.return_value = [0]

    reference = pd.DataFrame(
        {
            "age": list(range(20, 40)),
            "tenure": list(range(1, 21)),
            "monthly_charges": [50.0 + i for i in range(20)],
            "contract_type": ["monthly", "annual"] * 10,
            "support_tickets": list(range(20)),
        }
    )
    reference_path = tmp_path / "reference.csv"
    reference.to_csv(reference_path, index=False)

    records = reference.iloc[:10].to_dict(orient="records")

    with (
        patch("backend.app.api.api.settings.drift_reference_path", str(reference_path)),
        patch("backend.app.api.api.settings.drift_min_samples", 10),
        TestClient(app) as client,
    ):
        response = client.post(
            "/monitoring/drift",
            json={"records": records},
        )

    assert response.status_code == 200
    report = response.json()
    assert report["status"] in {"stable", "drift_detected"}
    assert report["reference_samples"] == 20
    assert report["current_samples"] == 10
    assert set(report["features"]) == {
        "age",
        "tenure",
        "monthly_charges",
        "contract_type",
        "support_tickets",
    }


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_drift_endpoint_reports_insufficient_data(mock_load_model, tmp_path):
    """Report insufficient data instead of claiming stability."""
    mock_load_model.return_value.predict.return_value = [0]

    reference = pd.DataFrame(
        {
            "age": list(range(20, 40)),
            "tenure": list(range(1, 21)),
            "monthly_charges": [50.0 + i for i in range(20)],
            "contract_type": ["monthly", "annual"] * 10,
            "support_tickets": list(range(20)),
        }
    )
    reference_path = tmp_path / "reference.csv"
    reference.to_csv(reference_path, index=False)

    records = reference.iloc[:5].to_dict(orient="records")

    with (
        patch("backend.app.api.api.settings.drift_reference_path", str(reference_path)),
        patch("backend.app.api.api.settings.drift_min_samples", 10),
        TestClient(app) as client,
    ):
        response = client.post(
            "/monitoring/drift",
            json={"records": records},
        )

    assert response.status_code == 200
    report = response.json()
    assert report["status"] == "insufficient_data"
    assert report["drift_detected"] is None
    assert report["current_samples"] == 5
    assert report["features"] == {}


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_drift_endpoint_returns_503_when_reference_is_missing(
    mock_load_model,
    tmp_path,
):
    """Return a safe error when the reference dataset cannot be loaded."""
    mock_load_model.return_value.predict.return_value = [0]

    missing_path = tmp_path / "missing.csv"
    records = [
        {
            "age": 35,
            "tenure": 12,
            "monthly_charges": 75.0,
            "contract_type": "monthly",
            "support_tickets": 2,
        }
    ]

    with (
        patch(
            "backend.app.api.api.settings.drift_reference_path",
            str(missing_path),
        ),
        TestClient(app) as client,
    ):
        response = client.post(
            "/monitoring/drift",
            json={"records": records},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Drift monitoring reference data is unavailable."
    }


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_drift_endpoint_rejects_empty_batch(mock_load_model):
    """Reject a batch that contains no customer records."""
    mock_load_model.return_value.predict.return_value = [0]

    with TestClient(app) as client:
        response = client.post(
            "/monitoring/drift",
            json={"records": []},
        )

    assert response.status_code == 422


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_drift_endpoint_rejects_invalid_record(mock_load_model):
    """Reject records that violate the customer feature schema."""
    mock_load_model.return_value.predict.return_value = [0]

    with TestClient(app) as client:
        response = client.post(
            "/monitoring/drift",
            json={
                "records": [
                    {
                        "age": 17,
                        "tenure": 12,
                        "monthly_charges": 75.0,
                        "contract_type": "monthly",
                        "support_tickets": 2,
                    }
                ]
            },
        )

    assert response.status_code == 422


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_drift_endpoint_returns_503_when_reference_is_invalid(
    mock_load_model,
    tmp_path,
):
    """Reject and avoid caching a reference dataset with missing features."""
    mock_load_model.return_value.predict.return_value = [0]

    invalid_path = tmp_path / "invalid.csv"
    pd.DataFrame({"age": [25, 30, 35]}).to_csv(invalid_path, index=False)

    records = [
        {
            "age": 35,
            "tenure": 12,
            "monthly_charges": 75.0,
            "contract_type": "monthly",
            "support_tickets": 2,
        }
    ]

    with (
        patch(
            "backend.app.api.api.settings.drift_reference_path",
            str(invalid_path),
        ),
        TestClient(app) as client,
    ):
        response = client.post(
            "/monitoring/drift",
            json={"records": records},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Drift monitoring reference data is invalid."}
    assert getattr(app.state, "drift_reference_path", None) != str(invalid_path)
