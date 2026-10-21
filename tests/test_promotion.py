from unittest.mock import patch

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode

from backend.app.ml.promotion import evaluate_promotion, promote_model


def test_approves_candidate_that_beats_champion():
    decision = evaluate_promotion(0.85, 0.80, minimum_f1=0.70)

    assert decision.approved is True
    assert decision.reason == "candidate_meets_quality_gate"


def test_rejects_candidate_below_champion():
    decision = evaluate_promotion(0.75, 0.80, minimum_f1=0.70)

    assert decision.approved is False
    assert decision.reason == "candidate_f1_below_champion"


def test_rejects_candidate_below_minimum():
    decision = evaluate_promotion(0.60, None, minimum_f1=0.70)

    assert decision.approved is False
    assert decision.reason == "candidate_below_minimum_f1"


def test_allows_initial_champion_that_meets_minimum():
    decision = evaluate_promotion(0.75, None, minimum_f1=0.70)

    assert decision.approved is True
    assert decision.champion_f1 is None


def test_rejects_invalid_candidate_metric():
    with pytest.raises(ValueError, match="candidate_f1"):
        evaluate_promotion(1.2, 0.8)


def test_rejects_invalid_minimum_metric():
    with pytest.raises(ValueError, match="minimum_f1"):
        evaluate_promotion(0.8, None, minimum_f1=-0.1)


def test_rejects_invalid_champion_metric():
    with pytest.raises(ValueError, match="champion_f1"):
        evaluate_promotion(0.8, 1.5)


def test_promotes_candidate_that_beats_champion():
    champion = type("ModelVersion", (), {"version": "2", "run_id": "run-2"})()
    run = type(
        "Run", (), {"data": type("Data", (), {"metrics": {"f1_score": 0.80}})()}
    )()

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        client.get_model_version_by_alias.return_value = champion
        client.get_run.return_value = run

        decision = promote_model(
            3,
            0.85,
            model_name="churn-classifier",
            alias="champion",
            minimum_f1=0.70,
        )

    assert decision.approved is True
    client.set_registered_model_alias.assert_called_once_with(
        "churn-classifier", "champion", "3"
    )


def test_does_not_promote_candidate_below_champion():
    champion = type("ModelVersion", (), {"version": "2", "run_id": "run-2"})()
    run = type(
        "Run", (), {"data": type("Data", (), {"metrics": {"f1_score": 0.80}})()}
    )()

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        client.get_model_version_by_alias.return_value = champion
        client.get_run.return_value = run

        decision = promote_model(
            3,
            0.75,
            model_name="churn-classifier",
            alias="champion",
            minimum_f1=0.70,
        )

    assert decision.approved is False
    assert decision.reason == "candidate_f1_below_champion"
    client.set_registered_model_alias.assert_not_called()


def test_allows_initial_champion_when_alias_is_missing():
    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        client.get_model_version_by_alias.side_effect = MlflowException(
            "Alias not found",
            error_code=ErrorCode.RESOURCE_DOES_NOT_EXIST,
        )

        decision = promote_model(
            1,
            0.75,
            model_name="churn-classifier",
            alias="champion",
            minimum_f1=0.70,
        )

    assert decision.approved is True
    client.set_registered_model_alias.assert_called_once_with(
        "churn-classifier", "champion", "1"
    )


def test_does_not_promote_initial_candidate_below_minimum():
    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        client.get_model_version_by_alias.side_effect = MlflowException(
            "Alias not found",
            error_code=ErrorCode.RESOURCE_DOES_NOT_EXIST,
        )

        decision = promote_model(
            1,
            0.60,
            model_name="churn-classifier",
            alias="champion",
            minimum_f1=0.70,
        )

    assert decision.approved is False
    assert decision.reason == "candidate_below_minimum_f1"
    client.set_registered_model_alias.assert_not_called()


def test_propagates_non_missing_alias_errors():
    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        client.get_model_version_by_alias.side_effect = MlflowException(
            "Permission denied",
            error_code=ErrorCode.PERMISSION_DENIED,
        )

        with pytest.raises(MlflowException, match="Permission denied"):
            promote_model(1, 0.85, model_name="churn-classifier")

        client.set_registered_model_alias.assert_not_called()


def test_rejects_champion_without_f1_metric():
    champion = type("ModelVersion", (), {"version": "2", "run_id": "run-2"})()
    run = type("Run", (), {"data": type("Data", (), {"metrics": {}})()})()

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        client.get_model_version_by_alias.return_value = champion
        client.get_run.return_value = run

        with pytest.raises(ValueError, match="no recorded f1_score"):
            promote_model(3, 0.85, model_name="churn-classifier")

        client.set_registered_model_alias.assert_not_called()
