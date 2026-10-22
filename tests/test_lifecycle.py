from unittest.mock import patch

import pandas as pd
import pytest

from backend.app.ml.lifecycle import run_drift_aware_lifecycle
from backend.app.ml.promotion import PromotionDecision


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "age": list(range(20, 40)),
            "tenure": list(range(1, 21)),
            "monthly_charges": [50.0 + i for i in range(20)],
            "contract_type": ["monthly", "annual"] * 10,
            "support_tickets": list(range(20)),
            "churn": [0, 1] * 10,
        }
    )


def test_stable_data_skips_training(sample_data, tmp_path):
    drift_result = {
        "status": "stable",
        "retraining_recommended": False,
        "reason": "no_feature_drift_detected",
        "drift_report": {"drift_detected": False},
    }

    with (
        patch(
            "backend.app.ml.lifecycle.assess_retraining_need",
            return_value=drift_result,
        ),
        patch("backend.app.ml.lifecycle.run_training_and_promote") as train,
    ):
        result = run_drift_aware_lifecycle(
            sample_data,
            sample_data.copy(),
            tmp_path / "candidate.joblib",
        )

    assert result["status"] == "stable"
    assert result["training_attempted"] is False
    train.assert_not_called()


def test_insufficient_data_skips_training(sample_data, tmp_path):
    drift_result = {
        "status": "insufficient_data",
        "retraining_recommended": False,
        "reason": "insufficient_data",
        "drift_report": {"drift_detected": None},
    }

    with (
        patch(
            "backend.app.ml.lifecycle.assess_retraining_need",
            return_value=drift_result,
        ),
        patch("backend.app.ml.lifecycle.run_training_and_promote") as train,
    ):
        result = run_drift_aware_lifecycle(
            sample_data,
            sample_data.copy(),
            tmp_path / "candidate.joblib",
        )

    assert result["status"] == "insufficient_data"
    assert result["training_attempted"] is False
    train.assert_not_called()


def test_detected_drift_trains_and_reports_approved_candidate(sample_data, tmp_path):
    drift_result = {
        "status": "retraining_recommended",
        "retraining_recommended": True,
        "reason": "feature_drift_detected",
        "drift_report": {"drift_detected": True},
    }
    metrics = {"accuracy": 0.9, "f1_score": 0.85}
    decision = PromotionDecision(
        approved=True,
        reason="candidate_meets_quality_gate",
        candidate_f1=0.85,
        champion_f1=0.8,
    )

    with (
        patch(
            "backend.app.ml.lifecycle.assess_retraining_need",
            return_value=drift_result,
        ),
        patch(
            "backend.app.ml.lifecycle.run_training_and_promote",
            return_value=(metrics, decision),
        ) as train,
    ):
        result = run_drift_aware_lifecycle(
            sample_data,
            sample_data.copy(),
            tmp_path / "candidate.joblib",
        )

    assert result["status"] == "promoted"
    assert result["training_attempted"] is True
    assert result["metrics"] == metrics
    assert result["promotion"]["approved"] is True
    train.assert_called_once()


def test_rejected_candidate_is_not_reported_as_promoted(sample_data, tmp_path):
    drift_result = {
        "status": "retraining_recommended",
        "retraining_recommended": True,
        "reason": "feature_drift_detected",
        "drift_report": {"drift_detected": True},
    }
    metrics = {"accuracy": 0.8, "f1_score": 0.75}
    decision = PromotionDecision(
        approved=False,
        reason="candidate_f1_below_champion",
        candidate_f1=0.75,
        champion_f1=0.8,
    )

    with (
        patch(
            "backend.app.ml.lifecycle.assess_retraining_need",
            return_value=drift_result,
        ),
        patch(
            "backend.app.ml.lifecycle.run_training_and_promote",
            return_value=(metrics, decision),
        ),
    ):
        result = run_drift_aware_lifecycle(
            sample_data,
            sample_data.copy(),
            tmp_path / "candidate.joblib",
        )

    assert result["status"] == "promotion_rejected"
    assert result["training_attempted"] is True
    assert result["promotion"]["approved"] is False


def test_training_error_propagates(sample_data, tmp_path):
    drift_result = {
        "status": "retraining_recommended",
        "retraining_recommended": True,
        "reason": "feature_drift_detected",
        "drift_report": {"drift_detected": True},
    }

    with (
        patch(
            "backend.app.ml.lifecycle.assess_retraining_need",
            return_value=drift_result,
        ),
        patch(
            "backend.app.ml.lifecycle.run_training_and_promote",
            side_effect=RuntimeError("training failed"),
        ),
        pytest.raises(RuntimeError, match="training failed"),
    ):
        run_drift_aware_lifecycle(
            sample_data,
            sample_data.copy(),
            tmp_path / "candidate.joblib",
        )


def test_stable_lifecycle_logs_assessment(sample_data, tmp_path, caplog):
    import logging

    drift_result = {
        "status": "stable",
        "retraining_recommended": False,
        "reason": "no_feature_drift_detected",
        "drift_report": {"drift_detected": False},
    }

    with (
        patch(
            "backend.app.ml.lifecycle.assess_retraining_need",
            return_value=drift_result,
        ),
        caplog.at_level(logging.INFO, logger="backend.app.ml.lifecycle"),
    ):
        result = run_drift_aware_lifecycle(
            sample_data,
            sample_data.copy(),
            tmp_path / "candidate.joblib",
        )

    assert result["status"] == "stable"
    assert "lifecycle_assessment_completed" in caplog.text


def test_promoted_lifecycle_logs_promotion(sample_data, tmp_path, caplog):
    import logging

    drift_result = {
        "status": "retraining_recommended",
        "retraining_recommended": True,
        "reason": "feature_drift_detected",
        "drift_report": {"drift_detected": True},
    }
    decision = PromotionDecision(
        approved=True,
        reason="candidate_meets_quality_gate",
        candidate_f1=0.85,
        champion_f1=0.8,
    )

    with (
        patch(
            "backend.app.ml.lifecycle.assess_retraining_need",
            return_value=drift_result,
        ),
        patch(
            "backend.app.ml.lifecycle.run_training_and_promote",
            return_value=({"f1_score": 0.85}, decision),
        ),
        caplog.at_level(logging.INFO, logger="backend.app.ml.lifecycle"),
    ):
        result = run_drift_aware_lifecycle(
            sample_data,
            sample_data.copy(),
            tmp_path / "candidate.joblib",
        )

    assert result["status"] == "promoted"
    assert "lifecycle_promotion_completed" in caplog.text


def test_training_failure_is_logged_and_propagated(sample_data, tmp_path, caplog):
    import logging

    drift_result = {
        "status": "retraining_recommended",
        "retraining_recommended": True,
        "reason": "feature_drift_detected",
        "drift_report": {"drift_detected": True},
    }

    with (
        patch(
            "backend.app.ml.lifecycle.assess_retraining_need",
            return_value=drift_result,
        ),
        patch(
            "backend.app.ml.lifecycle.run_training_and_promote",
            side_effect=RuntimeError("training failed"),
        ),
        caplog.at_level(logging.ERROR, logger="backend.app.ml.lifecycle"),
        pytest.raises(RuntimeError, match="training failed"),
    ):
        run_drift_aware_lifecycle(
            sample_data,
            sample_data.copy(),
            tmp_path / "candidate.joblib",
        )

    assert "lifecycle_training_failed" in caplog.text
