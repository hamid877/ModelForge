import pandas as pd
import pytest

from backend.app.ml.retraining import assess_retraining_need


@pytest.fixture
def reference_data():
    return pd.DataFrame(
        {
            "age": list(range(20, 40)),
            "tenure": list(range(1, 21)),
            "monthly_charges": [50.0 + i for i in range(20)],
            "contract_type": ["monthly", "annual"] * 10,
            "support_tickets": list(range(20)),
        }
    )


def test_stable_data_does_not_recommend_retraining(reference_data):
    result = assess_retraining_need(reference_data, reference_data.copy())

    assert result["status"] == "stable"
    assert result["retraining_recommended"] is False
    assert result["drift_report"]["drift_detected"] is False


def test_detected_drift_recommends_retraining(reference_data):
    current = reference_data.copy()
    current["age"] += 100

    result = assess_retraining_need(reference_data, current)

    assert result["status"] == "retraining_recommended"
    assert result["retraining_recommended"] is True
    assert result["drift_report"]["drift_detected"] is True


def test_insufficient_data_does_not_recommend_retraining(reference_data):
    small_data = reference_data.iloc[:5]

    result = assess_retraining_need(small_data, small_data.copy())

    assert result["status"] == "insufficient_data"
    assert result["retraining_recommended"] is False


def test_invalid_data_preserves_validation_error(reference_data):
    current = reference_data.drop(columns=["age"])

    with pytest.raises(ValueError, match="missing required features"):
        assess_retraining_need(reference_data, current)


def test_custom_drift_settings_are_passed_to_detector(
    reference_data,
):
    from unittest.mock import patch

    report = {
        "status": "stable",
        "drift_detected": False,
        "reference_samples": 20,
        "current_samples": 20,
        "features": {},
    }

    current = reference_data.copy()

    with patch(
        "backend.app.ml.retraining.detect_data_drift",
        return_value=report,
    ) as detector:
        result = assess_retraining_need(reference_data, current)

    detector.assert_called_once()
    args, kwargs = detector.call_args

    pd.testing.assert_frame_equal(args[0], reference_data)
    pd.testing.assert_frame_equal(args[1], current)
    assert kwargs == {
        "min_samples": 10,
        "p_value_threshold": 0.05,
    }
    assert result["reason"] == "no_feature_drift_detected"
    assert result["drift_report"] is report


def test_drift_recommendation_includes_reason(reference_data):
    current = reference_data.copy()
    current["age"] += 100

    result = assess_retraining_need(reference_data, current)

    assert result["retraining_recommended"] is True
    assert result["reason"] == "feature_drift_detected"
