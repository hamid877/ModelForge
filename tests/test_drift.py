import pandas as pd
import pytest

from backend.app.ml.drift import detect_data_drift


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


def test_stable_data_reports_no_drift(reference_data):
    report = detect_data_drift(
        reference_data,
        reference_data.copy(),
        min_samples=10,
    )

    assert report["status"] == "stable"
    assert report["drift_detected"] is False
    assert report["reference_samples"] == 20
    assert report["current_samples"] == 20
    assert set(report["features"]) == {
        "age",
        "tenure",
        "monthly_charges",
        "contract_type",
        "support_tickets",
    }


def test_numerical_shift_detects_drift(reference_data):
    current = reference_data.copy()
    current["age"] = current["age"] + 100

    report = detect_data_drift(
        reference_data,
        current,
        min_samples=10,
    )

    assert report["status"] == "drift_detected"
    assert report["drift_detected"] is True
    assert report["features"]["age"]["drift_detected"] is True
    assert report["features"]["age"]["test"] == "kolmogorov_smirnov"


def test_small_sample_reports_insufficient_data(reference_data):
    report = detect_data_drift(
        reference_data.iloc[:5],
        reference_data.iloc[:5].copy(),
        min_samples=10,
    )

    assert report["status"] == "insufficient_data"
    assert report["drift_detected"] is None
    assert report["features"] == {}


def test_missing_feature_raises_error(reference_data):
    current = reference_data.drop(columns=["age"])

    with pytest.raises(ValueError, match="missing required features"):
        detect_data_drift(reference_data, current)


def test_invalid_minimum_sample_size_raises_error(reference_data):
    with pytest.raises(ValueError, match="min_samples"):
        detect_data_drift(
            reference_data,
            reference_data,
            min_samples=1,
        )
