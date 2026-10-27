from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode

from backend.app.ml.promotion import (
    BenchmarkDataset,
    check_training_benchmark_overlap,
    evaluate_model_on_benchmark,
    evaluate_promotion,
    load_benchmark_dataset,
    promote_model,
    validate_benchmark_data,
)


@pytest.fixture
def training_data():
    """Valid churn training dataset completely disjoint from benchmark_data."""
    return pd.DataFrame(
        {
            "age": [19, 21, 23, 27, 29, 31, 33, 37, 39, 41],
            "tenure": [1, 2, 4, 5, 7, 9, 10, 11, 13, 14],
            "monthly_charges": [
                60.0,
                62.0,
                64.0,
                66.0,
                68.0,
                72.0,
                74.0,
                76.0,
                78.0,
                82.0,
            ],
            "contract_type": [
                "monthly",
                "annual",
                "monthly",
                "annual",
                "monthly",
                "annual",
                "monthly",
                "annual",
                "monthly",
                "annual",
            ],
            "support_tickets": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def benchmark_data():
    """Valid churn benchmark dataset satisfying all validation constraints."""
    return pd.DataFrame(
        {
            "age": [22, 28, 35, 42, 50, 25, 33, 45, 52, 60],
            "tenure": [3, 8, 15, 24, 36, 6, 12, 20, 30, 48],
            "monthly_charges": [
                70.0,
                85.0,
                55.0,
                90.0,
                65.0,
                75.0,
                80.0,
                60.0,
                95.0,
                50.0,
            ],
            "contract_type": [
                "monthly",
                "monthly",
                "annual",
                "monthly",
                "annual",
                "monthly",
                "monthly",
                "annual",
                "monthly",
                "annual",
            ],
            "support_tickets": [3, 4, 1, 5, 0, 2, 3, 0, 4, 1],
            "churn": [1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        }
    )


class PerfectEstimator:
    """Estimator predicting ground-truth labels on benchmark dataset."""

    def __init__(self, target_series):
        self._preds = np.array(target_series)

    def predict(self, X):
        return self._preds[: len(X)]


class DegradedEstimator:
    """Estimator with degraded performance for champion/candidate comparison."""

    def __init__(self, target_series, error_count: int = 4):
        preds = np.array(target_series).copy()
        preds[:error_count] = 1 - preds[:error_count]
        self._preds = preds

    def predict(self, X):
        return self._preds[: len(X)]


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


def test_promotes_candidate_that_beats_champion(training_data, benchmark_data):
    """Promotes candidate that beats active champion on shared benchmark."""
    candidate_model = PerfectEstimator(benchmark_data["churn"])
    champion_model = DegradedEstimator(benchmark_data["churn"], error_count=4)
    champion = type("ModelVersion", (), {"version": "2", "run_id": "run-2"})()

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        client.get_model_version_by_alias.return_value = champion

        decision = promote_model(
            candidate_version=3,
            candidate_training_df=training_data,
            candidate_model=candidate_model,
            champion_model=champion_model,
            benchmark_df=benchmark_data,
            benchmark_version="v1",
            model_name="churn-classifier",
            alias="champion",
            minimum_f1=0.70,
            min_improvement=0.01,
        )

    assert decision.approved is True
    assert decision.reason == "candidate_meets_quality_gate"
    client.set_registered_model_alias.assert_called_once_with(
        "churn-classifier", "champion", "3"
    )


def test_does_not_promote_candidate_below_champion(training_data, benchmark_data):
    """Rejects candidate performing below champion on shared benchmark."""
    candidate_model = DegradedEstimator(benchmark_data["churn"], error_count=2)
    champion_model = PerfectEstimator(benchmark_data["churn"])
    champion = type("ModelVersion", (), {"version": "2", "run_id": "run-2"})()

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        client.get_model_version_by_alias.return_value = champion

        decision = promote_model(
            candidate_version=3,
            candidate_training_df=training_data,
            candidate_model=candidate_model,
            champion_model=champion_model,
            benchmark_df=benchmark_data,
            benchmark_version="v1",
            model_name="churn-classifier",
            alias="champion",
            minimum_f1=0.70,
            min_improvement=0.01,
        )

    assert decision.approved is False
    assert decision.reason == "candidate_f1_below_champion"
    client.set_registered_model_alias.assert_not_called()


def test_allows_initial_champion_when_alias_is_missing(training_data, benchmark_data):
    """Initial model meeting benchmark threshold promoted when no champion exists."""
    candidate_model = PerfectEstimator(benchmark_data["churn"])

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        client.get_model_version_by_alias.side_effect = MlflowException(
            "Alias not found",
            error_code=ErrorCode.RESOURCE_DOES_NOT_EXIST,
        )

        decision = promote_model(
            candidate_version=1,
            candidate_training_df=training_data,
            candidate_model=candidate_model,
            benchmark_df=benchmark_data,
            benchmark_version="v1",
            model_name="churn-classifier",
            alias="champion",
            minimum_f1=0.70,
        )

    assert decision.approved is True
    client.set_registered_model_alias.assert_called_once_with(
        "churn-classifier", "champion", "1"
    )


def test_does_not_promote_initial_candidate_below_minimum(
    training_data, benchmark_data
):
    """Initial model failing minimum benchmark threshold is not promoted."""
    candidate_model = DegradedEstimator(benchmark_data["churn"], error_count=8)

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        client.get_model_version_by_alias.side_effect = MlflowException(
            "Alias not found",
            error_code=ErrorCode.RESOURCE_DOES_NOT_EXIST,
        )

        decision = promote_model(
            candidate_version=1,
            candidate_training_df=training_data,
            candidate_model=candidate_model,
            benchmark_df=benchmark_data,
            benchmark_version="v1",
            model_name="churn-classifier",
            alias="champion",
            minimum_f1=0.70,
        )

    assert decision.approved is False
    assert decision.reason == "candidate_below_minimum_f1"
    client.set_registered_model_alias.assert_not_called()


def test_propagates_non_missing_alias_errors(training_data, benchmark_data):
    """Propagates non-404 MLflow exceptions immediately."""
    candidate_model = PerfectEstimator(benchmark_data["churn"])

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        client.get_model_version_by_alias.side_effect = MlflowException(
            "Permission denied",
            error_code=ErrorCode.PERMISSION_DENIED,
        )

        with pytest.raises(MlflowException, match="Permission denied"):
            promote_model(
                candidate_version=1,
                candidate_training_df=training_data,
                candidate_model=candidate_model,
                benchmark_df=benchmark_data,
                model_name="churn-classifier",
            )

        client.set_registered_model_alias.assert_not_called()


def test_rejects_champion_when_benchmark_evaluation_fails(
    training_data, benchmark_data
):
    """Fails closed if active champion cannot be evaluated on benchmark records."""
    candidate_model = PerfectEstimator(benchmark_data["churn"])
    champion = type("ModelVersion", (), {"version": "2", "run_id": "run-2"})()

    with (
        patch("backend.app.ml.promotion.MlflowClient") as mock_client,
        patch(
            "backend.app.ml.promotion.load_registered_model_by_alias",
            side_effect=RuntimeError("artifact corrupted"),
        ),
    ):
        client = mock_client.return_value
        client.get_model_version_by_alias.return_value = champion

        decision = promote_model(
            candidate_version=3,
            candidate_training_df=training_data,
            candidate_model=candidate_model,
            benchmark_df=benchmark_data,
            model_name="churn-classifier",
        )

    assert decision.approved is False
    assert decision.reason == "champion_benchmark_evaluation_failed"
    client.set_registered_model_alias.assert_not_called()


# ----------------------------------------------------------------------
# Milestone 7, Phase 3: Safe Model Promotion & Shared Benchmark Tests
# ----------------------------------------------------------------------


def test_promotion_gate_rejects_ties():
    """Gate rejects candidate that ties champion F1; preserves champion alias."""
    decision = evaluate_promotion(
        0.80,
        0.80,
        minimum_f1=0.70,
        min_improvement=0.01,
    )

    assert decision.approved is False
    assert decision.reason == "candidate_f1_tied_champion"


def test_promotion_gate_rejects_insufficient_improvement():
    """Gate rejects candidate exceeding champion by less than margin."""
    decision = evaluate_promotion(
        0.805,
        0.800,
        minimum_f1=0.70,
        min_improvement=0.01,
    )

    assert decision.approved is False
    assert decision.reason == "candidate_f1_insufficient_improvement"


def test_promotion_gate_approves_when_margin_is_exceeded():
    """Gate approves candidate exceeding champion by at least the margin."""
    decision = evaluate_promotion(
        0.820,
        0.800,
        minimum_f1=0.70,
        min_improvement=0.01,
    )

    assert decision.approved is True
    assert decision.reason == "candidate_meets_quality_gate"


def test_promotion_rejects_different_evaluation_datasets():
    """Never silently compare metrics from different benchmark datasets or versions."""
    decision = evaluate_promotion(
        0.85,
        0.80,
        minimum_f1=0.70,
        min_improvement=0.01,
        candidate_benchmark_version="v2",
        champion_benchmark_version="v1",
    )

    assert decision.approved is False
    assert decision.reason == "benchmark_version_mismatch"


def test_promote_model_rejects_missing_benchmark(training_data):
    """Reject promotion when benchmark dataset is missing; preserves alias."""
    with patch("backend.app.ml.promotion.load_benchmark_dataset") as mock_load:
        mock_load.side_effect = FileNotFoundError("Benchmark not found")
        decision = promote_model(
            candidate_version=2,
            candidate_training_df=training_data,
            benchmark_df=None,
        )

    assert decision.approved is False
    assert decision.reason == "benchmark_missing"


def test_promote_model_rejects_empty_benchmark(training_data):
    """Reject promotion when benchmark dataset is empty; preserves alias."""
    empty_df = pd.DataFrame()
    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        decision = promote_model(
            candidate_version=2,
            candidate_training_df=training_data,
            benchmark_df=empty_df,
        )

    assert decision.approved is False
    assert decision.reason == "benchmark_missing"
    client.set_registered_model_alias.assert_not_called()


def test_promote_model_rejects_invalid_benchmark_schema(training_data, benchmark_data):
    """Reject promotion when schema lacks required columns; preserves alias."""
    invalid_benchmark = benchmark_data.drop(columns=["monthly_charges"])

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        decision = promote_model(
            candidate_version=2,
            candidate_training_df=training_data,
            benchmark_df=invalid_benchmark,
        )

    assert decision.approved is False
    assert decision.reason == "benchmark_invalid"
    client.set_registered_model_alias.assert_not_called()


def test_promote_model_rejects_insufficient_sample_size(training_data, benchmark_data):
    """Reject promotion when sample count is below minimum; preserves alias."""
    tiny_benchmark = benchmark_data.iloc[:5]

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        decision = promote_model(
            candidate_version=2,
            candidate_training_df=training_data,
            benchmark_df=tiny_benchmark,
            min_samples=10,
        )

    assert decision.approved is False
    assert decision.reason == "benchmark_insufficient_samples"
    client.set_registered_model_alias.assert_not_called()


def test_promote_model_rejects_training_benchmark_overlap(benchmark_data):
    """Fail closed when candidate was trained on records present in benchmark."""
    # Training data shares records with benchmark dataset
    overlapping_training_data = benchmark_data.copy()
    candidate_model = PerfectEstimator(benchmark_data["churn"])

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        decision = promote_model(
            candidate_version=2,
            candidate_training_df=overlapping_training_data,
            candidate_model=candidate_model,
            benchmark_df=benchmark_data,
        )

    assert decision.approved is False
    assert decision.reason == "training_benchmark_overlap"
    client.set_registered_model_alias.assert_not_called()


def test_promote_model_fails_closed_when_separation_not_established(
    benchmark_data,
):
    """Fail closed when training data is not provided to establish separation."""
    candidate_model = PerfectEstimator(benchmark_data["churn"])

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        decision = promote_model(
            candidate_version=2,
            candidate_training_df=None,  # Separation cannot be established
            candidate_model=candidate_model,
            benchmark_df=benchmark_data,
        )

    assert decision.approved is False
    assert decision.reason == "separation_not_established"
    client.set_registered_model_alias.assert_not_called()


def test_promote_model_rejects_changed_benchmark_contents_under_same_version(
    training_data, benchmark_data
):
    """Reject promotion when benchmark digest does not match expected digest."""
    candidate_model = PerfectEstimator(benchmark_data["churn"])

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        decision = promote_model(
            candidate_version=2,
            candidate_training_df=training_data,
            candidate_model=candidate_model,
            benchmark_df=benchmark_data,
            benchmark_version="v1",
            expected_benchmark_digest="incorrect_digest_value_indicates_tampered_file",
        )

    assert decision.approved is False
    assert decision.reason == "benchmark_digest_mismatch"
    client.set_registered_model_alias.assert_not_called()


def test_legacy_callers_cannot_bypass_benchmark_gates():
    """No bypass: calling promote_model without training data fails closed."""
    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        decision = promote_model(
            candidate_version=3,
            candidate_f1=0.85,  # legacy bypass attempt
            candidate_training_df=None,
        )

    assert decision.approved is False
    assert decision.reason == "separation_not_established"
    client.set_registered_model_alias.assert_not_called()


def test_promote_model_on_shared_benchmark_preserves_alias_on_tie(
    training_data, benchmark_data
):
    """When both candidate and champion tie on benchmark, alias is preserved."""

    class ConstantEstimator:
        def predict(self, X):
            return np.ones(len(X), dtype=int)

    candidate_model = ConstantEstimator()
    champion_model = ConstantEstimator()
    champion = type("ModelVersion", (), {"version": "1", "run_id": "run-1"})()

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        client.get_model_version_by_alias.return_value = champion

        decision = promote_model(
            candidate_version=2,
            candidate_training_df=training_data,
            candidate_model=candidate_model,
            champion_model=champion_model,
            benchmark_df=benchmark_data,
            benchmark_version="v1",
            minimum_f1=0.50,
            min_improvement=0.01,
        )

    assert decision.approved is False
    assert decision.reason == "candidate_f1_tied_champion"
    client.set_registered_model_alias.assert_not_called()


def test_promote_model_on_shared_benchmark_preserves_alias_on_insufficient_improvement(
    training_data, benchmark_data
):
    """When candidate improvement is below margin, alias is preserved."""
    candidate_model = PerfectEstimator(benchmark_data["churn"])
    champion_model = DegradedEstimator(benchmark_data["churn"], error_count=1)
    champion = type("ModelVersion", (), {"version": "1", "run_id": "run-1"})()

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        client.get_model_version_by_alias.return_value = champion

        decision = promote_model(
            candidate_version=2,
            candidate_training_df=training_data,
            candidate_model=candidate_model,
            champion_model=champion_model,
            benchmark_df=benchmark_data,
            benchmark_version="v1",
            min_improvement=0.50,  # Requires unrealistically large margin
        )

    assert decision.approved is False
    assert decision.reason == "candidate_f1_insufficient_improvement"
    client.set_registered_model_alias.assert_not_called()


def test_benchmark_validation_threshold_documentation():
    """Verify docs state threshold does not establish statistical sufficiency."""
    assert "statistical sufficiency" in validate_benchmark_data.__doc__
    assert "statistical sufficiency" in load_benchmark_dataset.__doc__
    assert "statistical significance" in evaluate_promotion.__doc__
    assert "statistical significance" in promote_model.__doc__


def test_validate_benchmark_data_none_raises():
    """Validating None benchmark data raises ValueError."""
    with pytest.raises(ValueError, match="missing"):
        validate_benchmark_data(None)


def test_validate_benchmark_data_non_dataframe_raises():
    """Validating non-DataFrame benchmark data raises TypeError."""
    with pytest.raises(TypeError, match="must be a DataFrame"):
        validate_benchmark_data([1, 2, 3])  # type: ignore[arg-type]


def test_validate_benchmark_data_sample_count_raises(benchmark_data):
    """Validating benchmark with sample count below threshold raises ValueError."""
    tiny_df = benchmark_data.iloc[:5]
    with pytest.raises(ValueError, match="below required minimum"):
        validate_benchmark_data(tiny_df, min_samples=10)


def test_load_benchmark_dataset_missing_file_raises(tmp_path):
    """Loading benchmark dataset from non-existent file raises FileNotFoundError."""
    missing_path = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_benchmark_dataset(path=missing_path)


def test_load_benchmark_dataset_success(benchmark_data, tmp_path):
    """Loading benchmark dataset returns BenchmarkDataset dataclass with SHA-256."""
    csv_path = tmp_path / "benchmark.csv"
    benchmark_data.to_csv(csv_path, index=False)

    dataset = load_benchmark_dataset(path=csv_path, version="v1.2")
    assert isinstance(dataset, BenchmarkDataset)
    assert dataset.version == "v1.2"
    assert len(dataset.data) == len(benchmark_data)
    assert len(dataset.digest) == 64  # valid sha256 hex string


def test_load_benchmark_dataset_rejects_content_tampering_under_same_version(
    benchmark_data, tmp_path
):
    """Tampering with benchmark file contents under same version raises ValueError."""
    csv_path = tmp_path / "benchmark.csv"
    benchmark_data.to_csv(csv_path, index=False)

    initial_dataset = load_benchmark_dataset(path=csv_path, version="v1.0")

    # Modify file contents under the same version label
    tampered_data = benchmark_data.copy()
    tampered_data.loc[0, "monthly_charges"] += 50.0
    tampered_data.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="content digest mismatch"):
        load_benchmark_dataset(
            path=csv_path,
            version="v1.0",
            expected_sha256=initial_dataset.digest,
        )


def test_evaluate_model_on_benchmark_invokes_evaluation_logic(benchmark_data):
    """evaluate_model_on_benchmark evaluates model on benchmark records."""
    model = PerfectEstimator(benchmark_data["churn"])
    metrics = evaluate_model_on_benchmark(model, benchmark_data)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0


def test_check_training_benchmark_overlap_detection(training_data, benchmark_data):
    """Verify check_training_benchmark_overlap detects disjoint and overlapping data."""
    # Disjoint datasets
    assert not check_training_benchmark_overlap(training_data, benchmark_data)

    # Overlapping datasets
    overlapping_df = pd.concat([training_data, benchmark_data.iloc[:1]])
    assert check_training_benchmark_overlap(overlapping_df, benchmark_data)


def test_load_benchmark_dataset_default_config_rejects_content_tampering_under_v1(
    benchmark_data, tmp_path, monkeypatch
):
    """Normal default configuration rejects tampered benchmark file under v1."""
    csv_path = tmp_path / "tampered_benchmark.csv"
    tampered_data = benchmark_data.copy()
    tampered_data.loc[0, "monthly_charges"] += 99.0
    tampered_data.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        "backend.app.ml.promotion.settings.benchmark_data_path",
        str(csv_path),
    )

    with pytest.raises(ValueError, match="content digest mismatch"):
        # Called without supplying expected_sha256; relies on default configuration
        load_benchmark_dataset()


def test_promote_model_default_config_rejects_tampered_benchmark_file(
    training_data, benchmark_data, tmp_path, monkeypatch
):
    """promote_model fails closed and preserves alias on tampered benchmark."""
    csv_path = tmp_path / "tampered_benchmark.csv"
    tampered_data = benchmark_data.copy()
    tampered_data.loc[0, "monthly_charges"] += 99.0
    tampered_data.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        "backend.app.ml.promotion.settings.benchmark_data_path",
        str(csv_path),
    )

    candidate_model = PerfectEstimator(tampered_data["churn"])

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        decision = promote_model(
            candidate_version=2,
            candidate_training_df=training_data,
            candidate_model=candidate_model,
        )

    assert decision.approved is False
    assert decision.reason == "benchmark_digest_mismatch"
    client.set_registered_model_alias.assert_not_called()
