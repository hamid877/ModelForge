import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.app.config import settings
from backend.app.ml.audit import (
    get_promotion_audit_records,
    record_promotion_audit,
)
from backend.app.ml.promotion import (
    compute_dataframe_sha256,
    compute_file_sha256,
    promote_model,
)
from backend.app.ml.provenance import (
    DatasetProvenance,
    build_split_details,
    extract_dataset_provenance,
)
from backend.app.ml.tracking import (
    log_dataset_provenance_to_mlflow,
    log_promotion_traceability_to_mlflow,
)
from backend.app.ml.workflow import run_training, run_training_and_promote


@pytest.fixture
def sample_training_df():
    """Sample churn training dataframe disjoint from benchmark data."""
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
def sample_benchmark_df():
    """Sample benchmark dataframe."""
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


class DeterministicEstimator:
    """Estimator that returns predetermined predictions."""

    def __init__(self, target_series, error_indices=None):
        preds = np.array(target_series).copy()
        if error_indices:
            for idx in error_indices:
                preds[idx] = 1 - preds[idx]
        self._preds = preds

    def predict(self, X):
        return self._preds[: len(X)]


def test_extract_dataset_provenance_with_split(sample_training_df):
    """Provenance logs SHA-256, counts, schema, and split without row leakage."""
    train_part = sample_training_df.iloc[:8].copy()
    test_part = sample_training_df.iloc[8:].copy()

    split_details = build_split_details(
        train_df=train_part,
        test_df=test_part,
        test_size=0.2,
        random_state=42,
        stratified=True,
    )
    prov = extract_dataset_provenance(sample_training_df, split_details=split_details)

    assert isinstance(prov, DatasetProvenance)
    data_dict = prov.to_dict()

    expected_sha = compute_dataframe_sha256(sample_training_df)
    assert data_dict["dataset_sha256"] == expected_sha
    assert data_dict["row_count"] == 10
    assert data_dict["feature_count"] == 5
    assert "churn" not in data_dict["feature_names"]
    assert "monthly_charges" in data_dict["feature_names"]
    assert data_dict["target_column"] == "churn"
    assert data_dict["target_classes"] == [0, 1]
    assert data_dict["class_distribution"] == {"0": 5, "1": 5}

    split = data_dict["split"]
    assert split["split_performed"] is True
    assert split["train_rows"] == 8
    assert split["test_rows"] == 2
    assert split["test_size"] == 0.2
    assert split["random_state"] == 42
    assert split["stratified"] is True
    assert "train_sha256" in split
    assert "test_sha256" in split

    # Ensure no raw row records leaked into the provenance dictionary
    dumped = str(data_dict)
    assert "60.0" not in dumped
    assert "monthly_charges: 60" not in dumped


def test_extract_dataset_provenance_without_split_documents_gap(
    sample_training_df,
):
    """When split details are omitted, provenance explicitly documents the gap."""
    prov = extract_dataset_provenance(sample_training_df, split_details=None)
    data_dict = prov.to_dict()

    assert data_dict["split"]["split_performed"] is False
    assert "gap_documentation" in data_dict["split"]
    assert (
        "without a caller train/test split" in (data_dict["split"]["gap_documentation"])
    )


def test_extract_dataset_provenance_rejects_non_dataframe():
    """Extracting provenance from non-DataFrame raises TypeError."""
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        extract_dataset_provenance({"not": "a dataframe"})  # type: ignore


def test_record_and_retrieve_promotion_audit(tmp_path):
    """Both approved and rejected decisions are saved to SQLite and retrievable."""
    db_path = tmp_path / "test_promotion_audit.db"

    approved_record = {
        "model_name": "churn-classifier",
        "candidate_version": "5",
        "candidate_run_id": "run-abc-123",
        "champion_version": "4",
        "approved": True,
        "reason": "candidate_meets_quality_gate",
        "candidate_f1": 0.88,
        "champion_f1": 0.82,
        "benchmark_version": "v1",
        "benchmark_digest": "digest123",
        "margin": 0.01,
        "gate_outcomes": {"quality_gate": {"passed": True}},
        "candidate_metrics": {"f1_score": 0.88, "accuracy": 0.90},
        "champion_metrics": {"f1_score": 0.82, "accuracy": 0.85},
    }

    rejected_record = {
        "model_name": "churn-classifier",
        "candidate_version": "6",
        "candidate_run_id": "run-def-456",
        "champion_version": "5",
        "approved": False,
        "reason": "candidate_f1_below_champion",
        "candidate_f1": 0.80,
        "champion_f1": 0.88,
        "benchmark_version": "v1",
        "benchmark_digest": "digest123",
        "margin": 0.01,
        "gate_outcomes": {
            "quality_gate": {
                "passed": False,
                "reason": "candidate_f1_below_champion",
            }
        },
    }

    meta1 = record_promotion_audit(approved_record, database_path=db_path)
    meta2 = record_promotion_audit(rejected_record, database_path=db_path)

    assert meta1["audit_id"] != meta2["audit_id"]

    records = get_promotion_audit_records(database_path=db_path)
    assert len(records) == 2

    # Most recent record is first
    assert records[0]["audit_id"] == meta2["audit_id"]
    assert records[0]["approved"] is False
    assert records[0]["reason"] == "candidate_f1_below_champion"
    assert records[0]["candidate_f1"] == 0.80
    assert records[0]["champion_f1"] == 0.88

    assert records[1]["audit_id"] == meta1["audit_id"]
    assert records[1]["approved"] is True
    assert records[1]["reason"] == "candidate_meets_quality_gate"
    assert records[1]["payload"]["candidate_metrics"]["f1_score"] == 0.88


def test_get_promotion_audit_records_nonexistent_db(tmp_path):
    """Retrieving from non-existent audit DB returns empty list cleanly."""
    records = get_promotion_audit_records(tmp_path / "nonexistent.db")
    assert records == []


def test_promote_model_successful_promotion_traceability(
    sample_training_df, sample_benchmark_df, tmp_path
):
    """Successful promotion records traceability in decision and SQLite audit."""
    db_path = tmp_path / "audit.db"
    candidate_model = DeterministicEstimator(sample_benchmark_df["churn"])
    champion_model = DeterministicEstimator(
        sample_benchmark_df["churn"], error_indices=[0, 1]
    )

    champion_stub = type(
        "ModelVersionStub", (), {"version": "2", "run_id": "champ-run"}
    )()

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        client.get_model_version_by_alias.return_value = champion_stub

        decision = promote_model(
            candidate_version=3,
            candidate_training_df=sample_training_df,
            candidate_model=candidate_model,
            champion_model=champion_model,
            benchmark_df=sample_benchmark_df,
            benchmark_version="v1",
            model_name="churn-classifier",
            alias="champion",
            minimum_f1=0.70,
            min_improvement=0.01,
            candidate_run_id="candidate-run-99",
            audit_db_path=db_path,
        )

    assert decision.approved is True
    assert decision.reason == "candidate_meets_quality_gate"
    assert decision.candidate_version == 3
    assert decision.champion_version == "2"
    assert decision.candidate_run_id == "candidate-run-99"
    assert decision.audit_id is not None
    assert decision.timestamp is not None
    assert decision.benchmark_digest is not None
    assert decision.candidate_metrics is not None
    assert decision.champion_metrics is not None
    assert decision.gate_outcomes is not None
    assert decision.gate_outcomes["quality_gate"]["passed"] is True

    client.set_registered_model_alias.assert_called_once_with(
        "churn-classifier", "champion", "3"
    )

    records = get_promotion_audit_records(db_path)
    assert len(records) == 1
    assert records[0]["audit_id"] == decision.audit_id
    assert records[0]["approved"] is True
    assert records[0]["candidate_run_id"] == "candidate-run-99"
    assert records[0]["champion_version"] == "2"


def test_promote_model_rejected_decision_is_audited(
    sample_training_df, sample_benchmark_df, tmp_path
):
    """Rejected candidate is recorded in audit database and alias is NOT updated."""
    db_path = tmp_path / "audit.db"
    candidate_model = DeterministicEstimator(
        sample_benchmark_df["churn"], error_indices=[0]
    )
    champion_model = DeterministicEstimator(sample_benchmark_df["churn"])

    champion_stub = type(
        "ModelVersionStub", (), {"version": "2", "run_id": "champ-run"}
    )()

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        client.get_model_version_by_alias.return_value = champion_stub

        decision = promote_model(
            candidate_version=4,
            candidate_training_df=sample_training_df,
            candidate_model=candidate_model,
            champion_model=champion_model,
            benchmark_df=sample_benchmark_df,
            benchmark_version="v1",
            model_name="churn-classifier",
            alias="champion",
            minimum_f1=0.70,
            min_improvement=0.01,
            candidate_run_id="cand-run-4",
            audit_db_path=db_path,
        )

    assert decision.approved is False
    assert decision.reason == "candidate_f1_below_champion"
    assert decision.audit_id is not None
    client.set_registered_model_alias.assert_not_called()

    records = get_promotion_audit_records(db_path)
    assert len(records) == 1
    assert records[0]["approved"] is False
    assert records[0]["reason"] == "candidate_f1_below_champion"
    assert records[0]["candidate_version"] == "4"


def test_promote_model_aborts_promotion_on_audit_logging_failure(
    sample_training_df, sample_benchmark_df, tmp_path
):
    """SAFETY: If audit logging fails, promotion fails closed and alias is intact."""
    candidate_model = DeterministicEstimator(sample_benchmark_df["churn"])
    champion_model = DeterministicEstimator(
        sample_benchmark_df["churn"], error_indices=[0, 1]
    )
    champion_stub = type("ModelVersionStub", (), {"version": "2"})()

    with (
        patch("backend.app.ml.promotion.MlflowClient") as mock_client,
        patch(
            "backend.app.ml.promotion.record_promotion_audit",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ),
    ):
        client = mock_client.return_value
        client.get_model_version_by_alias.return_value = champion_stub

        decision = promote_model(
            candidate_version=5,
            candidate_training_df=sample_training_df,
            candidate_model=candidate_model,
            champion_model=champion_model,
            benchmark_df=sample_benchmark_df,
            benchmark_version="v1",
            model_name="churn-classifier",
            alias="champion",
            minimum_f1=0.70,
            min_improvement=0.01,
            audit_db_path=tmp_path / "audit.db",
        )

    assert decision.approved is False
    assert decision.reason == "audit_logging_failed"
    # CRITICAL: live champion alias was never touched!
    client.set_registered_model_alias.assert_not_called()


def test_promote_model_aborts_promotion_on_mlflow_logging_failure(
    sample_training_df, sample_benchmark_df, tmp_path
):
    """SAFETY: If active MLflow traceability logging fails, promotion is aborted."""
    candidate_model = DeterministicEstimator(sample_benchmark_df["churn"])
    champion_model = DeterministicEstimator(
        sample_benchmark_df["churn"], error_indices=[0, 1]
    )
    champion_stub = type("ModelVersionStub", (), {"version": "2"})()

    active_run_mock = MagicMock()

    with (
        patch("backend.app.ml.promotion.MlflowClient") as mock_client,
        patch(
            "backend.app.ml.promotion.mlflow.active_run",
            return_value=active_run_mock,
        ),
        patch(
            "backend.app.ml.promotion.log_promotion_traceability_to_mlflow",
            side_effect=RuntimeError("MLflow server connection lost"),
        ),
    ):
        client = mock_client.return_value
        client.get_model_version_by_alias.return_value = champion_stub

        decision = promote_model(
            candidate_version=7,
            candidate_training_df=sample_training_df,
            candidate_model=candidate_model,
            champion_model=champion_model,
            benchmark_df=sample_benchmark_df,
            benchmark_version="v1",
            model_name="churn-classifier",
            alias="champion",
            minimum_f1=0.70,
            min_improvement=0.01,
            audit_db_path=tmp_path / "audit.db",
        )

    assert decision.approved is False
    assert decision.reason == "audit_logging_failed"
    client.set_registered_model_alias.assert_not_called()

    # CRITICAL: SQLite audit record was updated to reflect aborted status,
    # not false approval
    records = get_promotion_audit_records(tmp_path / "audit.db")
    assert len(records) == 1
    assert records[0]["approved"] is False
    assert records[0]["reason"] == "audit_logging_failed"


def test_promote_model_aborts_and_updates_audit_on_alias_mutation_failure(
    sample_training_df, sample_benchmark_df, tmp_path
):
    """SAFETY: If alias mutation fails, candidate is rejected and audit
    reflects failure.
    """
    db_path = tmp_path / "audit.db"
    candidate_model = DeterministicEstimator(sample_benchmark_df["churn"])
    champion_model = DeterministicEstimator(
        sample_benchmark_df["churn"], error_indices=[0, 1]
    )
    champion_stub = type("ModelVersionStub", (), {"version": "2"})()

    active_run_mock = MagicMock()

    with (
        patch("backend.app.ml.promotion.MlflowClient") as mock_client,
        patch(
            "backend.app.ml.promotion.mlflow.active_run",
            return_value=active_run_mock,
        ),
        patch(
            "backend.app.ml.promotion.log_promotion_traceability_to_mlflow"
        ) as mock_log_mlflow,
    ):
        client = mock_client.return_value
        client.get_model_version_by_alias.return_value = champion_stub
        client.set_registered_model_alias.side_effect = RuntimeError(
            "MLflow registry unavailable"
        )

        decision = promote_model(
            candidate_version=8,
            candidate_training_df=sample_training_df,
            candidate_model=candidate_model,
            champion_model=champion_model,
            benchmark_df=sample_benchmark_df,
            benchmark_version="v1",
            model_name="churn-classifier",
            alias="champion",
            minimum_f1=0.70,
            min_improvement=0.01,
            audit_db_path=db_path,
        )

    assert decision.approved is False
    assert decision.reason == "alias_mutation_failed"
    client.set_registered_model_alias.assert_called_once_with(
        "churn-classifier", "champion", "8"
    )

    # CRITICAL: Audit record in SQLite must NOT falsely claim success!
    records = get_promotion_audit_records(db_path)
    assert len(records) == 1
    assert records[0]["approved"] is False
    assert records[0]["reason"] == "alias_mutation_failed"
    assert records[0]["candidate_version"] == "8"

    # MLflow active run should receive compensated failure update
    assert mock_log_mlflow.call_count == 2
    second_call_record = mock_log_mlflow.call_args_list[1][0][0]
    assert second_call_record["approved"] is False
    assert second_call_record["reason"] == "alias_mutation_failed"


def test_promote_model_reconciles_when_candidate_alias_found_after_exception(
    sample_training_df, sample_benchmark_df, tmp_path
):
    """Reconciliation: If set_alias raises but re-read shows candidate,
    promotion succeeds.
    """
    db_path = tmp_path / "audit.db"
    candidate_model = DeterministicEstimator(sample_benchmark_df["churn"])
    champion_model = DeterministicEstimator(
        sample_benchmark_df["churn"], error_indices=[0, 1]
    )
    initial_champ_stub = type("ModelVersionStub", (), {"version": "2"})()
    reconciled_cand_stub = type("ModelVersionStub", (), {"version": "8"})()

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        # Initial champion check returns "2", post-mutation check returns "8"
        client.get_model_version_by_alias.side_effect = [
            initial_champ_stub,
            reconciled_cand_stub,
        ]
        client.set_registered_model_alias.side_effect = RuntimeError(
            "Socket timeout after server committed update"
        )

        decision = promote_model(
            candidate_version=8,
            candidate_training_df=sample_training_df,
            candidate_model=candidate_model,
            champion_model=champion_model,
            benchmark_df=sample_benchmark_df,
            benchmark_version="v1",
            model_name="churn-classifier",
            alias="champion",
            minimum_f1=0.70,
            min_improvement=0.01,
            audit_db_path=db_path,
        )

    assert decision.approved is True
    assert decision.reason == "candidate_meets_quality_gate"
    assert decision.candidate_version == 8

    records = get_promotion_audit_records(db_path)
    assert len(records) == 1
    assert records[0]["approved"] is True
    assert records[0]["reason"] == "candidate_meets_quality_gate"
    assert records[0]["candidate_version"] == "8"


def test_promote_model_records_unverified_when_alias_lookup_fails(
    sample_training_df, sample_benchmark_df, tmp_path
):
    """Reconciliation: If alias state cannot be verified, record unverified reason."""
    db_path = tmp_path / "audit.db"
    candidate_model = DeterministicEstimator(sample_benchmark_df["churn"])
    champion_model = DeterministicEstimator(
        sample_benchmark_df["churn"], error_indices=[0, 1]
    )
    initial_champ_stub = type("ModelVersionStub", (), {"version": "2"})()

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        # Initial check succeeds, post-mutation check raises error
        client.get_model_version_by_alias.side_effect = [
            initial_champ_stub,
            RuntimeError("Registry connection completely down"),
        ]
        client.set_registered_model_alias.side_effect = RuntimeError(
            "Registry write failed"
        )

        decision = promote_model(
            candidate_version=8,
            candidate_training_df=sample_training_df,
            candidate_model=candidate_model,
            champion_model=champion_model,
            benchmark_df=sample_benchmark_df,
            benchmark_version="v1",
            model_name="churn-classifier",
            alias="champion",
            minimum_f1=0.70,
            min_improvement=0.01,
            audit_db_path=db_path,
        )

    assert decision.approved is False
    assert decision.reason == "alias_state_unverified"

    records = get_promotion_audit_records(db_path)
    assert len(records) == 1
    assert records[0]["approved"] is False
    assert records[0]["reason"] == "alias_state_unverified"
    assert records[0]["candidate_version"] == "8"


def test_promote_model_records_unverified_when_unexpected_third_version_found(
    sample_training_df, sample_benchmark_df, tmp_path
):
    """Reconciliation: If alias points to an unexpected third version,
    classify as unverified.
    """
    db_path = tmp_path / "audit.db"
    candidate_model = DeterministicEstimator(sample_benchmark_df["churn"])
    champion_model = DeterministicEstimator(
        sample_benchmark_df["churn"], error_indices=[0, 1]
    )
    initial_champ_stub = type("ModelVersionStub", (), {"version": "2"})()
    unexpected_third_stub = type("ModelVersionStub", (), {"version": "99"})()

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        client.get_model_version_by_alias.side_effect = [
            initial_champ_stub,
            unexpected_third_stub,
        ]
        client.set_registered_model_alias.side_effect = RuntimeError("Connection reset")

        decision = promote_model(
            candidate_version=8,
            candidate_training_df=sample_training_df,
            candidate_model=candidate_model,
            champion_model=champion_model,
            benchmark_df=sample_benchmark_df,
            benchmark_version="v1",
            model_name="churn-classifier",
            alias="champion",
            minimum_f1=0.70,
            min_improvement=0.01,
            audit_db_path=db_path,
        )

    assert decision.approved is False
    assert decision.reason == "alias_state_unverified"

    records = get_promotion_audit_records(db_path)
    assert len(records) == 1
    assert records[0]["approved"] is False
    assert records[0]["reason"] == "alias_state_unverified"
    assert records[0]["candidate_version"] == "8"


def test_promote_model_handles_audit_persistence_failure_when_candidate_alias_confirmed(
    sample_training_df, sample_benchmark_df, tmp_path
):
    """Reconciliation: If candidate alias confirmed but audit write fails,
    abort approval.
    """
    db_path = tmp_path / "audit.db"
    candidate_model = DeterministicEstimator(sample_benchmark_df["churn"])
    champion_model = DeterministicEstimator(
        sample_benchmark_df["churn"], error_indices=[0, 1]
    )
    initial_champ_stub = type("ModelVersionStub", (), {"version": "2"})()
    reconciled_cand_stub = type("ModelVersionStub", (), {"version": "8"})()

    real_record_fn = record_promotion_audit
    call_count = 0

    def mock_record_promotion_audit(payload, database_path=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return real_record_fn(payload, database_path=database_path)
        raise sqlite3.OperationalError("disk I/O error during reconciliation")

    with (
        patch("backend.app.ml.promotion.MlflowClient") as mock_client,
        patch(
            "backend.app.ml.promotion.record_promotion_audit",
            side_effect=mock_record_promotion_audit,
        ),
    ):
        client = mock_client.return_value
        client.get_model_version_by_alias.side_effect = [
            initial_champ_stub,
            reconciled_cand_stub,
        ]
        client.set_registered_model_alias.side_effect = RuntimeError(
            "Socket timeout after commit"
        )

        decision = promote_model(
            candidate_version=8,
            candidate_training_df=sample_training_df,
            candidate_model=candidate_model,
            champion_model=champion_model,
            benchmark_df=sample_benchmark_df,
            benchmark_version="v1",
            model_name="churn-classifier",
            alias="champion",
            minimum_f1=0.70,
            min_improvement=0.01,
            audit_db_path=db_path,
        )

    # Must NOT report an entirely successful, fully audited promotion
    assert decision.approved is False
    assert decision.reason == "audit_logging_failed"
    # Preserves verified registry state in gate outcomes
    assert decision.candidate_version == 8
    assert decision.gate_outcomes["alias_reconciliation"]["alias_verified"] is True
    assert (
        decision.gate_outcomes["alias_reconciliation"]["audit_persistence_failed"]
        is True
    )


def test_promote_model_audits_gate_failures_due_to_overlap(
    sample_benchmark_df, tmp_path
):
    """Gate failure due to training/benchmark overlap is audited and rejected."""
    db_path = tmp_path / "audit.db"

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        decision = promote_model(
            candidate_version=10,
            candidate_training_df=sample_benchmark_df,  # Exact overlap!
            candidate_model=None,
            benchmark_df=sample_benchmark_df,
            audit_db_path=db_path,
        )

    assert decision.approved is False
    assert decision.reason == "training_benchmark_overlap"
    assert decision.gate_outcomes["separation_and_overlap"]["passed"] is False
    client.set_registered_model_alias.assert_not_called()

    records = get_promotion_audit_records(db_path)
    assert len(records) == 1
    assert records[0]["approved"] is False
    assert records[0]["reason"] == "training_benchmark_overlap"


def test_promote_model_audits_gate_failures_due_to_digest_mismatch(
    sample_training_df, sample_benchmark_df, tmp_path
):
    """Gate failure due to benchmark digest mismatch is audited and rejected."""
    db_path = tmp_path / "audit.db"

    with patch("backend.app.ml.promotion.MlflowClient") as mock_client:
        client = mock_client.return_value
        decision = promote_model(
            candidate_version=11,
            candidate_training_df=sample_training_df,
            benchmark_df=sample_benchmark_df,
            expected_benchmark_digest="wrong_digest_value",
            audit_db_path=db_path,
        )

    assert decision.approved is False
    assert decision.reason == "benchmark_digest_mismatch"
    assert decision.gate_outcomes["benchmark_integrity"]["passed"] is False
    client.set_registered_model_alias.assert_not_called()

    records = get_promotion_audit_records(db_path)
    assert len(records) == 1
    assert records[0]["reason"] == "benchmark_digest_mismatch"


def test_tracking_log_dataset_provenance_to_mlflow(tmp_path):
    """log_dataset_provenance_to_mlflow sets tags, params, and JSON artifact."""
    prov = {
        "dataset_sha256": "fake_sha256",
        "row_count": 100,
        "feature_count": 5,
        "target_column": "churn",
        "split": {
            "split_performed": True,
            "train_rows": 80,
            "test_rows": 20,
        },
    }

    with (
        patch("backend.app.ml.tracking.mlflow.set_tag") as mock_tag,
        patch("backend.app.ml.tracking.mlflow.log_param") as mock_param,
        patch("backend.app.ml.tracking.mlflow.log_artifact") as mock_artifact,
    ):
        log_dataset_provenance_to_mlflow(prov)

    mock_tag.assert_called_with("dataset_sha256", "fake_sha256")
    param_calls = {call[0][0]: call[0][1] for call in mock_param.call_args_list}
    assert param_calls["dataset_rows"] == 100
    assert param_calls["dataset_features"] == 5
    assert param_calls["dataset_target"] == "churn"
    assert param_calls["split_train_rows"] == 80
    assert param_calls["split_test_rows"] == 20
    mock_artifact.assert_called_once()
    assert mock_artifact.call_args[0][0].endswith("dataset_provenance.json")


def test_tracking_log_promotion_traceability_to_mlflow():
    """log_promotion_traceability_to_mlflow sets tags, params, metrics, artifact."""
    record = {
        "approved": True,
        "reason": "candidate_meets_quality_gate",
        "benchmark_version": "v1",
        "benchmark_digest": "sha_bench",
        "candidate_version": "4",
        "champion_version": "3",
        "margin": 0.01,
        "candidate_f1": 0.89,
        "champion_f1": 0.83,
    }

    with (
        patch("backend.app.ml.tracking.mlflow.set_tag") as mock_tag,
        patch("backend.app.ml.tracking.mlflow.log_param") as mock_param,
        patch("backend.app.ml.tracking.mlflow.log_metric") as mock_metric,
        patch("backend.app.ml.tracking.mlflow.log_artifact") as mock_artifact,
    ):
        log_promotion_traceability_to_mlflow(record)

    tag_calls = {call[0][0]: call[0][1] for call in mock_tag.call_args_list}
    assert tag_calls["promotion_approved"] == "true"
    assert tag_calls["promotion_reason"] == "candidate_meets_quality_gate"
    assert tag_calls["promotion_benchmark_version"] == "v1"
    assert tag_calls["promotion_benchmark_digest"] == "sha_bench"
    assert tag_calls["promotion_candidate_version"] == "4"
    assert tag_calls["promotion_champion_version"] == "3"

    mock_param.assert_called_with("promotion_margin", 0.01)

    metric_calls = {call[0][0]: call[0][1] for call in mock_metric.call_args_list}
    assert metric_calls["promotion_candidate_f1"] == 0.89
    assert metric_calls["promotion_champion_f1"] == 0.83

    mock_artifact.assert_called_once()
    assert mock_artifact.call_args[0][0].endswith("promotion_decision.json")


def test_workflow_run_training_logs_provenance(tmp_path):
    """run_training calls extract_dataset_provenance and logs provenance."""
    df = pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 25, 35, 45, 55, 65],
            "tenure": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "monthly_charges": [50.0] * 10,
            "contract_type": ["monthly"] * 10,
            "support_tickets": [0] * 10,
            "churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch("backend.app.ml.workflow.mlflow.start_run"),
        patch("backend.app.ml.workflow.mlflow.set_tag"),
        patch("backend.app.ml.workflow.mlflow.log_param"),
        patch("backend.app.ml.workflow.mlflow.log_metric"),
        patch("backend.app.ml.workflow.mlflow.log_artifact"),
        patch(
            "backend.app.ml.workflow.log_dataset_provenance_to_mlflow"
        ) as mock_log_prov,
    ):
        run_training(df, tmp_path / "model.joblib")

    mock_log_prov.assert_called_once()
    prov_arg = mock_log_prov.call_args[0][0]
    assert prov_arg["row_count"] == 10
    assert prov_arg["split"]["split_performed"] is True


def test_workflow_run_training_and_promote_logs_provenance_and_decision(
    tmp_path,
):
    """run_training_and_promote logs provenance and passes candidate_run_id."""
    df = pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 25, 35, 45, 55, 65],
            "tenure": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "monthly_charges": [50.0] * 10,
            "contract_type": ["monthly"] * 10,
            "support_tickets": [0] * 10,
            "churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    model_path = tmp_path / "candidate.joblib"
    model_info = type("ModelInfoStub", (), {"registered_model_version": 9})()

    mock_run = MagicMock()
    mock_run.__enter__.return_value.info.run_id = "run-uuid-123"

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=mock_run,
        ),
        patch("backend.app.ml.workflow.mlflow.set_tag"),
        patch("backend.app.ml.workflow.mlflow.log_param"),
        patch("backend.app.ml.workflow.mlflow.log_metric"),
        patch("backend.app.ml.workflow.mlflow.log_artifact"),
        patch(
            "backend.app.ml.workflow.mlflow.sklearn.log_model",
            return_value=model_info,
        ),
        patch(
            "backend.app.ml.workflow.log_dataset_provenance_to_mlflow"
        ) as mock_log_prov,
        patch("backend.app.ml.workflow.promote_model") as mock_promote,
    ):
        run_training_and_promote(df, model_path)

    mock_log_prov.assert_called_once()
    mock_promote.assert_called_once()
    assert mock_promote.call_args.kwargs["candidate_run_id"] == "run-uuid-123"
    assert mock_promote.call_args.kwargs["candidate_version"] == 9


def test_synthetic_benchmark_limitation_explicit():
    """Verify the 20-row synthetic benchmark limitation is accurately configured."""
    bench_path = Path("data/benchmark/churn_benchmark.csv")
    assert bench_path.exists()

    df = pd.read_csv(bench_path)
    # The benchmark dataset is explicitly limited to 20 synthetic rows
    assert len(df) == 20
    assert "churn" in df.columns

    # Verify SHA-256 matches settings default
    actual_digest = compute_file_sha256(bench_path)
    assert actual_digest == settings.benchmark_expected_sha256
