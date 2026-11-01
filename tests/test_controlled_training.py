"""Comprehensive tests for Milestone 9 / Phase 4 Controlled Training Integration."""

import sqlite3
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode

from backend.app.ml.audit import (
    AuditDatabaseError,
    get_promotion_audit_by_job_id,
    record_promotion_audit,
)
from backend.app.ml.jobs import (
    InvalidStateTransitionError,
    JobOwnershipError,
    claim_queued_job,
    create_queued_job,
    get_job,
    mark_promotion_started,
    reclaim_expired_job,
    recover_expired_jobs,
)
from backend.app.ml.migrations import migrate_audit_database_idempotent
from backend.app.ml.promotion import PromotionDecision, promote_model
from backend.app.ml.training import TrainingConfig
from backend.app.ml.training_handler import ControlledTrainingJobHandler
from backend.app.ml.worker import (
    JobExecutionContext,
    JobExecutionResult,
    RetrainingWorker,
    main,
)


@pytest.fixture
def migrated_db(tmp_path: Path) -> Path:
    """Isolated SQLite database migrated with all M9 schema tables and triggers."""
    db_path = tmp_path / "lifecycle_audit.db"
    migrate_audit_database_idempotent(db_path)
    return db_path


@pytest.fixture
def sample_training_df() -> pd.DataFrame:
    """Valid churn training dataset completely disjoint from benchmark data."""
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
def sample_benchmark_df() -> pd.DataFrame:
    """Valid churn benchmark dataset with distinct records."""
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
                72.0,
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


@pytest.fixture
def training_csv_path(tmp_path: Path, sample_training_df: pd.DataFrame) -> Path:
    """Write sample training data to a CSV path."""
    csv_path = tmp_path / "training_data.csv"
    sample_training_df.to_csv(csv_path, index=False)
    return csv_path


class MockVersion:
    def __init__(self, version: str) -> None:
        self.version = version


class MockMlflowClient:
    """In-memory mock MLflow client for isolated registry testing."""

    def __init__(self, champion_version: str | None = None) -> None:
        self.aliases: dict[tuple[str, str], str] = {}
        if champion_version is not None:
            self.aliases[("churn-classifier", "champion")] = champion_version

    def get_model_version_by_alias(self, model_name: str, alias: str) -> MockVersion:
        key = (model_name, alias)
        if key not in self.aliases:
            raise MlflowException(
                "Alias not found",
                error_code=ErrorCode.RESOURCE_DOES_NOT_EXIST,
            )
        return MockVersion(self.aliases[key])

    def set_registered_model_alias(
        self, model_name: str, alias: str, version: str
    ) -> None:
        self.aliases[(model_name, alias)] = str(version)


@pytest.fixture(autouse=True)
def mock_mlflow_tracking():
    """Autouse fixture isolating tests from real MLflow tracking and artifact
    storage."""
    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch("backend.app.ml.workflow.log_dataset_provenance_to_mlflow"),
        patch("backend.app.ml.promotion.log_promotion_traceability_to_mlflow"),
        patch("backend.app.ml.workflow.mlflow.start_run", return_value=nullcontext()),
        patch("backend.app.ml.workflow.mlflow.set_tag"),
        patch("backend.app.ml.workflow.mlflow.log_param"),
        patch("backend.app.ml.workflow.mlflow.log_metric"),
        patch("backend.app.ml.workflow.mlflow.log_artifact"),
    ):
        yield


# 1. Successful training and candidate registration
def test_successful_training_and_candidate_registration(
    migrated_db: Path,
    training_csv_path: Path,
    sample_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    job = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
    )
    jid = job["job_id"]

    mock_client = MockMlflowClient(champion_version=None)
    model_info = type(
        "ModelInfoStub",
        (),
        {"registered_model_version": "1"},
    )()

    handler = ControlledTrainingJobHandler(
        database_path=migrated_db,
        model_path=tmp_path / "cand.joblib",
        benchmark_df=sample_benchmark_df,
        data_path=training_csv_path,
        mlflow_client=mock_client,
        training_config=TrainingConfig(max_iter=50),
    )

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker.lock",
        worker_id="test-worker-1",
        handler=handler,
    )

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=nullcontext(),
        ),
        patch("backend.app.ml.workflow.mlflow.set_tag"),
        patch("backend.app.ml.workflow.mlflow.log_param"),
        patch("backend.app.ml.workflow.mlflow.log_metric"),
        patch("backend.app.ml.workflow.mlflow.log_artifact"),
        patch(
            "backend.app.ml.workflow.mlflow.sklearn.log_model",
            return_value=model_info,
        ),
        patch("backend.app.ml.promotion.MlflowClient", return_value=mock_client),
        worker,
    ):
        result = worker.run_once()

    assert result is not None
    assert result["job_id"] == jid
    assert result["status"] == "succeeded"
    assert result["is_active"] is None
    assert result["candidate_version"] == "1"
    assert result["promotion_approved"] is True
    assert mock_client.aliases[("churn-classifier", "champion")] == "1"

    # Confirm audit was recorded with exact job_id
    audit = get_promotion_audit_by_job_id(jid, database_path=migrated_db)
    assert audit is not None
    assert audit["job_id"] == jid
    assert audit["candidate_version"] == "1"
    assert audit["approved"] is True


# 2. Candidate metadata and previous champion persisted before promotion
def test_candidate_metadata_and_previous_champion_persisted_before_promotion(
    migrated_db: Path,
    training_csv_path: Path,
    sample_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    job = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
    )
    jid = job["job_id"]

    mock_client = MockMlflowClient(champion_version="5")
    model_info = type(
        "ModelInfoStub",
        (),
        {"registered_model_version": "6"},
    )()

    db_state_at_promotion_call: dict = {}

    real_promote_model = promote_model

    def spy_promote_model(*args, **kwargs):
        # Inspect database at the exact moment promote_model is invoked
        db_state_at_promotion_call["job"] = get_job(migrated_db, job_id=jid)
        return real_promote_model(*args, **kwargs)

    handler = ControlledTrainingJobHandler(
        database_path=migrated_db,
        model_path=tmp_path / "cand.joblib",
        benchmark_df=sample_benchmark_df,
        data_path=training_csv_path,
        mlflow_client=mock_client,
        training_config=TrainingConfig(max_iter=50),
    )

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker.lock",
        worker_id="test-worker-1",
        handler=handler,
    )

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=nullcontext(),
        ),
        patch("backend.app.ml.workflow.mlflow.set_tag"),
        patch("backend.app.ml.workflow.mlflow.log_param"),
        patch("backend.app.ml.workflow.mlflow.log_metric"),
        patch("backend.app.ml.workflow.mlflow.log_artifact"),
        patch(
            "backend.app.ml.workflow.mlflow.sklearn.log_model",
            return_value=model_info,
        ),
        patch("backend.app.ml.workflow.promote_model", side_effect=spy_promote_model),
        patch("backend.app.ml.promotion.MlflowClient", return_value=mock_client),
        worker,
    ):
        worker.run_once()

    observed = db_state_at_promotion_call.get("job")
    assert observed is not None
    assert observed["candidate_version"] == "6"
    assert observed["previous_champion_version"] == "5"
    assert observed["promotion_started_at"] is not None


# 3. Callback failure prevents promotion
def test_callback_failure_prevents_promotion(
    migrated_db: Path,
    training_csv_path: Path,
    sample_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    job = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
        max_attempts=3,
    )
    jid = job["job_id"]

    mock_client = MockMlflowClient(champion_version="1")
    model_info = type(
        "ModelInfoStub",
        (),
        {"registered_model_version": "2"},
    )()

    handler = ControlledTrainingJobHandler(
        database_path=migrated_db,
        model_path=tmp_path / "cand.joblib",
        benchmark_df=sample_benchmark_df,
        data_path=training_csv_path,
        mlflow_client=mock_client,
    )

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker.lock",
        worker_id="test-worker-1",
        handler=handler,
    )

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=nullcontext(),
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
            "backend.app.ml.training_handler.persist_candidate_identity",
            side_effect=sqlite3.OperationalError("Simulated DB lock during callback"),
        ),
        patch("backend.app.ml.workflow.promote_model") as mock_promote,
        worker,
    ):
        _ = worker.run_once()

    # promote_model must NEVER have been called
    mock_promote.assert_not_called()

    # Job must not have set promotion marker, but be requeued for retry
    # (Phase 3 retry semantics)
    persisted = get_job(migrated_db, job_id=jid)
    assert persisted is not None
    assert persisted["status"] == "queued"
    assert persisted["promotion_started_at"] is None
    assert persisted["attempt_count"] == 1
    assert "Simulated DB lock during callback" in persisted["failure_reason"]


# 4. Lease loss before marker prevents promotion
def test_lease_loss_before_marker_prevents_promotion(
    migrated_db: Path,
    training_csv_path: Path,
    sample_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    job = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
    )
    jid = job["job_id"]

    mock_client = MockMlflowClient(champion_version="1")
    model_info = type(
        "ModelInfoStub",
        (),
        {"registered_model_version": "2"},
    )()

    handler = ControlledTrainingJobHandler(
        database_path=migrated_db,
        model_path=tmp_path / "cand.joblib",
        benchmark_df=sample_benchmark_df,
        data_path=training_csv_path,
        mlflow_client=mock_client,
    )

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker.lock",
        worker_id="test-worker-1",
        handler=handler,
    )

    # Invalidate lease in database right before callback executes
    def revoke_lease_on_log_model(*args, **kwargs):
        with sqlite3.connect(migrated_db) as conn:
            conn.execute(
                "UPDATE retraining_jobs SET lease_token = 'stolen-token' "
                "WHERE job_id = ?",
                (jid,),
            )
        return model_info

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=nullcontext(),
        ),
        patch("backend.app.ml.workflow.mlflow.set_tag"),
        patch("backend.app.ml.workflow.mlflow.log_param"),
        patch("backend.app.ml.workflow.mlflow.log_metric"),
        patch("backend.app.ml.workflow.mlflow.log_artifact"),
        patch(
            "backend.app.ml.workflow.mlflow.sklearn.log_model",
            side_effect=revoke_lease_on_log_model,
        ),
        patch("backend.app.ml.workflow.promote_model") as mock_promote,
        worker,
    ):
        result = worker.run_once()

    mock_promote.assert_not_called()
    assert result is None  # Worker refuses to update DB after ownership lost


# 5. Marker committed before promotion invocation
def test_marker_committed_before_promotion_invocation(
    migrated_db: Path,
    training_csv_path: Path,
    sample_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    job = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
    )
    jid = job["job_id"]

    mock_client = MockMlflowClient(champion_version=None)
    model_info = type(
        "ModelInfoStub",
        (),
        {"registered_model_version": "1"},
    )()

    marker_at_entry: str | None = None

    def capture_marker(*args, **kwargs):
        nonlocal marker_at_entry
        row = get_job(migrated_db, job_id=jid)
        marker_at_entry = row["promotion_started_at"] if row else None
        return PromotionDecision(
            approved=True,
            reason="candidate_meets_quality_gate",
            candidate_f1=0.85,
            champion_f1=None,
            candidate_version="1",
        )

    handler = ControlledTrainingJobHandler(
        database_path=migrated_db,
        model_path=tmp_path / "cand.joblib",
        benchmark_df=sample_benchmark_df,
        data_path=training_csv_path,
        mlflow_client=mock_client,
    )

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker.lock",
        worker_id="test-worker-1",
        handler=handler,
    )

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=nullcontext(),
        ),
        patch("backend.app.ml.workflow.mlflow.set_tag"),
        patch("backend.app.ml.workflow.mlflow.log_param"),
        patch("backend.app.ml.workflow.mlflow.log_metric"),
        patch("backend.app.ml.workflow.mlflow.log_artifact"),
        patch(
            "backend.app.ml.workflow.mlflow.sklearn.log_model",
            return_value=model_info,
        ),
        patch("backend.app.ml.workflow.promote_model", side_effect=capture_marker),
        patch("backend.app.ml.promotion.MlflowClient", return_value=mock_client),
        worker,
    ):
        worker.run_once()

    assert marker_at_entry is not None
    assert len(marker_at_entry) > 0


# 6. Marker persistence failure prevents promotion
def test_marker_persistence_failure_prevents_promotion(
    migrated_db: Path,
    training_csv_path: Path,
    sample_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    job = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
    )
    jid = job["job_id"]

    mock_client = MockMlflowClient(champion_version=None)
    model_info = type(
        "ModelInfoStub",
        (),
        {"registered_model_version": "1"},
    )()

    handler = ControlledTrainingJobHandler(
        database_path=migrated_db,
        model_path=tmp_path / "cand.joblib",
        benchmark_df=sample_benchmark_df,
        data_path=training_csv_path,
        mlflow_client=mock_client,
    )

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker.lock",
        worker_id="test-worker-1",
        handler=handler,
    )

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=nullcontext(),
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
            "backend.app.ml.training_handler.mark_promotion_started",
            side_effect=sqlite3.OperationalError("Simulated write conflict on marker"),
        ),
        patch("backend.app.ml.workflow.promote_model") as mock_promote,
        worker,
    ):
        worker.run_once()

    mock_promote.assert_not_called()
    persisted = get_job(migrated_db, job_id=jid)
    assert persisted is not None
    assert persisted["promotion_started_at"] is None


# 7. Successful promotion and exact job_id audit correlation
def test_successful_promotion_and_exact_job_id_audit_correlation(
    migrated_db: Path,
    training_csv_path: Path,
    sample_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    job = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
    )
    jid = job["job_id"]

    mock_client = MockMlflowClient(champion_version=None)
    model_info = type(
        "ModelInfoStub",
        (),
        {"registered_model_version": "10"},
    )()

    handler = ControlledTrainingJobHandler(
        database_path=migrated_db,
        model_path=tmp_path / "cand.joblib",
        benchmark_df=sample_benchmark_df,
        data_path=training_csv_path,
        mlflow_client=mock_client,
        training_config=TrainingConfig(max_iter=50),
    )

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker.lock",
        worker_id="test-worker-1",
        handler=handler,
    )

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=nullcontext(),
        ),
        patch("backend.app.ml.workflow.mlflow.set_tag"),
        patch("backend.app.ml.workflow.mlflow.log_param"),
        patch("backend.app.ml.workflow.mlflow.log_metric"),
        patch("backend.app.ml.workflow.mlflow.log_artifact"),
        patch(
            "backend.app.ml.workflow.mlflow.sklearn.log_model",
            return_value=model_info,
        ),
        patch("backend.app.ml.promotion.MlflowClient", return_value=mock_client),
        worker,
    ):
        result = worker.run_once()

    assert result is not None
    assert result["status"] == "succeeded"
    assert result["promotion_audit_id"] is not None

    # Correlate exact audit record
    audit = get_promotion_audit_by_job_id(jid, database_path=migrated_db)
    assert audit is not None
    assert audit["job_id"] == jid
    assert audit["audit_id"] == result["promotion_audit_id"]
    assert audit["approved"] is True
    assert audit["candidate_version"] == "10"


# 8. Rejected promotion outcome
def test_rejected_promotion_outcome(
    migrated_db: Path,
    training_csv_path: Path,
    sample_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    job = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
    )
    jid = job["job_id"]

    # Active champion has F1=0.99, impossible for candidate to beat
    mock_client = MockMlflowClient(champion_version="1")
    model_info = type(
        "ModelInfoStub",
        (),
        {"registered_model_version": "2"},
    )()

    class SuperiorChampionEstimator:
        def predict(self, X):
            # Perfect predictions yielding 1.0 F1
            return np.array(sample_benchmark_df["churn"])

    handler = ControlledTrainingJobHandler(
        database_path=migrated_db,
        model_path=tmp_path / "cand.joblib",
        benchmark_df=sample_benchmark_df,
        data_path=training_csv_path,
        mlflow_client=mock_client,
        training_config=TrainingConfig(max_iter=10),
    )

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker.lock",
        worker_id="test-worker-1",
        handler=handler,
    )

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=nullcontext(),
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
            "backend.app.ml.promotion.load_registered_model_by_alias",
            return_value=SuperiorChampionEstimator(),
        ),
        patch("backend.app.ml.promotion.MlflowClient", return_value=mock_client),
        worker,
    ):
        result = worker.run_once()

    assert result is not None
    assert result["status"] == "rejected"
    assert result["is_active"] is None
    assert result["promotion_approved"] is False

    # Champion alias unchanged
    assert mock_client.aliases[("churn-classifier", "champion")] == "1"

    # Rejection audit preserved with exact job_id
    audit = get_promotion_audit_by_job_id(jid, database_path=migrated_db)
    assert audit is not None
    assert audit["job_id"] == jid
    assert audit["approved"] is False


# 9. Training failure before marker and retry behavior
def test_training_failure_before_marker_and_retry_behavior(
    migrated_db: Path,
    training_csv_path: Path,
    sample_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    job = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
        max_attempts=2,
    )
    jid = job["job_id"]

    handler = ControlledTrainingJobHandler(
        database_path=migrated_db,
        model_path=tmp_path / "cand.joblib",
        benchmark_df=sample_benchmark_df,
        data_path=training_csv_path,
    )

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker.lock",
        worker_id="test-worker-1",
        handler=handler,
    )

    # First attempt fails during train_churn_model -> requeued
    with (
        patch(
            "backend.app.ml.workflow.train_churn_model",
            side_effect=RuntimeError("Data preprocessing failure"),
        ),
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=nullcontext(),
        ),
        worker,
    ):
        res1 = worker.run_once()

    assert res1 is not None
    assert res1["job_id"] == jid
    assert res1["status"] == "queued"
    assert res1["attempt_count"] == 1
    assert "Data preprocessing failure" in res1["failure_reason"]

    # Second attempt fails -> attempts exhausted -> failed
    with (
        patch(
            "backend.app.ml.workflow.train_churn_model",
            side_effect=RuntimeError("Data preprocessing failure second time"),
        ),
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=nullcontext(),
        ),
        worker,
    ):
        res2 = worker.run_once()

    assert res2 is not None
    assert res2["job_id"] == jid
    assert res2["status"] == "failed"
    assert res2["is_active"] is None
    assert res2["attempt_count"] == 2


# 10. Exception after marker and quarantine
def test_exception_after_marker_and_quarantine(
    migrated_db: Path,
    training_csv_path: Path,
    sample_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    job = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
        max_attempts=3,
    )
    jid = job["job_id"]

    mock_client = MockMlflowClient(champion_version=None)
    model_info = type(
        "ModelInfoStub",
        (),
        {"registered_model_version": "1"},
    )()

    handler = ControlledTrainingJobHandler(
        database_path=migrated_db,
        model_path=tmp_path / "cand.joblib",
        benchmark_df=sample_benchmark_df,
        data_path=training_csv_path,
        mlflow_client=mock_client,
    )

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker.lock",
        worker_id="test-worker-1",
        handler=handler,
    )

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=nullcontext(),
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
            "backend.app.ml.workflow.promote_model",
            side_effect=RuntimeError("MLflow connection dropped during promote"),
        ),
        patch("backend.app.ml.promotion.MlflowClient", return_value=mock_client),
        worker,
    ):
        result = worker.run_once()

    assert result is not None
    assert result["job_id"] == jid
    assert result["status"] == "reconciliation_required"
    assert result["is_active"] == 1
    assert result["promotion_started_at"] is not None
    assert "Handler exception after promotion started" in result["failure_reason"]


# 11. MLflow timeout or ambiguous promotion result
def test_mlflow_timeout_or_ambiguous_promotion_result(
    migrated_db: Path,
    training_csv_path: Path,
    sample_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    job = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
    )
    jid = job["job_id"]

    mock_client = MockMlflowClient(champion_version=None)
    model_info = type(
        "ModelInfoStub",
        (),
        {"registered_model_version": "1"},
    )()

    handler = ControlledTrainingJobHandler(
        database_path=migrated_db,
        model_path=tmp_path / "cand.joblib",
        benchmark_df=sample_benchmark_df,
        data_path=training_csv_path,
        mlflow_client=mock_client,
    )

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker.lock",
        worker_id="test-worker-1",
        handler=handler,
    )

    ambiguous_decision = PromotionDecision(
        approved=False,
        reason="alias_state_unverified",
        candidate_f1=0.85,
        champion_f1=None,
        candidate_version="1",
        job_id=jid,
    )

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=nullcontext(),
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
            "backend.app.ml.workflow.promote_model",
            return_value=ambiguous_decision,
        ),
        patch("backend.app.ml.promotion.MlflowClient", return_value=mock_client),
        worker,
    ):
        result = worker.run_once()

    assert result is not None
    assert result["status"] == "reconciliation_required"
    assert result["is_active"] == 1
    assert "Ambiguous MLflow promotion outcome" in result["failure_reason"]


# 12. Alias changed but audit missing
def test_alias_changed_but_audit_missing_quarantined(
    migrated_db: Path,
    training_csv_path: Path,
    sample_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    job = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
    )
    jid = job["job_id"]

    mock_client = MockMlflowClient(champion_version=None)
    # Set alias to candidate manually to simulate alias changed
    mock_client.aliases[("churn-classifier", "champion")] = "1"

    model_info = type(
        "ModelInfoStub",
        (),
        {"registered_model_version": "1"},
    )()

    handler = ControlledTrainingJobHandler(
        database_path=migrated_db,
        model_path=tmp_path / "cand.joblib",
        benchmark_df=sample_benchmark_df,
        data_path=training_csv_path,
        mlflow_client=mock_client,
    )

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker.lock",
        worker_id="test-worker-1",
        handler=handler,
    )

    # Decision says approved, but audit record query returns None
    approved_decision = PromotionDecision(
        approved=True,
        reason="candidate_meets_quality_gate",
        candidate_f1=0.85,
        champion_f1=None,
        candidate_version="1",
        job_id=jid,
    )

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=nullcontext(),
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
            "backend.app.ml.workflow.promote_model",
            return_value=approved_decision,
        ),
        patch(
            "backend.app.ml.training_handler.get_promotion_audit_by_job_id",
            return_value=None,
        ),
        patch("backend.app.ml.promotion.MlflowClient", return_value=mock_client),
        worker,
    ):
        result = worker.run_once()

    assert result is not None
    assert result["status"] == "reconciliation_required"
    assert result["is_active"] == 1
    assert "audit record is missing or invalid" in result["failure_reason"]


# 13. Approved audit but alias mismatch
def test_approved_audit_but_alias_mismatch_quarantined(
    migrated_db: Path,
    training_csv_path: Path,
    sample_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    job = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
    )
    jid = job["job_id"]

    mock_client = MockMlflowClient(champion_version="99")  # Mismatch: 99 != 1
    model_info = type(
        "ModelInfoStub",
        (),
        {"registered_model_version": "1"},
    )()

    handler = ControlledTrainingJobHandler(
        database_path=migrated_db,
        model_path=tmp_path / "cand.joblib",
        benchmark_df=sample_benchmark_df,
        data_path=training_csv_path,
        mlflow_client=mock_client,
    )

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker.lock",
        worker_id="test-worker-1",
        handler=handler,
    )

    approved_decision = PromotionDecision(
        approved=True,
        reason="candidate_meets_quality_gate",
        candidate_f1=0.85,
        champion_f1=None,
        candidate_version="1",
        audit_id="audit-123",
        job_id=jid,
    )

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=nullcontext(),
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
            "backend.app.ml.workflow.promote_model",
            return_value=approved_decision,
        ),
        patch(
            "backend.app.ml.training_handler.get_promotion_audit_by_job_id",
            return_value={"job_id": jid, "approved": True, "audit_id": "audit-123"},
        ),
        patch("backend.app.ml.promotion.MlflowClient", return_value=mock_client),
        worker,
    ):
        result = worker.run_once()

    assert result is not None
    assert result["status"] == "reconciliation_required"
    assert result["is_active"] == 1
    assert (
        "alias points to version '99' instead of candidate" in result["failure_reason"]
    )


# 14. Stale worker cannot finalize
def test_stale_worker_cannot_finalize(
    migrated_db: Path,
    training_csv_path: Path,
    sample_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    job = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
    )
    jid = job["job_id"]

    class StealLeaseHandler:
        def __call__(
            self, job_dict: dict, context: JobExecutionContext
        ) -> JobExecutionResult:
            # Another worker steals lease token
            with sqlite3.connect(migrated_db) as conn:
                conn.execute(
                    "UPDATE retraining_jobs SET lease_token = 'stolen-token' "
                    "WHERE job_id = ?",
                    (jid,),
                )
            return JobExecutionResult(status="succeeded")

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker.lock",
        worker_id="test-worker-1",
        handler=StealLeaseHandler(),
    )

    with worker:
        result = worker.run_once()

    assert result is None  # Worker refused to finalize
    persisted = get_job(migrated_db, job_id=jid)
    assert persisted["status"] == "running"  # Status not overwritten by stale worker


# 15. Existing five promotion gates remain unchanged
def test_existing_five_promotion_gates_remain_unchanged(
    sample_training_df: pd.DataFrame,
    sample_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "gate_audit.db"
    migrate_audit_database_idempotent(db_path)

    class ZeroPredictor:
        def predict(self, X):
            return np.zeros(len(X))

    # Gate 1: Benchmark missing / invalid
    d1 = promote_model(
        candidate_version="1",
        candidate_training_df=sample_training_df,
        candidate_model=ZeroPredictor(),
        benchmark_df=pd.DataFrame(),
        audit_db_path=db_path,
    )
    assert d1.approved is False
    assert d1.gate_outcomes["benchmark_integrity"]["passed"] is False

    # Gate 2: Training / benchmark overlap
    overlapping_df = sample_benchmark_df.copy()
    d2 = promote_model(
        candidate_version="1",
        candidate_training_df=overlapping_df,
        candidate_model=ZeroPredictor(),
        benchmark_df=sample_benchmark_df,
        audit_db_path=db_path,
    )
    assert d2.approved is False
    assert d2.gate_outcomes["separation_and_overlap"]["passed"] is False

    # Gate 3: Candidate model missing
    d3 = promote_model(
        candidate_version="1",
        candidate_training_df=sample_training_df,
        candidate_model=None,
        benchmark_df=sample_benchmark_df,
        audit_db_path=db_path,
    )
    assert d3.approved is False
    assert d3.gate_outcomes["candidate_evaluation"]["passed"] is False

    # Gate 4: Champion benchmark evaluation fails
    mock_client = MockMlflowClient(champion_version="1")
    with (
        patch("backend.app.ml.promotion.MlflowClient", return_value=mock_client),
        patch(
            "backend.app.ml.promotion.load_registered_model_by_alias",
            side_effect=RuntimeError("Cannot load champion model"),
        ),
    ):
        d4 = promote_model(
            candidate_version="2",
            candidate_training_df=sample_training_df,
            candidate_model=MagicMock(predict=lambda X: np.zeros(len(X))),
            benchmark_df=sample_benchmark_df,
            audit_db_path=db_path,
        )
        assert d4.approved is False
        assert d4.gate_outcomes["champion_evaluation"]["passed"] is False

    # Gate 5: Quality gate (candidate below minimum F1)
    fresh_client = MockMlflowClient(champion_version=None)
    with patch("backend.app.ml.promotion.MlflowClient", return_value=fresh_client):
        d5 = promote_model(
            candidate_version="2",
            candidate_training_df=sample_training_df,
            candidate_model=ZeroPredictor(),
            benchmark_df=sample_benchmark_df,
            audit_db_path=db_path,
            minimum_f1=0.70,
        )
        assert d5.approved is False
        assert d5.gate_outcomes["quality_gate"]["passed"] is False


# 16. Worker CLI executes with production handler
def test_worker_cli_with_production_handler(migrated_db: Path, tmp_path: Path) -> None:
    exit_code = main(
        [
            "--db-path",
            str(migrated_db),
            "--lock-path",
            str(tmp_path / "cli.lock"),
            "--handler",
            "production",
            "--once",
        ]
    )
    assert exit_code == 0


# ==============================================================================
# Post-Implementation Safety Audit Tests (Cases 1 - 5)
# ==============================================================================


# Case 1: Dataset identity isolation and pre-training validation
def test_case1_dataset_identity_isolation_and_prevalidation(
    migrated_db: Path,
    sample_training_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verify handler strictly uses queued job dataset_id, avoids default fallback,
    and fails before candidate registration when identity cannot be resolved."""
    ds1_path = tmp_path / "dataset_alpha.csv"
    ds2_path = tmp_path / "dataset_beta.csv"
    sample_training_df.to_csv(ds1_path, index=False)
    sample_training_df.iloc[:20].to_csv(ds2_path, index=False)

    handler = ControlledTrainingJobHandler(
        database_path=migrated_db,
        # Configured data_path should NOT override job dataset_id
        data_path=tmp_path / "default_override.csv",
    )

    job1 = {"job_id": "job-1", "dataset_id": str(ds1_path)}
    job2 = {"job_id": "job-2", "dataset_id": str(ds2_path)}

    # Distinct dataset identities resolve to distinct files, not default_override
    resolved_1 = handler._resolve_dataset_path(job1)
    resolved_2 = handler._resolve_dataset_path(job2)
    assert resolved_1 == ds1_path
    assert resolved_2 == ds2_path
    assert resolved_1 != resolved_2

    # Nonexistent dataset_id MUST raise FileNotFoundError and never fall back to default
    job3 = {"job_id": "job-3", "dataset_id": "missing_dataset_xyz.csv"}
    with pytest.raises(FileNotFoundError, match="could not be resolved"):
        handler._resolve_dataset_path(job3)

    # Corrupt dataset fails validation before training or candidate registration
    corrupt_path = tmp_path / "corrupt.csv"
    corrupt_path.write_text("not,a,valid,churn,csv\n1,2,3,4,5\n")
    _ = create_queued_job(
        migrated_db,
        dataset_id=str(corrupt_path),
        trigger_source="manual",
    )
    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker.lock",
        worker_id="test-worker-1",
        handler=handler,
    )

    with patch("backend.app.ml.workflow.run_training_and_promote") as mock_run:
        res = worker.run_once()

    # Pipeline was never invoked
    mock_run.assert_not_called()
    assert res is not None
    assert res["status"] == "queued"  # Pre-marker error is requeued under Phase 3
    assert res["promotion_started_at"] is None
    assert "Missing required columns" in res["failure_reason"]


# Case 2: Previous champion lookup distinguishes absence from failure
def test_case2_previous_champion_genuine_absence_vs_connectivity_failure(
    migrated_db: Path,
    training_csv_path: Path,
    sample_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verify genuine absence produces previous_champion_version=None while
    connectivity/internal error halts promotion immediately."""
    model_info = type("ModelInfoStub", (), {"registered_model_version": "1"})()

    # Subcase A: Genuine absence (RESOURCE_DOES_NOT_EXIST)
    mock_absent_client = MagicMock()
    mock_absent_client.get_model_version_by_alias.side_effect = MlflowException(
        "Alias champion not found",
        error_code=ErrorCode.RESOURCE_DOES_NOT_EXIST,
    )

    job_a = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
    )
    jid_a = job_a["job_id"]

    handler_a = ControlledTrainingJobHandler(
        database_path=migrated_db,
        model_path=tmp_path / "cand_a.joblib",
        benchmark_df=sample_benchmark_df,
        data_path=training_csv_path,
        mlflow_client=mock_absent_client,
    )
    worker_a = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker_a.lock",
        worker_id="test-worker-1",
        handler=handler_a,
    )

    def fake_promote_a(*args, **kwargs):
        record_promotion_audit(
            {
                "job_id": jid_a,
                "approved": False,
                "candidate_version": "1",
                "champion_version": None,
                "reason": "quality_gate_failed",
                "candidate_f1": 0.60,
                "champion_f1": None,
                "benchmark_version": "v1",
                "benchmark_digest": "b" * 64,
                "margin": 0.0,
                "audit_id": "audit-absent",
            },
            database_path=migrated_db,
        )
        return PromotionDecision(
            approved=False,
            candidate_version="1",
            champion_version=None,
            reason="quality_gate_failed",
            candidate_f1=0.60,
            champion_f1=None,
            gate_outcomes={"quality_gate": {"passed": False}},
            audit_id="audit-absent",
            job_id=jid_a,
        )

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch("backend.app.ml.workflow.mlflow.start_run", return_value=nullcontext()),
        patch("backend.app.ml.workflow.mlflow.set_tag"),
        patch("backend.app.ml.workflow.mlflow.log_param"),
        patch("backend.app.ml.workflow.mlflow.log_metric"),
        patch("backend.app.ml.workflow.mlflow.log_artifact"),
        patch(
            "backend.app.ml.workflow.mlflow.sklearn.log_model",
            return_value=model_info,
        ),
        patch("backend.app.ml.workflow.promote_model", side_effect=fake_promote_a),
        patch("backend.app.ml.promotion.MlflowClient", return_value=mock_absent_client),
        worker_a,
    ):
        worker_a.run_once()

    # Successfully persisted previous_champion_version as None
    persisted_a = get_job(migrated_db, job_id=jid_a)
    assert persisted_a is not None
    assert persisted_a["candidate_version"] == "1"
    assert persisted_a["previous_champion_version"] is None
    assert persisted_a["promotion_started_at"] is not None

    # Subcase B: MLflow connectivity/lookup failure (INTERNAL_ERROR or ConnectionError)
    mock_failing_client = MagicMock()
    mock_failing_client.get_model_version_by_alias.side_effect = MlflowException(
        "Connection timeout to MLflow remote tracking server",
        error_code=ErrorCode.Name(ErrorCode.INTERNAL_ERROR),
    )

    job_b = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
    )
    jid_b = job_b["job_id"]

    handler_b = ControlledTrainingJobHandler(
        database_path=migrated_db,
        model_path=tmp_path / "cand_b.joblib",
        benchmark_df=sample_benchmark_df,
        data_path=training_csv_path,
        mlflow_client=mock_failing_client,
    )
    worker_b = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker_b.lock",
        worker_id="test-worker-1",
        handler=handler_b,
    )

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch("backend.app.ml.workflow.mlflow.start_run", return_value=nullcontext()),
        patch("backend.app.ml.workflow.mlflow.set_tag"),
        patch("backend.app.ml.workflow.mlflow.log_param"),
        patch("backend.app.ml.workflow.mlflow.log_metric"),
        patch("backend.app.ml.workflow.mlflow.log_artifact"),
        patch(
            "backend.app.ml.workflow.mlflow.sklearn.log_model",
            return_value=model_info,
        ),
        patch("backend.app.ml.workflow.promote_model") as mock_promote_b,
        worker_b,
    ):
        worker_b.run_once()

    # Promotion was NEVER called
    mock_promote_b.assert_not_called()
    # Marker was NEVER set
    persisted_b = get_job(migrated_db, job_id=jid_b)
    assert persisted_b is not None
    assert persisted_b["promotion_started_at"] is None
    assert "Connection timeout" in persisted_b["failure_reason"]


# Case 3: Promotion marker fencing, repeated attempts, and lost ownership
def test_case3_promotion_marker_fencing_repeated_attempts_and_lost_ownership(
    migrated_db: Path,
    training_csv_path: Path,
) -> None:
    """Verify mark_promotion_started is strictly fenced, disallows repeated
    setting, and rejects stolen or lost leases."""
    job = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
    )
    jid = job["job_id"]

    claimed = claim_queued_job(
        migrated_db, worker_id="worker-w1", lease_duration_seconds=60
    )
    assert claimed is not None
    token = claimed["lease_token"]

    # 1. Attempt with stolen/invalid lease token raises JobOwnershipError
    with pytest.raises(JobOwnershipError, match="is not active owner"):
        mark_promotion_started(
            migrated_db,
            job_id=jid,
            worker_id="worker-w1",
            lease_token="wrong-token",
            candidate_version="1",
        )

    # 2. First valid marker setting succeeds
    marked = mark_promotion_started(
        migrated_db,
        job_id=jid,
        worker_id="worker-w1",
        lease_token=token,
        candidate_version="1",
    )
    assert marked["promotion_started_at"] is not None

    # 3. Repeated attempt by same worker and token raises InvalidStateTransitionError
    with pytest.raises(
        InvalidStateTransitionError, match="Promotion marker already set"
    ):
        mark_promotion_started(
            migrated_db,
            job_id=jid,
            worker_id="worker-w1",
            lease_token=token,
            candidate_version="1",
        )


# Case 4: Audit uniqueness and duplicate record quarantine
def test_case4_audit_uniqueness_duplicate_records_cause_reconciliation(
    migrated_db: Path,
    training_csv_path: Path,
    sample_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verify get_promotion_audit_by_job_id requires exactly one record and
    quarantines if multiple or zero records exist for resolved promotion."""
    job = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
    )
    jid = job["job_id"]

    # Insert two records with the exact same job_id
    record_promotion_audit(
        {
            "job_id": jid,
            "approved": True,
            "candidate_version": "1",
            "champion_version": None,
            "reason": "first_promotion",
            "candidate_f1": 0.85,
            "champion_f1": None,
            "benchmark_version": "v1",
            "benchmark_digest": "b" * 64,
            "margin": 0.0,
            "audit_id": "audit-dup-1",
        },
        database_path=migrated_db,
    )
    record_promotion_audit(
        {
            "job_id": jid,
            "approved": True,
            "candidate_version": "1",
            "champion_version": None,
            "reason": "duplicate_promotion",
            "candidate_f1": 0.86,
            "champion_f1": None,
            "benchmark_version": "v1",
            "benchmark_digest": "b" * 64,
            "margin": 0.0,
            "audit_id": "audit-dup-2",
        },
        database_path=migrated_db,
    )

    # Direct query raises AuditDatabaseError
    with pytest.raises(AuditDatabaseError, match="Duplicate promotion audit records"):
        get_promotion_audit_by_job_id(jid, database_path=migrated_db)

    # In handler, duplicate audit causes reconciliation_required instead of picking one
    mock_client = MockMlflowClient(champion_version=None)
    model_info = type("ModelInfoStub", (), {"registered_model_version": "1"})()

    handler = ControlledTrainingJobHandler(
        database_path=migrated_db,
        model_path=tmp_path / "cand.joblib",
        benchmark_df=sample_benchmark_df,
        data_path=training_csv_path,
        mlflow_client=mock_client,
    )
    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker.lock",
        worker_id="test-worker-1",
        handler=handler,
    )

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch("backend.app.ml.workflow.mlflow.start_run", return_value=nullcontext()),
        patch("backend.app.ml.workflow.mlflow.set_tag"),
        patch("backend.app.ml.workflow.mlflow.log_param"),
        patch("backend.app.ml.workflow.mlflow.log_metric"),
        patch("backend.app.ml.workflow.mlflow.log_artifact"),
        patch(
            "backend.app.ml.workflow.mlflow.sklearn.log_model",
            return_value=model_info,
        ),
        patch("backend.app.ml.promotion.MlflowClient", return_value=mock_client),
        worker,
    ):
        result = worker.run_once()

    assert result is not None
    assert result["status"] == "reconciliation_required"
    assert result["is_active"] == 1
    assert "Duplicate promotion audit records" in result["failure_reason"]


# Case 5: Lease expiry during promotion and recovery
def test_case5_lease_expiry_during_promotion_stale_worker_and_recovery(
    migrated_db: Path,
    training_csv_path: Path,
    sample_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verify lease expiry during promote_model prevents stale worker finalization,
    quarantines via recovery, and blocks reclaim or auto-retry."""
    job = create_queued_job(
        migrated_db,
        dataset_id=str(training_csv_path),
        trigger_source="manual",
    )
    jid = job["job_id"]

    mock_client = MockMlflowClient(champion_version=None)
    model_info = type("ModelInfoStub", (), {"registered_model_version": "1"})()

    handler = ControlledTrainingJobHandler(
        database_path=migrated_db,
        model_path=tmp_path / "cand.joblib",
        benchmark_df=sample_benchmark_df,
        data_path=training_csv_path,
        mlflow_client=mock_client,
    )
    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "worker.lock",
        worker_id="test-worker-1",
        handler=handler,
    )

    # During promote_model, lease expires in database
    def expire_lease_during_promotion(*args, **kwargs):
        expired_ts = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        with sqlite3.connect(migrated_db) as conn:
            conn.execute(
                "UPDATE retraining_jobs SET lease_expires_at = ? WHERE job_id = ?",
                (expired_ts, jid),
            )
        return PromotionDecision(
            approved=True,
            candidate_version="1",
            champion_version=None,
            reason="first_model_promoted",
            gate_outcomes={},
            metrics={},
            audit_id="audit-exp-1",
            job_id=jid,
        )

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch("backend.app.ml.workflow.mlflow.start_run", return_value=nullcontext()),
        patch("backend.app.ml.workflow.mlflow.set_tag"),
        patch("backend.app.ml.workflow.mlflow.log_param"),
        patch("backend.app.ml.workflow.mlflow.log_metric"),
        patch("backend.app.ml.workflow.mlflow.log_artifact"),
        patch(
            "backend.app.ml.workflow.mlflow.sklearn.log_model",
            return_value=model_info,
        ),
        patch(
            "backend.app.ml.workflow.promote_model",
            side_effect=expire_lease_during_promotion,
        ),
        worker,
    ):
        result = worker.run_once()

    # Stale worker CANNOT finalize: worker returns None and does not mark succeeded
    assert result is None

    # Job still has promotion_started_at set and is in running state
    persisted = get_job(migrated_db, job_id=jid)
    assert persisted is not None
    assert persisted["status"] == "running"
    assert persisted["promotion_started_at"] is not None

    # Expired job recovery detects lease and quarantines into reconciliation_required
    recovery_outcomes = recover_expired_jobs(migrated_db)
    assert len(recovery_outcomes) == 1
    assert recovery_outcomes[0]["job_id"] == jid
    assert recovery_outcomes[0]["action"] == "quarantined"

    quarantined = get_job(migrated_db, job_id=jid)
    assert quarantined is not None
    assert quarantined["status"] == "reconciliation_required"
    assert quarantined["is_active"] == 1

    # Another worker CANNOT claim or automatically retry it
    claimed_next = claim_queued_job(migrated_db, worker_id="worker-w2")
    assert claimed_next is None

    with pytest.raises(
        InvalidStateTransitionError, match="promotion has already started"
    ):
        reclaim_expired_job(migrated_db, job_id=jid)
