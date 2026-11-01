"""Comprehensive test suite for Phase 5 Operator Reconciliation.

Validates process locking invariants, read-only evidence inspection,
fail-closed verification, strict safety invariants across all resolution paths,
bootstrap alias handling, failure injection, and atomic database finalization.
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode

from backend.app.ml.audit import (
    get_operator_resolution_audit,
    record_promotion_audit,
)
from backend.app.ml.jobs import get_job
from backend.app.ml.lock import (
    MaintenanceLockConflictError,
    WorkerLockConflictError,
    operator_maintenance_lock,
    worker_execution_lock,
)
from backend.app.ml.migrations import migrate_audit_database_idempotent
from backend.app.ml.reconciliation import (
    EvidenceVerificationError,
    JobNotQuarantinedError,
    MLflowOperationError,
    SafetyInvariantViolationError,
    inspect_quarantined_job,
    main,
    reconcile_job,
)
from backend.app.ml.worker import RetrainingWorker


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    """Isolated SQLite database migrated with all M9 schema tables."""
    db_path = tmp_path / "test_lifecycle.db"
    migrate_audit_database_idempotent(db_path)
    return db_path


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    """Isolated lock file path for process locking tests."""
    return tmp_path / "test_worker.lock"


def _create_quarantined_job(
    db_path: Path,
    job_id: str = "job-quarantined-1",
    candidate_version: str | None = "2",
    candidate_run_id: str | None = "run-cand-2",
    previous_champion_version: str | None = "1",
    promotion_started_at: str | None = "2026-11-01T12:00:00Z",
) -> dict:
    """Helper to insert a job directly in reconciliation_required status."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO retraining_jobs (
                job_id, created_at, status, is_active, dataset_id,
                trigger_source, candidate_version, candidate_run_id,
                previous_champion_version, promotion_started_at, payload
            ) VALUES (
                ?, datetime('now'), 'reconciliation_required', 1, 'ds-test',
                'manual', ?, ?, ?, ?, '{}'
            );
            """,
            (
                job_id,
                candidate_version,
                candidate_run_id,
                previous_champion_version,
                promotion_started_at,
            ),
        )
    return get_job(database_path=db_path, job_id=job_id)  # type: ignore


def _create_mock_mlflow_client(
    alias_version: str | None = None,
    candidate_version: str | None = "2",
    candidate_run_id: str | None = "run-cand-2",
    model_name: str = "churn-classifier",
    alias: str = "champion",
) -> MagicMock:
    """Create a mock MlflowClient simulating registry state."""
    client = MagicMock()

    # Alias retrieval
    if alias_version is not None:
        alias_run_id = (
            candidate_run_id
            if str(alias_version) == str(candidate_version)
            else f"run-{alias_version}"
        )
        alias_obj = type(
            "VersionStub",
            (),
            {
                "version": str(alias_version),
                "run_id": alias_run_id,
                "name": model_name,
            },
        )()
        client.get_model_version_by_alias.return_value = alias_obj
    else:
        client.get_model_version_by_alias.side_effect = MlflowException(
            f"Alias '{alias}' not found",
            error_code=ErrorCode.RESOURCE_DOES_NOT_EXIST,
        )

    # Version retrieval
    def _get_model_version(m_name: str, ver: str):
        if m_name != model_name:
            raise MlflowException(
                "Model not found", error_code=ErrorCode.RESOURCE_DOES_NOT_EXIST
            )
        run_id = candidate_run_id if ver == candidate_version else f"run-{ver}"
        return type(
            "VersionStub",
            (),
            {"version": str(ver), "run_id": run_id, "name": m_name},
        )()

    client.get_model_version.side_effect = _get_model_version
    return client


# 1. Process locking safety
def test_worker_lock_prevents_operator_reconciliation(test_db: Path, lock_path: Path):
    """Worker execution lock must prevent operator reconciliation."""
    _create_quarantined_job(test_db, "job-lock-1")

    with worker_execution_lock(lock_path):
        with pytest.raises(MaintenanceLockConflictError):
            reconcile_job(
                job_id="job-lock-1",
                selected_resolution="resolved_succeeded",
                operator="ops",
                reason="Testing lock conflict",
                database_path=test_db,
                lock_path=lock_path,
            )

    # Job remains quarantined
    job = get_job(database_path=test_db, job_id="job-lock-1")
    assert job["status"] == "reconciliation_required"
    assert job["is_active"] == 1


def test_maintenance_lock_prevents_worker_startup(test_db: Path, lock_path: Path):
    """Operator maintenance lock must prevent worker startup."""
    with operator_maintenance_lock(lock_path):
        worker = RetrainingWorker(
            database_path=test_db,
            lock_path=lock_path,
        )
        with pytest.raises(WorkerLockConflictError):
            worker.acquire_lock()


# 2. Successful resolution paths
def test_successful_resolution_alias_already_points_to_candidate(
    test_db: Path, lock_path: Path
):
    """Operator resolves success when approved audit and alias matches candidate."""
    jid = "job-succ-alias-match"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="2",
        candidate_run_id="run-2",
        previous_champion_version="1",
    )

    record_promotion_audit(
        {
            "job_id": jid,
            "model_name": "churn-classifier",
            "candidate_version": "2",
            "candidate_run_id": "run-2",
            "champion_version": "1",
            "approved": True,
            "reason": "gate_passed",
            "candidate_f1": 0.88,
            "champion_f1": 0.80,
            "margin": 0.01,
            "benchmark_version": "v1",
            "benchmark_digest": "digest-1",
        },
        database_path=test_db,
    )

    mock_client = _create_mock_mlflow_client(
        alias_version="2",
        candidate_version="2",
        candidate_run_id="run-2",
    )

    finalized = reconcile_job(
        job_id=jid,
        selected_resolution="resolved_succeeded",
        operator="ops_lead",
        reason="Verified candidate is live champion",
        database_path=test_db,
        lock_path=lock_path,
        client=mock_client,
    )

    assert finalized["status"] == "succeeded"
    assert finalized["is_active"] is None

    audit = get_operator_resolution_audit(jid, database_path=test_db)
    assert audit is not None
    assert audit["selected_resolution"] == "resolved_succeeded"
    assert audit["rollback_performed"] == 0
    assert audit["evidence_payload"]["action"] == "resolved_succeeded_alias_verified"
    # No alias mutation was performed
    mock_client.set_registered_model_alias.assert_not_called()


def test_successful_resolution_explicit_alias_update_and_verification(
    test_db: Path, lock_path: Path
):
    """Operator updates alias when approved audit exists and alias is at champion."""
    jid = "job-succ-update-alias"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="2",
        candidate_run_id="run-2",
        previous_champion_version="1",
    )

    record_promotion_audit(
        {
            "job_id": jid,
            "model_name": "churn-classifier",
            "candidate_version": "2",
            "candidate_run_id": "run-2",
            "champion_version": "1",
            "approved": True,
            "reason": "gate_passed",
            "candidate_f1": 0.88,
            "champion_f1": 0.80,
            "margin": 0.01,
            "benchmark_version": "v1",
            "benchmark_digest": "digest-1",
        },
        database_path=test_db,
    )

    # Initial state: alias points to champion "1"
    mock_client = _create_mock_mlflow_client(
        alias_version="1",
        candidate_version="2",
        candidate_run_id="run-2",
    )

    # Without update_alias flag -> refused
    with pytest.raises(
        SafetyInvariantViolationError, match="Explicit '--update-alias' required"
    ):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_succeeded",
            operator="ops_lead",
            reason="Promoting candidate",
            database_path=test_db,
            lock_path=lock_path,
            update_alias=False,
            client=mock_client,
        )

    # Simulate alias change on mutation
    def _set_alias(name: str, alias: str, ver: str):
        mock_client.get_model_version_by_alias.return_value = type(
            "VersionStub",
            (),
            {"version": str(ver), "run_id": f"run-{ver}", "name": name},
        )()

    mock_client.set_registered_model_alias.side_effect = _set_alias

    finalized = reconcile_job(
        job_id=jid,
        selected_resolution="resolved_succeeded",
        operator="ops_lead",
        reason="Authorized candidate alias promotion",
        database_path=test_db,
        lock_path=lock_path,
        update_alias=True,
        client=mock_client,
    )

    assert finalized["status"] == "succeeded"
    mock_client.set_registered_model_alias.assert_called_once_with(
        "churn-classifier", "champion", "2"
    )

    audit = get_operator_resolution_audit(jid, database_path=test_db)
    assert audit is not None
    assert audit["selected_resolution"] == "resolved_succeeded"
    assert audit["rollback_performed"] == 0
    assert audit["evidence_payload"]["action"] == "resolved_succeeded_alias_updated"


# 3. Evidence inspection errors
def test_missing_audit_evidence_leaves_job_quarantined(test_db: Path, lock_path: Path):
    """Missing promotion audit prevents successful resolution."""
    jid = "job-missing-audit"
    _create_quarantined_job(test_db, job_id=jid)

    mock_client = _create_mock_mlflow_client(alias_version="2")

    with pytest.raises(
        EvidenceVerificationError, match="promotion audit is not approved"
    ):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_succeeded",
            operator="ops",
            reason="Missing audit test",
            database_path=test_db,
            lock_path=lock_path,
            client=mock_client,
        )

    job = get_job(database_path=test_db, job_id=jid)
    assert job["status"] == "reconciliation_required"
    assert job["is_active"] == 1


def test_duplicate_audit_evidence_leaves_job_quarantined(
    test_db: Path, lock_path: Path
):
    """Duplicate promotion audit records prevent resolution."""
    jid = "job-dup-audit"
    _create_quarantined_job(test_db, job_id=jid)

    # Insert two records for the same job_id directly
    with sqlite3.connect(test_db) as conn:
        conn.execute(
            """
            INSERT INTO promotion_audit (
                audit_id, timestamp, model_name, candidate_version,
                candidate_run_id, champion_version, approved, reason,
                candidate_f1, champion_f1, benchmark_version,
                benchmark_digest, margin, job_id, payload
            ) VALUES
            (
                'aud-1', datetime('now', '-1 minute'), 'churn-classifier',
                '2', 'run-2', '1', 1, 'ok', 0.85, 0.80, 'v1', 'dig', 0.01,
                ?, '{}'
            ),
            (
                'aud-2', datetime('now'), 'churn-classifier',
                '2', 'run-2', '1', 1, 'ok', 0.85, 0.80, 'v1', 'dig', 0.01,
                ?, '{}'
            );
            """,
            (jid, jid),
        )

    mock_client = _create_mock_mlflow_client(alias_version="2")

    with pytest.raises(
        EvidenceVerificationError, match="Duplicate or corrupted promotion audit"
    ):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_succeeded",
            operator="ops",
            reason="Duplicate audit test",
            database_path=test_db,
            lock_path=lock_path,
            client=mock_client,
        )

    job = get_job(database_path=test_db, job_id=jid)
    assert job["status"] == "reconciliation_required"


def test_unexpected_alias_version_leaves_job_quarantined(
    test_db: Path, lock_path: Path
):
    """Unexpected alias version prevents both success and failure resolution."""
    jid = "job-unexpected-alias"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="2",
        previous_champion_version="1",
    )

    record_promotion_audit(
        {
            "job_id": jid,
            "model_name": "churn-classifier",
            "candidate_version": "2",
            "approved": True,
            "reason": "gate_passed",
            "candidate_f1": 0.85,
            "benchmark_version": "v1",
            "benchmark_digest": "dig",
        },
        database_path=test_db,
    )

    # Remote alias points to unexpected version "99"
    mock_client = _create_mock_mlflow_client(alias_version="99")

    with pytest.raises(SafetyInvariantViolationError, match="unexpected version '99'"):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_succeeded",
            operator="ops",
            reason="Testing unexpected version",
            database_path=test_db,
            lock_path=lock_path,
            client=mock_client,
        )

    with pytest.raises(SafetyInvariantViolationError, match="unexpected version '99'"):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_failed",
            operator="ops",
            reason="Testing unexpected version",
            database_path=test_db,
            lock_path=lock_path,
            client=mock_client,
        )

    job = get_job(database_path=test_db, job_id=jid)
    assert job["status"] == "reconciliation_required"


def test_mlflow_unavailable_or_auth_failure_leaves_job_quarantined(
    test_db: Path, lock_path: Path
):
    """MLflow server error or authentication failure halts resolution safely."""
    jid = "job-mlflow-down"
    _create_quarantined_job(test_db, job_id=jid)

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.side_effect = MlflowException(
        "500 Internal Server Error: Database locked",
        error_code=ErrorCode.INTERNAL_ERROR,
    )

    with pytest.raises(MLflowOperationError, match="Cannot verify remote MLflow alias"):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_failed",
            operator="ops",
            reason="Testing MLflow failure",
            database_path=test_db,
            lock_path=lock_path,
            client=mock_client,
        )

    job = get_job(database_path=test_db, job_id=jid)
    assert job["status"] == "reconciliation_required"


# 4. Failed resolution paths
def test_failed_resolution_alias_at_previous_champion_no_rollback(
    test_db: Path, lock_path: Path
):
    """Operator resolves failed cleanly when alias remained at previous champion."""
    jid = "job-fail-no-rollback"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="2",
        previous_champion_version="1",
    )

    mock_client = _create_mock_mlflow_client(alias_version="1")

    finalized = reconcile_job(
        job_id=jid,
        selected_resolution="resolved_failed",
        operator="ops_lead",
        reason="Retraining failed quality gates",
        database_path=test_db,
        lock_path=lock_path,
        client=mock_client,
    )

    assert finalized["status"] == "failed"
    assert finalized["is_active"] is None

    audit = get_operator_resolution_audit(jid, database_path=test_db)
    assert audit is not None
    assert audit["selected_resolution"] == "resolved_failed"
    assert audit["rollback_performed"] == 0
    assert audit["evidence_payload"]["action"] == "resolved_failed_alias_unchanged"
    mock_client.set_registered_model_alias.assert_not_called()


def test_failed_resolution_explicit_rollback_and_verification(
    test_db: Path, lock_path: Path
):
    """Operator rolls back alias to previous champion with explicit authorization."""
    jid = "job-fail-rollback"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="2",
        previous_champion_version="1",
    )

    # Alias points to candidate "2"
    mock_client = _create_mock_mlflow_client(
        alias_version="2",
        candidate_version="2",
    )

    # Without rollback_alias flag -> refused
    with pytest.raises(
        SafetyInvariantViolationError, match="Explicit '--rollback-alias' required"
    ):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_failed",
            operator="ops_lead",
            reason="Candidate model corrupted",
            database_path=test_db,
            lock_path=lock_path,
            rollback_alias=False,
            client=mock_client,
        )

    # Simulate rollback effect
    def _rollback(name: str, alias: str, ver: str):
        mock_client.get_model_version_by_alias.return_value = type(
            "VersionStub",
            (),
            {"version": str(ver), "run_id": f"run-{ver}", "name": name},
        )()

    mock_client.set_registered_model_alias.side_effect = _rollback

    finalized = reconcile_job(
        job_id=jid,
        selected_resolution="resolved_failed",
        operator="ops_lead",
        reason="Candidate model corrupted; restoring champion 1",
        database_path=test_db,
        lock_path=lock_path,
        rollback_alias=True,
        client=mock_client,
    )

    assert finalized["status"] == "failed"
    mock_client.set_registered_model_alias.assert_called_once_with(
        "churn-classifier", "champion", "1"
    )

    audit = get_operator_resolution_audit(jid, database_path=test_db)
    assert audit is not None
    assert audit["selected_resolution"] == "resolved_failed"
    assert audit["rollback_performed"] == 1
    assert audit["evidence_payload"]["action"] == "resolved_failed_rollback_performed"


# 5. Bootstrap case (no previous champion)
def test_bootstrap_alias_deletion_with_both_outcomes(test_db: Path, lock_path: Path):
    """Bootstrap failed resolution verifies deletion and already-absent outcomes."""
    # Case A: Alias was set to candidate "1", must be deleted and verified
    jid_a = "job-bootstrap-delete"
    _create_quarantined_job(
        test_db,
        job_id=jid_a,
        candidate_version="1",
        candidate_run_id="run-cand-1",
        previous_champion_version=None,
    )

    mock_client_a = _create_mock_mlflow_client(
        alias_version="1",
        candidate_version="1",
        candidate_run_id="run-cand-1",
    )

    # Missing flags
    with pytest.raises(
        SafetyInvariantViolationError, match="--force-bootstrap-removal"
    ):
        reconcile_job(
            job_id=jid_a,
            selected_resolution="resolved_failed",
            operator="ops",
            reason="Bootstrap candidate failed",
            database_path=test_db,
            lock_path=lock_path,
            delete_alias=True,
            force_bootstrap_removal=False,
            client=mock_client_a,
        )

    # Simulate deletion
    def _delete(name: str, alias: str):
        mock_client_a.get_model_version_by_alias.side_effect = MlflowException(
            "Alias deleted", error_code=ErrorCode.RESOURCE_DOES_NOT_EXIST
        )

    mock_client_a.delete_registered_model_alias.side_effect = _delete

    finalized_a = reconcile_job(
        job_id=jid_a,
        selected_resolution="resolved_failed",
        operator="ops_lead",
        reason="Bootstrap model failed post-training audit",
        database_path=test_db,
        lock_path=lock_path,
        delete_alias=True,
        force_bootstrap_removal=True,
        client=mock_client_a,
    )
    assert finalized_a["status"] == "failed"
    mock_client_a.delete_registered_model_alias.assert_called_once_with(
        "churn-classifier", "champion"
    )

    audit_a = get_operator_resolution_audit(jid_a, database_path=test_db)
    assert audit_a is not None
    assert audit_a["evidence_payload"]["action"] == "bootstrap_alias_deleted"
    assert audit_a["rollback_performed"] == 0

    # Case B: Alias was already absent
    jid_b = "job-bootstrap-absent"
    _create_quarantined_job(
        test_db,
        job_id=jid_b,
        candidate_version="1",
        candidate_run_id="run-cand-1",
        previous_champion_version=None,
    )

    mock_client_b = _create_mock_mlflow_client(
        alias_version=None,
        candidate_version="1",
        candidate_run_id="run-cand-1",
    )

    finalized_b = reconcile_job(
        job_id=jid_b,
        selected_resolution="resolved_failed",
        operator="ops_lead",
        reason="Bootstrap model failed before alias assignment",
        database_path=test_db,
        lock_path=lock_path,
        client=mock_client_b,
    )
    assert finalized_b["status"] == "failed"

    audit_b = get_operator_resolution_audit(jid_b, database_path=test_db)
    assert audit_b is not None
    assert audit_b["evidence_payload"]["action"] == "bootstrap_alias_already_absent"
    assert audit_b["rollback_performed"] == 0
    mock_client_b.delete_registered_model_alias.assert_not_called()


# 6. Failure injection and transactional safety
def test_alias_mutation_failure_leaves_quarantined(test_db: Path, lock_path: Path):
    """If MLflow alias update fails, job remains quarantined."""
    jid = "job-mutation-err"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="2",
        previous_champion_version="1",
    )

    record_promotion_audit(
        {
            "job_id": jid,
            "model_name": "churn-classifier",
            "candidate_version": "2",
            "approved": True,
            "reason": "gate_passed",
            "candidate_f1": 0.88,
            "benchmark_version": "v1",
            "benchmark_digest": "dig",
        },
        database_path=test_db,
    )

    mock_client = _create_mock_mlflow_client(
        alias_version="1",
        candidate_version="2",
    )
    mock_client.set_registered_model_alias.side_effect = MlflowException(
        "Registry write failure", error_code=ErrorCode.INTERNAL_ERROR
    )

    with pytest.raises(MLflowOperationError, match="Failed to update alias 'champion'"):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_succeeded",
            operator="ops",
            reason="Update alias",
            database_path=test_db,
            lock_path=lock_path,
            update_alias=True,
            client=mock_client,
        )

    job = get_job(database_path=test_db, job_id=jid)
    assert job["status"] == "reconciliation_required"


def test_failure_injection_verification_fails_no_compensating_mutation(
    test_db: Path, lock_path: Path
):
    """If verification fails after alias mutation, do not attempt rollback."""
    jid = "job-verif-fail"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="2",
        previous_champion_version="1",
    )

    record_promotion_audit(
        {
            "job_id": jid,
            "model_name": "churn-classifier",
            "candidate_version": "2",
            "approved": True,
            "reason": "gate_passed",
            "candidate_f1": 0.88,
            "benchmark_version": "v1",
            "benchmark_digest": "dig",
        },
        database_path=test_db,
    )

    mock_client = _create_mock_mlflow_client(
        alias_version="1",
        candidate_version="2",
    )

    # First call to get_model_version_by_alias returns champion "1"
    # Second call (verification) raises unexpected exception
    mock_client.get_model_version_by_alias.side_effect = [
        type(
            "VersionStub",
            (),
            {"version": "1", "run_id": "run-1", "name": "churn-classifier"},
        )(),
        MlflowException(
            "503 Service Unavailable", error_code=ErrorCode.TEMPORARILY_UNAVAILABLE
        ),
    ]

    with pytest.raises(
        MLflowOperationError, match="Failed to verify alias after update"
    ):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_succeeded",
            operator="ops",
            reason="Update alias",
            database_path=test_db,
            lock_path=lock_path,
            update_alias=True,
            client=mock_client,
        )

    # Mutation was called once; NO compensating rollback was attempted
    mock_client.set_registered_model_alias.assert_called_once_with(
        "churn-classifier", "champion", "2"
    )

    # Job is still quarantined
    job = get_job(database_path=test_db, job_id=jid)
    assert job["status"] == "reconciliation_required"
    assert job["is_active"] == 1


def test_failure_injection_db_finalization_fails_no_compensating_mutation(
    test_db: Path, lock_path: Path
):
    """If DB finalization fails, transaction rolls back with no rollback mutation."""
    jid = "job-db-fail"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="2",
        previous_champion_version="1",
    )

    record_promotion_audit(
        {
            "job_id": jid,
            "model_name": "churn-classifier",
            "candidate_version": "2",
            "approved": True,
            "reason": "gate_passed",
            "candidate_f1": 0.88,
            "benchmark_version": "v1",
            "benchmark_digest": "dig",
        },
        database_path=test_db,
    )

    mock_client = _create_mock_mlflow_client(
        alias_version="2",
        candidate_version="2",
    )

    # Patch finalize_operator_resolution_atomic to simulate disk failure
    with patch(
        "backend.app.ml.reconciliation.finalize_operator_resolution_atomic",
        side_effect=sqlite3.OperationalError("disk I/O error"),
    ):
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            reconcile_job(
                job_id=jid,
                selected_resolution="resolved_succeeded",
                operator="ops",
                reason="Resolving success",
                database_path=test_db,
                lock_path=lock_path,
                client=mock_client,
            )

    # Job remains in reconciliation_required and active
    job = get_job(database_path=test_db, job_id=jid)
    assert job["status"] == "reconciliation_required"
    assert job["is_active"] == 1


# 7. Invalid inputs and state transition validation
def test_invalid_resolution_and_rollback_validation(test_db: Path, lock_path: Path):
    """Invalid resolutions and non-integer rollback flags are strictly rejected."""
    jid = "job-invalid-res"
    _create_quarantined_job(test_db, job_id=jid)

    mock_client = _create_mock_mlflow_client(alias_version="1")

    with pytest.raises(ValueError, match="Invalid selected_resolution"):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_cancelled",
            operator="ops",
            reason="Invalid resolution",
            database_path=test_db,
            lock_path=lock_path,
            client=mock_client,
        )

    with pytest.raises(ValueError, match="operator must be a non-empty string"):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_failed",
            operator="",
            reason="No operator",
            database_path=test_db,
            lock_path=lock_path,
            client=mock_client,
        )

    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_failed",
            operator="ops",
            reason="   ",
            database_path=test_db,
            lock_path=lock_path,
            client=mock_client,
        )


def test_repeated_resolution_attempts_rejected(test_db: Path, lock_path: Path):
    """A job already resolved cannot be resolved again."""
    jid = "job-repeat"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="2",
        previous_champion_version="1",
    )

    mock_client = _create_mock_mlflow_client(alias_version="1")

    # Resolve failed once
    reconcile_job(
        job_id=jid,
        selected_resolution="resolved_failed",
        operator="ops",
        reason="Initial failure resolution",
        database_path=test_db,
        lock_path=lock_path,
        client=mock_client,
    )

    # Attempt to resolve again
    with pytest.raises(JobNotQuarantinedError):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_failed",
            operator="ops",
            reason="Second resolution",
            database_path=test_db,
            lock_path=lock_path,
            client=mock_client,
        )


# 8. Read-only inspection tests
def test_inspect_quarantined_job_read_only(test_db: Path):
    """inspect_quarantined_job reads state without altering job or audit data."""
    jid = "job-inspect-ro"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="2",
        candidate_run_id="run-2",
        previous_champion_version="1",
    )

    record_promotion_audit(
        {
            "job_id": jid,
            "model_name": "churn-classifier",
            "candidate_version": "2",
            "candidate_run_id": "run-2",
            "champion_version": "1",
            "approved": True,
            "reason": "gate_passed",
            "candidate_f1": 0.88,
            "benchmark_version": "v1",
            "benchmark_digest": "dig",
        },
        database_path=test_db,
    )

    mock_client = _create_mock_mlflow_client(
        alias_version="2", candidate_run_id="run-2"
    )

    evidence = inspect_quarantined_job(
        job_id=jid,
        database_path=test_db,
        client=mock_client,
    )

    assert evidence.job_id == jid
    assert evidence.job_status == "reconciliation_required"
    assert evidence.is_active == 1
    assert evidence.candidate_version == "2"
    assert evidence.observed_alias_version == "2"
    assert evidence.recommended_action == "ready_for_success_resolution"

    # Confirm job in database is untouched
    job = get_job(database_path=test_db, job_id=jid)
    assert job["status"] == "reconciliation_required"
    assert job["is_active"] == 1


# 9. CLI integration tests
def test_cli_inspect_and_resolve(test_db: Path, lock_path: Path, capsys):
    """CLI inspect and resolve subcommands execute according to safety rules."""
    jid = "job-cli-test"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="2",
        previous_champion_version="1",
    )

    record_promotion_audit(
        {
            "job_id": jid,
            "model_name": "churn-classifier",
            "candidate_version": "2",
            "champion_version": "1",
            "approved": False,
            "reason": "gate_failed",
            "candidate_f1": 0.65,
            "benchmark_version": "v1",
            "benchmark_digest": "dig",
        },
        database_path=test_db,
    )

    mock_client = _create_mock_mlflow_client(alias_version="1")

    with patch("backend.app.ml.reconciliation.MlflowClient", return_value=mock_client):
        # 1. Inspect
        ret = main(["inspect", "--job-id", jid, "--db-path", str(test_db)])
        assert ret == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["job_id"] == jid
        assert data["recommended_action"] == "ready_for_failure_resolution"

        # 2. Resolve without --confirm -> aborts with code 1
        ret_no_confirm = main(
            [
                "resolve",
                "--job-id",
                jid,
                "--resolution",
                "resolved_failed",
                "--operator",
                "sre_lead",
                "--reason",
                "Failing candidate",
                "--db-path",
                str(test_db),
                "--lock-path",
                str(lock_path),
            ]
        )
        assert ret_no_confirm == 1
        assert "requires '--confirm' flag" in capsys.readouterr().err

        # 3. Resolve with --confirm -> succeeds with code 0
        ret_confirm = main(
            [
                "resolve",
                "--job-id",
                jid,
                "--resolution",
                "resolved_failed",
                "--operator",
                "sre_lead",
                "--reason",
                "Failing candidate",
                "--db-path",
                str(test_db),
                "--lock-path",
                str(lock_path),
                "--confirm",
            ]
        )
        assert ret_confirm == 0
        assert (
            f"Successfully resolved job '{jid}' to status 'failed'"
            in capsys.readouterr().out
        )


def test_missing_previous_champion_deletion_not_explicitly_authorized(
    test_db: Path, lock_path: Path
):
    """Attempting bootstrap deletion without both authorization flags fails."""
    jid = "job-bootstrap-noauth"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="1",
        previous_champion_version=None,
    )

    mock_client = _create_mock_mlflow_client(
        alias_version="1",
        candidate_version="1",
    )

    # 1. No flags at all
    with pytest.raises(SafetyInvariantViolationError, match="--delete-alias"):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_failed",
            operator="ops",
            reason="Testing no flags",
            database_path=test_db,
            lock_path=lock_path,
            client=mock_client,
        )

    # 2. Only delete_alias
    with pytest.raises(
        SafetyInvariantViolationError, match="--force-bootstrap-removal"
    ):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_failed",
            operator="ops",
            reason="Testing missing force flag",
            database_path=test_db,
            lock_path=lock_path,
            delete_alias=True,
            force_bootstrap_removal=False,
            client=mock_client,
        )

    # 3. Only force_bootstrap_removal
    with pytest.raises(SafetyInvariantViolationError, match="--delete-alias"):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_failed",
            operator="ops",
            reason="Testing missing delete flag",
            database_path=test_db,
            lock_path=lock_path,
            delete_alias=False,
            force_bootstrap_removal=True,
            client=mock_client,
        )

    job = get_job(database_path=test_db, job_id=jid)
    assert job["status"] == "reconciliation_required"


def test_job_state_changing_before_finalization(test_db: Path, lock_path: Path):
    """If job status changes before finalization, transaction fails and rolls back."""
    jid = "job-race-status"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="2",
        previous_champion_version="1",
    )

    record_promotion_audit(
        {
            "job_id": jid,
            "model_name": "churn-classifier",
            "candidate_version": "2",
            "approved": True,
            "reason": "gate_passed",
            "candidate_f1": 0.88,
            "benchmark_version": "v1",
            "benchmark_digest": "dig",
        },
        database_path=test_db,
    )

    mock_client = _create_mock_mlflow_client(
        alias_version="2",
        candidate_version="2",
    )

    # Hook into client to modify job status in SQLite just before finalization
    def _tamper_job(*args, **kwargs):
        with sqlite3.connect(test_db) as conn:
            conn.execute(
                "UPDATE retraining_jobs SET status = 'running' WHERE job_id = ?;",
                (jid,),
            )
            return type(
                "VersionStub",
                (),
                {"version": "2", "run_id": "run-cand-2", "name": "churn-classifier"},
            )()

    mock_client.get_model_version.side_effect = _tamper_job

    with pytest.raises(RuntimeError, match="Job finalization failed"):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_succeeded",
            operator="ops",
            reason="Testing state change during finalization",
            database_path=test_db,
            lock_path=lock_path,
            client=mock_client,
        )

    # Operator resolution audit must be empty
    audit = get_operator_resolution_audit(jid, database_path=test_db)
    assert audit is None


def test_invalid_rollback_flag_types_rejected():
    """validate_operator_resolution_inputs strictly rejects booleans and floats."""
    from backend.app.ml.migrations import validate_operator_resolution_inputs

    # Valid
    validate_operator_resolution_inputs("resolved_succeeded", 0)
    validate_operator_resolution_inputs("resolved_failed", 1)

    # Booleans rejected
    with pytest.raises(ValueError, match="Invalid rollback_performed"):
        validate_operator_resolution_inputs("resolved_succeeded", True)  # type: ignore

    with pytest.raises(ValueError, match="Invalid rollback_performed"):
        validate_operator_resolution_inputs("resolved_failed", False)  # type: ignore

    # Floats / other ints rejected
    with pytest.raises(ValueError, match="Invalid rollback_performed"):
        validate_operator_resolution_inputs("resolved_failed", 1.0)  # type: ignore

    with pytest.raises(ValueError, match="Invalid rollback_performed"):
        validate_operator_resolution_inputs("resolved_failed", 2)


def test_candidate_model_run_id_mismatch_blocks_rollback(
    test_db: Path, lock_path: Path
):
    """Candidate model run ID mismatch in MLflow halts rollback before mutation."""
    jid = "job-cand-run-mismatch"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="2",
        candidate_run_id="run-expected-2",
        previous_champion_version="1",
    )

    mock_client = MagicMock()
    # Live alias points to "2"
    mock_client.get_model_version_by_alias.return_value = type(
        "VersionStub",
        (),
        {"version": "2", "run_id": "run-expected-2", "name": "churn-classifier"},
    )()

    # Candidate version in MLflow has mismatched run_id
    mock_client.get_model_version.return_value = type(
        "VersionStub",
        (),
        {"version": "2", "run_id": "run-tampered-2", "name": "churn-classifier"},
    )()

    with pytest.raises(
        EvidenceVerificationError, match="Candidate model run_id mismatch in MLflow"
    ):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_failed",
            operator="ops",
            reason="Testing candidate run_id mismatch",
            database_path=test_db,
            lock_path=lock_path,
            rollback_alias=True,
            client=mock_client,
        )

    # Invariants: No mutation occurred; job remains quarantined
    mock_client.set_registered_model_alias.assert_not_called()
    job = get_job(database_path=test_db, job_id=jid)
    assert job["status"] == "reconciliation_required"
    assert job["is_active"] == 1


def test_candidate_model_run_id_mismatch_blocks_bootstrap_deletion(
    test_db: Path, lock_path: Path
):
    """Candidate run ID mismatch blocks bootstrap deletion before alias delete call."""
    jid = "job-boot-cand-mismatch"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="1",
        candidate_run_id="run-expected-1",
        previous_champion_version=None,
    )

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = type(
        "VersionStub",
        (),
        {"version": "1", "run_id": "run-expected-1", "name": "churn-classifier"},
    )()
    mock_client.get_model_version.return_value = type(
        "VersionStub",
        (),
        {"version": "1", "run_id": "run-unrelated-1", "name": "churn-classifier"},
    )()

    with pytest.raises(
        EvidenceVerificationError, match="Candidate model run_id mismatch in MLflow"
    ):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_failed",
            operator="ops",
            reason="Testing candidate run mismatch in bootstrap",
            database_path=test_db,
            lock_path=lock_path,
            delete_alias=True,
            force_bootstrap_removal=True,
            client=mock_client,
        )

    mock_client.delete_registered_model_alias.assert_not_called()
    job = get_job(database_path=test_db, job_id=jid)
    assert job["status"] == "reconciliation_required"
    assert job["is_active"] == 1


def test_alias_resolved_run_id_mismatch_blocks_mutation(test_db: Path, lock_path: Path):
    """Live alias pointing to candidate with mismatched run ID halts mutation."""
    jid = "job-alias-run-mismatch"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="2",
        candidate_run_id="run-cand-correct",
        previous_champion_version="1",
    )

    mock_client = MagicMock()
    # Candidate version model has correct run ID
    mock_client.get_model_version.return_value = type(
        "VersionStub",
        (),
        {"version": "2", "run_id": "run-cand-correct", "name": "churn-classifier"},
    )()

    # Live alias points to version "2" but has different run ID
    mock_client.get_model_version_by_alias.return_value = type(
        "VersionStub",
        (),
        {"version": "2", "run_id": "run-stale-different", "name": "churn-classifier"},
    )()

    with pytest.raises(
        EvidenceVerificationError,
        match="Live alias 'champion' points to version '2' with run_id",
    ):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_failed",
            operator="ops",
            reason="Testing alias run mismatch",
            database_path=test_db,
            lock_path=lock_path,
            rollback_alias=True,
            client=mock_client,
        )

    mock_client.set_registered_model_alias.assert_not_called()
    job = get_job(database_path=test_db, job_id=jid)
    assert job["status"] == "reconciliation_required"
    assert job["is_active"] == 1


def test_previous_champion_model_identity_mismatch_blocks_rollback(
    test_db: Path, lock_path: Path
):
    """Previous champion belonging to different model name blocks rollback."""
    jid = "job-champ-model-mismatch"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="2",
        candidate_run_id="run-cand-2",
        previous_champion_version="1",
    )

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = type(
        "VersionStub",
        (),
        {"version": "2", "run_id": "run-cand-2", "name": "churn-classifier"},
    )()

    def _get_ver(m_name: str, ver: str):
        if ver == "2":
            return type(
                "VersionStub",
                (),
                {"version": "2", "run_id": "run-cand-2", "name": "churn-classifier"},
            )()
        if ver == "1":
            # Previous champion has mismatched model name
            return type(
                "VersionStub",
                (),
                {"version": "1", "run_id": "run-champ-1", "name": "foreign-classifier"},
            )()
        raise ValueError(f"Unknown version {ver}")

    mock_client.get_model_version.side_effect = _get_ver

    with pytest.raises(
        EvidenceVerificationError, match="Previous champion model name mismatch"
    ):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_failed",
            operator="ops",
            reason="Testing champion model name mismatch",
            database_path=test_db,
            lock_path=lock_path,
            rollback_alias=True,
            client=mock_client,
        )

    mock_client.set_registered_model_alias.assert_not_called()
    job = get_job(database_path=test_db, job_id=jid)
    assert job["status"] == "reconciliation_required"
    assert job["is_active"] == 1


def test_missing_job_candidate_run_id_with_conflicting_audit_run_id_blocks_mutation(
    test_db: Path, lock_path: Path
):
    """When job lacks candidate_run_id, conflicting audit run_id blocks mutation."""
    jid = "job-missing-run-id-conflicting-audit"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="2",
        candidate_run_id=None,  # Missing candidate_run_id in job
        previous_champion_version="1",
    )

    record_promotion_audit(
        {
            "job_id": jid,
            "model_name": "churn-classifier",
            "candidate_version": "2",
            "candidate_run_id": "run-audit-123",
            "champion_version": "1",
            "approved": True,
            "reason": "gate_passed",
            "candidate_f1": 0.90,
            "benchmark_version": "v1",
            "benchmark_digest": "dig",
        },
        database_path=test_db,
    )

    # MLflow returns a different run ID for candidate version 2
    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = type(
        "VersionStub",
        (),
        {"version": "1", "run_id": "run-champ-1", "name": "churn-classifier"},
    )()
    mock_client.get_model_version.return_value = type(
        "VersionStub",
        (),
        {
            "version": "2",
            "run_id": "run-mlflow-conflicting",
            "name": "churn-classifier",
        },
    )()

    # 1. Inspect must report quarantine due to identity mismatch
    evidence = inspect_quarantined_job(
        job_id=jid,
        database_path=test_db,
        client=mock_client,
    )
    assert evidence.recommended_action == "quarantine_due_to_identity_mismatch"
    assert "audit_mlflow_run_id_mismatch" in evidence.details

    # 2. Reconcile must fail and perform NO alias mutations
    with pytest.raises(
        EvidenceVerificationError, match="Candidate model run_id mismatch in MLflow"
    ):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_succeeded",
            operator="ops_lead",
            reason="Resolving success with missing job run_id but conflicting audit",
            database_path=test_db,
            lock_path=lock_path,
            update_alias=True,
            client=mock_client,
        )

    mock_client.set_registered_model_alias.assert_not_called()
    mock_client.delete_registered_model_alias.assert_not_called()

    # Job remains strictly quarantined
    job = get_job(database_path=test_db, job_id=jid)
    assert job["status"] == "reconciliation_required"
    assert job["is_active"] == 1


def test_missing_job_candidate_run_id_cross_validates_audit_and_alias_run_id(
    test_db: Path, lock_path: Path
):
    """When job lacks candidate_run_id, audit run_id is cross-validated
    against live alias."""
    jid = "job-missing-run-id-alias-audit-mismatch"
    _create_quarantined_job(
        test_db,
        job_id=jid,
        candidate_version="2",
        candidate_run_id=None,  # Missing candidate_run_id in job
        previous_champion_version="1",
    )

    record_promotion_audit(
        {
            "job_id": jid,
            "model_name": "churn-classifier",
            "candidate_version": "2",
            "candidate_run_id": "run-audit-correct",
            "champion_version": "1",
            "approved": False,
            "reason": "gate_failed",
            "candidate_f1": 0.60,
            "benchmark_version": "v1",
            "benchmark_digest": "dig",
        },
        database_path=test_db,
    )

    # Candidate object in MLflow matches the audit run ID
    # BUT live alias points to candidate version 2 with a contradictory/rogue run ID
    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = type(
        "VersionStub",
        (),
        {
            "version": "2",
            "run_id": "run-rogue-mismatched",
            "name": "churn-classifier",
        },
    )()
    mock_client.get_model_version.return_value = type(
        "VersionStub",
        (),
        {"version": "2", "run_id": "run-audit-correct", "name": "churn-classifier"},
    )()

    # 1. Inspect reports identity mismatch
    evidence = inspect_quarantined_job(
        job_id=jid,
        database_path=test_db,
        client=mock_client,
    )
    assert evidence.recommended_action == "quarantine_due_to_identity_mismatch"

    # 2. Reconcile with rollback must fail before any alias mutation
    with pytest.raises(
        EvidenceVerificationError, match="expected candidate run_id 'run-audit-correct'"
    ):
        reconcile_job(
            job_id=jid,
            selected_resolution="resolved_failed",
            operator="ops_lead",
            reason="Resolving failed candidate with contradictory alias run_id",
            database_path=test_db,
            lock_path=lock_path,
            rollback_alias=True,
            client=mock_client,
        )

    mock_client.set_registered_model_alias.assert_not_called()
    mock_client.delete_registered_model_alias.assert_not_called()

    job = get_job(database_path=test_db, job_id=jid)
    assert job["status"] == "reconciliation_required"
    assert job["is_active"] == 1
