"""Tests for Docker Compose worker service integration (Phase 6F & 6F.1).

Validates:
1. Docker Compose configuration schema and worker service definition.
2. Volume mounts, shared paths, and environment variable parity between API and worker.
3. Single-worker locking exclusivity and conflict rejection.
4. Worker standby backoff behavior avoiding uncontrolled Docker restart loops.
5. CLI entrypoint invocation with --handler production, --once, and --no-standby.
6. Shared SQLite database and lock file accessibility.
7. Graceful shutdown, containment, and lease recovery behavior.
8. Persistent database isolation regression safeguard.
"""

import hashlib
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from backend.app.config import Settings, settings
from backend.app.ml.db import get_db_connection, resolve_shared_database_path
from backend.app.ml.jobs import (
    claim_queued_job,
    create_queued_job,
    get_job,
    mark_promotion_started,
    recover_expired_jobs,
)
from backend.app.ml.lock import (
    MaintenanceLock,
    MaintenanceLockConflictError,
    WorkerLock,
    WorkerLockConflictError,
    resolve_shared_lock_path,
)
from backend.app.ml.migrations import migrate_audit_database_idempotent
from backend.app.ml.worker import (
    JobExecutionResult,
    RetrainingWorker,
    main,
)

COMPOSE_FILE = Path(__file__).resolve().parent.parent / "docker-compose.yml"
PERSISTENT_DB = Path(__file__).resolve().parent.parent / "data/lifecycle_audit.db"


# ============================================================================
# Regression Safeguard: Isolate Settings Paths for All Tests
# ============================================================================


@pytest.fixture(autouse=True)
def isolate_compose_worker_paths(tmp_path, monkeypatch):
    """Ensure all tests in this module use temporary isolated databases and locks.

    Protects persistent data/lifecycle_audit.db from accidental access or migrations.
    """
    isolated_db = tmp_path / "lifecycle_audit.db"
    isolated_lock = tmp_path / "worker.lock"
    monkeypatch.setattr(settings, "lifecycle_audit_db_path", str(isolated_db))
    monkeypatch.setattr(settings, "promotion_audit_db_path", str(isolated_db))
    monkeypatch.setattr(settings, "worker_lock_path", str(isolated_lock))
    return isolated_db, isolated_lock


def test_regression_safeguard_persistent_db_untouched():
    """Verify persistent database is never modified during test executions."""
    if not PERSISTENT_DB.exists():
        pytest.skip("Persistent DB not found")
    initial_stat = PERSISTENT_DB.stat()
    initial_hash = hashlib.sha256(PERSISTENT_DB.read_bytes()).hexdigest()

    # Verify that default path resolution under the fixture points to isolated tmp_path
    resolved_db = resolve_shared_database_path()
    assert resolved_db != PERSISTENT_DB
    assert "tmp" in str(resolved_db)

    # Confirm persistent file remains identical
    assert PERSISTENT_DB.stat().st_size == initial_stat.st_size
    assert hashlib.sha256(PERSISTENT_DB.read_bytes()).hexdigest() == initial_hash


# ============================================================================
# 1. Compose Configuration and Service Validation
# ============================================================================


def test_compose_file_exists_and_parses():
    """Verify docker-compose.yml exists and parses as valid YAML."""
    assert COMPOSE_FILE.exists(), f"Missing docker-compose.yml at {COMPOSE_FILE}"
    with open(COMPOSE_FILE, encoding="utf-8") as f:
        compose = yaml.safe_load(f)
    assert "services" in compose
    assert "worker" in compose["services"]
    assert "api" in compose["services"]
    assert "mlflow" in compose["services"]


def test_compose_worker_service_definition():
    """Verify worker service requirements in docker-compose.yml."""
    with open(COMPOSE_FILE, encoding="utf-8") as f:
        compose = yaml.safe_load(f)

    worker = compose["services"]["worker"]
    api = compose["services"]["api"]

    # 1. Image parity
    assert worker["image"] == api["image"]
    assert worker.get("user") == api.get("user") == "1000:1000"

    # 2. Command invokes production worker CLI
    expected_command = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "backend.app.ml.worker",
        "--handler",
        "production",
    ]
    assert worker["command"] == expected_command

    # 3. Environment parity
    worker_env = worker.get("environment", {})
    api_env = api.get("environment", {})
    assert (
        worker_env["MODELFORGE_MLFLOW_TRACKING_URI"]
        == api_env["MODELFORGE_MLFLOW_TRACKING_URI"]
    )
    assert worker_env["MODELFORGE_MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    assert worker_env.get("MODELFORGE_MODEL_NAME") == api_env.get(
        "MODELFORGE_MODEL_NAME"
    )
    assert worker_env.get("MODELFORGE_MODEL_ALIAS") == api_env.get(
        "MODELFORGE_MODEL_ALIAS"
    )

    # 4. Dependency on healthy MLflow
    assert "depends_on" in worker
    assert "mlflow" in worker["depends_on"]
    assert worker["depends_on"]["mlflow"]["condition"] == "service_healthy"

    # 5. Restart policy and grace period (60s aligned with lease duration)
    assert worker.get("restart") == "unless-stopped"
    assert worker.get("stop_grace_period") == "60s"

    # 6. Single replica deployment
    assert worker.get("deploy", {}).get("replicas") == 1


def test_compose_shared_volumes_parity():
    """Verify API and worker mount identical host paths to same targets."""
    with open(COMPOSE_FILE, encoding="utf-8") as f:
        compose = yaml.safe_load(f)

    worker_volumes = compose["services"]["worker"].get("volumes", [])
    api_volumes = compose["services"]["api"].get("volumes", [])

    def _normalize_volumes(vols):
        res = {}
        for v in vols:
            if isinstance(v, str):
                parts = v.split(":")
                res[parts[1]] = parts[0]
            elif isinstance(v, dict):
                res[v["target"]] = v["source"]
        return res

    worker_norm = _normalize_volumes(worker_volumes)
    api_norm = _normalize_volumes(api_volumes)

    # Required shared mounts: /app/data, /app/models, /mlruns
    assert "/app/data" in worker_norm
    assert "/app/data" in api_norm
    assert worker_norm["/app/data"] == api_norm["/app/data"] == "./data"

    assert "/app/models" in worker_norm
    assert "/app/models" in api_norm
    assert worker_norm["/app/models"] == api_norm["/app/models"] == "./models"

    assert "/mlruns" in worker_norm
    assert "/mlruns" in api_norm
    assert worker_norm["/mlruns"] == api_norm["/mlruns"] == "./mlruns"


def test_docker_compose_config_cli():
    """Run `docker compose config` to ensure compose specification is valid."""
    result = subprocess.run(
        ["docker", "compose", "config"],
        cwd=COMPOSE_FILE.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    assert "worker:" in result.stdout
    assert "api:" in result.stdout


# ============================================================================
# 2. Path Resolution Parity
# ============================================================================


def test_path_resolution_parity():
    """Ensure API and Worker resolve identical SQLite, lock, and data paths."""
    test_settings = Settings()

    # In container, PROJECT_ROOT is /app
    fake_project_root = Path("/app")

    with (
        patch("backend.app.ml.lock.PROJECT_ROOT", fake_project_root),
        patch("backend.app.ml.db.PROJECT_ROOT", fake_project_root),
    ):
        db_path = resolve_shared_database_path("data/lifecycle_audit.db")
        lock_path = resolve_shared_lock_path("data/worker.lock")

        assert db_path == Path("/app/data/lifecycle_audit.db")
        assert lock_path == Path("/app/data/worker.lock")

        # Datasets
        drift_path = fake_project_root / test_settings.drift_reference_path
        benchmark_path = fake_project_root / test_settings.benchmark_data_path
        assert drift_path == Path("/app/data/raw/churn.csv")
        assert benchmark_path == Path("/app/data/benchmark/churn_benchmark.csv")

        # Models
        candidate_model_path = fake_project_root / "models/candidate_model.joblib"
        assert candidate_model_path == Path("/app/models/candidate_model.joblib")


# ============================================================================
# 3. Single-Worker Locking Enforcement & Standby Behavior
# ============================================================================


def test_worker_lock_exclusive_acquisition_and_conflict(tmp_path):
    """Confirm a worker acquires lock and a second worker cannot acquire it."""
    isolated_db = tmp_path / "isolated.db"
    isolated_lock = tmp_path / "worker.lock"

    worker1 = RetrainingWorker(
        database_path=isolated_db,
        lock_path=isolated_lock,
        worker_id="worker-1",
    )
    worker2 = RetrainingWorker(
        database_path=isolated_db,
        lock_path=isolated_lock,
        worker_id="worker-2",
    )

    # Worker 1 acquires lock
    worker1.acquire_lock()
    assert worker1.is_locked

    # Worker 2 attempts to acquire lock and fails with WorkerLockConflictError
    with pytest.raises(WorkerLockConflictError) as exc_info:
        worker2.acquire_lock()
    assert "Cannot start worker" in str(exc_info.value)
    assert not worker2.is_locked

    # Worker 1 releases lock
    worker1.release_lock()
    assert not worker1.is_locked

    # Worker 2 can now acquire the lock
    worker2.acquire_lock()
    assert worker2.is_locked
    worker2.release_lock()


def test_maintenance_lock_conflict_with_worker(tmp_path):
    """Confirm maintenance lock and worker lock are mutually exclusive."""
    isolated_db = tmp_path / "isolated.db"
    isolated_lock = tmp_path / "worker.lock"

    worker = RetrainingWorker(
        database_path=isolated_db,
        lock_path=isolated_lock,
        worker_id="worker-1",
    )
    maint_lock = MaintenanceLock(lock_path=isolated_lock)

    # Worker active -> maintenance cannot acquire
    with worker:
        with pytest.raises(MaintenanceLockConflictError):
            maint_lock.acquire()

    # Maintenance active -> worker cannot start
    with maint_lock:
        with pytest.raises(WorkerLockConflictError):
            worker.acquire_lock()


def test_worker_run_standby_on_lock_conflict_avoids_crash_loop(tmp_path):
    """Verify second worker in daemon mode waits in standby instead of crash-looping."""
    isolated_db = tmp_path / "standby.db"
    isolated_lock = tmp_path / "standby.lock"
    migrate_audit_database_idempotent(isolated_db)

    # Worker 1 acquires lock
    worker1 = RetrainingWorker(
        database_path=isolated_db,
        lock_path=isolated_lock,
        worker_id="worker-primary",
    )
    worker1.acquire_lock()
    assert worker1.is_locked

    # Worker 2 starts in background daemon mode with standby=True
    worker2 = RetrainingWorker(
        database_path=isolated_db,
        lock_path=isolated_lock,
        worker_id="worker-standby",
        poll_interval_seconds=0.1,
        standby=True,
    )

    t2 = threading.Thread(target=worker2.run)
    t2.start()

    time.sleep(0.3)
    # Worker 2 is alive (waiting in standby) and does NOT hold the lock
    assert t2.is_alive()
    assert not worker2.is_locked

    # Signal stop to worker 2 (simulating docker stop)
    worker2.stop()
    t2.join(timeout=2.0)
    assert not t2.is_alive()
    assert not worker2.is_locked

    worker1.release_lock()


def test_worker_run_standby_acquires_lock_when_primary_releases(tmp_path):
    """Verify standby worker automatically assumes lock when primary terminates."""
    isolated_db = tmp_path / "standby_failover.db"
    isolated_lock = tmp_path / "standby_failover.lock"
    migrate_audit_database_idempotent(isolated_db)

    worker1 = RetrainingWorker(
        database_path=isolated_db,
        lock_path=isolated_lock,
        worker_id="primary-w",
    )
    worker1.acquire_lock()
    assert worker1.is_locked

    worker2 = RetrainingWorker(
        database_path=isolated_db,
        lock_path=isolated_lock,
        worker_id="standby-w",
        poll_interval_seconds=0.1,
        standby=True,
    )

    t2 = threading.Thread(target=worker2.run)
    t2.start()

    time.sleep(0.2)
    assert not worker2.is_locked

    # Primary crashes or releases lock
    worker1.release_lock()

    # Standby worker acquires lock on next polling iteration
    time.sleep(0.3)
    assert worker2.is_locked

    worker2.stop()
    t2.join(timeout=2.0)
    assert not worker2.is_locked


def test_worker_standby_logging_behavior(tmp_path, caplog):
    """Verify standby emits warning on entry without warning spam on polls."""
    import logging

    isolated_db = tmp_path / "standby_log.db"
    isolated_lock = tmp_path / "standby_log.lock"
    migrate_audit_database_idempotent(isolated_db)

    worker1 = RetrainingWorker(
        database_path=isolated_db,
        lock_path=isolated_lock,
        worker_id="primary-w",
    )
    worker1.acquire_lock()

    worker2 = RetrainingWorker(
        database_path=isolated_db,
        lock_path=isolated_lock,
        worker_id="standby-w",
        poll_interval_seconds=0.05,
        standby=True,
    )

    with caplog.at_level(logging.DEBUG):
        t2 = threading.Thread(target=worker2.run)
        t2.start()

        # Let worker 2 run several standby iterations (~5-6 iterations)
        time.sleep(0.3)
        assert not worker2.is_locked

        # Release primary lock so standby acquires it
        worker1.release_lock()
        time.sleep(0.2)
        assert worker2.is_locked

        worker2.stop()
        t2.join(timeout=2.0)

    # Check logs: warning must be emitted exactly once on entering standby
    conflict_warnings = [
        r
        for r in caplog.records
        if r.levelname == "WARNING"
        and "worker_startup_lock_conflict_standby" in r.message
    ]
    assert len(conflict_warnings) == 1

    # Intermediate polling cycles must NOT spam warnings
    other_warnings = [
        r
        for r in caplog.records
        if r.levelname == "WARNING"
        and r.name == "backend.app.ml.worker"
        and r != conflict_warnings[0]
    ]
    assert len(other_warnings) == 0

    # Transition to active must be logged at INFO
    acquisition_infos = [
        r
        for r in caplog.records
        if r.levelname == "INFO" and "worker_standby_lock_acquired" in r.message
    ]
    assert len(acquisition_infos) == 1


# ============================================================================
# 4. Worker CLI Invocation with Production Handler & Flags
# ============================================================================


def test_worker_cli_production_handler_argument(tmp_path):
    """Test worker CLI parses --handler production and executes single run cleanly."""
    isolated_db = tmp_path / "isolated.db"
    isolated_lock = tmp_path / "isolated.lock"
    migrate_audit_database_idempotent(isolated_db)

    # Verify invalid handler is rejected
    with pytest.raises(SystemExit) as exit_exc:
        main(
            [
                "--handler",
                "invalid",
                "--db-path",
                str(isolated_db),
                "--lock-path",
                str(isolated_lock),
            ]
        )
    assert exit_exc.value.code != 0

    # Verify production handler starts cleanly with --once
    exit_code = main(
        [
            "--handler",
            "production",
            "--db-path",
            str(isolated_db),
            "--lock-path",
            str(isolated_lock),
            "--worker-id",
            "test-prod-worker",
            "--once",
        ]
    )
    assert exit_code == 0
    assert isolated_lock.exists()


def test_worker_cli_once_lock_conflict_exits_with_code_1(tmp_path):
    """When lock is held and --once is passed, worker CLI fails fast with code 1."""
    isolated_lock = tmp_path / "active.lock"
    holding_lock = WorkerLock(isolated_lock)
    holding_lock.acquire()

    try:
        isolated_db = tmp_path / "isolated.db"
        exit_code = main(
            [
                "--handler",
                "production",
                "--db-path",
                str(isolated_db),
                "--lock-path",
                str(isolated_lock),
                "--worker-id",
                "conflicting-worker",
                "--once",
            ]
        )
        assert exit_code == 1
    finally:
        holding_lock.release()


def test_worker_cli_no_standby_flag_exits_with_code_1(tmp_path):
    """When --no-standby is passed and lock is held, daemon mode fails with code 1."""
    isolated_lock = tmp_path / "active.lock"
    holding_lock = WorkerLock(isolated_lock)
    holding_lock.acquire()

    try:
        isolated_db = tmp_path / "isolated.db"
        exit_code = main(
            [
                "--handler",
                "production",
                "--db-path",
                str(isolated_db),
                "--lock-path",
                str(isolated_lock),
                "--worker-id",
                "failfast-worker",
                "--no-standby",
            ]
        )
        assert exit_code == 1
    finally:
        holding_lock.release()


# ============================================================================
# 5. Shared SQLite Database and Schema Initialization
# ============================================================================


def test_worker_and_api_shared_database_access(tmp_path):
    """Confirm worker and API access same SQLite database without conflict."""
    isolated_db = tmp_path / "shared_lifecycle.db"
    isolated_lock = tmp_path / "shared_worker.lock"

    # Simulate API / migration initializing DB
    migrate_audit_database_idempotent(isolated_db)

    # API enqueues a retraining job
    with get_db_connection(isolated_db):
        job = create_queued_job(
            isolated_db,
            dataset_id="churn",
            trigger_source="drift_monitor",
            payload={"test": "data"},
        )
    assert job["status"] == "queued"
    job_id = job["job_id"]

    # Worker runs single poll with NoOp handler
    worker = RetrainingWorker(
        database_path=isolated_db,
        lock_path=isolated_lock,
        worker_id="test-worker",
    )
    result = worker.run_once()
    assert result is not None
    assert result["job_id"] == job_id
    assert result["status"] == "succeeded"

    # API verifies status
    updated_job = get_job(isolated_db, job_id=job_id)
    assert updated_job["status"] == "succeeded"
    worker.release_lock()


# ============================================================================
# 6. Shutdown, Containment, and Lease Recovery
# ============================================================================


def test_graceful_shutdown_releases_lock(tmp_path):
    """Verify worker releases lock cleanly on graceful stop."""
    isolated_db = tmp_path / "shutdown.db"
    isolated_lock = tmp_path / "shutdown.lock"
    migrate_audit_database_idempotent(isolated_db)

    worker = RetrainingWorker(
        database_path=isolated_db,
        lock_path=isolated_lock,
        poll_interval_seconds=0.1,
    )

    t = threading.Thread(target=worker.run)
    t.start()

    time.sleep(0.2)
    assert worker.is_locked
    assert worker.is_running

    # Signal stop (simulating SIGTERM)
    worker.stop()
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert not worker.is_locked


def test_containment_preserves_lock_when_handler_running(tmp_path):
    """Verify Phase 6E.1 containment: worker preserves lock if handler active."""
    isolated_db = tmp_path / "containment.db"
    isolated_lock = tmp_path / "containment.lock"
    migrate_audit_database_idempotent(isolated_db)

    block_event = threading.Event()

    def stubborn_handler(job, context):
        block_event.wait()
        return JobExecutionResult(status="succeeded")

    worker = RetrainingWorker(
        database_path=isolated_db,
        lock_path=isolated_lock,
        handler=stubborn_handler,
    )
    worker.acquire_lock()
    assert worker.is_locked

    # Claim a job
    create_queued_job(isolated_db, dataset_id="churn", trigger_source="test")
    claimed = claim_queued_job(
        isolated_db,
        worker_id=worker.worker_id,
        lease_duration_seconds=10,
    )

    # Start processing in thread
    t = threading.Thread(target=worker.process_job, args=(claimed,))
    t.start()

    time.sleep(0.2)
    assert worker.is_executing

    # Attempt to release lock without force while work is active: must be refused
    worker.release_lock(force=False)
    assert worker._lock.is_held, "Worker lock must NOT be released while handler alive"

    # Unblock handler to let test complete cleanly
    block_event.set()
    t.join(timeout=3.0)
    worker.release_lock(force=True)


def test_lease_expiry_recovery_quarantine_after_sigkill(tmp_path):
    """Simulate SIGKILL during promotion: recovery quarantines into reconciliation."""
    isolated_db = tmp_path / "sigkill_recovery.db"
    migrate_audit_database_idempotent(isolated_db)

    # Job was running and set promotion_started_at before simulated crash/SIGKILL
    job = create_queued_job(isolated_db, dataset_id="churn", trigger_source="drift")
    job_id = job["job_id"]
    claimed = claim_queued_job(
        isolated_db,
        worker_id="killed-worker",
        lease_duration_seconds=1,
    )
    lease_token = claimed["lease_token"]

    mark_promotion_started(
        isolated_db,
        job_id=job_id,
        worker_id="killed-worker",
        lease_token=lease_token,
        candidate_version="2",
        candidate_run_id="run-123",
        previous_champion_version="1",
    )

    # Wait for lease to expire (1 second lease)
    time.sleep(1.2)

    # When supervisor or next worker executes recover_expired_jobs:
    recovered = recover_expired_jobs(isolated_db)
    assert any(
        r["job_id"] == job_id and r["action"] == "quarantined" for r in recovered
    )

    final_job = get_job(isolated_db, job_id=job_id)
    assert final_job["status"] == "reconciliation_required"
    assert "reconciliation required" in (final_job.get("failure_reason") or "").lower()
