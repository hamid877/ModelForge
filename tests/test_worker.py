"""Tests for worker execution infrastructure, lifecycle, and safety mechanisms."""

import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from backend.app.config import settings
from backend.app.ml.jobs import (
    claim_queued_job,
    create_queued_job,
    get_job,
    mark_promotion_started,
)
from backend.app.ml.lock import WorkerLock, WorkerLockConflictError
from backend.app.ml.migrations import migrate_audit_database_idempotent
from backend.app.ml.worker import (
    JobExecutionContext,
    JobExecutionResult,
    RetrainingWorker,
    main,
)


@pytest.fixture(autouse=True)
def isolate_worker_settings(tmp_path, monkeypatch):
    """Ensure worker tests are isolated from default database and lock files."""
    isolated_db = tmp_path / "isolated_worker_default.db"
    isolated_lock = tmp_path / "isolated_worker_default.lock"
    monkeypatch.setattr(settings, "lifecycle_audit_db_path", str(isolated_db))
    monkeypatch.setattr(settings, "promotion_audit_db_path", str(isolated_db))
    monkeypatch.setattr(settings, "worker_lock_path", str(isolated_lock))
    return isolated_db


@pytest.fixture
def migrated_db(tmp_path):
    """Fixture providing an initialized SQLite audit database."""
    db_path = tmp_path / "test_worker_audit.db"
    migrate_audit_database_idempotent(db_path)
    return str(db_path)


def test_worker_initialization_and_validation(tmp_path):
    """Worker validates inputs and uses configured defaults."""
    worker = RetrainingWorker(
        database_path=tmp_path / "db.sqlite",
        lock_path=tmp_path / "test.lock",
        worker_id="test-worker-1",
    )
    assert worker.worker_id == "test-worker-1"
    assert worker.poll_interval_seconds == 5.0
    assert worker.heartbeat_interval_seconds == 15.0
    assert worker.lease_duration_seconds == 60
    assert not worker.is_locked
    assert not worker.is_running

    with pytest.raises(ValueError, match="worker_id must be a non-empty string"):
        RetrainingWorker(worker_id="   ")

    with pytest.raises(ValueError, match="poll_interval_seconds must be > 0"):
        RetrainingWorker(poll_interval_seconds=0)

    with pytest.raises(ValueError, match="heartbeat_interval_seconds must be > 0"):
        RetrainingWorker(heartbeat_interval_seconds=-1)

    with pytest.raises(ValueError, match="lease_duration_seconds must be >= 1"):
        RetrainingWorker(lease_duration_seconds=0)

    with pytest.raises(ValueError, match="job_timeout_seconds must be > 0"):
        RetrainingWorker(job_timeout_seconds=0)

    with pytest.raises(ValueError, match="job_timeout_seconds must be > 0"):
        RetrainingWorker(job_timeout_seconds=-10.0)


def test_worker_startup_and_exclusive_lock_acquisition(tmp_path):
    """Worker acquires and retains exclusive lock descriptor throughout its lifetime."""
    lock_file = tmp_path / "exclusive_worker.lock"
    db_file = tmp_path / "exclusive_worker.db"
    worker = RetrainingWorker(database_path=db_file, lock_path=lock_file)

    assert not worker.is_locked
    fd = worker.acquire_lock()
    assert isinstance(fd, int)
    assert fd > 0
    assert worker.is_locked

    # Idempotent acquire on same worker instance
    assert worker.acquire_lock() == fd

    worker.release_lock()
    assert not worker.is_locked

    # Context manager cleans up lock
    with worker:
        assert worker.is_locked
    assert not worker.is_locked


def test_worker_refusing_startup_when_lock_held(tmp_path):
    """Worker refuses startup if another worker or process holds the exclusive lock."""
    lock_file = tmp_path / "conflict_worker.lock"
    db_file = tmp_path / "conflict_worker.db"
    worker1 = RetrainingWorker(
        database_path=db_file, lock_path=lock_file, worker_id="worker-primary"
    )
    worker2 = RetrainingWorker(
        database_path=db_file, lock_path=lock_file, worker_id="worker-secondary"
    )

    worker1.acquire_lock()
    try:
        assert worker1.is_locked
        with pytest.raises(WorkerLockConflictError) as exc_info:
            worker2.acquire_lock()
        assert "Lock is currently held" in str(exc_info.value)
        assert not worker2.is_locked
    finally:
        worker1.release_lock()

    # After worker1 releases, worker2 can acquire
    worker2.acquire_lock()
    assert worker2.is_locked
    worker2.release_lock()


def test_empty_queue_polling(migrated_db, tmp_path):
    """Polling an empty queue returns None without error and maintains lock."""
    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "empty_queue.lock",
    )
    with worker:
        result = worker.run_once()
        assert result is None
        assert worker.is_locked


def test_successful_job_claiming_and_handler_invocation(migrated_db, tmp_path):
    """Worker claims queued job, passes context to handler, and finalizes succeeded."""
    job_record = create_queued_job(
        migrated_db,
        dataset_id="churn_test_v1",
        trigger_source="unit_test",
        payload={"param": 42},
    )
    jid = job_record["job_id"]

    invoked_jobs = []

    class CustomHandler:
        def __call__(
            self, job: dict, context: JobExecutionContext
        ) -> JobExecutionResult:
            invoked_jobs.append(job)
            assert context.job_id == jid
            assert context.worker_id == "w-handler-1"
            assert not context.is_cancelled
            return JobExecutionResult(
                status="succeeded",
                candidate_version="candidate-1.0",
                candidate_run_id="run-xyz",
                payload_updates={"handler_ran": True},
            )

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "claim.lock",
        worker_id="w-handler-1",
        handler=CustomHandler(),
    )

    with worker:
        finalized = worker.run_once()

    assert len(invoked_jobs) == 1
    assert invoked_jobs[0]["job_id"] == jid
    assert finalized is not None
    assert finalized["job_id"] == jid
    assert finalized["status"] == "succeeded"
    assert finalized["is_active"] is None
    assert finalized["completed_at"] is not None
    assert finalized["candidate_version"] == "candidate-1.0"
    assert finalized["candidate_run_id"] == "run-xyz"
    assert finalized["payload"].get("handler_ran") is True

    # Job in DB reflects terminal succeeded state
    db_job = get_job(migrated_db, job_id=jid)
    assert db_job["status"] == "succeeded"
    assert db_job["is_active"] is None


def test_heartbeat_extension(migrated_db, tmp_path):
    """Worker heartbeat loop extends lease expiration during job execution."""
    job_record = create_queued_job(
        migrated_db,
        dataset_id="churn_test_v1",
        trigger_source="unit_test",
    )
    jid = job_record["job_id"]

    heartbeat_count = 0

    class SlowHandler:
        def __call__(
            self, job: dict, context: JobExecutionContext
        ) -> JobExecutionResult:
            nonlocal heartbeat_count
            # Sleep cooperatively for 0.15s with 0.04s heartbeat interval
            time.sleep(0.15)
            # Read DB to verify heartbeat updated
            updated = get_job(migrated_db, job_id=jid)
            if updated and updated.get("heartbeat_at"):
                heartbeat_count += 1
            return JobExecutionResult(status="succeeded")

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "hb.lock",
        worker_id="w-hb",
        handler=SlowHandler(),
        heartbeat_interval_seconds=0.04,
        lease_duration_seconds=5,
    )

    with worker:
        finalized = worker.run_once()

    assert finalized is not None
    assert finalized["status"] == "succeeded"
    assert heartbeat_count > 0


def test_heartbeat_failure_and_cooperative_cancellation(migrated_db, tmp_path):
    """Heartbeat failure on lost lease sets cancel_event and aborts finalization."""
    job_record = create_queued_job(
        migrated_db,
        dataset_id="churn_test_v1",
        trigger_source="unit_test",
    )
    jid = job_record["job_id"]

    handler_started = threading.Event()
    handler_observed_cancellation = threading.Event()

    class CancellableHandler:
        def __call__(
            self, job: dict, context: JobExecutionContext
        ) -> JobExecutionResult:
            handler_started.set()
            # Wait for cancellation signal
            if context.cancel_event.wait(timeout=2.0):
                handler_observed_cancellation.set()
            return JobExecutionResult(status="succeeded")

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "hb_fail.lock",
        worker_id="w-canceller",
        handler=CancellableHandler(),
        heartbeat_interval_seconds=0.04,
        lease_duration_seconds=10,
    )

    def _interfere():
        handler_started.wait(timeout=2.0)
        # Steal ownership in DB while worker is processing
        conn = sqlite3.connect(migrated_db)
        conn.execute(
            "UPDATE retraining_jobs SET worker_id = 'intruder-worker' "
            "WHERE job_id = ?;",
            (jid,),
        )
        conn.commit()
        conn.close()

    interferer = threading.Thread(target=_interfere)
    interferer.start()

    with worker:
        finalized = worker.run_once()

    interferer.join()

    assert handler_observed_cancellation.is_set()
    assert finalized is None

    # Verify stale worker did NOT overwrite the intruder's ownership or mark failed
    current_job = get_job(migrated_db, job_id=jid)
    assert current_job["worker_id"] == "intruder-worker"
    assert current_job["status"] == "running"


def test_graceful_shutdown_while_idle(migrated_db, tmp_path):
    """Worker terminates cleanly and releases lock when stopped while idle."""
    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "idle_shutdown.lock",
        poll_interval_seconds=2.0,
    )

    worker_thread = threading.Thread(target=worker.run)
    worker_thread.start()

    # Wait until worker acquires lock
    deadline = time.time() + 3.0
    while not worker.is_locked and time.time() < deadline:
        time.sleep(0.05)
    assert worker.is_locked

    # Signal shutdown
    worker.stop()
    worker_thread.join(timeout=3.0)

    assert not worker_thread.is_alive()
    assert not worker.is_locked


def test_graceful_shutdown_during_job_execution(migrated_db, tmp_path):
    """Shutdown during job execution signals cancellation and releases lock cleanly."""
    job_record = create_queued_job(
        migrated_db,
        dataset_id="churn_test_v1",
        trigger_source="unit_test",
    )
    jid = job_record["job_id"]

    handler_started = threading.Event()
    handler_completed = threading.Event()

    class LongRunningHandler:
        def __call__(
            self, job: dict, context: JobExecutionContext
        ) -> JobExecutionResult:
            handler_started.set()
            context.cancel_event.wait(timeout=5.0)
            handler_completed.set()
            return JobExecutionResult(status="succeeded")

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "exec_shutdown.lock",
        worker_id="w-shutdown-exec",
        handler=LongRunningHandler(),
        poll_interval_seconds=2.0,
    )

    worker_thread = threading.Thread(target=worker.run)
    worker_thread.start()

    assert handler_started.wait(timeout=3.0)

    # Stop worker while job is actively running
    worker.stop()
    worker_thread.join(timeout=3.0)

    assert not worker_thread.is_alive()
    assert handler_completed.is_set()
    assert not worker.is_locked

    # Job was cancelled; not marked succeeded
    current_job = get_job(migrated_db, job_id=jid)
    assert current_job["status"] == "running"


def test_handler_deliberately_ignoring_cancellation_preserves_lock(
    migrated_db, tmp_path
):
    """Worker retains exclusive process lock while handler ignores cancellation."""
    create_queued_job(
        migrated_db,
        dataset_id="churn_test_v1",
        trigger_source="unit_test",
    )

    handler_started = threading.Event()
    allow_handler_exit = threading.Event()

    class StubbornHandler:
        def __call__(
            self, job: dict, context: JobExecutionContext
        ) -> JobExecutionResult:
            handler_started.set()
            # Deliberately ignore cancellation; wait only on allow_handler_exit
            while not allow_handler_exit.is_set():
                time.sleep(0.05)
            return JobExecutionResult(status="succeeded")

    lock_file = tmp_path / "stubborn.lock"
    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=lock_file,
        worker_id="w-stubborn",
        handler=StubbornHandler(),
        poll_interval_seconds=1.0,
        shutdown_timeout=0.1,  # Prompt timeout
    )

    worker_thread = threading.Thread(target=worker.run)
    worker_thread.start()

    assert handler_started.wait(timeout=3.0)

    # Signal shutdown with timeout that expires before handler exits
    worker.stop()

    # Worker remains alive in containment, retaining lock and heartbeats
    assert worker_thread.is_alive()
    assert worker.is_executing
    assert worker.is_locked

    # Another worker CANNOT acquire the lock because it is retained
    worker2 = RetrainingWorker(
        database_path=migrated_db, lock_path=lock_file, worker_id="w-intruder"
    )
    with pytest.raises(WorkerLockConflictError):
        worker2.acquire_lock()

    # release_lock() refuses to release while handler is executing
    worker.release_lock()
    assert worker.is_locked

    # Allow stubborn handler to exit cleanly
    allow_handler_exit.set()
    worker_thread.join(timeout=3.0)
    assert not worker_thread.is_alive()
    assert not worker.is_executing

    # Now release_lock() succeeds and frees the lock
    worker.release_lock()
    assert not worker.is_locked

    # Intruder worker can now acquire lock
    worker2.acquire_lock()
    assert worker2.is_locked
    worker2.release_lock()


def test_unexpected_handler_exception_requeues_when_attempts_remain(
    migrated_db, tmp_path
):
    """Pre-promotion exception with attempts remaining requeues job for retry."""
    job_record = create_queued_job(
        migrated_db,
        dataset_id="churn_test_v1",
        trigger_source="unit_test",
        max_attempts=3,
    )
    jid = job_record["job_id"]

    class ExplodingHandler:
        def __call__(
            self, job: dict, context: JobExecutionContext
        ) -> JobExecutionResult:
            raise RuntimeError("Transient computation failure")

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "retry_requeue.lock",
        worker_id="w-retry-test",
        handler=ExplodingHandler(),
    )

    with worker:
        requeued = worker.run_once()

    # Must be requeued, not failed!
    assert requeued is not None
    assert requeued["status"] == "queued"
    assert requeued["is_active"] == 1
    assert requeued["worker_id"] is None
    assert requeued["lease_token"] is None
    assert requeued["attempt_count"] == 1
    assert "Transient computation failure" in requeued["failure_reason"]

    # Verify DB state
    db_job = get_job(migrated_db, job_id=jid)
    assert db_job["status"] == "queued"
    assert db_job["is_active"] == 1
    assert db_job["attempt_count"] == 1


def test_unexpected_handler_exception_fails_when_attempts_exhausted(
    migrated_db, tmp_path
):
    """Pre-promotion exception with attempts exhausted transitions job to failed."""
    job_record = create_queued_job(
        migrated_db,
        dataset_id="churn_test_v1",
        trigger_source="unit_test",
        max_attempts=1,
    )
    jid = job_record["job_id"]

    class ExplodingHandler:
        def __call__(
            self, job: dict, context: JobExecutionContext
        ) -> JobExecutionResult:
            raise RuntimeError("Fatal unrecoverable error")

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "exhausted_fail.lock",
        worker_id="w-exhaust-test",
        handler=ExplodingHandler(),
    )

    with worker:
        finalized = worker.run_once()

    assert finalized is not None
    assert finalized["status"] == "failed"
    assert finalized["is_active"] is None
    assert "Fatal unrecoverable error" in finalized["failure_reason"]
    assert "attempts exhausted" in finalized["failure_reason"].lower()

    # Verify DB state
    db_job = get_job(migrated_db, job_id=jid)
    assert db_job["status"] == "failed"
    assert db_job["is_active"] is None
    assert db_job["completed_at"] is not None


def test_unexpected_handler_exception_quarantines_after_promotion_started(
    migrated_db, tmp_path
):
    """Exception after promotion_started_at quarantines into reconciliation_required."""
    job_record = create_queued_job(
        migrated_db,
        dataset_id="churn_test_v1",
        trigger_source="unit_test",
        max_attempts=3,
    )
    jid = job_record["job_id"]

    class PostPromotionFailHandler:
        def __call__(
            self, job: dict, context: JobExecutionContext
        ) -> JobExecutionResult:
            # Set promotion started marker
            mark_promotion_started(
                migrated_db,
                job_id=jid,
                worker_id=context.worker_id,
                lease_token=context.lease_token,
                candidate_version="v-cand-crash",
            )
            raise RuntimeError("Crash during post-promotion gate")

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "post_promo_fail.lock",
        worker_id="w-post-promo",
        handler=PostPromotionFailHandler(),
    )

    with worker:
        quarantined = worker.run_once()

    # Never retried, never terminal failed: quarantined!
    assert quarantined is not None
    assert quarantined["status"] == "reconciliation_required"
    assert quarantined["is_active"] == 1
    assert "Crash during post-promotion gate" in quarantined["failure_reason"]

    # Verify DB state
    db_job = get_job(migrated_db, job_id=jid)
    assert db_job["status"] == "reconciliation_required"
    assert db_job["is_active"] == 1


def test_unexpected_handler_exception_ignored_if_ownership_lost(migrated_db, tmp_path):
    """Exception is ignored and state is untouched if lease ownership was lost."""
    job_record = create_queued_job(
        migrated_db,
        dataset_id="churn_test_v1",
        trigger_source="unit_test",
    )
    jid = job_record["job_id"]

    handler_started = threading.Event()
    lease_stolen = threading.Event()

    class StolenLeaseHandler:
        def __call__(
            self, job: dict, context: JobExecutionContext
        ) -> JobExecutionResult:
            handler_started.set()
            lease_stolen.wait(timeout=2.0)
            raise RuntimeError("Handler failed after lease stolen")

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "stolen_exc.lock",
        worker_id="w-victim",
        handler=StolenLeaseHandler(),
        heartbeat_interval_seconds=10.0,
    )

    def _steal():
        handler_started.wait(timeout=2.0)
        conn = sqlite3.connect(migrated_db)
        conn.execute(
            "UPDATE retraining_jobs SET worker_id = 'thief' WHERE job_id = ?;",
            (jid,),
        )
        conn.commit()
        conn.close()
        lease_stolen.set()

    t = threading.Thread(target=_steal)
    t.start()

    with worker:
        res = worker.run_once()

    t.join()

    # Worker must NOT update DB state because ownership was lost
    assert res is None
    db_job = get_job(migrated_db, job_id=jid)
    assert db_job["worker_id"] == "thief"
    assert db_job["status"] == "running"


def test_stale_worker_unable_to_finalize(migrated_db, tmp_path):
    """Worker whose lease token is invalidated before completion cannot finalize."""
    job_record = create_queued_job(
        migrated_db,
        dataset_id="churn_test_v1",
        trigger_source="unit_test",
    )
    jid = job_record["job_id"]

    handler_started = threading.Event()
    invalidate_done = threading.Event()

    class StaleHandler:
        def __call__(
            self, job: dict, context: JobExecutionContext
        ) -> JobExecutionResult:
            handler_started.set()
            invalidate_done.wait(timeout=2.0)
            return JobExecutionResult(status="succeeded")

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "stale_finalize.lock",
        worker_id="w-stale-finalize",
        handler=StaleHandler(),
        heartbeat_interval_seconds=10.0,
    )

    def _invalidate():
        handler_started.wait(timeout=2.0)
        conn = sqlite3.connect(migrated_db)
        conn.execute(
            "UPDATE retraining_jobs SET lease_token = 'replaced-token' "
            "WHERE job_id = ?;",
            (jid,),
        )
        conn.commit()
        conn.close()
        invalidate_done.set()

    t = threading.Thread(target=_invalidate)
    t.start()

    with worker:
        finalized = worker.run_once()

    t.join()

    assert finalized is None
    db_job = get_job(migrated_db, job_id=jid)
    # Status should remain running with replaced token, not succeeded
    assert db_job["status"] == "running"
    assert db_job["lease_token"] == "replaced-token"


def test_lock_release_after_process_termination(tmp_path):
    """Terminating worker process releases flock so another worker can start."""
    lock_file = tmp_path / "proc_term.lock"
    ready_file = tmp_path / "worker_ready.txt"
    db_file = tmp_path / "proc_term.db"

    code = f"""
import sys, time
from pathlib import Path
from backend.app.ml.worker import RetrainingWorker

worker = RetrainingWorker(database_path=r'{db_file}', lock_path=r'{lock_file}')
worker.acquire_lock()
Path(r'{ready_file}').write_text("ready")
while True:
    time.sleep(0.05)
"""
    proc = subprocess.Popen([sys.executable, "-c", code])
    try:
        deadline = time.time() + 5.0
        while not ready_file.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert ready_file.exists()

        # Another worker in this process cannot acquire lock
        worker2 = RetrainingWorker(database_path=db_file, lock_path=lock_file)
        with pytest.raises(WorkerLockConflictError):
            worker2.acquire_lock()

        # Terminate worker process
        proc.terminate()
        proc.wait(timeout=3.0)

        # Now lock should be acquired without error
        worker2.acquire_lock()
        assert worker2.is_locked
        worker2.release_lock()
    finally:
        if proc.poll() is None:
            proc.kill()


def test_no_automatic_retry_after_promotion_marker(migrated_db, tmp_path):
    """Job with promotion marker is quarantined on lease expiry, not retried."""
    create_queued_job(
        migrated_db,
        dataset_id="churn_promo_test",
        trigger_source="unit_test",
        max_attempts=3,
    )
    claimed = claim_queued_job(
        migrated_db,
        worker_id="w-promo-old",
        lease_duration_seconds=5,
    )
    assert claimed is not None
    jid = claimed["job_id"]

    # Mark promotion started
    mark_promotion_started(
        migrated_db,
        job_id=jid,
        worker_id="w-promo-old",
        lease_token=claimed["lease_token"],
        candidate_version="v-promoted",
    )

    # Simulate expired lease in SQLite
    past_iso = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
    conn = sqlite3.connect(migrated_db)
    conn.execute(
        "UPDATE retraining_jobs SET lease_expires_at = ? WHERE job_id = ?;",
        (past_iso, jid),
    )
    conn.commit()
    conn.close()

    # New worker runs polling iteration
    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "promo_recovery.lock",
        worker_id="w-new-recovery",
    )
    with worker:
        res = worker.run_once()

    # Worker should NOT have claimed or retried the job
    assert res is None

    # Job must be quarantined in reconciliation_required, NOT queued
    db_job = get_job(migrated_db, job_id=jid)
    assert db_job["status"] == "reconciliation_required"
    assert db_job["is_active"] == 1
    assert db_job["attempt_count"] == 1


def test_worker_main_cli_entry_point(migrated_db, tmp_path):
    """CLI entry point supports --once flag and returns appropriate exit codes."""
    lock_file = tmp_path / "cli.lock"

    # Run single iteration on empty queue
    ret = main(
        [
            "--db-path",
            migrated_db,
            "--lock-path",
            str(lock_file),
            "--worker-id",
            "cli-w1",
            "--handler",
            "noop",
            "--once",
        ]
    )
    assert ret == 0

    # Test lock conflict exit code
    with WorkerLock(lock_file):
        conflict_ret = main(
            [
                "--db-path",
                migrated_db,
                "--lock-path",
                str(lock_file),
                "--worker-id",
                "cli-w2",
                "--handler",
                "noop",
                "--once",
            ]
        )
        assert conflict_ret == 1


def test_subprocess_sigterm_graceful_shutdown(tmp_path):
    """Subprocess running RetrainingWorker handles SIGTERM gracefully and exits 0."""
    lock_file = tmp_path / "sigterm.lock"
    ready_file = tmp_path / "ready.txt"
    db_file = tmp_path / "sigterm.db"

    code = f"""
import sys, time
from pathlib import Path
from backend.app.ml.worker import RetrainingWorker

worker = RetrainingWorker(
    database_path=r'{db_file}',
    lock_path=r'{lock_file}',
    poll_interval_seconds=1.0,
)
Path(r'{ready_file}').write_text("ready")
worker.run()
"""
    proc = subprocess.Popen([sys.executable, "-c", code])
    try:
        deadline = time.time() + 5.0
        while not ready_file.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert ready_file.exists()

        # Allow worker.run() to enter poll loop
        time.sleep(0.2)

        # Send SIGTERM
        proc.send_signal(signal.SIGTERM)
        exit_code = proc.wait(timeout=5.0)
        assert exit_code == 0

        # Lock was released
        w = RetrainingWorker(database_path=db_file, lock_path=lock_file)
        w.acquire_lock()
        assert w.is_locked
        w.release_lock()
    finally:
        if proc.poll() is None:
            proc.kill()


def test_worker_migrates_only_after_acquiring_exclusive_lock(tmp_path):
    """Worker invokes schema migration only after successfully acquiring WorkerLock."""
    db_path = tmp_path / "unmigrated.db"
    lock_file = tmp_path / "worker.lock"
    worker = RetrainingWorker(database_path=db_path, lock_path=lock_file)

    migration_lock_state = None

    def spy_migrate(path):
        nonlocal migration_lock_state
        migration_lock_state = worker.is_locked
        migrate_audit_database_idempotent(path)

    with patch(
        "backend.app.ml.worker.migrate_audit_database_idempotent",
        side_effect=spy_migrate,
    ):
        with worker:
            pass

    assert migration_lock_state is True
    assert worker.is_locked is False


def test_worker_lock_failure_does_not_migrate(tmp_path):
    """A worker that fails to acquire the exclusive lock never invokes migration."""
    db_path = tmp_path / "unmigrated.db"
    lock_file = tmp_path / "conflict.lock"
    worker1 = RetrainingWorker(
        database_path=db_path, lock_path=lock_file, worker_id="w1"
    )
    worker2 = RetrainingWorker(
        database_path=db_path, lock_path=lock_file, worker_id="w2"
    )

    worker1.acquire_lock()
    try:
        migrate_called = False

        def spy_migrate(path):
            nonlocal migrate_called
            migrate_called = True

        with patch(
            "backend.app.ml.worker.migrate_audit_database_idempotent",
            side_effect=spy_migrate,
        ):
            with pytest.raises(WorkerLockConflictError):
                with worker2:
                    pass

        assert not migrate_called
    finally:
        worker1.release_lock()


def test_worker_single_iteration_mode_initializes_schema_safely(tmp_path):
    """Worker single iteration execution initializes schema before polling."""
    db_path = tmp_path / "once_fresh.db"
    lock_file = tmp_path / "once.lock"
    assert not db_path.exists()

    worker = RetrainingWorker(database_path=db_path, lock_path=lock_file)
    with worker:
        res = worker.run_once()

    assert res is None
    assert db_path.exists()

    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        assert "retraining_jobs" in tables
        assert "drift_alerts" in tables
        assert "promotion_audit" in tables


def test_worker_main_once_flag_initializes_schema(tmp_path):
    """Worker main() with --once flag safely migrates fresh DB and exits 0."""
    db_path = tmp_path / "cli_once.db"
    lock_file = tmp_path / "cli_once.lock"
    assert not db_path.exists()

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "--lock-path",
            str(lock_file),
            "--handler",
            "noop",
            "--once",
        ]
    )
    assert exit_code == 0
    assert db_path.exists()

    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        assert "retraining_jobs" in tables


def test_worker_main_cli_requires_handler(migrated_db, tmp_path):
    """Worker CLI rejects invocation when --handler is omitted."""
    lock_file = tmp_path / "cli_no_handler.lock"
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--db-path",
                migrated_db,
                "--lock-path",
                str(lock_file),
                "--once",
            ]
        )
    assert exc_info.value.code == 2


def test_worker_execution_timeout_pre_marker_requeues_when_attempts_remain(
    migrated_db, tmp_path
):
    """Execution timeout before promotion marker requeues job when attempts remain."""
    job = create_queued_job(
        migrated_db,
        dataset_id="ds-timeout-1",
        trigger_source="manual",
        max_attempts=3,
    )
    jid = job["job_id"]

    cancelled_detected = threading.Event()

    def slow_handler(j, ctx):
        while not ctx.is_cancelled:
            time.sleep(0.01)
        cancelled_detected.set()
        raise RuntimeError("Cancelled by timeout")

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "timeout1.lock",
        worker_id="w-timeout-1",
        handler=slow_handler,
        job_timeout_seconds=0.1,
        shutdown_timeout=1.0,
    )

    with worker:
        requeued = worker.run_once()

    assert cancelled_detected.is_set()
    assert requeued is not None
    assert requeued["status"] == "queued"
    assert requeued["attempt_count"] == 1
    assert "Job execution timed out" in requeued["failure_reason"]

    persisted = get_job(migrated_db, job_id=jid)
    assert persisted["status"] == "queued"
    assert persisted["is_active"] == 1


def test_worker_execution_timeout_pre_marker_fails_when_attempts_exhausted(
    migrated_db, tmp_path
):
    """Execution timeout before promotion marker fails job when max attempts reached."""
    # Pre-set attempt_count so this run reaches max_attempts
    job = create_queued_job(
        migrated_db,
        dataset_id="ds-timeout-2",
        trigger_source="manual",
        max_attempts=2,
    )
    jid = job["job_id"]

    # First attempt fails to reach attempt 1
    with sqlite3.connect(migrated_db) as conn:
        conn.execute(
            "UPDATE retraining_jobs SET attempt_count = 1 WHERE job_id = ?",
            (jid,),
        )

    cancelled_detected = threading.Event()

    def slow_handler(j, ctx):
        while not ctx.is_cancelled:
            time.sleep(0.01)
        cancelled_detected.set()
        raise RuntimeError("Cancelled by timeout")

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "timeout2.lock",
        worker_id="w-timeout-2",
        handler=slow_handler,
        job_timeout_seconds=0.1,
        shutdown_timeout=1.0,
    )

    with worker:
        failed = worker.run_once()

    assert cancelled_detected.is_set()
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["is_active"] is None
    assert "attempts exhausted" in failed["failure_reason"]

    persisted = get_job(migrated_db, job_id=jid)
    assert persisted["status"] == "failed"
    assert persisted["is_active"] is None


def test_worker_execution_timeout_post_marker_quarantines(migrated_db, tmp_path):
    """Timeout after promotion_started_at commits quarantines job."""
    job = create_queued_job(
        migrated_db,
        dataset_id="ds-timeout-3",
        trigger_source="manual",
        max_attempts=3,
    )
    jid = job["job_id"]

    cancelled_detected = threading.Event()

    def post_marker_handler(j, ctx):
        mark_promotion_started(
            migrated_db,
            job_id=jid,
            worker_id=ctx.worker_id,
            lease_token=ctx.lease_token,
            candidate_version="5",
        )
        while not ctx.is_cancelled:
            time.sleep(0.01)
        cancelled_detected.set()
        raise RuntimeError("Cancelled by timeout after promotion marker")

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "timeout3.lock",
        worker_id="w-timeout-3",
        handler=post_marker_handler,
        job_timeout_seconds=0.1,
        shutdown_timeout=1.0,
    )

    with worker:
        quarantined = worker.run_once()

    assert cancelled_detected.is_set()
    assert quarantined is not None
    assert quarantined["status"] == "reconciliation_required"
    assert quarantined["is_active"] == 1
    assert "after promotion started" in quarantined["failure_reason"]

    # Invariants: no automatic retry
    persisted = get_job(migrated_db, job_id=jid)
    assert persisted["status"] == "reconciliation_required"
    assert persisted["attempt_count"] == 1


def test_worker_execution_timeout_handler_ignoring_cancellation_preserves_lock(
    migrated_db, tmp_path
):
    """Handler ignoring cancellation preserves lock and avoids DB mutation."""
    job = create_queued_job(
        migrated_db,
        dataset_id="ds-timeout-ignore",
        trigger_source="manual",
    )
    jid = job["job_id"]

    hang_event = threading.Event()

    def ignoring_handler(j, ctx):
        hang_event.wait(timeout=2.0)
        return JobExecutionResult(status="succeeded")

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "timeout_ignore.lock",
        worker_id="w-ignore-1",
        handler=ignoring_handler,
        job_timeout_seconds=0.1,
        shutdown_timeout=0.1,
    )

    worker.acquire_lock()
    try:
        # Single iteration: timeout will trigger, handler will ignore it
        result = worker.run_once()
        # Worker returns None because handler is still alive
        assert result is None
        assert worker.is_executing is True
        assert worker.is_locked is True

        # Database state was NOT mutated to failed or requeued!
        db_job = get_job(migrated_db, job_id=jid)
        assert db_job["status"] == "running"
        assert db_job["is_active"] == 1

        # WorkerLock cannot be released while work is active
        worker.release_lock()
        assert worker.is_locked is True

        # Worker refuses subsequent poll cycles while handler is active
        subsequent = worker.run_once()
        assert subsequent is None
    finally:
        hang_event.set()
        time.sleep(0.1)
        worker.release_lock(force=True)


def test_worker_execution_timeout_heartbeat_preserved_while_ignoring_cancellation(
    migrated_db, tmp_path
):
    """Heartbeat loop continues extending lease while handler is active."""
    job = create_queued_job(
        migrated_db,
        dataset_id="ds-timeout-hb",
        trigger_source="manual",
    )
    jid = job["job_id"]

    hang_event = threading.Event()

    def hanging_handler(j, ctx):
        hang_event.wait(timeout=0.8)
        return JobExecutionResult(status="succeeded")

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "timeout_hb.lock",
        worker_id="w-hb-1",
        handler=hanging_handler,
        heartbeat_interval_seconds=0.05,
        job_timeout_seconds=0.1,
        shutdown_timeout=0.05,
        lease_duration_seconds=5,
    )

    worker.acquire_lock()
    try:
        worker.run_once()
        # While handler is hanging past timeout, heartbeat continues
        time.sleep(0.15)
        job_after = get_job(migrated_db, job_id=jid)
        assert job_after is not None
        assert job_after["heartbeat_at"] is not None
        assert job_after["status"] == "running"
    finally:
        hang_event.set()
        time.sleep(0.1)
        worker.release_lock(force=True)


def test_worker_execution_timeout_shutdown_interaction(migrated_db, tmp_path):
    """Worker receives shutdown signal during timeout and cancels cleanly."""
    create_queued_job(
        migrated_db,
        dataset_id="ds-timeout-shutdown",
        trigger_source="manual",
    )

    cancelled_event = threading.Event()

    def slow_handler(j, ctx):
        while not ctx.is_cancelled:
            time.sleep(0.01)
        cancelled_event.set()
        raise RuntimeError("Cancelled")

    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=tmp_path / "timeout_shutdown.lock",
        worker_id="w-shutdown-1",
        handler=slow_handler,
        job_timeout_seconds=0.1,
        shutdown_timeout=0.5,
    )

    worker.acquire_lock()
    try:
        # Simulate simultaneous shutdown signal
        worker.stop()
        worker.process_job(claim_queued_job(migrated_db, worker_id=worker.worker_id))
        assert cancelled_event.is_set()
    finally:
        worker.release_lock()
    assert not worker.is_locked


def test_worker_sigterm_uncooperative_handler_containment_and_lock_retention(
    migrated_db, tmp_path
):
    """Worker in containment retains lock and heartbeats without state overwrite."""
    job = create_queued_job(
        migrated_db,
        dataset_id="ds-uncoop-contain",
        trigger_source="manual",
    )
    jid = job["job_id"]

    hang_event = threading.Event()

    def uncooperative_handler(j, ctx):
        hang_event.wait(timeout=2.0)
        return JobExecutionResult(status="succeeded")

    lock_file = tmp_path / "uncoop_contain.lock"
    worker = RetrainingWorker(
        database_path=migrated_db,
        lock_path=lock_file,
        worker_id="w-uncoop-1",
        handler=uncooperative_handler,
        heartbeat_interval_seconds=0.05,
        shutdown_timeout=0.05,
        job_timeout_seconds=0.1,
        lease_duration_seconds=5,
    )

    worker.acquire_lock()
    try:
        # Worker claims job and starts handler
        worker.run_once()

        # Simulate SIGTERM / shutdown signal while handler is alive
        worker.stop()

        # Verify containment properties while handler is alive:
        assert worker.is_executing is True
        assert worker.is_locked is True

        # 1. release_lock() refuses to release
        worker.release_lock()
        assert worker.is_locked is True

        # 2. Conflicting worker cannot acquire the held lock
        w_other = RetrainingWorker(
            database_path=migrated_db,
            lock_path=lock_file,
            worker_id="w-other",
        )
        with pytest.raises(WorkerLockConflictError):
            w_other.acquire_lock()

        # 3. SQLite job remains running with active lease (never failed/requeued)
        db_job = get_job(migrated_db, job_id=jid)
        assert db_job["status"] == "running"
        assert db_job["is_active"] == 1
        hb1 = db_job["heartbeat_at"]

        # 4. Heartbeat loop continues extending the lease in containment
        time.sleep(0.12)
        db_job2 = get_job(migrated_db, job_id=jid)
        assert db_job2["status"] == "running"
        assert db_job2["heartbeat_at"] is not None
        assert db_job2["heartbeat_at"] > hb1

        # 5. wait_for_containment with bounded timeout returns False while handler alive
        assert worker.wait_for_containment(timeout=0.05) is False

        # Now allow uncooperative handler to finish
        hang_event.set()

        # Containment resolves once handler terminates
        assert worker.wait_for_containment(timeout=2.0) is True
        assert worker.is_executing is False

        # Lock can now be released normally
        worker.release_lock()
        assert worker.is_locked is False
    finally:
        hang_event.set()
        time.sleep(0.05)
        worker.release_lock(force=True)


def test_subprocess_sigterm_uncooperative_handler_does_not_exit_zero(tmp_path):
    """Subprocess running worker does not exit 0 while handler is alive."""
    lock_file = tmp_path / "subproc_uncoop.lock"
    db_file = tmp_path / "subproc_uncoop.db"
    started_file = tmp_path / "started.txt"
    unblock_file = tmp_path / "unblock.txt"

    migrate_audit_database_idempotent(db_file)
    create_queued_job(
        db_file,
        dataset_id="ds-subproc-uncoop",
        trigger_source="manual",
    )

    code = f"""
import sys, time
from pathlib import Path
from backend.app.ml.worker import RetrainingWorker, JobExecutionResult

def stubborn_handler(job, ctx):
    Path(r'{started_file}').write_text("running")
    deadline = time.time() + 3.0
    while not Path(r'{unblock_file}').exists() and time.time() < deadline:
        time.sleep(0.05)
    return JobExecutionResult(status="succeeded")

worker = RetrainingWorker(
    database_path=r'{db_file}',
    lock_path=r'{lock_file}',
    handler=stubborn_handler,
    poll_interval_seconds=0.1,
    shutdown_timeout=0.1,
    job_timeout_seconds=5.0,
    heartbeat_interval_seconds=0.05,
)
worker.run()
"""
    proc = subprocess.Popen([sys.executable, "-c", code])
    try:
        # Wait for handler to start executing
        deadline = time.time() + 5.0
        while not started_file.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert started_file.exists()

        # Send SIGTERM to the worker subprocess while handler is running
        proc.send_signal(signal.SIGTERM)

        # Allow time for shutdown_timeout to elapse
        time.sleep(0.4)

        # Process MUST NOT have exited 0 while the handler is alive!
        assert proc.poll() is None

        # Lock must remain held by the subprocess
        w_other = RetrainingWorker(database_path=db_file, lock_path=lock_file)
        with pytest.raises(WorkerLockConflictError):
            w_other.acquire_lock()

        # Now signal the stubborn handler to finish
        unblock_file.write_text("done")

        # Subprocess should now resolve containment and exit 0 cleanly
        exit_code = proc.wait(timeout=5.0)
        assert exit_code == 0

        # After clean exit, lock is released
        w_other.acquire_lock()
        assert w_other.is_locked
        w_other.release_lock()
    finally:
        if proc.poll() is None:
            proc.kill()


def test_worker_main_returns_nonzero_when_handler_remains_executing(tmp_path):
    """main() returns 1 (never 0) if execution threads remain active after timeout."""
    db_file = tmp_path / "main_contain.db"
    lock_file = tmp_path / "main_contain.lock"
    migrate_audit_database_idempotent(db_file)
    create_queued_job(
        db_file,
        dataset_id="ds-main-contain",
        trigger_source="manual",
    )

    hang_event = threading.Event()

    def hanging_handler(job, ctx):
        hang_event.wait(timeout=2.0)
        return JobExecutionResult(status="succeeded")

    with patch(
        "backend.app.ml.training_handler.ControlledTrainingJobHandler",
        return_value=hanging_handler,
    ):
        # Mock wait_for_containment to simulate timeout elapsed while thread alive
        with patch.object(RetrainingWorker, "wait_for_containment", return_value=False):
            code = main(
                [
                    "--db-path",
                    str(db_file),
                    "--lock-path",
                    str(lock_file),
                    "--handler",
                    "production",
                    "--once",
                    "--job-timeout",
                    "0.05",
                    "--shutdown-timeout",
                    "0.05",
                ]
            )
            # Must return non-zero exit code (1), never 0!
            assert code == 1

    hang_event.set()
