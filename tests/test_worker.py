"""Tests for worker execution infrastructure, lifecycle, and safety mechanisms."""

import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

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


def test_worker_startup_and_exclusive_lock_acquisition(tmp_path):
    """Worker acquires and retains exclusive lock descriptor throughout its lifetime."""
    lock_file = tmp_path / "exclusive_worker.lock"
    worker = RetrainingWorker(lock_path=lock_file)

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
    worker1 = RetrainingWorker(lock_path=lock_file, worker_id="worker-primary")
    worker2 = RetrainingWorker(lock_path=lock_file, worker_id="worker-secondary")

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
    worker_thread.join(timeout=2.0)

    # Worker polling thread stopped, but handler is still alive in background
    assert not worker_thread.is_alive()
    assert worker.is_executing
    assert worker.is_locked

    # Another worker CANNOT acquire the lock because it is retained
    worker2 = RetrainingWorker(lock_path=lock_file, worker_id="w-intruder")
    with pytest.raises(WorkerLockConflictError):
        worker2.acquire_lock()

    # release_lock() refuses to release while handler is executing
    worker.release_lock()
    assert worker.is_locked

    # Allow stubborn handler to exit cleanly
    allow_handler_exit.set()
    deadline = time.time() + 2.0
    while worker.is_executing and time.time() < deadline:
        time.sleep(0.05)
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

    code = f"""
import sys, time
from pathlib import Path
from backend.app.ml.worker import RetrainingWorker

worker = RetrainingWorker(lock_path=r'{lock_file}')
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
        worker2 = RetrainingWorker(lock_path=lock_file)
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
                "--once",
            ]
        )
        assert conflict_ret == 1


def test_subprocess_sigterm_graceful_shutdown(tmp_path):
    """Subprocess running RetrainingWorker handles SIGTERM gracefully and exits 0."""
    lock_file = tmp_path / "sigterm.lock"
    ready_file = tmp_path / "ready.txt"

    code = f"""
import sys, time
from pathlib import Path
from backend.app.ml.worker import RetrainingWorker

worker = RetrainingWorker(lock_path=r'{lock_file}', poll_interval_seconds=1.0)
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
        w = RetrainingWorker(lock_path=lock_file)
        w.acquire_lock()
        assert w.is_locked
        w.release_lock()
    finally:
        if proc.poll() is None:
            proc.kill()
