"""Tests for job state machine, lease management, fencing, and recovery."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from backend.app.ml.db import db_connection, get_db_connection
from backend.app.ml.jobs import (
    ActiveJobExistsError,
    InvalidStateTransitionError,
    JobClaimError,
    JobNotFoundError,
    JobOwnershipError,
    StaleLeaseError,
    claim_queued_job,
    create_queued_job,
    detect_expired_leases,
    get_active_job,
    get_job,
    heartbeat_job,
    mark_exhausted_job_failed,
    mark_promotion_started,
    quarantine_expired_job,
    reclaim_expired_job,
    recover_expired_jobs,
    requeue_job_for_retry,
    transition_job_status,
    validate_job_ownership,
)
from backend.app.ml.migrations import migrate_audit_database_idempotent


@pytest.fixture
def migrated_db(tmp_path):
    """Fixture providing an initialized SQLite audit database."""
    db_path = tmp_path / "test_retraining.db"
    migrate_audit_database_idempotent(db_path)
    return str(db_path)


def test_foreign_keys_enforced_on_normal_connection(migrated_db):
    """Foreign keys are strictly enforced on standard get_db_connection instances."""
    with get_db_connection(migrated_db) as conn:
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            conn.execute(
                """
                INSERT INTO drift_alert_actions (
                    action_id, alert_id, timestamp, action_type, actor
                ) VALUES (
                    'act-1', 'nonexistent-alert', datetime('now'), 'test', 'tester'
                );
                """
            )
        assert "foreign key constraint failed" in str(exc_info.value).lower()


def test_db_connection_context_manager(migrated_db):
    """db_connection context manager automatically closes the SQLite connection."""
    with db_connection(migrated_db) as conn:
        cursor = conn.execute("SELECT 1;")
        assert cursor.fetchone() == (1,)

    # Connection should be closed after exiting context manager
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1;")


def test_create_and_fetch_queued_job(migrated_db):
    """Create a queued job and verify initial state machine attributes."""
    job = create_queued_job(
        migrated_db,
        dataset_id="dataset_2026_q4",
        trigger_source="manual_test",
        max_attempts=3,
        payload={"note": "first run"},
    )

    assert job["status"] == "queued"
    assert job["is_active"] == 1
    assert job["dataset_id"] == "dataset_2026_q4"
    assert job["trigger_source"] == "manual_test"
    assert job["attempt_count"] == 0
    assert job["max_attempts"] == 3
    assert job["worker_id"] is None
    assert job["lease_token"] is None
    assert job["started_at"] is None
    assert job["completed_at"] is None
    assert job["payload"] == {"note": "first run"}

    fetched = get_job(migrated_db, job_id=job["job_id"])
    assert fetched == job

    active = get_active_job(migrated_db)
    assert active == job


def test_unique_active_job_constraint(migrated_db):
    """Single active job constraint prevents creating multiple active jobs."""
    create_queued_job(
        migrated_db,
        dataset_id="dataset_1",
        trigger_source="manual",
    )

    # Second active job creation must be refused
    with pytest.raises(ActiveJobExistsError):
        create_queued_job(
            migrated_db,
            dataset_id="dataset_2",
            trigger_source="drift",
        )


def test_concurrent_job_creation(migrated_db):
    """Concurrent attempts to create active jobs result in exactly 1 active job."""
    errors: list[Exception] = []

    def _attempt_create(i: int):
        try:
            create_queued_job(
                migrated_db,
                dataset_id=f"dataset_{i}",
                trigger_source=f"thread_{i}",
            )
            return True
        except ActiveJobExistsError as exc:
            errors.append(exc)
            return False

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(_attempt_create, range(5)))

    created_count = sum(1 for r in results if r)
    assert created_count == 1
    assert len(errors) == 4

    active = get_active_job(migrated_db)
    assert active is not None
    assert active["status"] == "queued"


def test_claim_queued_job_lifecycle(migrated_db):
    """Worker claims queued job, receives unique lease token, and increments attempt."""
    job = create_queued_job(
        migrated_db,
        dataset_id="dataset_v1",
        trigger_source="drift",
    )

    claimed = claim_queued_job(
        migrated_db,
        worker_id="worker-node-1",
        lease_duration_seconds=30,
    )

    assert claimed is not None
    assert claimed["job_id"] == job["job_id"]
    assert claimed["status"] == "running"
    assert claimed["is_active"] == 1
    assert claimed["worker_id"] == "worker-node-1"
    assert claimed["lease_token"] is not None
    assert claimed["attempt_count"] == 1
    assert claimed["started_at"] is not None
    assert claimed["heartbeat_at"] is not None
    assert claimed["lease_expires_at"] is not None

    # Claiming again returns None because no jobs are in queued status
    assert claim_queued_job(migrated_db, worker_id="worker-node-2") is None


def test_claim_specific_job_error_when_not_queued(migrated_db):
    """Claiming a specific job ID raises JobClaimError if not in queued status."""
    job = create_queued_job(migrated_db, dataset_id="ds1", trigger_source="manual")
    claimed = claim_queued_job(migrated_db, worker_id="w1", job_id=job["job_id"])
    assert claimed is not None

    # Attempting to claim the already running job raises JobClaimError
    with pytest.raises(JobClaimError):
        claim_queued_job(migrated_db, worker_id="w2", job_id=job["job_id"])


def test_concurrent_job_claiming(migrated_db):
    """Multiple workers competing to claim the same queued job; exactly 1 succeeds."""
    create_queued_job(
        migrated_db,
        dataset_id="dataset_competing",
        trigger_source="manual",
    )

    def _attempt_claim(worker_idx: int):
        return claim_queued_job(
            migrated_db,
            worker_id=f"worker-{worker_idx}",
            lease_duration_seconds=45,
        )

    with ThreadPoolExecutor(max_workers=5) as pool:
        claims = list(pool.map(_attempt_claim, range(5)))

    successful_claims = [c for c in claims if c is not None]
    assert len(successful_claims) == 1
    winner = successful_claims[0]
    assert winner["status"] == "running"
    assert winner["attempt_count"] == 1


def test_heartbeat_updates_lease_expiry(migrated_db):
    """Active worker successfully extends lease expiry via heartbeat."""
    create_queued_job(
        migrated_db,
        dataset_id="ds1",
        trigger_source="manual",
    )
    claimed = claim_queued_job(
        migrated_db,
        worker_id="worker-1",
        lease_duration_seconds=10,
    )
    assert claimed is not None

    initial_expiry = claimed["lease_expires_at"]

    # Heartbeat with longer lease duration
    updated = heartbeat_job(
        migrated_db,
        job_id=claimed["job_id"],
        worker_id="worker-1",
        lease_token=claimed["lease_token"],
        lease_duration_seconds=60,
    )

    assert updated["lease_expires_at"] > initial_expiry
    assert updated["heartbeat_at"] >= claimed["heartbeat_at"]


def test_stale_heartbeat_rejection(migrated_db):
    """Heartbeat with wrong worker ID or stale lease token is rejected."""
    create_queued_job(migrated_db, dataset_id="ds1", trigger_source="manual")
    claimed = claim_queued_job(migrated_db, worker_id="worker-1")
    assert claimed is not None

    # Wrong worker ID
    with pytest.raises(JobOwnershipError):
        heartbeat_job(
            migrated_db,
            job_id=claimed["job_id"],
            worker_id="imposter-worker",
            lease_token=claimed["lease_token"],
        )

    # Wrong lease token
    with pytest.raises(JobOwnershipError):
        heartbeat_job(
            migrated_db,
            job_id=claimed["job_id"],
            worker_id="worker-1",
            lease_token="lease-invalid-token",
        )

    # Non-existent job
    with pytest.raises(JobNotFoundError):
        heartbeat_job(
            migrated_db,
            job_id="job-nonexistent",
            worker_id="worker-1",
            lease_token="lease-any",
        )


def test_validate_ownership_and_expiry(migrated_db):
    """validate_job_ownership verifies active holder and detects expired leases."""
    create_queued_job(migrated_db, dataset_id="ds1", trigger_source="manual")
    claimed = claim_queued_job(
        migrated_db,
        worker_id="worker-1",
        lease_duration_seconds=10,
    )
    assert claimed is not None

    # Valid ownership
    valid = validate_job_ownership(
        migrated_db,
        job_id=claimed["job_id"],
        worker_id="worker-1",
        lease_token=claimed["lease_token"],
    )
    assert valid["job_id"] == claimed["job_id"]

    # Future time check passes
    future_time = datetime.fromisoformat(claimed["lease_expires_at"]) + timedelta(
        seconds=5
    )
    with pytest.raises(StaleLeaseError):
        validate_job_ownership(
            migrated_db,
            job_id=claimed["job_id"],
            worker_id="worker-1",
            lease_token=claimed["lease_token"],
            now=future_time,
        )


def test_mark_promotion_started_fencing(migrated_db):
    """Setting promotion marker is fenced and irreversible."""
    create_queued_job(migrated_db, dataset_id="ds1", trigger_source="manual")
    claimed = claim_queued_job(migrated_db, worker_id="worker-1")
    assert claimed is not None
    jid = claimed["job_id"]
    token = claimed["lease_token"]

    # Fails with wrong token
    with pytest.raises(JobOwnershipError):
        mark_promotion_started(
            migrated_db,
            job_id=jid,
            worker_id="worker-1",
            lease_token="wrong-token",
        )

    # Successfully marked
    marked = mark_promotion_started(
        migrated_db,
        job_id=jid,
        worker_id="worker-1",
        lease_token=token,
        candidate_version="2",
        candidate_run_id="run-abc-123",
        previous_champion_version="1",
    )
    assert marked["promotion_started_at"] is not None
    assert marked["candidate_version"] == "2"

    # Second call fails because promotion_started_at IS NOT NULL
    with pytest.raises(InvalidStateTransitionError):
        mark_promotion_started(
            migrated_db,
            job_id=jid,
            worker_id="worker-1",
            lease_token=token,
        )


def test_fenced_state_transitions(migrated_db):
    """Fenced transitions to terminal states clear is_active and set completed_at."""
    create_queued_job(migrated_db, dataset_id="ds1", trigger_source="manual")
    claimed = claim_queued_job(migrated_db, worker_id="worker-1")
    assert claimed is not None
    jid = claimed["job_id"]
    token = claimed["lease_token"]

    # Transition to succeeded
    succeeded = transition_job_status(
        migrated_db,
        job_id=jid,
        worker_id="worker-1",
        lease_token=token,
        target_status="succeeded",
        promotion_approved=True,
        promotion_reason="exceeded champion f1",
        payload_updates={"eval_score": 0.88},
    )

    assert succeeded["status"] == "succeeded"
    assert succeeded["is_active"] is None
    assert succeeded["completed_at"] is not None
    assert succeeded["promotion_approved"] is True
    assert succeeded["payload"]["eval_score"] == 0.88

    # Now no job is active; a new job can be created
    new_job = create_queued_job(
        migrated_db,
        dataset_id="ds2",
        trigger_source="drift",
    )
    assert new_job["is_active"] == 1


def test_detect_expired_leases(migrated_db):
    """detect_expired_leases identifies running jobs past their lease expiry."""
    create_queued_job(migrated_db, dataset_id="ds1", trigger_source="manual")
    claimed = claim_queued_job(
        migrated_db,
        worker_id="worker-1",
        lease_duration_seconds=10,
    )
    assert claimed is not None

    # Before expiry: 0 expired
    assert len(detect_expired_leases(migrated_db)) == 0

    # Simulate time 20 seconds later
    future_time = datetime.now(UTC) + timedelta(seconds=20)
    expired = detect_expired_leases(migrated_db, now=future_time)
    assert len(expired) == 1
    assert expired[0]["job_id"] == claimed["job_id"]


def test_pre_promotion_lease_reclaim(migrated_db):
    """Expired running job pre-promotion is safely reclaimed to queued."""
    create_queued_job(
        migrated_db,
        dataset_id="ds1",
        trigger_source="manual",
        max_attempts=3,
    )
    claimed = claim_queued_job(
        migrated_db,
        worker_id="worker-crash-1",
        lease_duration_seconds=5,
    )
    assert claimed is not None
    jid = claimed["job_id"]

    # Simulate passage of time past lease expiry
    future_time = datetime.now(UTC) + timedelta(seconds=10)

    # Reclaim to queued
    reclaimed = reclaim_expired_job(migrated_db, job_id=jid, now=future_time)
    assert reclaimed["status"] == "queued"
    assert reclaimed["is_active"] == 1
    assert reclaimed["worker_id"] is None
    assert reclaimed["lease_token"] is None
    assert reclaimed["lease_expires_at"] is None
    assert reclaimed["attempt_count"] == 1  # Retained attempt count

    # Another worker can now claim attempt 2
    claimed2 = claim_queued_job(migrated_db, worker_id="worker-recovered")
    assert claimed2 is not None
    assert claimed2["job_id"] == jid
    assert claimed2["attempt_count"] == 2


def test_blocked_reclaim_when_promotion_started(migrated_db):
    """Job with promotion_started_at set cannot be automatically reclaimed."""
    create_queued_job(migrated_db, dataset_id="ds1", trigger_source="manual")
    claimed = claim_queued_job(
        migrated_db,
        worker_id="worker-1",
        lease_duration_seconds=5,
    )
    assert claimed is not None
    jid = claimed["job_id"]

    # Mark promotion started
    mark_promotion_started(
        migrated_db,
        job_id=jid,
        worker_id="worker-1",
        lease_token=claimed["lease_token"],
    )

    future_time = datetime.now(UTC) + timedelta(seconds=10)

    # Reclaim attempt must be rejected
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        reclaim_expired_job(migrated_db, job_id=jid, now=future_time)
    assert "promotion has already started" in str(exc_info.value).lower()
    assert "quarantined" in str(exc_info.value).lower()


def test_post_promotion_marker_quarantine(migrated_db):
    """Expired job where promotion started is quarantined into reconciliation."""
    create_queued_job(migrated_db, dataset_id="ds1", trigger_source="manual")
    claimed = claim_queued_job(
        migrated_db,
        worker_id="worker-1",
        lease_duration_seconds=5,
    )
    assert claimed is not None
    jid = claimed["job_id"]

    mark_promotion_started(
        migrated_db,
        job_id=jid,
        worker_id="worker-1",
        lease_token=claimed["lease_token"],
        candidate_version="5",
    )

    future_time = datetime.now(UTC) + timedelta(seconds=10)

    # Quarantine into reconciliation_required
    quarantined = quarantine_expired_job(migrated_db, job_id=jid, now=future_time)
    assert quarantined["status"] == "reconciliation_required"
    assert quarantined["is_active"] == 1  # Remains active to block new jobs
    assert "manual" in quarantined["failure_reason"].lower()

    # Verify no new job can be created while quarantined
    with pytest.raises(ActiveJobExistsError):
        create_queued_job(migrated_db, dataset_id="ds2", trigger_source="manual")


def test_exhausted_attempts_marked_failed(migrated_db):
    """Pre-promotion job that exhausts max_attempts is marked terminal failed."""
    create_queued_job(
        migrated_db,
        dataset_id="ds1",
        trigger_source="manual",
        max_attempts=2,
    )

    # Claim attempt 1
    c1 = claim_queued_job(migrated_db, worker_id="w1", lease_duration_seconds=5)
    assert c1 is not None
    jid = c1["job_id"]

    t1 = datetime.now(UTC) + timedelta(seconds=10)
    reclaim_expired_job(migrated_db, job_id=jid, now=t1)

    # Claim attempt 2 (max_attempts reached)
    c2 = claim_queued_job(migrated_db, worker_id="w2", lease_duration_seconds=5)
    assert c2 is not None
    assert c2["attempt_count"] == 2

    t2 = datetime.now(UTC) + timedelta(seconds=20)

    # Cannot reclaim because attempts are exhausted
    with pytest.raises(InvalidStateTransitionError):
        reclaim_expired_job(migrated_db, job_id=jid, now=t2)

    # Mark exhausted failed
    failed = mark_exhausted_job_failed(migrated_db, job_id=jid, now=t2)
    assert failed["status"] == "failed"
    assert failed["is_active"] is None
    assert failed["completed_at"] is not None
    assert "exhausted" in failed["failure_reason"].lower()

    # Now that job is terminal failed, new active jobs are allowed
    new_job = create_queued_job(
        migrated_db, dataset_id="ds_retry", trigger_source="manual"
    )
    assert new_job["is_active"] == 1


def test_recover_expired_jobs_dispatcher(migrated_db):
    """recover_expired_jobs routes expired jobs to correct recovery action."""
    create_queued_job(
        migrated_db,
        dataset_id="ds1",
        trigger_source="manual",
        max_attempts=3,
    )
    claimed = claim_queued_job(migrated_db, worker_id="w1", lease_duration_seconds=5)
    assert claimed is not None

    future_time = datetime.now(UTC) + timedelta(seconds=15)

    # First recovery: reclaims to queued (attempt 1 < 3)
    outcomes = recover_expired_jobs(migrated_db, now=future_time)
    assert len(outcomes) == 1
    assert outcomes[0]["action"] == "reclaimed_queued"
    assert outcomes[0]["job"]["status"] == "queued"


def test_requeue_job_for_retry_success(migrated_db):
    """Running job is requeued for retry preserving attempt count and active status."""
    create_queued_job(
        migrated_db, dataset_id="ds1", trigger_source="manual", max_attempts=3
    )
    claimed = claim_queued_job(
        migrated_db, worker_id="w-requeue", lease_duration_seconds=60
    )
    assert claimed is not None
    assert claimed["attempt_count"] == 1
    jid = claimed["job_id"]

    requeued = requeue_job_for_retry(
        migrated_db,
        job_id=jid,
        worker_id="w-requeue",
        lease_token=claimed["lease_token"],
        failure_reason="Transient failure on attempt 1",
    )

    assert requeued["status"] == "queued"
    assert requeued["is_active"] == 1
    assert requeued["worker_id"] is None
    assert requeued["lease_token"] is None
    assert requeued["heartbeat_at"] is None
    assert requeued["lease_expires_at"] is None
    assert requeued["attempt_count"] == 1
    assert "Transient failure" in requeued["failure_reason"]

    # Can be claimed again for attempt 2
    claimed2 = claim_queued_job(migrated_db, worker_id="w-retry-2")
    assert claimed2 is not None
    assert claimed2["job_id"] == jid
    assert claimed2["attempt_count"] == 2


def test_requeue_job_for_retry_blocked_after_promotion_started(migrated_db):
    """Job cannot be requeued for retry after promotion_started_at is set."""
    create_queued_job(
        migrated_db, dataset_id="ds1", trigger_source="manual", max_attempts=3
    )
    claimed = claim_queued_job(
        migrated_db, worker_id="w-requeue", lease_duration_seconds=60
    )
    assert claimed is not None
    jid = claimed["job_id"]

    mark_promotion_started(
        migrated_db,
        job_id=jid,
        worker_id="w-requeue",
        lease_token=claimed["lease_token"],
        candidate_version="v-cand",
    )

    with pytest.raises(InvalidStateTransitionError) as exc_info:
        requeue_job_for_retry(
            migrated_db,
            job_id=jid,
            worker_id="w-requeue",
            lease_token=claimed["lease_token"],
            failure_reason="Should fail",
        )
    assert "promotion has already started" in str(exc_info.value).lower()


def test_requeue_job_for_retry_blocked_when_attempts_exhausted(migrated_db):
    """Job cannot be requeued when attempts have reached max_attempts."""
    create_queued_job(
        migrated_db, dataset_id="ds1", trigger_source="manual", max_attempts=1
    )
    claimed = claim_queued_job(
        migrated_db, worker_id="w-requeue", lease_duration_seconds=60
    )
    assert claimed is not None
    assert claimed["attempt_count"] == 1
    jid = claimed["job_id"]

    with pytest.raises(InvalidStateTransitionError) as exc_info:
        requeue_job_for_retry(
            migrated_db,
            job_id=jid,
            worker_id="w-requeue",
            lease_token=claimed["lease_token"],
            failure_reason="Attempts exhausted",
        )
    assert "exhausted" in str(exc_info.value).lower()
