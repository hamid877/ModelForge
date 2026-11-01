"""Job state machine and persistence operations for controlled retraining orchestration.

Provides atomic, fenced state transitions, lease management, heartbeat updates,
and recovery mechanics for retraining jobs in SQLite.
"""

import json
import logging
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.config import settings
from backend.app.ml.db import db_connection, immediate_transaction
from backend.app.ml.migrations import (
    TERMINAL_JOB_STATUSES,
)

logger = logging.getLogger(__name__)


# Domain Exceptions
class JobError(RuntimeError):
    """Base exception for retraining job operations."""


class ActiveJobExistsError(JobError):
    """Raised when an operation would violate the single active job constraint."""


class JobNotFoundError(JobError):
    """Raised when a specified job does not exist."""


class JobOwnershipError(JobError):
    """Raised when worker identity or lease token does not match active job owner."""


class StaleLeaseError(JobOwnershipError):
    """Raised when a lease has expired or is invalid for the requested operation."""


class JobClaimError(JobError):
    """Raised when a worker cannot claim a job."""


class JobStateConflictError(JobError):
    """Raised when a job is not in the expected state for an operation."""


class InvalidStateTransitionError(JobError):
    """Raised when an illegal or unsafe state transition is attempted."""


def _row_to_dict(cursor: sqlite3.Cursor, row: tuple[Any, ...]) -> dict[str, Any]:
    """Convert an SQLite row tuple into a typed dictionary."""
    col_names = [col[0] for col in cursor.description]
    item = dict(zip(col_names, row, strict=False))
    if "payload" in item and isinstance(item["payload"], str):
        try:
            item["payload"] = json.loads(item["payload"])
        except (json.JSONDecodeError, TypeError):
            pass
    if "promotion_approved" in item and item["promotion_approved"] is not None:
        item["promotion_approved"] = bool(item["promotion_approved"])
    return item


def create_queued_job(
    database_path: str | Path | None = None,
    *,
    dataset_id: str,
    trigger_source: str,
    alert_id: str | None = None,
    max_attempts: int | None = None,
    payload: Mapping[str, Any] | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Create a new retraining job in queued state.

    Enforces single-active-job constraint. If an active job already exists,
    raises ActiveJobExistsError.

    Args:
        database_path: Optional path to SQLite database.
        dataset_id: Dataset identifier for retraining.
        trigger_source: Source that requested retraining (e.g. 'manual', 'drift').
        alert_id: Optional drift alert ID foreign key reference.
        max_attempts: Maximum retry attempts before failure. Defaults to settings.
        payload: Optional arbitrary metadata payload mapping.
        job_id: Optional custom job ID. If None, generated as 'job-<uuid4>'.

    Returns:
        Dictionary representing the created job record.
    """
    if not dataset_id or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string.")
    if not trigger_source or not trigger_source.strip():
        raise ValueError("trigger_source must be a non-empty string.")

    effective_max_attempts = (
        max_attempts
        if max_attempts is not None
        else getattr(settings, "retraining_max_attempts", 3)
    )
    if effective_max_attempts < 1:
        raise ValueError("max_attempts must be >= 1.")

    jid = job_id or f"job-{uuid4()}"
    created_at = datetime.now(UTC).isoformat()
    serialized_payload = json.dumps(dict(payload or {}), allow_nan=False)

    with db_connection(database_path, isolation_level=None) as conn:
        with immediate_transaction(conn):
            try:
                conn.execute(
                    """
                    INSERT INTO retraining_jobs (
                        job_id, created_at, status, is_active, dataset_id,
                        trigger_source, alert_id, attempt_count, max_attempts, payload
                    ) VALUES (
                        ?, ?, 'queued', 1, ?,
                        ?, ?, 0, ?, ?
                    );
                    """,
                    (
                        jid,
                        created_at,
                        dataset_id.strip(),
                        trigger_source.strip(),
                        alert_id,
                        effective_max_attempts,
                        serialized_payload,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                exc_str = str(exc).lower()
                if (
                    "idx_single_active_retraining_job" in exc_str
                    or "retraining_jobs.is_active" in exc_str
                ):
                    raise ActiveJobExistsError(
                        "Cannot create queued job: An active retraining job "
                        "already exists in the system."
                    ) from exc
                raise

            cursor = conn.execute(
                "SELECT * FROM retraining_jobs WHERE job_id = ?;", (jid,)
            )
            row = cursor.fetchone()
            if row is None:
                raise JobNotFoundError(f"Created job '{jid}' could not be fetched.")
            return _row_to_dict(cursor, row)


def claim_queued_job(
    database_path: str | Path | None = None,
    *,
    worker_id: str,
    lease_duration_seconds: int | None = None,
    job_id: str | None = None,
) -> dict[str, Any] | None:
    """Atomically claim a queued job, assigning a worker ID, lease token, and attempt.

    Args:
        database_path: Optional path to SQLite database.
        worker_id: Unique identifier for the claiming worker.
        lease_duration_seconds: Duration in seconds for lease expiration.
        job_id: Optional specific queued job ID to claim.

    Returns:
        Updated job record as dictionary if claimed, or None if unavailable.
    """
    if not worker_id or not worker_id.strip():
        raise ValueError("worker_id must be a non-empty string.")

    lease_dur = (
        lease_duration_seconds
        if lease_duration_seconds is not None
        else getattr(settings, "retraining_lease_duration_seconds", 60)
    )
    if lease_dur < 1:
        raise ValueError("lease_duration_seconds must be >= 1.")

    with db_connection(database_path, isolation_level=None) as conn:
        with immediate_transaction(conn):
            target_job_id: str | None = None
            if job_id is None:
                cursor = conn.execute(
                    """
                    SELECT job_id
                    FROM retraining_jobs
                    WHERE status = 'queued' AND is_active = 1
                    ORDER BY created_at ASC
                    LIMIT 1;
                    """
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                target_job_id = row[0]
            else:
                target_job_id = job_id

            lease_token = f"lease-{uuid4()}"
            now = datetime.now(UTC)
            now_iso = now.isoformat()
            expires_iso = (now + timedelta(seconds=lease_dur)).isoformat()

            cursor = conn.execute(
                """
                UPDATE retraining_jobs
                SET status = 'running',
                    worker_id = ?,
                    lease_token = ?,
                    heartbeat_at = ?,
                    lease_expires_at = ?,
                    started_at = COALESCE(started_at, ?),
                    attempt_count = attempt_count + 1
                WHERE job_id = ?
                  AND status = 'queued'
                  AND is_active = 1;
                """,
                (
                    worker_id.strip(),
                    lease_token,
                    now_iso,
                    expires_iso,
                    now_iso,
                    target_job_id,
                ),
            )

            if cursor.rowcount != 1:
                if job_id is not None:
                    raise JobClaimError(
                        f"Job '{job_id}' cannot be claimed: not in 'queued' state."
                    )
                return None

            cursor = conn.execute(
                "SELECT * FROM retraining_jobs WHERE job_id = ?;", (target_job_id,)
            )
            claimed_row = cursor.fetchone()
            if claimed_row is None:
                raise JobNotFoundError(f"Claimed job '{target_job_id}' not found.")
            return _row_to_dict(cursor, claimed_row)


def heartbeat_job(
    database_path: str | Path | None = None,
    *,
    job_id: str,
    worker_id: str,
    lease_token: str,
    lease_duration_seconds: int | None = None,
) -> dict[str, Any]:
    """Update heartbeat timestamp and extend lease expiry for an actively owned job.

    Fenced by job_id, worker_id, lease_token, status='running', and is_active=1.
    Rejects stale or unowned heartbeat attempts.

    Args:
        database_path: Optional path to SQLite database.
        job_id: ID of the running job.
        worker_id: Worker ID asserting ownership.
        lease_token: Unique lease token issued upon claiming.
        lease_duration_seconds: Extended lease duration in seconds from current time.

    Returns:
        Updated job record as dictionary.
    """
    if not job_id:
        raise ValueError("job_id must be provided.")
    if not worker_id:
        raise ValueError("worker_id must be provided.")
    if not lease_token:
        raise ValueError("lease_token must be provided.")

    lease_dur = (
        lease_duration_seconds
        if lease_duration_seconds is not None
        else getattr(settings, "retraining_lease_duration_seconds", 60)
    )
    if lease_dur < 1:
        raise ValueError("lease_duration_seconds must be >= 1.")

    now = datetime.now(UTC)
    now_iso = now.isoformat()
    expires_iso = (now + timedelta(seconds=lease_dur)).isoformat()

    with db_connection(database_path, isolation_level=None) as conn:
        with immediate_transaction(conn):
            cursor = conn.execute(
                """
                UPDATE retraining_jobs
                SET heartbeat_at = ?,
                    lease_expires_at = ?
                WHERE job_id = ?
                  AND worker_id = ?
                  AND lease_token = ?
                  AND status = 'running'
                  AND is_active = 1;
                """,
                (now_iso, expires_iso, job_id, worker_id, lease_token),
            )

            if cursor.rowcount != 1:
                chk = conn.execute(
                    "SELECT status, is_active, worker_id, lease_token "
                    "FROM retraining_jobs WHERE job_id = ?;",
                    (job_id,),
                ).fetchone()
                if chk is None:
                    raise JobNotFoundError(f"Job '{job_id}' does not exist.")
                raise JobOwnershipError(
                    f"Heartbeat rejected: Worker '{worker_id}' with token "
                    f"'{lease_token}' is not active owner of running job '{job_id}' "
                    f"(current status='{chk[0]}', active={chk[1]}, "
                    f"worker='{chk[2]}', token='{chk[3]}')."
                )

            cursor = conn.execute(
                "SELECT * FROM retraining_jobs WHERE job_id = ?;", (job_id,)
            )
            row = cursor.fetchone()
            if row is None:
                raise JobNotFoundError(f"Job '{job_id}' not found.")
            return _row_to_dict(cursor, row)


def validate_job_ownership(
    database_path: str | Path | None = None,
    *,
    job_id: str,
    worker_id: str,
    lease_token: str,
    check_lease_expiry: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate that the worker and lease token hold active ownership of a job.

    Args:
        database_path: Optional path to SQLite database.
        job_id: ID of the job to validate.
        worker_id: Worker identifier claiming ownership.
        lease_token: Lease token claiming ownership.
        check_lease_expiry: If True, validates that lease_expires_at > now.
        now: Optional reference timestamp. Defaults to UTC now.

    Returns:
        Job record dictionary if valid.

    Raises:
        JobNotFoundError: If job does not exist.
        JobOwnershipError: If status != 'running', is_active != 1, or tokens mismatch.
        StaleLeaseError: If lease has expired.
    """
    with db_connection(database_path, isolation_level=None) as conn:
        cursor = conn.execute(
            "SELECT * FROM retraining_jobs WHERE job_id = ?;", (job_id,)
        )
        row = cursor.fetchone()
        if row is None:
            raise JobNotFoundError(f"Job '{job_id}' does not exist.")

        job = _row_to_dict(cursor, row)

    if job["status"] != "running" or job["is_active"] != 1:
        raise JobOwnershipError(
            f"Job '{job_id}' is not in active running state "
            f"(status='{job['status']}', active={job['is_active']})."
        )

    if job.get("worker_id") != worker_id or job.get("lease_token") != lease_token:
        raise JobOwnershipError(
            f"Ownership mismatch for job '{job_id}': expected worker="
            f"'{job.get('worker_id')}', token='{job.get('lease_token')}'; "
            f"provided worker='{worker_id}', token='{lease_token}'."
        )

    if check_lease_expiry and job.get("lease_expires_at"):
        curr = now or datetime.now(UTC)
        expiry = datetime.fromisoformat(job["lease_expires_at"])
        if expiry <= curr:
            raise StaleLeaseError(
                f"Lease for job '{job_id}' expired at {job['lease_expires_at']} "
                f"(current time: {curr.isoformat()})."
            )

    return job


def persist_candidate_identity(
    database_path: str | Path | None = None,
    *,
    job_id: str,
    worker_id: str,
    lease_token: str,
    candidate_version: str,
    candidate_run_id: str | None = None,
    previous_champion_version: str | None = None,
) -> dict[str, Any]:
    """Persist candidate and previous champion identities for an actively owned job.

    Fenced by job_id, worker_id, lease_token, status='running', and is_active=1.
    Strictly requires exactly one affected row.

    Args:
        database_path: Optional path to SQLite database.
        job_id: ID of the running job.
        worker_id: Active worker identifier.
        lease_token: Active lease token.
        candidate_version: Registered candidate model version string.
        candidate_run_id: Optional MLflow run ID string.
        previous_champion_version: Optional observed champion version string.

    Returns:
        Updated job record as dictionary.

    Raises:
        JobNotFoundError: If job does not exist.
        JobOwnershipError: If worker/token mismatch or job not in running state.
    """
    if not job_id:
        raise ValueError("job_id must be provided.")
    if not worker_id:
        raise ValueError("worker_id must be provided.")
    if not lease_token:
        raise ValueError("lease_token must be provided.")
    if not candidate_version:
        raise ValueError("candidate_version must be provided.")

    with db_connection(database_path, isolation_level=None) as conn:
        with immediate_transaction(conn):
            cursor = conn.execute(
                """
                UPDATE retraining_jobs
                SET candidate_version = ?,
                    candidate_run_id = ?,
                    previous_champion_version = ?
                WHERE job_id = ?
                  AND worker_id = ?
                  AND lease_token = ?
                  AND status = 'running'
                  AND is_active = 1;
                """,
                (
                    str(candidate_version),
                    str(candidate_run_id) if candidate_run_id is not None else None,
                    (
                        str(previous_champion_version)
                        if previous_champion_version is not None
                        else None
                    ),
                    job_id,
                    worker_id,
                    lease_token,
                ),
            )

            if cursor.rowcount != 1:
                chk = conn.execute(
                    "SELECT status, is_active, worker_id, lease_token "
                    "FROM retraining_jobs WHERE job_id = ?;",
                    (job_id,),
                ).fetchone()
                if chk is None:
                    raise JobNotFoundError(f"Job '{job_id}' does not exist.")
                raise JobOwnershipError(
                    f"Failed to persist candidate identity: Worker '{worker_id}' "
                    f"with token '{lease_token}' is not active owner of running job "
                    f"'{job_id}' (current status='{chk[0]}', active={chk[1]})."
                )

            cursor = conn.execute(
                "SELECT * FROM retraining_jobs WHERE job_id = ?;", (job_id,)
            )
            row = cursor.fetchone()
            if row is None:
                raise JobNotFoundError(f"Job '{job_id}' not found.")
            return _row_to_dict(cursor, row)


def mark_promotion_started(
    database_path: str | Path | None = None,
    *,
    job_id: str,
    worker_id: str,
    lease_token: str,
    candidate_version: str | None = None,
    candidate_run_id: str | None = None,
    previous_champion_version: str | None = None,
    promotion_started_at: str | None = None,
) -> dict[str, Any]:
    """Mark that promotion evaluation has begun for a running job.

    This irreversible marker prevents any subsequent automatic lease reclamation.
    Fenced by ownership and status='running' with promotion_started_at IS NULL.

    Args:
        database_path: Optional path to SQLite database.
        job_id: ID of the running job.
        worker_id: Active worker identifier.
        lease_token: Active lease token.
        candidate_version: Optional candidate model version string.
        candidate_run_id: Optional MLflow run ID for candidate model.
        previous_champion_version: Optional observed champion version string.
        promotion_started_at: Optional ISO timestamp. Defaults to UTC now.

    Returns:
        Updated job record as dictionary.
    """
    ts = promotion_started_at or datetime.now(UTC).isoformat()

    with db_connection(database_path, isolation_level=None) as conn:
        with immediate_transaction(conn):
            cursor = conn.execute(
                """
                UPDATE retraining_jobs
                SET promotion_started_at = ?,
                    candidate_version = COALESCE(?, candidate_version),
                    candidate_run_id = COALESCE(?, candidate_run_id),
                    previous_champion_version = COALESCE(?, previous_champion_version)
                WHERE job_id = ?
                  AND worker_id = ?
                  AND lease_token = ?
                  AND status = 'running'
                  AND is_active = 1
                  AND promotion_started_at IS NULL;
                """,
                (
                    ts,
                    candidate_version,
                    candidate_run_id,
                    previous_champion_version,
                    job_id,
                    worker_id,
                    lease_token,
                ),
            )

            if cursor.rowcount != 1:
                chk = conn.execute(
                    "SELECT status, is_active, worker_id, lease_token, "
                    "promotion_started_at FROM retraining_jobs WHERE job_id = ?;",
                    (job_id,),
                ).fetchone()
                if chk is None:
                    raise JobNotFoundError(f"Job '{job_id}' does not exist.")
                if chk[4] is not None:
                    raise InvalidStateTransitionError(
                        f"Promotion marker already set for job '{job_id}' at {chk[4]}."
                    )
                raise JobOwnershipError(
                    f"Failed to set promotion marker: Worker '{worker_id}' with token "
                    f"'{lease_token}' is not active owner of running job '{job_id}'."
                )

            cursor = conn.execute(
                "SELECT * FROM retraining_jobs WHERE job_id = ?;", (job_id,)
            )
            row = cursor.fetchone()
            if row is None:
                raise JobNotFoundError(f"Job '{job_id}' not found.")
            return _row_to_dict(cursor, row)


def transition_job_status(
    database_path: str | Path | None = None,
    *,
    job_id: str,
    worker_id: str,
    lease_token: str,
    target_status: str,
    failure_reason: str | None = None,
    promotion_approved: bool | None = None,
    promotion_reason: str | None = None,
    promotion_audit_id: str | None = None,
    candidate_version: str | None = None,
    candidate_run_id: str | None = None,
    previous_champion_version: str | None = None,
    payload_updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform fenced state transition for a running job.

    Transitions to 'succeeded', 'rejected', or 'failed' set is_active=NULL.
    Transitions to 'reconciliation_required' retain is_active=1.
    Fenced by job_id, worker_id, lease_token, status='running', and is_active=1.

    Args:
        database_path: Optional path to SQLite database.
        job_id: ID of the running job.
        worker_id: Active worker identifier.
        lease_token: Active lease token.
        target_status: Target status ('succeeded', 'rejected', 'failed', etc.).
        failure_reason: Optional failure or rejection message.
        promotion_approved: Optional boolean indicating promotion gate outcome.
        promotion_reason: Optional explanation of promotion decision.
        promotion_audit_id: Optional ID of the associated promotion audit record.
        candidate_version: Optional candidate model version string.
        candidate_run_id: Optional candidate MLflow run ID string.
        previous_champion_version: Optional previous champion version string.
        payload_updates: Optional dictionary of metadata updates merged into payload.

    Returns:
        Updated job record as dictionary.
    """
    valid_targets = {"succeeded", "rejected", "failed", "reconciliation_required"}
    if target_status not in valid_targets:
        raise InvalidStateTransitionError(
            f"Invalid target_status '{target_status}'. Must be one of: "
            f"{sorted(valid_targets)}."
        )

    now_iso = datetime.now(UTC).isoformat()
    if target_status in TERMINAL_JOB_STATUSES:
        new_is_active = None
        completed_at = now_iso
    else:  # reconciliation_required
        new_is_active = 1
        completed_at = None

    approved_int = (
        (1 if promotion_approved else 0) if promotion_approved is not None else None
    )

    with db_connection(database_path, isolation_level=None) as conn:
        with immediate_transaction(conn):
            cursor = conn.execute(
                "SELECT payload FROM retraining_jobs "
                "WHERE job_id = ? AND worker_id = ? AND lease_token = ? "
                "AND status = 'running' AND is_active = 1;",
                (job_id, worker_id, lease_token),
            )
            curr = cursor.fetchone()
            if curr is None:
                chk = conn.execute(
                    "SELECT status, is_active, worker_id, lease_token "
                    "FROM retraining_jobs WHERE job_id = ?;",
                    (job_id,),
                ).fetchone()
                if chk is None:
                    raise JobNotFoundError(f"Job '{job_id}' does not exist.")
                raise JobOwnershipError(
                    f"Fenced transition rejected: Worker '{worker_id}' with token "
                    f"'{lease_token}' is not active owner of running job '{job_id}' "
                    f"(current status='{chk[0]}', active={chk[1]})."
                )

            curr_payload = {}
            if curr[0]:
                try:
                    curr_payload = json.loads(curr[0])
                except (json.JSONDecodeError, TypeError):
                    curr_payload = {}
            if payload_updates:
                curr_payload.update(dict(payload_updates))
            updated_payload_json = json.dumps(curr_payload, allow_nan=False)

            cursor = conn.execute(
                """
                UPDATE retraining_jobs
                SET status = ?,
                    is_active = ?,
                    completed_at = COALESCE(?, completed_at),
                    failure_reason = COALESCE(?, failure_reason),
                    promotion_approved = COALESCE(?, promotion_approved),
                    promotion_reason = COALESCE(?, promotion_reason),
                    promotion_audit_id = COALESCE(?, promotion_audit_id),
                    candidate_version = COALESCE(?, candidate_version),
                    candidate_run_id = COALESCE(?, candidate_run_id),
                    previous_champion_version = COALESCE(?, previous_champion_version),
                    payload = ?,
                    lease_expires_at = NULL
                WHERE job_id = ?
                  AND worker_id = ?
                  AND lease_token = ?
                  AND status = 'running'
                  AND is_active = 1;
                """,
                (
                    target_status,
                    new_is_active,
                    completed_at,
                    failure_reason,
                    approved_int,
                    promotion_reason,
                    promotion_audit_id,
                    candidate_version,
                    candidate_run_id,
                    previous_champion_version,
                    updated_payload_json,
                    job_id,
                    worker_id,
                    lease_token,
                ),
            )

            if cursor.rowcount != 1:
                raise JobOwnershipError(
                    f"Expected 1 row update transitioning job '{job_id}' to "
                    f"'{target_status}', matched {cursor.rowcount}."
                )

            cursor = conn.execute(
                "SELECT * FROM retraining_jobs WHERE job_id = ?;", (job_id,)
            )
            row = cursor.fetchone()
            if row is None:
                raise JobNotFoundError(f"Job '{job_id}' not found.")
            return _row_to_dict(cursor, row)


def requeue_job_for_retry(
    database_path: str | Path | None = None,
    *,
    job_id: str,
    worker_id: str,
    lease_token: str,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """Requeue a running job back to queued state for retry after handler failure.

    Fenced by job_id, worker_id, lease_token, status='running', and is_active=1.
    Strictly enforces:
    - promotion_started_at IS NULL (never retries after promotion started)
    - attempt_count < max_attempts (attempts must not be exhausted)

    Args:
        database_path: Optional path to SQLite database.
        job_id: ID of the running job.
        worker_id: Active worker identifier.
        lease_token: Active lease token.
        failure_reason: Optional failure reason to record.

    Returns:
        Updated job record as dictionary in 'queued' status.
    """
    if not job_id:
        raise ValueError("job_id must be provided.")
    if not worker_id:
        raise ValueError("worker_id must be provided.")
    if not lease_token:
        raise ValueError("lease_token must be provided.")

    with db_connection(database_path, isolation_level=None) as conn:
        with immediate_transaction(conn):
            cursor = conn.execute(
                "SELECT promotion_started_at, attempt_count, max_attempts, status, "
                "is_active FROM retraining_jobs WHERE job_id = ?;",
                (job_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise JobNotFoundError(f"Job '{job_id}' does not exist.")

            promo_started, attempts, max_att, status, is_act = row

            if promo_started is not None:
                raise InvalidStateTransitionError(
                    f"Cannot requeue job '{job_id}': promotion has already started at "
                    f"{promo_started}. Job must be quarantined for reconciliation."
                )

            if attempts >= max_att:
                raise InvalidStateTransitionError(
                    f"Cannot requeue job '{job_id}': max attempts ({max_att}) "
                    f"exhausted (attempt_count={attempts}). Job must be marked failed."
                )

            if status != "running" or is_act != 1:
                raise JobStateConflictError(
                    f"Cannot requeue job '{job_id}': not in active running state "
                    f"(status='{status}', is_active={is_act})."
                )

            cursor = conn.execute(
                """
                UPDATE retraining_jobs
                SET status = 'queued',
                    worker_id = NULL,
                    lease_token = NULL,
                    heartbeat_at = NULL,
                    lease_expires_at = NULL,
                    failure_reason = COALESCE(?, failure_reason)
                WHERE job_id = ?
                  AND worker_id = ?
                  AND lease_token = ?
                  AND status = 'running'
                  AND is_active = 1
                  AND promotion_started_at IS NULL
                  AND attempt_count < max_attempts;
                """,
                (failure_reason, job_id, worker_id, lease_token),
            )

            if cursor.rowcount != 1:
                chk = conn.execute(
                    "SELECT status, is_active, worker_id, lease_token "
                    "FROM retraining_jobs WHERE job_id = ?;",
                    (job_id,),
                ).fetchone()
                raise JobOwnershipError(
                    f"Requeue rejected: Worker '{worker_id}' with token "
                    f"'{lease_token}' is not active owner of running job '{job_id}' "
                    f"(current status='{chk[0]}', active={chk[1]}, "
                    f"worker='{chk[2]}', token='{chk[3]}')."
                )

            cursor = conn.execute(
                "SELECT * FROM retraining_jobs WHERE job_id = ?;", (job_id,)
            )
            updated_row = cursor.fetchone()
            if updated_row is None:
                raise JobNotFoundError(f"Job '{job_id}' not found.")
            return _row_to_dict(cursor, updated_row)


def detect_expired_leases(
    database_path: str | Path | None = None,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Detect all active running jobs whose lease has expired.

    Args:
        database_path: Optional path to SQLite database.
        now: Optional reference timestamp. Defaults to UTC now.

    Returns:
        List of expired job record dictionaries ordered by lease_expires_at ascending.
    """
    now_iso = (now or datetime.now(UTC)).isoformat()

    with db_connection(database_path, isolation_level=None) as conn:
        cursor = conn.execute(
            """
            SELECT *
            FROM retraining_jobs
            WHERE status = 'running'
              AND is_active = 1
              AND lease_expires_at IS NOT NULL
              AND lease_expires_at < ?
            ORDER BY lease_expires_at ASC;
            """,
            (now_iso,),
        )
        return [_row_to_dict(cursor, row) for row in cursor.fetchall()]


def reclaim_expired_job(
    database_path: str | Path | None = None,
    *,
    job_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Reclaim an expired running job back to queued state.

    CRITICAL SAFETY CONSTRAINT:
    Reclaims ONLY jobs where promotion_started_at IS NULL and
    attempt_count < max_attempts.
    If promotion_started_at IS NOT NULL, raises InvalidStateTransitionError.

    Args:
        database_path: Optional path to SQLite database.
        job_id: ID of the expired running job.
        now: Optional reference timestamp. Defaults to UTC now.

    Returns:
        Updated job record as dictionary in 'queued' status.
    """
    now_iso = (now or datetime.now(UTC)).isoformat()

    with db_connection(database_path, isolation_level=None) as conn:
        with immediate_transaction(conn):
            cursor = conn.execute(
                "SELECT promotion_started_at, attempt_count, max_attempts, status, "
                "is_active, lease_expires_at FROM retraining_jobs WHERE job_id = ?;",
                (job_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise JobNotFoundError(f"Job '{job_id}' does not exist.")

            promo_started, attempts, max_att, status, is_act, lease_exp = row

            if promo_started is not None:
                raise InvalidStateTransitionError(
                    f"Cannot reclaim job '{job_id}': promotion has already started at "
                    f"{promo_started}. Job must be quarantined for reconciliation."
                )

            if attempts >= max_att:
                raise InvalidStateTransitionError(
                    f"Cannot reclaim job '{job_id}': max attempts ({max_att}) "
                    f"exhausted (attempt_count={attempts}). Job must be marked failed."
                )

            if status != "running" or is_act != 1:
                raise JobStateConflictError(
                    f"Cannot reclaim job '{job_id}': not in active running state "
                    f"(status='{status}', is_active={is_act})."
                )

            if lease_exp is not None and lease_exp >= now_iso:
                raise JobStateConflictError(
                    f"Cannot reclaim job '{job_id}': lease has not expired yet "
                    f"(expires={lease_exp}, now={now_iso})."
                )

            cursor = conn.execute(
                """
                UPDATE retraining_jobs
                SET status = 'queued',
                    worker_id = NULL,
                    lease_token = NULL,
                    heartbeat_at = NULL,
                    lease_expires_at = NULL
                WHERE job_id = ?
                  AND status = 'running'
                  AND is_active = 1
                  AND promotion_started_at IS NULL
                  AND attempt_count < max_attempts
                  AND (lease_expires_at IS NULL OR lease_expires_at < ?);
                """,
                (job_id, now_iso),
            )

            if cursor.rowcount != 1:
                raise JobStateConflictError(
                    f"Expected 1 row update reclaiming job '{job_id}', "
                    f"matched {cursor.rowcount}."
                )

            cursor = conn.execute(
                "SELECT * FROM retraining_jobs WHERE job_id = ?;", (job_id,)
            )
            updated_row = cursor.fetchone()
            if updated_row is None:
                raise JobNotFoundError(f"Job '{job_id}' not found.")
            return _row_to_dict(cursor, updated_row)


def quarantine_expired_job(
    database_path: str | Path | None = None,
    *,
    job_id: str,
    reason: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Quarantine an expired job where promotion may have started.

    Transitions job to 'reconciliation_required' with is_active=1.

    Args:
        database_path: Optional path to SQLite database.
        job_id: ID of the expired job where promotion_started_at IS NOT NULL.
        reason: Optional diagnostic explanation.
        now: Optional reference timestamp. Defaults to UTC now.

    Returns:
        Updated job record as dictionary in 'reconciliation_required' status.
    """
    now_iso = (now or datetime.now(UTC)).isoformat()
    failure_msg = (
        reason
        or "Lease expired after promotion started; manual reconciliation required."
    )

    with db_connection(database_path, isolation_level=None) as conn:
        with immediate_transaction(conn):
            cursor = conn.execute(
                "SELECT promotion_started_at, status, is_active, lease_expires_at "
                "FROM retraining_jobs WHERE job_id = ?;",
                (job_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise JobNotFoundError(f"Job '{job_id}' does not exist.")

            promo_started, status, is_act, lease_exp = row

            if promo_started is None:
                raise InvalidStateTransitionError(
                    f"Cannot quarantine job '{job_id}' as post-promotion: "
                    "promotion_started_at is NULL."
                )

            if status != "running" or is_act != 1:
                raise JobStateConflictError(
                    f"Cannot quarantine job '{job_id}': not in active running state "
                    f"(status='{status}', is_active={is_act})."
                )

            if lease_exp is not None and lease_exp >= now_iso:
                raise JobStateConflictError(
                    f"Cannot quarantine job '{job_id}': lease has not expired yet "
                    f"(expires={lease_exp}, now={now_iso})."
                )

            cursor = conn.execute(
                """
                UPDATE retraining_jobs
                SET status = 'reconciliation_required',
                    is_active = 1,
                    failure_reason = ?,
                    lease_expires_at = NULL
                WHERE job_id = ?
                  AND status = 'running'
                  AND is_active = 1
                  AND promotion_started_at IS NOT NULL
                  AND (lease_expires_at IS NULL OR lease_expires_at < ?);
                """,
                (failure_msg, job_id, now_iso),
            )

            if cursor.rowcount != 1:
                raise JobStateConflictError(
                    f"Expected 1 row update quarantining job '{job_id}', "
                    f"matched {cursor.rowcount}."
                )

            cursor = conn.execute(
                "SELECT * FROM retraining_jobs WHERE job_id = ?;", (job_id,)
            )
            updated_row = cursor.fetchone()
            if updated_row is None:
                raise JobNotFoundError(f"Job '{job_id}' not found.")
            return _row_to_dict(cursor, updated_row)


def mark_exhausted_job_failed(
    database_path: str | Path | None = None,
    *,
    job_id: str,
    reason: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Mark an expired pre-promotion job failed when retry attempts are exhausted.

    Transitions job to terminal 'failed' status with is_active=NULL.

    Args:
        database_path: Optional path to SQLite database.
        job_id: ID of the expired pre-promotion job.
        reason: Optional failure reason description.
        now: Optional reference timestamp. Defaults to UTC now.

    Returns:
        Updated job record as dictionary in 'failed' status.
    """
    now_iso = (now or datetime.now(UTC)).isoformat()

    with db_connection(database_path, isolation_level=None) as conn:
        with immediate_transaction(conn):
            cursor = conn.execute(
                "SELECT promotion_started_at, attempt_count, max_attempts, status, "
                "is_active, lease_expires_at FROM retraining_jobs WHERE job_id = ?;",
                (job_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise JobNotFoundError(f"Job '{job_id}' does not exist.")

            promo_started, attempts, max_att, status, is_act, lease_exp = row

            if promo_started is not None:
                raise InvalidStateTransitionError(
                    f"Cannot fail job '{job_id}' as exhausted: promotion has already "
                    f"started at {promo_started}. Job must be quarantined."
                )

            if attempts < max_att:
                raise InvalidStateTransitionError(
                    f"Cannot fail job '{job_id}' as exhausted: attempts not exhausted "
                    f"({attempts} < {max_att})."
                )

            if status != "running" or is_act != 1:
                raise JobStateConflictError(
                    f"Cannot fail job '{job_id}': not in active running state "
                    f"(status='{status}', is_active={is_act})."
                )

            if lease_exp is not None and lease_exp >= now_iso:
                raise JobStateConflictError(
                    f"Cannot fail job '{job_id}': lease has not expired yet "
                    f"(expires={lease_exp}, now={now_iso})."
                )

            failure_msg = (
                reason
                or f"Job attempts exhausted ({attempts}/{max_att}) without completion."
            )

            cursor = conn.execute(
                """
                UPDATE retraining_jobs
                SET status = 'failed',
                    is_active = NULL,
                    completed_at = ?,
                    failure_reason = ?,
                    lease_expires_at = NULL
                WHERE job_id = ?
                  AND status = 'running'
                  AND is_active = 1
                  AND promotion_started_at IS NULL
                  AND attempt_count >= max_attempts
                  AND (lease_expires_at IS NULL OR lease_expires_at < ?);
                """,
                (now_iso, failure_msg, job_id, now_iso),
            )

            if cursor.rowcount != 1:
                raise JobStateConflictError(
                    f"Expected 1 row update failing exhausted job '{job_id}', "
                    f"matched {cursor.rowcount}."
                )

            cursor = conn.execute(
                "SELECT * FROM retraining_jobs WHERE job_id = ?;", (job_id,)
            )
            updated_row = cursor.fetchone()
            if updated_row is None:
                raise JobNotFoundError(f"Job '{job_id}' not found.")
            return _row_to_dict(cursor, updated_row)


def recover_expired_jobs(
    database_path: str | Path | None = None,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Scan and recover all expired jobs according to safety invariants.

    - If promotion_started_at IS NOT NULL: Quarantines into 'reconciliation_required'.
    - Else if attempt_count >= max_attempts: Marks job as terminal 'failed'.
    - Else: Reclaims job back to 'queued' for retry.

    Args:
        database_path: Optional path to SQLite database.
        now: Optional reference timestamp. Defaults to UTC now.

    Returns:
        List of recovery outcome dictionaries with action and updated job record.
    """
    ref_time = now or datetime.now(UTC)
    expired = detect_expired_leases(database_path, now=ref_time)
    results: list[dict[str, Any]] = []

    for job in expired:
        jid = job["job_id"]
        if job.get("promotion_started_at") is not None:
            updated = quarantine_expired_job(database_path, job_id=jid, now=ref_time)
            results.append({"job_id": jid, "action": "quarantined", "job": updated})
        elif job.get("attempt_count", 0) >= job.get("max_attempts", 3):
            updated = mark_exhausted_job_failed(database_path, job_id=jid, now=ref_time)
            results.append(
                {"job_id": jid, "action": "exhausted_failed", "job": updated}
            )
        else:
            updated = reclaim_expired_job(database_path, job_id=jid, now=ref_time)
            results.append(
                {"job_id": jid, "action": "reclaimed_queued", "job": updated}
            )

    return results


def get_job(
    database_path: str | Path | None = None,
    *,
    job_id: str,
) -> dict[str, Any] | None:
    """Retrieve a job by ID, returning None if not found."""
    with db_connection(database_path, isolation_level=None) as conn:
        cursor = conn.execute(
            "SELECT * FROM retraining_jobs WHERE job_id = ?;", (job_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_dict(cursor, row)


def get_active_job(
    database_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Retrieve currently active job (is_active = 1), returning None if none active."""
    with db_connection(database_path, isolation_level=None) as conn:
        cursor = conn.execute("SELECT * FROM retraining_jobs WHERE is_active = 1;")
        row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_dict(cursor, row)
