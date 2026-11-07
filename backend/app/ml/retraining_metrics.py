"""Observability metrics collection for retraining and worker orchestration.

Provides durable, read-only collection of retraining queue depth, active jobs,
cumulative outcome counters, stateless SQL-derived execution duration histograms,
and worker heartbeat freshness.
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from backend.app.ml.db import (
    get_db_connection,
    immediate_transaction,
    resolve_shared_database_path,
)

logger = logging.getLogger(__name__)

# Standard duration histogram bucket boundaries (in seconds) for retraining workloads
DURATION_BUCKETS: tuple[float, ...] = (
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    1200.0,
    1800.0,
    3600.0,
)

VALID_WORKER_HEARTBEAT_STATUSES: frozenset[str] = frozenset({"active", "stopped"})


@dataclass(frozen=True)
class RetrainingMetricsSnapshot:
    """Read-only snapshot of retraining and worker orchestration metrics."""

    queue_depth: int = 0
    active_jobs: int = 0
    jobs_succeeded_total: int = 0
    jobs_rejected_total: int = 0
    jobs_failed_total: int = 0
    jobs_reconciliation_required_total: int = 0
    duration_buckets: dict[float, int] = field(default_factory=dict)
    duration_sum_seconds: float = 0.0
    duration_count: int = 0
    worker_heartbeat_timestamp_seconds: float = 0.0
    is_available: bool = True


def record_worker_heartbeat(
    database_path: str | Path | None = None,
    *,
    worker_id: str,
    status: str = "active",
) -> None:
    """Record or update worker heartbeat and operational status.

    Args:
        database_path: Optional path to SQLite database.
        worker_id: Unique identifier of the worker.
        status: Operational status, either 'active' or 'stopped'.
    """
    if not worker_id or not worker_id.strip():
        raise ValueError("worker_id must be a non-empty string.")

    if status not in VALID_WORKER_HEARTBEAT_STATUSES:
        raise ValueError(
            f"Invalid worker heartbeat status '{status}'. "
            f"Must be one of: {sorted(VALID_WORKER_HEARTBEAT_STATUSES)}."
        )

    now_iso = datetime.now(UTC).isoformat()

    try:
        conn = get_db_connection(database_path, isolation_level=None)
        try:
            with immediate_transaction(conn):
                conn.execute(
                    """
                    INSERT INTO worker_heartbeats (
                        worker_id, heartbeat_at, started_at, status
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(worker_id) DO UPDATE SET
                        heartbeat_at = excluded.heartbeat_at,
                        status = excluded.status;
                    """,
                    (worker_id.strip(), now_iso, now_iso, status),
                )
        finally:
            conn.close()
    except Exception as exc:
        logger.warning(
            "failed_to_record_worker_heartbeat",
            extra={
                "worker_id": worker_id,
                "status": status,
                "error": str(exc),
            },
        )


def _parse_iso_timestamp(ts_str: str | None) -> datetime | None:
    """Parse ISO timestamp string safely returning UTC datetime or None."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except (ValueError, TypeError):
        return None


def get_retraining_metrics_snapshot(
    database_path: str | Path | None = None,
    *,
    max_duration_jobs: int = 10000,
) -> RetrainingMetricsSnapshot:
    """Collect read-only metrics snapshot from SQLite without mutating state.

    Uses SQLite URI 'mode=ro' to enforce read-only access. Safely handles missing
    database files, unmigrated schemas, and corrupted files.

    Note on Histogram Semantics:
    The duration histogram is derived statelessly from completed jobs retained in
    the database. Repeated scrapes do not accumulate or double-count observations.
    The histogram reflects the cumulative retained job history in durable storage.

    Args:
        database_path: Path to SQLite database. Defaults to shared audit DB.
        max_duration_jobs: Safety limit on rows evaluated for duration calculations.

    Returns:
        RetrainingMetricsSnapshot with current metric values.
    """
    resolved_path = resolve_shared_database_path(database_path)

    # Handle missing database file gracefully
    if not resolved_path.exists():
        return RetrainingMetricsSnapshot(
            duration_buckets={le: 0 for le in DURATION_BUCKETS},
            is_available=False,
        )

    # Connect strictly in read-only mode using SQLite URI
    uri = f"file:{resolved_path.as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
        logger.warning(
            "retraining_metrics_database_connection_failed",
            extra={"path": str(resolved_path), "error": str(exc)},
        )
        return RetrainingMetricsSnapshot(
            duration_buckets={le: 0 for le in DURATION_BUCKETS},
            is_available=False,
        )

    try:
        # Check if tables exist before querying
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='retraining_jobs';"
        )
        if cursor.fetchone() is None:
            return RetrainingMetricsSnapshot(
                duration_buckets={le: 0 for le in DURATION_BUCKETS},
                is_available=True,
            )

        # 1. Queue depth (queued and active)
        cursor = conn.execute(
            """
            SELECT COUNT(*) FROM retraining_jobs
            WHERE status = 'queued' AND is_active = 1;
            """
        )
        queue_depth = int(cursor.fetchone()[0])

        # 2. Active running jobs (running and active)
        cursor = conn.execute(
            """
            SELECT COUNT(*) FROM retraining_jobs
            WHERE status = 'running' AND is_active = 1;
            """
        )
        active_jobs = int(cursor.fetchone()[0])

        # 3. Terminal job outcomes (succeeded, rejected, failed)
        cursor = conn.execute(
            """
            SELECT status, COUNT(*) FROM retraining_jobs
            WHERE is_active IS NULL AND status IN ('succeeded', 'rejected', 'failed')
            GROUP BY status;
            """
        )
        outcome_counts = {row[0]: int(row[1]) for row in cursor.fetchall()}
        jobs_succeeded = outcome_counts.get("succeeded", 0)
        jobs_rejected = outcome_counts.get("rejected", 0)
        jobs_failed = outcome_counts.get("failed", 0)

        # 4. Reconciliation required counter
        # Tracks distinct jobs that ever entered reconciliation quarantine,
        # preserving monotonicity even when quarantined jobs are later resolved.
        # Deduplication via UNION guarantees distinct job counting.
        has_operator_audit = (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND "
                "name='operator_resolution_audit';"
            ).fetchone()
            is not None
        )

        if has_operator_audit:
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT job_id) FROM (
                    SELECT job_id FROM retraining_jobs
                    WHERE status = 'reconciliation_required'
                    UNION
                    SELECT job_id FROM operator_resolution_audit
                    WHERE previous_state = 'reconciliation_required'
                );
                """
            )
            jobs_reconciliation_required = int(cursor.fetchone()[0])
        else:
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM retraining_jobs
                WHERE status = 'reconciliation_required';
                """
            )
            jobs_reconciliation_required = int(cursor.fetchone()[0])

        # 5. Execution duration histogram (stateless SQL aggregation)
        cursor = conn.execute(
            """
            SELECT started_at, completed_at FROM retraining_jobs
            WHERE status IN ('succeeded', 'rejected', 'failed')
              AND started_at IS NOT NULL
              AND completed_at IS NOT NULL
            ORDER BY completed_at DESC
            LIMIT ?;
            """,
            (max_duration_jobs,),
        )

        bucket_counts: dict[float, int] = {le: 0 for le in DURATION_BUCKETS}
        duration_sum = 0.0
        duration_count = 0

        for row in cursor.fetchall():
            start_dt = _parse_iso_timestamp(row[0])
            comp_dt = _parse_iso_timestamp(row[1])
            if start_dt is None or comp_dt is None:
                continue

            duration = (comp_dt - start_dt).total_seconds()
            if duration < 0:
                # Handle unexpected clock skew safely
                duration = 0.0

            duration_sum += duration
            duration_count += 1

            for le in DURATION_BUCKETS:
                if duration <= le:
                    bucket_counts[le] += 1

        # 6. Worker heartbeat timestamp
        worker_heartbeat_ts = 0.0
        worker_is_stopped = False
        has_heartbeats_table = (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND "
                "name='worker_heartbeats';"
            ).fetchone()
            is not None
        )

        if has_heartbeats_table:
            cursor = conn.execute(
                """
                SELECT heartbeat_at, status FROM worker_heartbeats
                ORDER BY heartbeat_at DESC LIMIT 1;
                """
            )
            hb_row = cursor.fetchone()
            if hb_row is not None:
                hb_at_str, hb_status = hb_row
                # If worker explicitly shut down gracefully, report 0.0 so that
                # a stopped worker is not mistakenly interpreted as healthy
                if hb_status == "stopped":
                    worker_is_stopped = True
                    worker_heartbeat_ts = 0.0
                elif hb_status == "active":
                    hb_dt = _parse_iso_timestamp(hb_at_str)
                    if hb_dt is not None:
                        worker_heartbeat_ts = hb_dt.timestamp()

        # If worker is not explicitly stopped, check if an actively running job
        # has a newer lease heartbeat
        if not worker_is_stopped:
            cursor = conn.execute(
                """
                SELECT heartbeat_at FROM retraining_jobs
                WHERE status = 'running' AND is_active = 1
                  AND heartbeat_at IS NOT NULL;
                """
            )
            job_hb_row = cursor.fetchone()
            if job_hb_row is not None and job_hb_row[0]:
                job_hb_dt = _parse_iso_timestamp(job_hb_row[0])
                if job_hb_dt is not None:
                    worker_heartbeat_ts = max(
                        worker_heartbeat_ts, job_hb_dt.timestamp()
                    )

        return RetrainingMetricsSnapshot(
            queue_depth=queue_depth,
            active_jobs=active_jobs,
            jobs_succeeded_total=jobs_succeeded,
            jobs_rejected_total=jobs_rejected,
            jobs_failed_total=jobs_failed,
            jobs_reconciliation_required_total=jobs_reconciliation_required,
            duration_buckets=bucket_counts,
            duration_sum_seconds=duration_sum,
            duration_count=duration_count,
            worker_heartbeat_timestamp_seconds=worker_heartbeat_ts,
            is_available=True,
        )
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
        logger.warning(
            "retraining_metrics_query_failed",
            extra={"path": str(resolved_path), "error": str(exc)},
        )
        return RetrainingMetricsSnapshot(
            duration_buckets={le: 0 for le in DURATION_BUCKETS},
            is_available=False,
        )
    finally:
        conn.close()


def format_retraining_prometheus_metrics(
    snapshot: RetrainingMetricsSnapshot,
) -> list[str]:
    """Format retraining metrics snapshot lines for Prometheus exposition format."""
    if not snapshot.is_available:
        return [
            (
                "# HELP modelforge_retraining_collector_up Retraining metrics "
                "collector availability"
            ),
            "# TYPE modelforge_retraining_collector_up gauge",
            "modelforge_retraining_collector_up 0",
        ]

    lines: list[str] = [
        (
            "# HELP modelforge_retraining_collector_up Retraining metrics "
            "collector availability"
        ),
        "# TYPE modelforge_retraining_collector_up gauge",
        "modelforge_retraining_collector_up 1",
        (
            "# HELP modelforge_retraining_queue_depth Current number of queued "
            "retraining jobs waiting for execution."
        ),
        "# TYPE modelforge_retraining_queue_depth gauge",
        f"modelforge_retraining_queue_depth {snapshot.queue_depth}",
        (
            "# HELP modelforge_retraining_active_jobs Current number of retraining "
            "jobs actively executing."
        ),
        "# TYPE modelforge_retraining_active_jobs gauge",
        f"modelforge_retraining_active_jobs {snapshot.active_jobs}",
        (
            "# HELP modelforge_retraining_jobs_total Cumulative number of retraining "
            "jobs by outcome or quarantine status."
        ),
        "# TYPE modelforge_retraining_jobs_total counter",
        (
            f'modelforge_retraining_jobs_total{{status="succeeded"}} '
            f"{snapshot.jobs_succeeded_total}"
        ),
        (
            f'modelforge_retraining_jobs_total{{status="rejected"}} '
            f"{snapshot.jobs_rejected_total}"
        ),
        (
            f'modelforge_retraining_jobs_total{{status="failed"}} '
            f"{snapshot.jobs_failed_total}"
        ),
        (
            f'modelforge_retraining_jobs_total{{status="reconciliation_required"}} '
            f"{snapshot.jobs_reconciliation_required_total}"
        ),
        (
            "# HELP modelforge_retraining_execution_duration_seconds Retraining "
            "job execution duration in seconds for completed jobs."
        ),
        "# TYPE modelforge_retraining_execution_duration_seconds histogram",
    ]

    for le in DURATION_BUCKETS:
        count = snapshot.duration_buckets.get(le, 0)
        lines.append(
            f'modelforge_retraining_execution_duration_seconds_bucket{{le="{le:.1f}"}} '
            f"{count}"
        )

    lines.append(
        'modelforge_retraining_execution_duration_seconds_bucket{le="+Inf"} '
        f"{snapshot.duration_count}"
    )
    lines.append(
        "modelforge_retraining_execution_duration_seconds_sum "
        f"{snapshot.duration_sum_seconds}"
    )
    lines.append(
        "modelforge_retraining_execution_duration_seconds_count "
        f"{snapshot.duration_count}"
    )

    lines.extend(
        [
            (
                "# HELP modelforge_worker_heartbeat_timestamp_seconds Unix timestamp "
                "in seconds of the most recent worker heartbeat."
            ),
            "# TYPE modelforge_worker_heartbeat_timestamp_seconds gauge",
            (
                "modelforge_worker_heartbeat_timestamp_seconds "
                f"{snapshot.worker_heartbeat_timestamp_seconds}"
            ),
        ]
    )

    return lines
