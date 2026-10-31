"""SQLite database schema definitions and idempotent migrations.

Provides persistence foundations for controlled retraining orchestration.
"""

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from backend.app.ml.db import get_db_connection

logger = logging.getLogger(__name__)

# Valid state machine statuses
VALID_JOB_STATUSES: frozenset[str] = frozenset(
    {
        "queued",
        "running",
        "succeeded",
        "rejected",
        "failed",
        "reconciliation_required",
    }
)

ACTIVE_JOB_STATUSES: frozenset[str] = frozenset(
    {"queued", "running", "reconciliation_required"}
)

TERMINAL_JOB_STATUSES: frozenset[str] = frozenset({"succeeded", "rejected", "failed"})

VALID_RESOLUTIONS: frozenset[str] = frozenset({"resolved_succeeded", "resolved_failed"})


LIFECYCLE_AUDIT_SQL = """
CREATE TABLE IF NOT EXISTS lifecycle_audit (
    run_id TEXT PRIMARY KEY CHECK (length(run_id) > 0),
    timestamp TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL
);
"""

PROMOTION_AUDIT_SQL = """
CREATE TABLE IF NOT EXISTS promotion_audit (
    audit_id TEXT PRIMARY KEY CHECK (length(audit_id) > 0),
    timestamp TEXT NOT NULL,
    model_name TEXT NOT NULL,
    candidate_version TEXT,
    candidate_run_id TEXT,
    champion_version TEXT,
    approved INTEGER NOT NULL CHECK (approved IN (0, 1)),
    reason TEXT NOT NULL,
    candidate_f1 REAL NOT NULL,
    champion_f1 REAL,
    benchmark_version TEXT NOT NULL,
    benchmark_digest TEXT NOT NULL,
    margin REAL NOT NULL,
    job_id TEXT,
    payload TEXT NOT NULL
);
"""

DRIFT_ALERTS_SQL = """
CREATE TABLE IF NOT EXISTS drift_alerts (
    alert_id TEXT PRIMARY KEY CHECK (length(alert_id) > 0),
    detected_at TEXT NOT NULL,
    drift_detected INTEGER NOT NULL CHECK (drift_detected IN (0, 1)),
    reference_samples INTEGER NOT NULL CHECK (reference_samples >= 0),
    current_samples INTEGER NOT NULL CHECK (current_samples >= 0),
    p_value_threshold REAL NOT NULL CHECK (
        p_value_threshold > 0.0 AND p_value_threshold <= 1.0
    ),
    features_payload TEXT NOT NULL CHECK (length(features_payload) > 0)
);
"""

DRIFT_ALERTS_TRIGGER_UPDATE = """
CREATE TRIGGER IF NOT EXISTS trg_drift_alerts_immutable_update
BEFORE UPDATE ON drift_alerts
BEGIN
    SELECT RAISE(
        ABORT,
        'drift_alerts records are immutable and cannot be updated'
    );
END;
"""

DRIFT_ALERTS_TRIGGER_DELETE = """
CREATE TRIGGER IF NOT EXISTS trg_drift_alerts_immutable_delete
BEFORE DELETE ON drift_alerts
BEGIN
    SELECT RAISE(
        ABORT,
        'drift_alerts records are immutable and cannot be deleted'
    );
END;
"""

RETRAINING_JOBS_SQL = """
CREATE TABLE IF NOT EXISTS retraining_jobs (
    job_id TEXT PRIMARY KEY CHECK (length(job_id) > 0),
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    status TEXT NOT NULL CHECK (
        status IN (
            'queued',
            'running',
            'succeeded',
            'rejected',
            'failed',
            'reconciliation_required'
        )
    ),
    is_active INTEGER CHECK (is_active IS NULL OR is_active = 1),
    dataset_id TEXT NOT NULL CHECK (length(dataset_id) > 0),
    trigger_source TEXT NOT NULL CHECK (length(trigger_source) > 0),
    worker_id TEXT,
    lease_token TEXT,
    heartbeat_at TEXT,
    lease_expires_at TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    max_attempts INTEGER NOT NULL DEFAULT 3 CHECK (max_attempts >= 1),
    candidate_version TEXT,
    candidate_run_id TEXT,
    previous_champion_version TEXT,
    promotion_started_at TEXT,
    promotion_approved INTEGER CHECK (
        promotion_approved IS NULL OR promotion_approved IN (0, 1)
    ),
    promotion_reason TEXT,
    promotion_audit_id TEXT,
    failure_reason TEXT,
    alert_id TEXT REFERENCES drift_alerts(alert_id),
    payload TEXT NOT NULL,
    CHECK (
        (
            is_active IS NOT NULL
            AND is_active = 1
            AND status IN (
                'queued',
                'running',
                'reconciliation_required'
            )
        )
        OR
        (
            is_active IS NULL
            AND status IN ('succeeded', 'rejected', 'failed')
        )
    )
);
"""

RETRAINING_JOBS_PARTIAL_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_single_active_retraining_job
ON retraining_jobs (is_active)
WHERE is_active = 1;
"""

DRIFT_ALERT_ACTIONS_SQL = """
CREATE TABLE IF NOT EXISTS drift_alert_actions (
    action_id TEXT PRIMARY KEY CHECK (length(action_id) > 0),
    alert_id TEXT NOT NULL REFERENCES drift_alerts(alert_id),
    timestamp TEXT NOT NULL,
    action_type TEXT NOT NULL CHECK (length(action_type) > 0),
    actor TEXT NOT NULL CHECK (length(actor) > 0),
    job_id TEXT REFERENCES retraining_jobs(job_id),
    notes TEXT
);
"""

DRIFT_ALERT_ACTIONS_TRIGGER_UPDATE = """
CREATE TRIGGER IF NOT EXISTS trg_drift_alert_actions_no_update
BEFORE UPDATE ON drift_alert_actions
BEGIN
    SELECT RAISE(
        ABORT,
        'drift_alert_actions records are append-only and cannot be updated'
    );
END;
"""

DRIFT_ALERT_ACTIONS_TRIGGER_DELETE = """
CREATE TRIGGER IF NOT EXISTS trg_drift_alert_actions_no_delete
BEFORE DELETE ON drift_alert_actions
BEGIN
    SELECT RAISE(
        ABORT,
        'drift_alert_actions records are append-only and cannot be deleted'
    );
END;
"""

OPERATOR_RESOLUTION_AUDIT_SQL = """
CREATE TABLE IF NOT EXISTS operator_resolution_audit (
    resolution_id TEXT PRIMARY KEY CHECK (length(resolution_id) > 0),
    job_id TEXT NOT NULL REFERENCES retraining_jobs(job_id),
    timestamp TEXT NOT NULL,
    operator TEXT NOT NULL CHECK (length(operator) > 0),
    previous_state TEXT NOT NULL,
    selected_resolution TEXT NOT NULL CHECK (
        selected_resolution IN ('resolved_succeeded', 'resolved_failed')
    ),
    candidate_version TEXT,
    observed_champion_version TEXT,
    target_champion_version TEXT,
    rollback_performed INTEGER NOT NULL CHECK (rollback_performed IN (0, 1)),
    resolution_reason TEXT NOT NULL CHECK (length(resolution_reason) > 0),
    evidence_payload TEXT NOT NULL
);
"""

OPERATOR_RESOLUTION_AUDIT_TRIGGER_UPDATE = """
CREATE TRIGGER IF NOT EXISTS trg_operator_resolution_audit_no_update
BEFORE UPDATE ON operator_resolution_audit
BEGIN
    SELECT RAISE(
        ABORT,
        'operator_resolution_audit records are append-only and cannot be updated'
    );
END;
"""

OPERATOR_RESOLUTION_AUDIT_TRIGGER_DELETE = """
CREATE TRIGGER IF NOT EXISTS trg_operator_resolution_audit_no_delete
BEFORE DELETE ON operator_resolution_audit
BEGIN
    SELECT RAISE(
        ABORT,
        'operator_resolution_audit records are append-only and cannot be deleted'
    );
END;
"""


def migrate_audit_database_idempotent(database_path: str | Path) -> None:
    """Ensure audit database schema is upgraded idempotently without data loss.

    Applies all table schemas, constraints, indexes, and triggers across:
    1. Fresh database where tables do not yet exist.
    2. Pre-existing database with lifecycle_audit and promotion_audit without
       job_id.
    3. Already-upgraded database with existing job_id column.

    All operations execute inside an atomic SQLite transaction. On failure,
    all changes roll back and existing data is preserved intact.
    """
    conn = get_db_connection(database_path, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE;")
        try:
            # Step 1: Ensure lifecycle_audit exists
            conn.execute(LIFECYCLE_AUDIT_SQL)

            # Step 2: Ensure promotion_audit exists and has job_id
            cursor = conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='promotion_audit';"
            )
            if cursor.fetchone() is not None:
                col_cursor = conn.execute("PRAGMA table_info(promotion_audit);")
                columns = {row[1] for row in col_cursor.fetchall()}
                if "job_id" not in columns:
                    conn.execute("ALTER TABLE promotion_audit ADD COLUMN job_id TEXT;")
                    conn.execute("PRAGMA table_info(promotion_audit);")
            else:
                conn.execute(PROMOTION_AUDIT_SQL)

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_promotion_audit_job_id "
                "ON promotion_audit(job_id);"
            )

            # Step 3: Drift alerts (immutable)
            conn.execute(DRIFT_ALERTS_SQL)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_drift_alerts_detected_at "
                "ON drift_alerts(detected_at);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_drift_alerts_drift_detected "
                "ON drift_alerts(drift_detected);"
            )
            conn.execute(DRIFT_ALERTS_TRIGGER_UPDATE)
            conn.execute(DRIFT_ALERTS_TRIGGER_DELETE)

            # Step 4: Retraining jobs (partial unique index on is_active = 1)
            cursor = conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='retraining_jobs';"
            )
            if cursor.fetchone() is not None:
                col_cursor = conn.execute("PRAGMA table_info(retraining_jobs);")
                job_columns = {row[1] for row in col_cursor.fetchall()}
                expected_columns = {
                    "worker_id": "TEXT",
                    "lease_token": "TEXT",
                    "heartbeat_at": "TEXT",
                    "lease_expires_at": "TEXT",
                    "attempt_count": "INTEGER NOT NULL DEFAULT 0",
                    "max_attempts": "INTEGER NOT NULL DEFAULT 3",
                    "candidate_version": "TEXT",
                    "candidate_run_id": "TEXT",
                    "previous_champion_version": "TEXT",
                    "promotion_started_at": "TEXT",
                    "promotion_approved": "INTEGER",
                    "promotion_reason": "TEXT",
                    "promotion_audit_id": "TEXT",
                    "failure_reason": "TEXT",
                    "alert_id": "TEXT REFERENCES drift_alerts(alert_id)",
                }
                for col_name, col_type in expected_columns.items():
                    if col_name not in job_columns:
                        conn.execute(
                            f"ALTER TABLE retraining_jobs ADD COLUMN "
                            f"{col_name} {col_type};"
                        )
                conn.execute("PRAGMA table_info(retraining_jobs);")
            else:
                conn.execute(RETRAINING_JOBS_SQL)

            conn.execute(RETRAINING_JOBS_PARTIAL_INDEX_SQL)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_retraining_jobs_status "
                "ON retraining_jobs(status);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_retraining_jobs_created_at "
                "ON retraining_jobs(created_at);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_retraining_jobs_dataset_id "
                "ON retraining_jobs(dataset_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_retraining_jobs_alert_id "
                "ON retraining_jobs(alert_id);"
            )

            # Step 5: Drift alert actions (append-only)
            conn.execute(DRIFT_ALERT_ACTIONS_SQL)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_drift_alert_actions_alert_id "
                "ON drift_alert_actions(alert_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_drift_alert_actions_job_id "
                "ON drift_alert_actions(job_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_drift_alert_actions_timestamp "
                "ON drift_alert_actions(timestamp);"
            )
            conn.execute(DRIFT_ALERT_ACTIONS_TRIGGER_UPDATE)
            conn.execute(DRIFT_ALERT_ACTIONS_TRIGGER_DELETE)

            # Step 6: Operator resolution audit (append-only)
            conn.execute(OPERATOR_RESOLUTION_AUDIT_SQL)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_operator_resolution_job_id "
                "ON operator_resolution_audit(job_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_operator_resolution_timestamp "
                "ON operator_resolution_audit(timestamp);"
            )
            conn.execute(OPERATOR_RESOLUTION_AUDIT_TRIGGER_UPDATE)
            conn.execute(OPERATOR_RESOLUTION_AUDIT_TRIGGER_DELETE)

            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
    finally:
        conn.close()


def validate_operator_resolution_inputs(
    selected_resolution: str,
    rollback_performed: int,
) -> None:
    """Validate operator resolution inputs prior to opening DB connection."""
    if selected_resolution not in VALID_RESOLUTIONS:
        raise ValueError(
            f"Invalid selected_resolution '{selected_resolution}'. "
            f"Must be exactly one of: {sorted(VALID_RESOLUTIONS)}."
        )

    if rollback_performed not in (0, 1) or isinstance(rollback_performed, bool):
        raise ValueError(
            f"Invalid rollback_performed value '{rollback_performed}'. "
            "Must be integer 0 or 1."
        )


def finalize_operator_resolution_atomic(
    database_path: str | Path,
    *,
    resolution_id: str,
    job_id: str,
    operator: str,
    selected_resolution: str,  # 'resolved_succeeded' or 'resolved_failed'
    candidate_version: str | None,
    observed_champion_version: str | None,
    target_champion_version: str | None,
    rollback_performed: int,  # 0 or 1
    resolution_reason: str,
    evidence_payload: Mapping[str, Any],
) -> None:
    """Commit resolution audit and job finalization atomically."""
    validate_operator_resolution_inputs(selected_resolution, rollback_performed)

    conn = get_db_connection(database_path, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE;")
        try:
            # Insert immutable operator resolution record
            conn.execute(
                """
                INSERT INTO operator_resolution_audit (
                    resolution_id, job_id, timestamp, operator, previous_state,
                    selected_resolution, candidate_version,
                    observed_champion_version, target_champion_version,
                    rollback_performed, resolution_reason, evidence_payload
                ) VALUES (
                    ?, ?, datetime('now'), ?, 'reconciliation_required',
                    ?, ?, ?, ?, ?, ?, ?
                );
                """,
                (
                    resolution_id,
                    job_id,
                    operator,
                    selected_resolution,
                    candidate_version,
                    observed_champion_version,
                    target_champion_version,
                    rollback_performed,
                    resolution_reason,
                    json.dumps(evidence_payload),
                ),
            )

            terminal_status = (
                "succeeded" if selected_resolution == "resolved_succeeded" else "failed"
            )
            failure_reason = resolution_reason if terminal_status == "failed" else None

            # Update retraining job to terminal state
            cursor = conn.execute(
                """
                UPDATE retraining_jobs
                SET status = ?,
                    is_active = NULL,
                    completed_at = datetime('now'),
                    failure_reason = ?,
                    payload = json_set(
                        payload,
                        '$.operator_resolution',
                        json_object(
                            'resolution_id', ?,
                            'operator', ?,
                            'action', ?
                        )
                    )
                WHERE job_id = ?
                  AND status = 'reconciliation_required'
                  AND is_active = 1;
                """,
                (
                    terminal_status,
                    failure_reason,
                    resolution_id,
                    operator,
                    evidence_payload.get("action", selected_resolution),
                    job_id,
                ),
            )

            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"Job finalization failed: Expected to update 1 row "
                    f"for job '{job_id}' in state 'reconciliation_required', "
                    f"but matched {cursor.rowcount} rows."
                )

            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
    finally:
        conn.close()
