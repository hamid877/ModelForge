"""SQLite persistence for model lifecycle audit records."""

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.config import settings
from backend.app.ml.db import db_connection, get_db_connection
from backend.app.ml.migrations import (
    ACTIVE_JOB_STATUSES,
    TERMINAL_JOB_STATUSES,
    VALID_JOB_STATUSES,
    VALID_RESOLUTIONS,
    finalize_operator_resolution_atomic,
    migrate_audit_database_idempotent,
    validate_operator_resolution_inputs,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ACTIVE_JOB_STATUSES",
    "TERMINAL_JOB_STATUSES",
    "VALID_JOB_STATUSES",
    "VALID_RESOLUTIONS",
    "AuditDatabaseError",
    "db_connection",
    "finalize_operator_resolution_atomic",
    "get_db_connection",
    "get_promotion_audit_by_job_id",
    "get_promotion_audit_records",
    "migrate_audit_database_idempotent",
    "record_lifecycle_run",
    "record_promotion_audit",
    "validate_operator_resolution_inputs",
]


class AuditDatabaseError(RuntimeError):
    """Raised when the audit database cannot be read due to corruption or failure."""


def record_lifecycle_run(
    record: dict[str, Any],
    database_path: str | Path,
) -> dict[str, str]:
    """Persist a minimal lifecycle audit record."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid4())
    timestamp = datetime.now(UTC).isoformat()

    payload: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "status": record["status"],
        "reason": record.get("reason"),
        "training_attempted": record.get("training_attempted", False),
    }

    if "promotion" in record:
        payload["promotion"] = record["promotion"]

    if "metrics" in record:
        payload["metrics"] = record["metrics"]

    if "failure" in record:
        payload["failure"] = {
            "type": record["failure"]["type"],
        }

    serialized = json.dumps(payload, allow_nan=False)

    with db_connection(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS lifecycle_audit (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO lifecycle_audit (run_id, timestamp, status, payload)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, timestamp, str(record["status"]), serialized),
        )

    return {"run_id": run_id, "timestamp": timestamp}


def record_promotion_audit(
    record: dict[str, Any],
    database_path: str | Path | None = None,
) -> dict[str, str]:
    """Persist a promotion decision audit record to SQLite.

    Ensures both approved and rejected promotion attempts are auditable,
    including candidate/champion metrics, benchmark digest, gate outcomes,
    and timestamps.
    """
    db_path = Path(
        database_path if database_path is not None else settings.promotion_audit_db_path
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)

    audit_id = record.get("audit_id") or str(uuid4())
    timestamp = record.get("timestamp") or datetime.now(UTC).isoformat()

    model_name = str(record.get("model_name") or settings.model_name)
    candidate_version = (
        str(record["candidate_version"])
        if record.get("candidate_version") is not None
        else None
    )
    candidate_run_id = (
        str(record["candidate_run_id"])
        if record.get("candidate_run_id") is not None
        else None
    )
    champion_version = (
        str(record["champion_version"])
        if record.get("champion_version") is not None
        else None
    )
    approved = 1 if record.get("approved") else 0
    reason = str(record.get("reason", "unknown"))
    candidate_f1 = float(record.get("candidate_f1", 0.0))
    champion_f1 = (
        float(record["champion_f1"]) if record.get("champion_f1") is not None else None
    )
    benchmark_version = str(
        record.get("benchmark_version") or settings.benchmark_version
    )
    benchmark_digest = str(record.get("benchmark_digest", ""))
    margin = float(
        record.get("margin")
        if record.get("margin") is not None
        else settings.promotion_f1_margin
    )
    job_id = str(record["job_id"]) if record.get("job_id") is not None else None

    payload_dict: dict[str, Any] = {
        "audit_id": audit_id,
        "timestamp": timestamp,
        "model_name": model_name,
        "candidate_version": candidate_version,
        "candidate_run_id": candidate_run_id,
        "champion_version": champion_version,
        "approved": bool(approved),
        "reason": reason,
        "candidate_f1": candidate_f1,
        "champion_f1": champion_f1,
        "benchmark_version": benchmark_version,
        "benchmark_digest": benchmark_digest,
        "margin": margin,
        "gate_outcomes": record.get("gate_outcomes", {}),
    }

    if job_id is not None:
        payload_dict["job_id"] = job_id
    if "candidate_metrics" in record and record["candidate_metrics"] is not None:
        payload_dict["candidate_metrics"] = record["candidate_metrics"]
    if "champion_metrics" in record and record["champion_metrics"] is not None:
        payload_dict["champion_metrics"] = record["champion_metrics"]
    if "dataset_provenance" in record and record["dataset_provenance"] is not None:
        payload_dict["dataset_provenance"] = record["dataset_provenance"]

    serialized = json.dumps(payload_dict, allow_nan=False)

    with db_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS promotion_audit (
                audit_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                candidate_version TEXT,
                candidate_run_id TEXT,
                champion_version TEXT,
                approved INTEGER NOT NULL,
                reason TEXT NOT NULL,
                candidate_f1 REAL NOT NULL,
                champion_f1 REAL,
                benchmark_version TEXT NOT NULL,
                benchmark_digest TEXT NOT NULL,
                margin REAL NOT NULL,
                job_id TEXT,
                payload TEXT NOT NULL
            )
            """
        )
        cursor = connection.execute("PRAGMA table_info(promotion_audit);")
        columns = {row[1] for row in cursor.fetchall()}
        if "job_id" in columns:
            connection.execute(
                """
                INSERT OR REPLACE INTO promotion_audit (
                    audit_id, timestamp, model_name, candidate_version,
                    candidate_run_id, champion_version, approved, reason,
                    candidate_f1, champion_f1, benchmark_version,
                    benchmark_digest, margin, job_id, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    timestamp,
                    model_name,
                    candidate_version,
                    candidate_run_id,
                    champion_version,
                    approved,
                    reason,
                    candidate_f1,
                    champion_f1,
                    benchmark_version,
                    benchmark_digest,
                    margin,
                    job_id,
                    serialized,
                ),
            )
        else:
            connection.execute(
                """
                INSERT OR REPLACE INTO promotion_audit (
                    audit_id, timestamp, model_name, candidate_version,
                    candidate_run_id, champion_version, approved, reason,
                    candidate_f1, champion_f1, benchmark_version,
                    benchmark_digest, margin, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    timestamp,
                    model_name,
                    candidate_version,
                    candidate_run_id,
                    champion_version,
                    approved,
                    reason,
                    candidate_f1,
                    champion_f1,
                    benchmark_version,
                    benchmark_digest,
                    margin,
                    serialized,
                ),
            )

    return {"audit_id": audit_id, "timestamp": timestamp}


def get_promotion_audit_records(
    database_path: str | Path | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Retrieve recent promotion audit records ordered by timestamp descending."""
    db_path = Path(
        database_path if database_path is not None else settings.promotion_audit_db_path
    )
    if not db_path.exists():
        return []

    try:
        with db_connection(db_path, row_factory=sqlite3.Row) as connection:
            cursor = connection.execute(
                """
                SELECT audit_id, timestamp, model_name, candidate_version,
                       candidate_run_id, champion_version, approved, reason,
                       candidate_f1, champion_f1, benchmark_version,
                       benchmark_digest, margin, payload
                FROM promotion_audit
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return []
        logger.exception("promotion_audit_read_failed")
        raise AuditDatabaseError("Failed to read promotion audit database.") from exc
    except (sqlite3.DatabaseError, OSError) as exc:
        logger.exception("promotion_audit_read_failed")
        raise AuditDatabaseError("Failed to read promotion audit database.") from exc

    results = []
    try:
        for row in rows:
            item = dict(row)
            item["approved"] = bool(item["approved"])
            item["payload"] = json.loads(item["payload"])
            results.append(item)
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        logger.exception("promotion_audit_payload_corrupted")
        raise AuditDatabaseError("Failed to parse promotion audit payload.") from exc

    return results


def get_promotion_audit_by_job_id(
    job_id: str,
    database_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Retrieve the latest promotion audit record associated with an exact job_id."""
    db_path = Path(
        database_path if database_path is not None else settings.promotion_audit_db_path
    )
    if not db_path.exists():
        return None

    try:
        with db_connection(db_path, row_factory=sqlite3.Row) as connection:
            cursor = connection.execute("PRAGMA table_info(promotion_audit);")
            columns = {row[1] for row in cursor.fetchall()}
            if not columns:
                return None

            if "job_id" in columns:
                cursor = connection.execute(
                    """
                    SELECT audit_id, timestamp, model_name, candidate_version,
                           candidate_run_id, champion_version, approved, reason,
                           candidate_f1, champion_f1, benchmark_version,
                           benchmark_digest, margin, job_id, payload
                    FROM promotion_audit
                    WHERE job_id = ?
                    ORDER BY timestamp DESC
                    """,
                    (str(job_id),),
                )
                matching_rows = [dict(r) for r in cursor.fetchall()]
            else:
                cursor = connection.execute(
                    """
                    SELECT audit_id, timestamp, model_name, candidate_version,
                           candidate_run_id, champion_version, approved, reason,
                           candidate_f1, champion_f1, benchmark_version,
                           benchmark_digest, margin, payload
                    FROM promotion_audit
                    ORDER BY timestamp DESC
                    """
                )
                matching_rows = []
                for r in cursor.fetchall():
                    r_dict = dict(r)
                    payload_data = json.loads(r_dict["payload"])
                    if payload_data.get("job_id") == str(job_id):
                        r_dict["job_id"] = str(job_id)
                        matching_rows.append(r_dict)

            if len(matching_rows) == 0:
                return None

            if len(matching_rows) > 1:
                raise AuditDatabaseError(
                    f"Duplicate promotion audit records found for job_id '{job_id}' "
                    f"(count={len(matching_rows)}). Exactly one record required."
                )

            item = matching_rows[0]
            item["approved"] = bool(item["approved"])
            if isinstance(item["payload"], str):
                item["payload"] = json.loads(item["payload"])
            return item
    except (sqlite3.DatabaseError, OSError, json.JSONDecodeError) as exc:
        logger.exception("get_promotion_audit_by_job_id_failed job_id=%s", job_id)
        raise AuditDatabaseError("Failed to query promotion audit database.") from exc
