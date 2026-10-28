"""SQLite persistence for model lifecycle audit records."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.config import settings


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

    with sqlite3.connect(path) as connection:
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

    if "candidate_metrics" in record and record["candidate_metrics"] is not None:
        payload_dict["candidate_metrics"] = record["candidate_metrics"]
    if "champion_metrics" in record and record["champion_metrics"] is not None:
        payload_dict["champion_metrics"] = record["champion_metrics"]
    if "dataset_provenance" in record and record["dataset_provenance"] is not None:
        payload_dict["dataset_provenance"] = record["dataset_provenance"]

    serialized = json.dumps(payload_dict, allow_nan=False)

    with sqlite3.connect(db_path) as connection:
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
                payload TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO promotion_audit (
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

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        try:
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
        except sqlite3.OperationalError:
            return []

    results = []
    for row in rows:
        item = dict(row)
        item["approved"] = bool(item["approved"])
        item["payload"] = json.loads(item["payload"])
        results.append(item)
    return results
