"""SQLite persistence for model lifecycle audit records."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


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
