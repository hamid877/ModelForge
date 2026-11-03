"""Persistence layer for drift alerts and append-only drift alert actions.

Provides transactional persistence, retrieval, listing/filtering, and audit action
recording for model drift detection events in SQLite.
"""

import json
import logging
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.ml.db import (
    db_connection,
    immediate_transaction,
    resolve_shared_database_path,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ActionAppendOnlyError",
    "AlertImmutableError",
    "AlertNotFoundError",
    "DriftAlertError",
    "DuplicateActionError",
    "DuplicateAlertError",
    "get_alert_actions",
    "get_drift_alert",
    "list_alert_actions",
    "list_drift_alerts",
    "persist_drift_alert",
    "persist_drift_alert_with_action",
    "record_alert_action",
    "record_drift_alert",
]

# Sensitive or oversized payload keys to strip from stored features
SENSITIVE_KEY_NAMES: frozenset[str] = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "credential",
        "credentials",
        "credit_card",
        "dataframe",
        "password",
        "passwd",
        "pii",
        "raw_data",
        "raw_records",
        "raw_sample",
        "records",
        "sample_data",
        "secret",
        "ssn",
        "token",
    }
)


# Domain Exceptions
class DriftAlertError(RuntimeError):
    """Base exception for drift alert persistence operations."""


class AlertNotFoundError(DriftAlertError, sqlite3.IntegrityError):
    """Raised when an alert record cannot be found or referenced."""


class DuplicateAlertError(DriftAlertError, sqlite3.IntegrityError):
    """Raised when an alert with the same ID already exists."""


class DuplicateActionError(DriftAlertError, sqlite3.IntegrityError):
    """Raised when an alert action with the same ID already exists."""


class AlertImmutableError(DriftAlertError, sqlite3.DatabaseError):
    """Raised when an update or delete is attempted on an immutable drift alert."""


class ActionAppendOnlyError(DriftAlertError, sqlite3.DatabaseError):
    """Raised when an update or delete is attempted on an append-only alert action."""


def _normalize_timestamp(ts: str | datetime | None = None) -> str:
    """Normalize timestamp input to standard UTC ISO 8601 string representation."""
    if ts is None:
        return datetime.now(UTC).isoformat()

    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=UTC).isoformat()
        return ts.astimezone(UTC).isoformat()

    if isinstance(ts, str):
        cleaned = ts.strip()
        if not cleaned:
            raise ValueError("Timestamp string cannot be empty.")
        try:
            dt = datetime.fromisoformat(cleaned)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid timestamp format '{ts}'. Must be ISO 8601."
            ) from exc

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = dt.astimezone(UTC)
        return dt.isoformat()

    raise ValueError(f"Unsupported timestamp type: {type(ts).__name__}")


def _sanitize_features_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """Sanitize feature metrics payload by removing sensitive or oversized data."""
    cleaned: dict[str, Any] = {}

    for key, value in data.items():
        key_str = str(key)
        key_lower = key_str.lower()

        # Reject sensitive field names
        if key_lower in SENSITIVE_KEY_NAMES or key_lower.endswith(
            ("_secret", "_token", "_password", "_key")
        ):
            continue

        if isinstance(value, Mapping):
            cleaned[key_str] = _sanitize_features_dict(value)
        elif isinstance(value, (list, tuple)):
            # Discard large raw sample arrays to preserve storage and avoid leaking data
            if len(value) > 50:
                continue
            cleaned[key_str] = list(value)
        else:
            cleaned[key_str] = value

    return cleaned


def _alert_row_to_dict(
    cursor: sqlite3.Cursor,
    row: tuple[Any, ...],
) -> dict[str, Any]:
    """Convert an SQLite row tuple from drift_alerts into a typed dictionary."""
    col_names = [col[0] for col in cursor.description]
    item = dict(zip(col_names, row, strict=False))

    if "drift_detected" in item and item["drift_detected"] is not None:
        item["drift_detected"] = bool(item["drift_detected"])

    if "p_value_threshold" in item and item["p_value_threshold"] is not None:
        item["p_value_threshold"] = float(item["p_value_threshold"])

    if "features_payload" in item and isinstance(item["features_payload"], str):
        try:
            item["features"] = json.loads(item["features_payload"])
        except (json.JSONDecodeError, TypeError):
            item["features"] = {}

    return item


def _action_row_to_dict(
    cursor: sqlite3.Cursor,
    row: tuple[Any, ...],
) -> dict[str, Any]:
    """Convert an SQLite row tuple from drift_alert_actions into a typed dictionary."""
    col_names = [col[0] for col in cursor.description]
    return dict(zip(col_names, row, strict=False))


def persist_drift_alert(
    database_path: str | Path | None = None,
    alert_data: Mapping[str, Any] | None = None,
    *,
    alert_id: str | None = None,
    detected_at: str | datetime | None = None,
    drift_detected: bool | int | None = None,
    reference_samples: int | None = None,
    current_samples: int | None = None,
    p_value_threshold: float | None = None,
    features: Mapping[str, Any] | None = None,
    features_payload: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Persist a drift alert record into the database.

    Args:
        database_path: Optional path to SQLite database.
        alert_data: Optional dictionary containing alert fields.
        alert_id: Optional custom alert ID. Generated if omitted.
        detected_at: Optional detection timestamp. Defaults to UTC now.
        drift_detected: Whether data drift was detected (bool or 0/1).
        reference_samples: Non-negative integer count of reference samples.
        current_samples: Non-negative integer count of current samples.
        p_value_threshold: Significance threshold in (0.0, 1.0]. Defaults to 0.05.
        features: Mapping of feature-level drift statistics.
        features_payload: Pre-serialized JSON string of features.
        conn: Optional active SQLite connection for transactional composition.

    Returns:
        Dictionary representing the persisted alert record.
    """
    data = dict(alert_data) if alert_data is not None else {}

    # 1. Resolve alert_id
    raw_id = alert_id if alert_id is not None else data.get("alert_id")
    if raw_id is not None:
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError("alert_id must be a non-empty string.")
        effective_alert_id = raw_id.strip()
    else:
        effective_alert_id = f"alert-{uuid4()}"

    # 2. Resolve detected_at
    raw_detected_at = (
        detected_at if detected_at is not None else data.get("detected_at")
    )
    effective_detected_at = _normalize_timestamp(raw_detected_at)

    # 3. Resolve drift_detected
    raw_drift_detected = (
        drift_detected if drift_detected is not None else data.get("drift_detected")
    )
    if (
        raw_drift_detected is None
        or type(raw_drift_detected) not in (bool, int)
        or raw_drift_detected not in (0, 1, True, False)
    ):
        raise ValueError(
            f"drift_detected must be a boolean or 0/1, got: {raw_drift_detected!r}"
        )
    effective_drift_detected = 1 if raw_drift_detected else 0

    # 4. Resolve sample counts
    raw_ref_samples = (
        reference_samples
        if reference_samples is not None
        else data.get("reference_samples")
    )
    if type(raw_ref_samples) is not int or raw_ref_samples < 0:
        raise ValueError(
            f"reference_samples must be a non-negative integer, got: "
            f"{raw_ref_samples!r}"
        )

    raw_cur_samples = (
        current_samples if current_samples is not None else data.get("current_samples")
    )
    if type(raw_cur_samples) is not int or raw_cur_samples < 0:
        raise ValueError(
            f"current_samples must be a non-negative integer, got: {raw_cur_samples!r}"
        )

    # 5. Resolve p_value_threshold
    raw_p_value = (
        p_value_threshold
        if p_value_threshold is not None
        else data.get("p_value_threshold", 0.05)
    )
    if (
        raw_p_value is None
        or isinstance(raw_p_value, bool)
        or not isinstance(raw_p_value, (int, float))
    ):
        raise ValueError(
            f"p_value_threshold must be a numeric value, got: {raw_p_value!r}"
        )
    float_p_value = float(raw_p_value)
    if not (0.0 < float_p_value <= 1.0):
        raise ValueError(
            f"p_value_threshold must be in the range (0.0, 1.0], got: {float_p_value}"
        )

    # 6. Resolve features and features_payload
    raw_features = features if features is not None else data.get("features")
    raw_features_payload = (
        features_payload
        if features_payload is not None
        else data.get("features_payload")
    )

    if raw_features_payload is not None:
        if (
            not isinstance(raw_features_payload, str)
            or not raw_features_payload.strip()
        ):
            raise ValueError("features_payload must be a non-empty JSON string.")
        try:
            parsed_features = json.loads(raw_features_payload)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("features_payload must be valid JSON.") from exc

        if isinstance(parsed_features, Mapping):
            sanitized_features = _sanitize_features_dict(parsed_features)
        else:
            sanitized_features = parsed_features
    elif raw_features is not None:
        if not isinstance(raw_features, Mapping):
            raise ValueError("features must be a mapping/dict.")
        sanitized_features = _sanitize_features_dict(raw_features)
    else:
        raise ValueError(
            "Either features or features_payload must be provided and non-empty."
        )

    try:
        effective_features_payload = json.dumps(sanitized_features, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "features payload could not be serialized to JSON without NaN/Inf."
        ) from exc

    if not effective_features_payload or not effective_features_payload.strip():
        raise ValueError("features_payload cannot be empty.")

    # 7. Persistence execution
    def _execute_insert(target_conn: sqlite3.Connection) -> None:
        try:
            target_conn.execute(
                """
                INSERT INTO drift_alerts (
                    alert_id, detected_at, drift_detected, reference_samples,
                    current_samples, p_value_threshold, features_payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    effective_alert_id,
                    effective_detected_at,
                    effective_drift_detected,
                    raw_ref_samples,
                    raw_cur_samples,
                    float_p_value,
                    effective_features_payload,
                ),
            )
        except sqlite3.IntegrityError as exc:
            exc_str = str(exc).lower()
            if "unique" in exc_str or "primary key" in exc_str:
                raise DuplicateAlertError(
                    f"Drift alert with ID '{effective_alert_id}' already exists."
                ) from exc
            raise

    if conn is not None:
        if conn.in_transaction:
            _execute_insert(conn)
        else:
            with immediate_transaction(conn):
                _execute_insert(conn)
    else:
        with db_connection(database_path, isolation_level=None) as active_conn:
            with immediate_transaction(active_conn):
                _execute_insert(active_conn)

    return {
        "alert_id": effective_alert_id,
        "detected_at": effective_detected_at,
        "drift_detected": bool(effective_drift_detected),
        "reference_samples": raw_ref_samples,
        "current_samples": raw_cur_samples,
        "p_value_threshold": float_p_value,
        "features_payload": effective_features_payload,
        "features": sanitized_features,
    }


# Alias for persist_drift_alert
record_drift_alert = persist_drift_alert


def get_drift_alert(
    alert_id: str | Path | None = None,
    database_path: str | Path | None = None,
    *,
    id: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    """Retrieve a drift alert record by ID.

    Args:
        alert_id: Identifier of the alert to retrieve.
        database_path: Optional path to SQLite database.
        id: Optional keyword alias for alert_id.
        conn: Optional active SQLite connection.

    Returns:
        Dictionary representing the alert record, or None if not found.
    """
    if id is not None:
        target_id = id
        target_db = database_path or alert_id
    elif isinstance(alert_id, Path):
        target_db = alert_id
        target_id = str(database_path) if database_path is not None else None
    else:
        target_id = alert_id
        target_db = database_path

    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("alert_id must be a non-empty string.")

    target_id_str = target_id.strip()

    def _query(target_conn: sqlite3.Connection) -> dict[str, Any] | None:
        cursor = target_conn.execute(
            """
            SELECT alert_id, detected_at, drift_detected, reference_samples,
                   current_samples, p_value_threshold, features_payload
            FROM drift_alerts
            WHERE alert_id = ?;
            """,
            (target_id_str,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return _alert_row_to_dict(cursor, row)

    if conn is not None:
        return _query(conn)

    resolved_path = resolve_shared_database_path(target_db)
    if not resolved_path.exists():
        return None

    try:
        with db_connection(resolved_path, isolation_level=None) as active_conn:
            return _query(active_conn)
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return None
        raise


def list_drift_alerts(
    database_path: str | Path | None = None,
    *,
    drift_detected: bool | int | None = None,
    start_time: str | datetime | None = None,
    end_time: str | datetime | None = None,
    since: str | datetime | None = None,
    until: str | datetime | None = None,
    limit: int = 100,
    offset: int = 0,
    descending: bool = True,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """List drift alert records with filtering and pagination.

    Args:
        database_path: Optional path to SQLite database.
        drift_detected: Optional filter for drift detection outcome (bool or 0/1).
        start_time: Optional start ISO timestamp / datetime (inclusive).
        end_time: Optional end ISO timestamp / datetime (inclusive).
        since: Alias for start_time.
        until: Alias for end_time.
        limit: Maximum number of records to return (positive integer).
        offset: Number of records to skip (non-negative integer).
        descending: Sort by detected_at descending if True, ascending if False.
        conn: Optional active SQLite connection.

    Returns:
        List of drift alert dictionaries.
    """
    if type(limit) is not int or limit <= 0:
        raise ValueError(f"limit must be a positive integer, got: {limit!r}")
    if type(offset) is not int or offset < 0:
        raise ValueError(f"offset must be a non-negative integer, got: {offset!r}")

    conditions: list[str] = []
    params: list[Any] = []

    if drift_detected is not None:
        if type(drift_detected) not in (bool, int) or drift_detected not in (
            0,
            1,
            True,
            False,
        ):
            raise ValueError(
                f"drift_detected filter must be a boolean or 0/1, got: "
                f"{drift_detected!r}"
            )
        conditions.append("drift_detected = ?")
        params.append(1 if drift_detected else 0)

    effective_start = start_time if start_time is not None else since
    if effective_start is not None:
        start_iso = _normalize_timestamp(effective_start)
        conditions.append("detected_at >= ?")
        params.append(start_iso)

    effective_end = end_time if end_time is not None else until
    if effective_end is not None:
        end_iso = _normalize_timestamp(effective_end)
        conditions.append("detected_at <= ?")
        params.append(end_iso)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    order_direction = "DESC" if descending else "ASC"

    query = f"""
        SELECT alert_id, detected_at, drift_detected, reference_samples,
               current_samples, p_value_threshold, features_payload
        FROM drift_alerts
        {where_clause}
        ORDER BY detected_at {order_direction}
        LIMIT ? OFFSET ?;
    """
    params.extend([limit, offset])

    def _query(target_conn: sqlite3.Connection) -> list[dict[str, Any]]:
        cursor = target_conn.execute(query, tuple(params))
        rows = cursor.fetchall()
        return [_alert_row_to_dict(cursor, row) for row in rows]

    if conn is not None:
        return _query(conn)

    resolved_path = resolve_shared_database_path(database_path)
    if not resolved_path.exists():
        return []

    try:
        with db_connection(resolved_path, isolation_level=None) as active_conn:
            return _query(active_conn)
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return []
        raise


def record_alert_action(
    database_path: str | Path | None = None,
    *,
    alert_id: str,
    action_type: str,
    actor: str,
    action_id: str | None = None,
    job_id: str | None = None,
    notes: str | None = None,
    timestamp: str | datetime | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Record an append-only action taken on a drift alert.

    Args:
        database_path: Optional path to SQLite database.
        alert_id: Identifier of the drift alert being acted upon.
        action_type: Non-empty action descriptor (e.g. 'triggered_retraining').
        actor: Non-empty identity of the actor (system or user).
        action_id: Optional custom action ID. Generated as 'act-<uuid4>' if omitted.
        job_id: Optional retraining job foreign key reference.
        notes: Optional additional context or notes.
        timestamp: Optional action timestamp. Defaults to UTC now.
        conn: Optional active SQLite connection for transactional composition.

    Returns:
        Dictionary representing the persisted action record.
    """
    # 1. Validate inputs
    if not isinstance(alert_id, str) or not alert_id.strip():
        raise ValueError("alert_id must be a non-empty string.")
    clean_alert_id = alert_id.strip()

    if not isinstance(action_type, str) or not action_type.strip():
        raise ValueError("action_type must be a non-empty string.")
    clean_action_type = action_type.strip()

    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("actor must be a non-empty string.")
    clean_actor = actor.strip()

    if action_id is not None:
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError("action_id must be a non-empty string.")
        effective_action_id = action_id.strip()
    else:
        effective_action_id = f"act-{uuid4()}"

    clean_job_id: str | None = None
    if job_id is not None:
        if not isinstance(job_id, str):
            raise ValueError("job_id must be a string or None.")
        clean_job_id = job_id.strip() if job_id.strip() else None

    clean_notes: str | None = None
    if notes is not None:
        if not isinstance(notes, str):
            raise ValueError("notes must be a string or None.")
        clean_notes = notes

    effective_timestamp = _normalize_timestamp(timestamp)

    def _execute_insert(target_conn: sqlite3.Connection) -> None:
        # Check referenced drift alert exists
        cursor = target_conn.execute(
            "SELECT 1 FROM drift_alerts WHERE alert_id = ?;",
            (clean_alert_id,),
        )
        if cursor.fetchone() is None:
            raise AlertNotFoundError(
                f"Referenced drift alert '{clean_alert_id}' does not exist."
            )

        # Check referenced retraining job exists if job_id is provided
        if clean_job_id is not None:
            job_cursor = target_conn.execute(
                "SELECT 1 FROM retraining_jobs WHERE job_id = ?;",
                (clean_job_id,),
            )
            if job_cursor.fetchone() is None:
                raise sqlite3.IntegrityError(
                    f"Referenced retraining job '{clean_job_id}' does not exist."
                )

        try:
            target_conn.execute(
                """
                INSERT INTO drift_alert_actions (
                    action_id, alert_id, timestamp, action_type, actor, job_id, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    effective_action_id,
                    clean_alert_id,
                    effective_timestamp,
                    clean_action_type,
                    clean_actor,
                    clean_job_id,
                    clean_notes,
                ),
            )
        except sqlite3.IntegrityError as exc:
            exc_str = str(exc).lower()
            if "unique" in exc_str or "primary key" in exc_str:
                raise DuplicateActionError(
                    f"Drift alert action with ID '{effective_action_id}' "
                    "already exists."
                ) from exc
            if "foreign key" in exc_str:
                raise AlertNotFoundError(
                    f"Foreign key constraint failed for drift alert '{clean_alert_id}'."
                ) from exc
            raise

    if conn is not None:
        if conn.in_transaction:
            _execute_insert(conn)
        else:
            with immediate_transaction(conn):
                _execute_insert(conn)
    else:
        with db_connection(database_path, isolation_level=None) as active_conn:
            with immediate_transaction(active_conn):
                _execute_insert(active_conn)

    return {
        "action_id": effective_action_id,
        "alert_id": clean_alert_id,
        "timestamp": effective_timestamp,
        "action_type": clean_action_type,
        "actor": clean_actor,
        "job_id": clean_job_id,
        "notes": clean_notes,
    }


def get_alert_actions(
    alert_id: str | Path | None = None,
    database_path: str | Path | None = None,
    *,
    id: str | None = None,
    limit: int = 100,
    offset: int = 0,
    descending: bool = False,
    require_alert_exists: bool = False,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Retrieve actions recorded for a given drift alert.

    Args:
        alert_id: Identifier of the drift alert.
        database_path: Optional path to SQLite database.
        id: Optional keyword alias for alert_id.
        limit: Maximum number of action records to return (positive integer).
        offset: Number of action records to skip (non-negative integer).
        descending: Sort by timestamp descending if True, ascending if False.
        require_alert_exists: If True, raises AlertNotFoundError if alert missing.
        conn: Optional active SQLite connection.

    Returns:
        List of action dictionaries for the alert.
    """
    if id is not None:
        target_id = id
        target_db = database_path or alert_id
    elif isinstance(alert_id, Path):
        target_db = alert_id
        target_id = str(database_path) if database_path is not None else None
    else:
        target_id = alert_id
        target_db = database_path

    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("alert_id must be a non-empty string.")

    if type(limit) is not int or limit <= 0:
        raise ValueError(f"limit must be a positive integer, got: {limit!r}")
    if type(offset) is not int or offset < 0:
        raise ValueError(f"offset must be a non-negative integer, got: {offset!r}")

    target_id_str = target_id.strip()
    order_direction = "DESC" if descending else "ASC"

    query = f"""
        SELECT action_id, alert_id, timestamp, action_type, actor, job_id, notes
        FROM drift_alert_actions
        WHERE alert_id = ?
        ORDER BY timestamp {order_direction}
        LIMIT ? OFFSET ?;
    """

    def _query(target_conn: sqlite3.Connection) -> list[dict[str, Any]]:
        if require_alert_exists:
            cursor = target_conn.execute(
                "SELECT 1 FROM drift_alerts WHERE alert_id = ?;",
                (target_id_str,),
            )
            if cursor.fetchone() is None:
                raise AlertNotFoundError(
                    f"Drift alert '{target_id_str}' does not exist."
                )

        cursor = target_conn.execute(query, (target_id_str, limit, offset))
        rows = cursor.fetchall()
        return [_action_row_to_dict(cursor, row) for row in rows]

    if conn is not None:
        return _query(conn)

    resolved_path = resolve_shared_database_path(target_db)
    if not resolved_path.exists():
        if require_alert_exists:
            raise AlertNotFoundError(f"Drift alert '{target_id_str}' does not exist.")
        return []

    try:
        with db_connection(resolved_path, isolation_level=None) as active_conn:
            return _query(active_conn)
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            if require_alert_exists:
                raise AlertNotFoundError(
                    f"Drift alert '{target_id_str}' does not exist."
                )
            return []
        raise


# Alias for get_alert_actions
list_alert_actions = get_alert_actions


def persist_drift_alert_with_action(
    database_path: str | Path | None = None,
    alert_data: Mapping[str, Any] | None = None,
    *,
    alert_id: str | None = None,
    detected_at: str | datetime | None = None,
    drift_detected: bool | int | None = None,
    reference_samples: int | None = None,
    current_samples: int | None = None,
    p_value_threshold: float | None = None,
    features: Mapping[str, Any] | None = None,
    features_payload: str | None = None,
    action_type: str,
    actor: str,
    action_id: str | None = None,
    job_id: str | None = None,
    notes: str | None = None,
    action_timestamp: str | datetime | None = None,
    conn: sqlite3.Connection | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Persist a drift alert and an initial action in a single atomic transaction.

    If either operation fails, changes are completely rolled back and no partial
    records are committed.

    Returns:
        Tuple of (persisted_alert_record, persisted_action_record).
    """

    def _transaction_work(
        target_conn: sqlite3.Connection,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        saved_alert = persist_drift_alert(
            alert_data=alert_data,
            alert_id=alert_id,
            detected_at=detected_at,
            drift_detected=drift_detected,
            reference_samples=reference_samples,
            current_samples=current_samples,
            p_value_threshold=p_value_threshold,
            features=features,
            features_payload=features_payload,
            conn=target_conn,
        )

        saved_action = record_alert_action(
            alert_id=saved_alert["alert_id"],
            action_type=action_type,
            actor=actor,
            action_id=action_id,
            job_id=job_id,
            notes=notes,
            timestamp=action_timestamp,
            conn=target_conn,
        )

        return saved_alert, saved_action

    if conn is not None:
        if conn.in_transaction:
            return _transaction_work(conn)
        with immediate_transaction(conn):
            return _transaction_work(conn)

    with db_connection(database_path, isolation_level=None) as active_conn:
        with immediate_transaction(active_conn):
            return _transaction_work(active_conn)
