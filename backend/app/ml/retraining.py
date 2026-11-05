"""Retraining eligibility policy, drift assessment, and candidate training.

This module owns:
- ``assess_retraining_need``: Drift-based retraining recommendation (read-only).
- ``evaluate_retraining_policy``: Full eligibility policy with cooldown evaluation
  and structured policy results. Used by the queue-enabled drift-assess endpoint.
- ``retrain_model_without_promotion``: Candidate training decoupled from promotion.
"""

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.config import settings
from backend.app.ml.db import db_connection, immediate_transaction
from backend.app.ml.drift import detect_data_drift
from backend.app.ml.drift_alerts import (
    get_drift_alert,
    persist_drift_alert,
    record_alert_action,
)
from backend.app.ml.jobs import (
    ActiveJobExistsError,
    create_queued_job,
)
from backend.app.ml.training import TrainingConfig
from backend.app.ml.workflow import run_training

logger = logging.getLogger(__name__)


def assess_retraining_need(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict[str, Any]:
    """Recommend retraining when sufficient data shows feature drift."""
    drift_report = detect_data_drift(
        reference,
        current,
        min_samples=settings.drift_min_samples,
        p_value_threshold=settings.drift_p_value_threshold,
    )

    if drift_report["status"] == "insufficient_data":
        return {
            "status": "insufficient_data",
            "retraining_recommended": False,
            "reason": "insufficient_data",
            "drift_report": drift_report,
        }

    if drift_report["drift_detected"]:
        return {
            "status": "retraining_recommended",
            "retraining_recommended": True,
            "reason": "feature_drift_detected",
            "drift_report": drift_report,
        }

    return {
        "status": "stable",
        "retraining_recommended": False,
        "reason": "no_feature_drift_detected",
        "drift_report": drift_report,
    }


def _get_last_successful_job_completed_at(
    conn: sqlite3.Connection,
) -> str | None:
    """Return the ``completed_at`` timestamp of the most recently succeeded job.

    Only considers successful jobs (``status = 'succeeded'``). Failed and rejected
    jobs must not extend the cooldown. Returns ``None`` if no such job exists.
    """
    cursor = conn.execute(
        """
        SELECT completed_at
        FROM retraining_jobs
        WHERE status = 'succeeded'
          AND completed_at IS NOT NULL
        ORDER BY completed_at DESC
        LIMIT 1;
        """
    )
    row = cursor.fetchone()
    return row[0] if row else None


def evaluate_retraining_policy(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    dataset_id: str,
    trigger_source: str,
    alert_id: str | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate drift and eligibility policy, optionally queuing a retraining job.

    Policy evaluation is performed atomically inside a single ``IMMEDIATE``
    transaction so that cooldown evaluation, alert/action persistence, and job
    creation are consistent under concurrent requests.

    Decision rules (evaluated in order):
    1. If data is insufficient, return ``insufficient_data`` – no job queued,
       and no database writes performed.
    2. If no drift is detected, persist the drift alert and action, and return
       ``stable`` – no job queued.
    3. If an active job already exists, persist the alert and action, and return
       ``active_job_exists`` with ``active_job_id`` – no job queued.
    4. If cooldown period has not elapsed since the last *succeeded* job,
       persist the alert and action, and return ``cooldown_active`` with
       ``cooldown_remaining_seconds`` – no job queued.
    5. Otherwise, persist the drift alert, queue a new retraining job linking
       the alert, record the action linking the alert and job, and return
       ``queued`` with the created job record.

    Args:
        reference: Reference feature DataFrame for drift comparison.
        current: Current feature DataFrame from the incoming batch.
        dataset_id: Dataset identifier passed to the new retraining job.
        trigger_source: Source label for the new job (e.g. ``'drift_assess'``).
        alert_id: Optional drift alert ID to link or persist.
        database_path: Optional path to the SQLite lifecycle database.

    Returns:
        Structured policy result dictionary.
    """
    if not dataset_id or not str(dataset_id).strip():
        raise ValueError("dataset_id must be a non-empty string.")
    if not trigger_source or not str(trigger_source).strip():
        raise ValueError("trigger_source must be a non-empty string.")

    clean_dataset_id = str(dataset_id).strip()
    clean_trigger_source = str(trigger_source).strip()

    # --- Step 1: Drift detection (pure; no DB interaction) ---
    drift_report = detect_data_drift(
        reference,
        current,
        min_samples=settings.drift_min_samples,
        p_value_threshold=settings.drift_p_value_threshold,
    )

    # 1. Insufficient data: strictly no DB writes performed.
    if drift_report["status"] == "insufficient_data":
        return {
            "status": "insufficient_data",
            "retraining_recommended": False,
            "drift_detected": None,
            "drift_report": drift_report,
            "job": None,
            "active_job_id": None,
            "alert_id": None,
            "cooldown_remaining_seconds": None,
            "reason": "insufficient_data",
        }

    # --- Step 2: Atomic transaction for alert/action persistence and job creation ---
    cooldown_seconds: int = getattr(settings, "retraining_cooldown_seconds", 3600)

    with db_connection(database_path, isolation_level=None) as conn:
        with immediate_transaction(conn):
            # Resolve or persist drift alert record
            saved_alert = None
            if alert_id is not None:
                saved_alert = get_drift_alert(alert_id=alert_id, conn=conn)

            if saved_alert is None:
                saved_alert = persist_drift_alert(
                    conn=conn,
                    alert_id=alert_id,
                    drift_detected=drift_report["drift_detected"],
                    reference_samples=drift_report["reference_samples"],
                    current_samples=drift_report["current_samples"],
                    p_value_threshold=drift_report.get(
                        "p_value_threshold", settings.drift_p_value_threshold
                    ),
                    features=drift_report.get("features", {}),
                )
            current_alert_id = saved_alert["alert_id"]

            # 2. Stable: distribution matches reference; no retraining needed.
            if not drift_report["drift_detected"]:
                record_alert_action(
                    conn=conn,
                    alert_id=current_alert_id,
                    action_type="drift_assessed_stable",
                    actor=clean_trigger_source,
                    job_id=None,
                    notes="No feature drift detected; retraining not recommended.",
                )
                return {
                    "status": "stable",
                    "retraining_recommended": False,
                    "drift_detected": False,
                    "drift_report": drift_report,
                    "job": None,
                    "active_job_id": None,
                    "alert_id": current_alert_id,
                    "cooldown_remaining_seconds": None,
                    "reason": "no_feature_drift_detected",
                }

            # 3. Active job check: if active job exists, suppress queuing.
            active_cursor = conn.execute(
                "SELECT job_id FROM retraining_jobs WHERE is_active = 1 LIMIT 1;"
            )
            active_row = active_cursor.fetchone()
            if active_row is not None:
                active_job_id = active_row[0]
                record_alert_action(
                    conn=conn,
                    alert_id=current_alert_id,
                    action_type="active_job_conflict",
                    actor=clean_trigger_source,
                    job_id=active_job_id,
                    notes=(
                        f"Retraining suppressed: active job '{active_job_id}' exists."
                    ),
                )
                return {
                    "status": "active_job_exists",
                    "retraining_recommended": False,
                    "drift_detected": True,
                    "drift_report": drift_report,
                    "job": None,
                    "active_job_id": active_job_id,
                    "alert_id": current_alert_id,
                    "cooldown_remaining_seconds": None,
                    "reason": "active_job_exists",
                }

            # 4. Cooldown evaluation: check only most recent *succeeded* job.
            if cooldown_seconds > 0:
                last_completed_at = _get_last_successful_job_completed_at(conn)
                if last_completed_at is not None:
                    try:
                        last_dt = datetime.fromisoformat(last_completed_at)
                        if last_dt.tzinfo is None:
                            last_dt = last_dt.replace(tzinfo=UTC)
                    except (ValueError, TypeError):
                        last_dt = None

                    if last_dt is not None:
                        now = datetime.now(UTC)
                        elapsed = (now - last_dt).total_seconds()
                        if elapsed < cooldown_seconds:
                            remaining = max(1, int(cooldown_seconds - elapsed))
                            record_alert_action(
                                conn=conn,
                                alert_id=current_alert_id,
                                action_type="cooldown_active",
                                actor=clean_trigger_source,
                                job_id=None,
                                notes=(
                                    f"Retraining suppressed: {remaining}s cooldown "
                                    f"remaining."
                                ),
                            )
                            return {
                                "status": "cooldown_active",
                                "retraining_recommended": False,
                                "drift_detected": True,
                                "drift_report": drift_report,
                                "job": None,
                                "active_job_id": None,
                                "alert_id": current_alert_id,
                                "cooldown_remaining_seconds": remaining,
                                "reason": (
                                    f"cooldown_active: {remaining}s remaining "
                                    f"of {cooldown_seconds}s"
                                ),
                            }

            # 5. Eligible: Queue a new retraining job linking the alert.
            try:
                job = create_queued_job(
                    conn=conn,
                    dataset_id=clean_dataset_id,
                    trigger_source=clean_trigger_source,
                    alert_id=current_alert_id,
                )
            except ActiveJobExistsError:
                # Concurrent race condition caught by unique index
                active_row = conn.execute(
                    "SELECT job_id FROM retraining_jobs WHERE is_active = 1 LIMIT 1;"
                ).fetchone()
                active_job_id = active_row[0] if active_row else None
                record_alert_action(
                    conn=conn,
                    alert_id=current_alert_id,
                    action_type="active_job_conflict",
                    actor=clean_trigger_source,
                    job_id=active_job_id,
                    notes=(
                        f"Retraining suppressed: active job '{active_job_id}' exists."
                    ),
                )
                return {
                    "status": "active_job_exists",
                    "retraining_recommended": False,
                    "drift_detected": True,
                    "drift_report": drift_report,
                    "job": None,
                    "active_job_id": active_job_id,
                    "alert_id": current_alert_id,
                    "cooldown_remaining_seconds": None,
                    "reason": "active_job_exists",
                }

            record_alert_action(
                conn=conn,
                alert_id=current_alert_id,
                action_type="triggered_retraining",
                actor=clean_trigger_source,
                job_id=job["job_id"],
                notes=(
                    f"Retraining job '{job['job_id']}' queued for dataset "
                    f"'{clean_dataset_id}'."
                ),
            )

    return {
        "status": "queued",
        "retraining_recommended": True,
        "drift_detected": True,
        "drift_report": drift_report,
        "job": job,
        "active_job_id": None,
        "alert_id": current_alert_id,
        "cooldown_remaining_seconds": None,
        "reason": "feature_drift_detected_job_queued",
    }


def retrain_model_without_promotion(
    df: pd.DataFrame,
    model_path: str | Path,
    config: TrainingConfig | None = None,
    register_model: bool = True,
) -> dict[str, float]:
    """Retrain and optionally register candidate model without promoting to champion.

    Decouples model retraining from promotion to prevent drift batches from
    automatically promoting models without benchmark validation.
    """
    return run_training(
        df,
        model_path,
        config=config,
        register_model=register_model,
    )
