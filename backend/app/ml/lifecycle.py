import logging
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.config import settings
from backend.app.ml.audit import record_lifecycle_run
from backend.app.ml.retraining import assess_retraining_need
from backend.app.ml.workflow import run_training_and_promote

logger = logging.getLogger(__name__)


def run_drift_aware_lifecycle(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    model_path: str | Path,
) -> dict[str, Any]:
    """Assess drift, optionally retrain, and persist lifecycle outcomes."""
    assessment = assess_retraining_need(reference, current)

    result: dict[str, Any] = {
        "status": assessment["status"],
        "training_attempted": False,
        "reason": assessment["reason"],
        "drift_report": assessment["drift_report"],
    }

    logger.info(
        "lifecycle_assessment_completed status=%s reason=%s retraining_recommended=%s",
        assessment["status"],
        assessment["reason"],
        assessment["retraining_recommended"],
    )

    if not assessment["retraining_recommended"]:
        _record_audit(result)
        return result

    try:
        metrics, decision = run_training_and_promote(current, model_path)
    except Exception as exc:
        logger.exception(
            "lifecycle_training_failed reason=%s",
            assessment["reason"],
        )
        failure_record = {
            **result,
            "status": "training_failed",
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        try:
            _record_audit(failure_record)
        except Exception:
            logger.exception("lifecycle_failure_audit_recording_failed")
        raise

    result["training_attempted"] = True
    result["metrics"] = metrics
    result["promotion"] = {
        "approved": decision.approved,
        "reason": decision.reason,
        "candidate_f1": decision.candidate_f1,
        "champion_f1": decision.champion_f1,
    }
    result["status"] = "promoted" if decision.approved else "promotion_rejected"

    logger.info(
        "lifecycle_promotion_completed status=%s approved=%s "
        "reason=%s candidate_f1=%.4f champion_f1=%s",
        result["status"],
        decision.approved,
        decision.reason,
        decision.candidate_f1,
        (f"{decision.champion_f1:.4f}" if decision.champion_f1 is not None else "none"),
    )

    _record_audit(result)
    return result


def _record_audit(record: dict[str, Any]) -> None:
    """Persist an audit record and log its identifier."""
    metadata = record_lifecycle_run(
        record,
        settings.lifecycle_audit_db_path,
    )
    logger.info(
        "lifecycle_audit_recorded run_id=%s status=%s",
        metadata["run_id"],
        record["status"],
    )
