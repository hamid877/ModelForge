import logging
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.ml.retraining import assess_retraining_need
from backend.app.ml.workflow import run_training_and_promote

logger = logging.getLogger(__name__)


def run_drift_aware_lifecycle(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    model_path: str | Path,
) -> dict[str, Any]:
    """Assess feature drift and retrain only when drift warrants it."""
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
        return result

    try:
        metrics, decision = run_training_and_promote(
            current,
            model_path,
        )
    except Exception:
        logger.exception(
            "lifecycle_training_failed reason=%s",
            assessment["reason"],
        )
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

    return result
