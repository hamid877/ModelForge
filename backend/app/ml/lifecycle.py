from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.ml.retraining import assess_retraining_need
from backend.app.ml.workflow import run_training_and_promote


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

    if not assessment["retraining_recommended"]:
        return result

    metrics, decision = run_training_and_promote(
        current,
        model_path,
    )

    result["training_attempted"] = True
    result["metrics"] = metrics
    result["promotion"] = {
        "approved": decision.approved,
        "reason": decision.reason,
        "candidate_f1": decision.candidate_f1,
        "champion_f1": decision.champion_f1,
    }
    result["status"] = "promoted" if decision.approved else "promotion_rejected"

    return result
