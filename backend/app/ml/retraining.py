from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.config import settings
from backend.app.ml.drift import detect_data_drift
from backend.app.ml.training import TrainingConfig
from backend.app.ml.workflow import run_training


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
