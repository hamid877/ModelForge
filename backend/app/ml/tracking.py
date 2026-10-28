import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import mlflow

from backend.app.config import settings

EXPERIMENT_NAME = "ModelForge Churn Prediction"


def setup_mlflow() -> None:
    """Configure MLflow tracking and the experiment."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_dataset_provenance_to_mlflow(
    provenance: dict[str, Any],
) -> None:
    """Attach dataset provenance metadata, tags, and JSON artifact to active run."""
    if "dataset_sha256" in provenance:
        mlflow.set_tag("dataset_sha256", str(provenance["dataset_sha256"]))

    if "row_count" in provenance:
        mlflow.log_param("dataset_rows", provenance["row_count"])
    if "feature_count" in provenance:
        mlflow.log_param("dataset_features", provenance["feature_count"])
    if provenance.get("target_column"):
        mlflow.log_param("dataset_target", str(provenance["target_column"]))

    split = provenance.get("split", {})
    if isinstance(split, dict) and split.get("split_performed"):
        if "train_rows" in split:
            mlflow.log_param("split_train_rows", split["train_rows"])
        if "test_rows" in split:
            mlflow.log_param("split_test_rows", split["test_rows"])

    temp_dir = tempfile.mkdtemp(prefix="provenance_")
    try:
        prov_file = Path(temp_dir) / "dataset_provenance.json"
        prov_file.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(prov_file))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def log_promotion_traceability_to_mlflow(
    decision_record: dict[str, Any],
) -> None:
    """Attach promotion metrics, gate outcomes, and decision artifact to MLflow."""
    if "approved" in decision_record:
        mlflow.set_tag("promotion_approved", str(decision_record["approved"]).lower())
    if "reason" in decision_record:
        mlflow.set_tag("promotion_reason", str(decision_record["reason"]))
    if decision_record.get("benchmark_version"):
        mlflow.set_tag(
            "promotion_benchmark_version", str(decision_record["benchmark_version"])
        )
    if decision_record.get("benchmark_digest"):
        mlflow.set_tag(
            "promotion_benchmark_digest", str(decision_record["benchmark_digest"])
        )
    if decision_record.get("candidate_version") is not None:
        mlflow.set_tag(
            "promotion_candidate_version", str(decision_record["candidate_version"])
        )
    if decision_record.get("champion_version") is not None:
        mlflow.set_tag(
            "promotion_champion_version", str(decision_record["champion_version"])
        )

    if "margin" in decision_record and decision_record["margin"] is not None:
        mlflow.log_param("promotion_margin", decision_record["margin"])

    if (
        "candidate_f1" in decision_record
        and decision_record["candidate_f1"] is not None
    ):
        mlflow.log_metric(
            "promotion_candidate_f1", float(decision_record["candidate_f1"])
        )
    if decision_record.get("champion_f1") is not None:
        mlflow.log_metric(
            "promotion_champion_f1", float(decision_record["champion_f1"])
        )

    temp_dir = tempfile.mkdtemp(prefix="promotion_")
    try:
        dec_file = Path(temp_dir) / "promotion_decision.json"
        dec_file.write_text(json.dumps(decision_record, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(dec_file))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
