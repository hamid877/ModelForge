import hashlib
import io
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode

from backend.app.config import settings
from backend.app.data.validator import REQUIRED_COLUMNS, validate_churn_data
from backend.app.ml.audit import record_promotion_audit
from backend.app.ml.evaluation import evaluate_churn_model
from backend.app.ml.provenance import (
    compute_dataframe_sha256,
    extract_dataset_provenance,
)
from backend.app.ml.registry import load_registered_model_by_alias
from backend.app.ml.tracking import log_promotion_traceability_to_mlflow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromotionDecision:
    """Result of evaluating whether a candidate model should be promoted."""

    approved: bool
    reason: str
    candidate_f1: float
    champion_f1: float | None
    benchmark_version: str | None = None
    margin: float = 0.0
    benchmark_digest: str | None = None
    candidate_metrics: dict[str, float] | None = None
    champion_metrics: dict[str, float] | None = None
    gate_outcomes: dict[str, Any] | None = None
    candidate_version: int | str | None = None
    champion_version: str | None = None
    candidate_run_id: str | None = None
    timestamp: str | None = None
    audit_id: str | None = None


@dataclass(frozen=True)
class BenchmarkDataset:
    """Explicitly versioned benchmark records with cryptographic content identity.

    Note: Standardized benchmark for evaluation and quality gate verification;
    synthetic benchmark data does not demonstrate real-world performance.
    """

    data: pd.DataFrame
    version: str
    digest: str


def compute_file_sha256(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_training_benchmark_overlap(
    training_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
) -> bool:
    """Check whether any candidate training records overlap with benchmark records.

    Returns True if overlap is detected, False otherwise.
    """
    common_columns = [
        c
        for c in training_df.columns
        if c in benchmark_df.columns and c in REQUIRED_COLUMNS
    ]
    if not common_columns:
        return False
    overlap = pd.merge(
        training_df[common_columns],
        benchmark_df[common_columns],
        how="inner",
    )
    return not overlap.empty


def validate_benchmark_data(
    benchmark_df: pd.DataFrame | None,
    *,
    min_samples: int | None = None,
    expected_digest: str | None = None,
) -> None:
    """Validate benchmark presence, structural integrity, and sample count.

    Note: Meeting the configurable minimum sample count validates basic data
    presence and operational evaluability, but does NOT establish
    statistical sufficiency or statistical power for comparative benchmarking.
    """
    if benchmark_df is None:
        raise ValueError("Benchmark dataset is missing.")

    if not isinstance(benchmark_df, pd.DataFrame):
        raise TypeError(
            f"Benchmark dataset must be a DataFrame, got {type(benchmark_df).__name__}."
        )

    if benchmark_df.empty:
        raise ValueError("Benchmark dataset must not be empty.")

    min_samples = settings.benchmark_min_samples if min_samples is None else min_samples
    if len(benchmark_df) < min_samples:
        raise ValueError(
            f"Benchmark dataset sample count ({len(benchmark_df)}) is below "
            f"required minimum of {min_samples}. (Note: meeting this threshold "
            "does not establish statistical sufficiency)."
        )

    if expected_digest is not None:
        actual_digest = compute_dataframe_sha256(benchmark_df)
        if actual_digest != expected_digest:
            raise ValueError(
                f"Benchmark content digest mismatch: expected {expected_digest}, "
                f"got {actual_digest}."
            )

    validate_churn_data(benchmark_df)


def load_benchmark_dataset(
    path: str | Path | None = None,
    version: str | None = None,
    expected_sha256: str | None = None,
    min_samples: int | None = None,
) -> BenchmarkDataset:
    """Load and validate benchmark dataset from disk with content verification.

    Note: Meeting the minimum sample count does NOT establish statistical sufficiency.
    """
    path = Path(path or settings.benchmark_data_path)
    version = version or settings.benchmark_version
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {path}")

    content_bytes = path.read_bytes()
    actual_file_digest = hashlib.sha256(content_bytes).hexdigest()

    expected = expected_sha256
    if expected is None:
        try:
            is_default_target = (
                str(path) == str(settings.benchmark_data_path)
                or path.resolve() == Path(settings.benchmark_data_path).resolve()
            )
        except Exception:
            is_default_target = str(path) == str(settings.benchmark_data_path)

        if is_default_target or version == settings.benchmark_version:
            expected = settings.benchmark_expected_sha256

    if expected is not None and actual_file_digest != expected:
        raise ValueError(
            f"Benchmark dataset content digest mismatch for version '{version}': "
            f"expected {expected}, got {actual_file_digest}."
        )

    df = pd.read_csv(io.BytesIO(content_bytes))
    validate_benchmark_data(df, min_samples=min_samples)
    return BenchmarkDataset(data=df, version=version, digest=actual_file_digest)


def evaluate_model_on_benchmark(
    model: Any,
    benchmark_df: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate candidate or champion on benchmark records using identical logic."""
    X = benchmark_df.drop(columns=["churn"])
    y = benchmark_df["churn"]
    return evaluate_churn_model(model, X, y)


def evaluate_promotion(
    candidate_f1: float,
    champion_f1: float | None,
    *,
    minimum_f1: float = 0.0,
    min_improvement: float = 0.0,
    candidate_benchmark_version: str | None = None,
    champion_benchmark_version: str | None = None,
) -> PromotionDecision:
    """Approve promotion only when candidate meets quality gate and beats champion.

    Note: The minimum improvement margin protects against promoting models
    on noise or ties; it does NOT establish or claim statistical significance.
    """
    for name, value in (
        ("candidate_f1", candidate_f1),
        ("minimum_f1", minimum_f1),
        ("min_improvement", min_improvement),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")

    if champion_f1 is not None and not 0.0 <= champion_f1 <= 1.0:
        raise ValueError("champion_f1 must be between 0 and 1.")

    # Guard against comparing evaluations across different benchmark versions/datasets
    if (
        candidate_benchmark_version is not None
        and champion_benchmark_version is not None
        and candidate_benchmark_version != champion_benchmark_version
    ):
        return PromotionDecision(
            approved=False,
            reason="benchmark_version_mismatch",
            candidate_f1=candidate_f1,
            champion_f1=champion_f1,
            benchmark_version=candidate_benchmark_version,
            margin=min_improvement,
        )

    if candidate_f1 < minimum_f1:
        return PromotionDecision(
            approved=False,
            reason="candidate_below_minimum_f1",
            candidate_f1=candidate_f1,
            champion_f1=champion_f1,
            benchmark_version=candidate_benchmark_version,
            margin=min_improvement,
        )

    if champion_f1 is not None:
        if candidate_f1 < champion_f1:
            return PromotionDecision(
                approved=False,
                reason="candidate_f1_below_champion",
                candidate_f1=candidate_f1,
                champion_f1=champion_f1,
                benchmark_version=candidate_benchmark_version,
                margin=min_improvement,
            )

        if candidate_f1 == champion_f1:
            return PromotionDecision(
                approved=False,
                reason="candidate_f1_tied_champion",
                candidate_f1=candidate_f1,
                champion_f1=champion_f1,
                benchmark_version=candidate_benchmark_version,
                margin=min_improvement,
            )

        if (candidate_f1 - champion_f1) < (min_improvement - 1e-9):
            return PromotionDecision(
                approved=False,
                reason="candidate_f1_insufficient_improvement",
                candidate_f1=candidate_f1,
                champion_f1=champion_f1,
                benchmark_version=candidate_benchmark_version,
                margin=min_improvement,
            )

    return PromotionDecision(
        approved=True,
        reason="candidate_meets_quality_gate",
        candidate_f1=candidate_f1,
        champion_f1=champion_f1,
        benchmark_version=candidate_benchmark_version,
        margin=min_improvement,
    )


def _persist_audit_and_finalize(
    decision: PromotionDecision,
    *,
    model_name: str,
    alias: str,
    candidate_version: int | str,
    candidate_run_id: str | None,
    champion_version: str | None,
    audit_db_path: str | Path | None,
    candidate_training_df: pd.DataFrame | None,
    client: MlflowClient | None = None,
) -> PromotionDecision:
    """Safely record audit and set alias only if approved and audit succeeds."""
    timestamp = datetime.now(UTC).isoformat()
    audit_id = str(uuid4())

    if candidate_run_id is None:
        try:
            active = mlflow.active_run()
            if active is not None:
                candidate_run_id = active.info.run_id
        except Exception:
            pass

    provenance_dict = None
    if candidate_training_df is not None:
        try:
            provenance_dict = extract_dataset_provenance(
                candidate_training_df
            ).to_dict()
        except Exception:
            pass

    audit_payload: dict[str, Any] = {
        "audit_id": audit_id,
        "timestamp": timestamp,
        "model_name": model_name,
        "candidate_version": (
            str(candidate_version) if candidate_version is not None else None
        ),
        "candidate_run_id": candidate_run_id,
        "champion_version": champion_version,
        "approved": decision.approved,
        "reason": decision.reason,
        "candidate_f1": decision.candidate_f1,
        "champion_f1": decision.champion_f1,
        "benchmark_version": decision.benchmark_version or "",
        "benchmark_digest": decision.benchmark_digest or "",
        "margin": decision.margin,
        "gate_outcomes": decision.gate_outcomes or {},
        "candidate_metrics": decision.candidate_metrics,
        "champion_metrics": decision.champion_metrics,
        "dataset_provenance": provenance_dict,
    }

    if decision.approved:
        # 1. Audit SQLite recording must succeed BEFORE alias update
        try:
            record_promotion_audit(audit_payload, database_path=audit_db_path)
        except Exception as exc:
            logger.exception(
                "Promotion audit logging failed: %s; aborting promotion", exc
            )
            return PromotionDecision(
                approved=False,
                reason="audit_logging_failed",
                candidate_f1=decision.candidate_f1,
                champion_f1=decision.champion_f1,
                benchmark_version=decision.benchmark_version,
                margin=decision.margin,
                benchmark_digest=decision.benchmark_digest,
                candidate_metrics=decision.candidate_metrics,
                champion_metrics=decision.champion_metrics,
                gate_outcomes=decision.gate_outcomes,
                candidate_version=candidate_version,
                champion_version=champion_version,
                candidate_run_id=candidate_run_id,
                timestamp=timestamp,
                audit_id=audit_id,
            )

        # 2. MLflow traceability logging if active run exists
        try:
            active_run = mlflow.active_run()
            if active_run is not None:
                log_promotion_traceability_to_mlflow(audit_payload)
        except Exception as exc:
            logger.exception(
                "MLflow promotion traceability logging failed: %s; aborting promotion",
                exc,
            )
            return PromotionDecision(
                approved=False,
                reason="audit_logging_failed",
                candidate_f1=decision.candidate_f1,
                champion_f1=decision.champion_f1,
                benchmark_version=decision.benchmark_version,
                margin=decision.margin,
                benchmark_digest=decision.benchmark_digest,
                candidate_metrics=decision.candidate_metrics,
                champion_metrics=decision.champion_metrics,
                gate_outcomes=decision.gate_outcomes,
                candidate_version=candidate_version,
                champion_version=champion_version,
                candidate_run_id=candidate_run_id,
                timestamp=timestamp,
                audit_id=audit_id,
            )

        # 3. Both succeed: safe to update champion alias
        if client is not None:
            client.set_registered_model_alias(
                model_name,
                alias,
                str(candidate_version),
            )

        return PromotionDecision(
            approved=True,
            reason=decision.reason,
            candidate_f1=decision.candidate_f1,
            champion_f1=decision.champion_f1,
            benchmark_version=decision.benchmark_version,
            margin=decision.margin,
            benchmark_digest=decision.benchmark_digest,
            candidate_metrics=decision.candidate_metrics,
            champion_metrics=decision.champion_metrics,
            gate_outcomes=decision.gate_outcomes,
            candidate_version=candidate_version,
            champion_version=champion_version,
            candidate_run_id=candidate_run_id,
            timestamp=timestamp,
            audit_id=audit_id,
        )

    # For rejected decisions: persist audit and return decision
    try:
        record_promotion_audit(audit_payload, database_path=audit_db_path)
    except Exception as exc:
        logger.exception("Failed to record rejected promotion audit: %s", exc)

    try:
        active_run = mlflow.active_run()
        if active_run is not None:
            log_promotion_traceability_to_mlflow(audit_payload)
    except Exception as exc:
        logger.exception(
            "Failed to log rejected promotion traceability to MLflow: %s", exc
        )

    return PromotionDecision(
        approved=False,
        reason=decision.reason,
        candidate_f1=decision.candidate_f1,
        champion_f1=decision.champion_f1,
        benchmark_version=decision.benchmark_version,
        margin=decision.margin,
        benchmark_digest=decision.benchmark_digest,
        candidate_metrics=decision.candidate_metrics,
        champion_metrics=decision.champion_metrics,
        gate_outcomes=decision.gate_outcomes,
        candidate_version=candidate_version,
        champion_version=champion_version,
        candidate_run_id=candidate_run_id,
        timestamp=timestamp,
        audit_id=audit_id,
    )


def promote_model(
    candidate_version: int | str,
    candidate_f1: float | None = None,
    *,
    candidate_training_df: pd.DataFrame | None = None,
    candidate_model: Any | None = None,
    champion_model: Any | None = None,
    benchmark_df: pd.DataFrame | None = None,
    benchmark_version: str | None = None,
    expected_benchmark_digest: str | None = None,
    model_name: str | None = None,
    alias: str | None = None,
    minimum_f1: float | None = None,
    min_improvement: float | None = None,
    min_samples: int | None = None,
    candidate_benchmark_version: str | None = None,
    champion_benchmark_version: str | None = None,
    candidate_run_id: str | None = None,
    audit_db_path: str | Path | None = None,
) -> PromotionDecision:
    """Promote a registered model version only if every promotion gate passes.

    Preserves the MLflow champion alias unless all validation and quality gates pass.
    Note: Thresholds validate minimum data presence and improvement margins;
    they do NOT establish statistical sufficiency or statistical significance.
    """
    model_name = model_name or settings.model_name
    alias = alias or settings.model_alias
    minimum_f1 = settings.promotion_minimum_f1 if minimum_f1 is None else minimum_f1
    min_improvement = (
        settings.promotion_f1_margin if min_improvement is None else min_improvement
    )
    min_samples = settings.benchmark_min_samples if min_samples is None else min_samples

    gate_outcomes: dict[str, Any] = {}
    actual_benchmark_digest: str | None = None

    # 1. Resolve and validate shared benchmark dataset
    resolved_benchmark_df = benchmark_df
    resolved_benchmark_version = benchmark_version or settings.benchmark_version

    if resolved_benchmark_df is None:
        try:
            bench_dataset = load_benchmark_dataset(
                settings.benchmark_data_path,
                version=resolved_benchmark_version,
                expected_sha256=expected_benchmark_digest
                or settings.benchmark_expected_sha256,
                min_samples=min_samples,
            )
            resolved_benchmark_df = bench_dataset.data
            resolved_benchmark_version = bench_dataset.version
            actual_benchmark_digest = bench_dataset.digest
        except FileNotFoundError:
            gate_outcomes["benchmark_integrity"] = {
                "passed": False,
                "reason": "benchmark_missing",
            }
            dec = PromotionDecision(
                approved=False,
                reason="benchmark_missing",
                candidate_f1=0.0,
                champion_f1=None,
                benchmark_version=resolved_benchmark_version,
                margin=min_improvement,
                gate_outcomes=gate_outcomes,
            )
            return _persist_audit_and_finalize(
                dec,
                model_name=model_name,
                alias=alias,
                candidate_version=candidate_version,
                candidate_run_id=candidate_run_id,
                champion_version=None,
                audit_db_path=audit_db_path,
                candidate_training_df=candidate_training_df,
            )
        except ValueError as exc:
            msg = str(exc).lower()
            reason = (
                "benchmark_digest_mismatch"
                if "digest mismatch" in msg
                else "benchmark_invalid"
            )
            gate_outcomes["benchmark_integrity"] = {
                "passed": False,
                "reason": reason,
            }
            dec = PromotionDecision(
                approved=False,
                reason=reason,
                candidate_f1=0.0,
                champion_f1=None,
                benchmark_version=resolved_benchmark_version,
                margin=min_improvement,
                gate_outcomes=gate_outcomes,
            )
            return _persist_audit_and_finalize(
                dec,
                model_name=model_name,
                alias=alias,
                candidate_version=candidate_version,
                candidate_run_id=candidate_run_id,
                champion_version=None,
                audit_db_path=audit_db_path,
                candidate_training_df=candidate_training_df,
            )
    else:
        actual_benchmark_digest = compute_dataframe_sha256(resolved_benchmark_df)
        try:
            validate_benchmark_data(
                resolved_benchmark_df,
                min_samples=min_samples,
                expected_digest=expected_benchmark_digest,
            )
        except ValueError as exc:
            msg = str(exc).lower()
            if "benchmark dataset is missing" in msg or "empty" in msg:
                reason = "benchmark_missing"
            elif "digest mismatch" in msg:
                reason = "benchmark_digest_mismatch"
            elif "below required minimum" in msg or "insufficient" in msg:
                reason = "benchmark_insufficient_samples"
            else:
                reason = "benchmark_invalid"
            gate_outcomes["benchmark_integrity"] = {
                "passed": False,
                "reason": reason,
            }
            dec = PromotionDecision(
                approved=False,
                reason=reason,
                candidate_f1=0.0,
                champion_f1=None,
                benchmark_version=resolved_benchmark_version,
                margin=min_improvement,
                benchmark_digest=actual_benchmark_digest,
                gate_outcomes=gate_outcomes,
            )
            return _persist_audit_and_finalize(
                dec,
                model_name=model_name,
                alias=alias,
                candidate_version=candidate_version,
                candidate_run_id=candidate_run_id,
                champion_version=None,
                audit_db_path=audit_db_path,
                candidate_training_df=candidate_training_df,
            )
        except TypeError:
            gate_outcomes["benchmark_integrity"] = {
                "passed": False,
                "reason": "benchmark_invalid",
            }
            dec = PromotionDecision(
                approved=False,
                reason="benchmark_invalid",
                candidate_f1=0.0,
                champion_f1=None,
                benchmark_version=resolved_benchmark_version,
                margin=min_improvement,
                benchmark_digest=actual_benchmark_digest,
                gate_outcomes=gate_outcomes,
            )
            return _persist_audit_and_finalize(
                dec,
                model_name=model_name,
                alias=alias,
                candidate_version=candidate_version,
                candidate_run_id=candidate_run_id,
                champion_version=None,
                audit_db_path=audit_db_path,
                candidate_training_df=candidate_training_df,
            )

    gate_outcomes["benchmark_integrity"] = {
        "passed": True,
        "benchmark_version": resolved_benchmark_version,
        "benchmark_digest": actual_benchmark_digest,
        "sample_count": len(resolved_benchmark_df),
    }

    # 2. Enforce training/benchmark separation (Fail closed)
    if candidate_training_df is None:
        gate_outcomes["separation_and_overlap"] = {
            "passed": False,
            "reason": "separation_not_established",
        }
        dec = PromotionDecision(
            approved=False,
            reason="separation_not_established",
            candidate_f1=0.0,
            champion_f1=None,
            benchmark_version=resolved_benchmark_version,
            margin=min_improvement,
            benchmark_digest=actual_benchmark_digest,
            gate_outcomes=gate_outcomes,
        )
        return _persist_audit_and_finalize(
            dec,
            model_name=model_name,
            alias=alias,
            candidate_version=candidate_version,
            candidate_run_id=candidate_run_id,
            champion_version=None,
            audit_db_path=audit_db_path,
            candidate_training_df=candidate_training_df,
        )

    if check_training_benchmark_overlap(candidate_training_df, resolved_benchmark_df):
        gate_outcomes["separation_and_overlap"] = {
            "passed": False,
            "reason": "training_benchmark_overlap",
        }
        dec = PromotionDecision(
            approved=False,
            reason="training_benchmark_overlap",
            candidate_f1=0.0,
            champion_f1=None,
            benchmark_version=resolved_benchmark_version,
            margin=min_improvement,
            benchmark_digest=actual_benchmark_digest,
            gate_outcomes=gate_outcomes,
        )
        return _persist_audit_and_finalize(
            dec,
            model_name=model_name,
            alias=alias,
            candidate_version=candidate_version,
            candidate_run_id=candidate_run_id,
            champion_version=None,
            audit_db_path=audit_db_path,
            candidate_training_df=candidate_training_df,
        )

    gate_outcomes["separation_and_overlap"] = {"passed": True}

    # 3. Evaluate candidate model on shared benchmark records
    if candidate_model is None:
        gate_outcomes["candidate_evaluation"] = {
            "passed": False,
            "reason": "candidate_model_missing",
        }
        dec = PromotionDecision(
            approved=False,
            reason="candidate_model_missing",
            candidate_f1=0.0,
            champion_f1=None,
            benchmark_version=resolved_benchmark_version,
            margin=min_improvement,
            benchmark_digest=actual_benchmark_digest,
            gate_outcomes=gate_outcomes,
        )
        return _persist_audit_and_finalize(
            dec,
            model_name=model_name,
            alias=alias,
            candidate_version=candidate_version,
            candidate_run_id=candidate_run_id,
            champion_version=None,
            audit_db_path=audit_db_path,
            candidate_training_df=candidate_training_df,
        )

    cand_eval = evaluate_model_on_benchmark(candidate_model, resolved_benchmark_df)
    computed_candidate_f1 = cand_eval["f1_score"]
    gate_outcomes["candidate_evaluation"] = {
        "passed": True,
        "metrics": cand_eval,
    }

    # 4. Lookup champion and evaluate on the exact same benchmark records
    client = MlflowClient()
    champion_version = None
    champ_eval = None

    try:
        champ_ver_obj = client.get_model_version_by_alias(model_name, alias)
        champion_version = str(champ_ver_obj.version)
    except MlflowException as exc:
        missing_alias_code = ErrorCode.Name(ErrorCode.RESOURCE_DOES_NOT_EXIST)
        if exc.error_code != missing_alias_code:
            raise
        champion_f1 = None
        gate_outcomes["champion_evaluation"] = {
            "passed": True,
            "champion_present": False,
        }
    else:
        # Champion exists: evaluate on the shared benchmark
        if champion_model is not None:
            champ_eval = evaluate_model_on_benchmark(
                champion_model, resolved_benchmark_df
            )
            champion_f1 = champ_eval["f1_score"]
        else:
            try:
                loaded_champion = load_registered_model_by_alias(model_name, alias)
                champ_eval = evaluate_model_on_benchmark(
                    loaded_champion, resolved_benchmark_df
                )
                champion_f1 = champ_eval["f1_score"]
            except Exception:
                gate_outcomes["champion_evaluation"] = {
                    "passed": False,
                    "champion_present": True,
                    "champion_version": champion_version,
                    "reason": "champion_benchmark_evaluation_failed",
                }
                dec = PromotionDecision(
                    approved=False,
                    reason="champion_benchmark_evaluation_failed",
                    candidate_f1=computed_candidate_f1,
                    champion_f1=None,
                    benchmark_version=resolved_benchmark_version,
                    margin=min_improvement,
                    benchmark_digest=actual_benchmark_digest,
                    candidate_metrics=cand_eval,
                    gate_outcomes=gate_outcomes,
                )
                return _persist_audit_and_finalize(
                    dec,
                    model_name=model_name,
                    alias=alias,
                    candidate_version=candidate_version,
                    candidate_run_id=candidate_run_id,
                    champion_version=champion_version,
                    audit_db_path=audit_db_path,
                    candidate_training_df=candidate_training_df,
                    client=client,
                )

        gate_outcomes["champion_evaluation"] = {
            "passed": True,
            "champion_present": True,
            "champion_version": champion_version,
            "metrics": champ_eval,
        }

    # 5. Evaluate promotion gate
    gate_decision = evaluate_promotion(
        computed_candidate_f1,
        champion_f1,
        minimum_f1=minimum_f1,
        min_improvement=min_improvement,
        candidate_benchmark_version=candidate_benchmark_version
        or resolved_benchmark_version,
        champion_benchmark_version=champion_benchmark_version
        or resolved_benchmark_version,
    )

    gate_outcomes["quality_gate"] = {
        "passed": gate_decision.approved,
        "reason": gate_decision.reason,
        "candidate_f1": computed_candidate_f1,
        "champion_f1": champion_f1,
        "minimum_f1": minimum_f1,
        "margin": min_improvement,
    }

    dec = PromotionDecision(
        approved=gate_decision.approved,
        reason=gate_decision.reason,
        candidate_f1=gate_decision.candidate_f1,
        champion_f1=gate_decision.champion_f1,
        benchmark_version=gate_decision.benchmark_version,
        margin=gate_decision.margin,
        benchmark_digest=actual_benchmark_digest,
        candidate_metrics=cand_eval,
        champion_metrics=champ_eval,
        gate_outcomes=gate_outcomes,
        candidate_version=candidate_version,
        champion_version=champion_version,
        candidate_run_id=candidate_run_id,
    )

    return _persist_audit_and_finalize(
        dec,
        model_name=model_name,
        alias=alias,
        candidate_version=candidate_version,
        candidate_run_id=candidate_run_id,
        champion_version=champion_version,
        audit_db_path=audit_db_path,
        candidate_training_df=candidate_training_df,
        client=client,
    )
