"""Production job handler integrating worker orchestration with controlled training.

Bridges the background worker execution loop with model training, MLflow candidate
registration, durable identity fencing, irreversible promotion markers, quality gate
evaluation, and post-promotion registry verification.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode

from backend.app.config import settings
from backend.app.data.pipeline import load_and_validate_churn_data
from backend.app.ml.audit import get_promotion_audit_by_job_id
from backend.app.ml.jobs import (
    JobNotFoundError,
    JobOwnershipError,
    get_job,
    mark_promotion_started,
    persist_candidate_identity,
    validate_job_ownership,
)
from backend.app.ml.training import TrainingConfig
from backend.app.ml.worker import JobExecutionContext, JobExecutionResult
from backend.app.ml.workflow import run_training_and_promote

logger = logging.getLogger(__name__)

__all__ = [
    "ControlledTrainingJobHandler",
    "TrainingJobHandler",
]


class ControlledTrainingJobHandler:
    """Production job handler integrating worker orchestration with training and
    promotion."""

    def __init__(
        self,
        database_path: str | Path | None = None,
        model_path: str | Path | None = None,
        model_name: str | None = None,
        alias: str | None = None,
        training_config: TrainingConfig | None = None,
        benchmark_df: pd.DataFrame | None = None,
        benchmark_version: str | None = None,
        data_path: str | Path | None = None,
        mlflow_client: MlflowClient | None = None,
    ) -> None:
        self.database_path: Path = Path(
            database_path
            if database_path is not None
            else settings.lifecycle_audit_db_path
        )
        self.model_path: Path = Path(
            model_path if model_path is not None else "models/candidate_model.joblib"
        )
        self.model_name: str = model_name or settings.model_name
        self.alias: str = alias or settings.model_alias
        self.training_config: TrainingConfig | None = training_config
        self.benchmark_df: pd.DataFrame | None = benchmark_df
        self.benchmark_version: str | None = benchmark_version
        self.data_path: Path | None = Path(data_path) if data_path is not None else None
        self._mlflow_client: MlflowClient | None = mlflow_client

    def _get_client(self) -> MlflowClient:
        """Return the active or configured MLflow client."""
        if self._mlflow_client is not None:
            return self._mlflow_client
        return MlflowClient()

    def _resolve_dataset_path(self, job: Mapping[str, Any]) -> Path:
        """Resolve training dataset path using job payload or dataset_id strictly.

        If a specific dataset identity is given, it must be resolved directly.
        Two jobs specifying different dataset identities must never silently fall
        back to the same default dataset.
        """
        payload = job.get("payload")
        if isinstance(payload, Mapping) and payload.get("data_path"):
            custom_path = Path(str(payload["data_path"]))
            if custom_path.exists():
                return custom_path
            raise FileNotFoundError(
                f"Custom data_path '{custom_path}' specified in job payload "
                f"does not exist for job '{job.get('job_id')}'."
            )

        dataset_id = str(job.get("dataset_id", "")).strip()
        if dataset_id:
            direct_path = Path(dataset_id)
            if direct_path.exists():
                return direct_path

            raw_path = Path("data/raw") / dataset_id
            if raw_path.exists():
                return raw_path

            csv_path = Path("data/raw") / f"{dataset_id}.csv"
            if csv_path.exists():
                return csv_path

            # Explicit dataset_id was provided but not found: fail immediately
            raise FileNotFoundError(
                f"Dataset identity '{dataset_id}' specified for job "
                f"'{job.get('job_id')}' could not be resolved to an existing file."
            )

        # Only when no dataset_id or payload was specified: use configured defaults
        if self.data_path is not None and self.data_path.exists():
            return self.data_path

        default_ref = Path(settings.drift_reference_path)
        if default_ref.exists():
            return default_ref

        raise FileNotFoundError(
            f"Could not resolve any dataset file for job '{job.get('job_id')}' "
            "(no dataset_id specified and no default dataset available)."
        )

    def __call__(
        self,
        job: dict[str, Any],
        context: JobExecutionContext,
    ) -> JobExecutionResult:
        """Execute a claimed retraining job with durable fencing and promotion gates."""
        job_id = context.job_id
        worker_id = context.worker_id
        lease_token = context.lease_token

        # 1. Pre-execution lease ownership validation
        validate_job_ownership(
            self.database_path,
            job_id=job_id,
            worker_id=worker_id,
            lease_token=lease_token,
        )
        if context.is_cancelled:
            raise JobOwnershipError(
                f"Job '{job_id}' execution cancelled due to lease loss or shutdown."
            )

        # 2. Resolve and validate dataset using existing validator
        data_file_path = self._resolve_dataset_path(job)
        df = load_and_validate_churn_data(data_file_path)

        # 3. Check for cancellation before expensive training steps
        if context.is_cancelled:
            raise JobOwnershipError(
                f"Job '{job_id}' cancelled before model training invocation."
            )

        candidate_metadata: dict[str, Any] = {}

        def on_candidate_registered(
            candidate_version: str,
            candidate_run_id: str | None,
        ) -> None:
            """Pre-promotion callback to persist identities and set marker."""
            # Step A: Revalidate current lease ownership
            validate_job_ownership(
                self.database_path,
                job_id=job_id,
                worker_id=worker_id,
                lease_token=lease_token,
            )
            if context.is_cancelled:
                raise JobOwnershipError(
                    f"Job '{job_id}' lost ownership before candidate persistence."
                )

            # Step B: Lookup current champion version from registry
            client = self._get_client()
            prev_champion_version: str | None = None
            try:
                champ_obj = client.get_model_version_by_alias(
                    self.model_name,
                    self.alias,
                )
                prev_champion_version = str(champ_obj.version)
            except MlflowException as exc:
                missing_codes = {
                    ErrorCode.Name(ErrorCode.RESOURCE_DOES_NOT_EXIST),
                    str(ErrorCode.RESOURCE_DOES_NOT_EXIST),
                    ErrorCode.RESOURCE_DOES_NOT_EXIST,
                    "404",
                }
                if exc.error_code not in missing_codes:
                    raise

            # Step C: Durably persist candidate version, run ID, and previous champion
            # Fenced by job_id, worker_id, lease_token, status='running', is_active=1
            persist_candidate_identity(
                self.database_path,
                job_id=job_id,
                worker_id=worker_id,
                lease_token=lease_token,
                candidate_version=candidate_version,
                candidate_run_id=candidate_run_id,
                previous_champion_version=prev_champion_version,
            )

            candidate_metadata["candidate_version"] = candidate_version
            candidate_metadata["candidate_run_id"] = candidate_run_id
            candidate_metadata["previous_champion_version"] = prev_champion_version

            # Step D: Immediately before calling promote_model():
            # 1. Revalidate lease ownership
            validate_job_ownership(
                self.database_path,
                job_id=job_id,
                worker_id=worker_id,
                lease_token=lease_token,
            )
            if context.is_cancelled:
                raise JobOwnershipError(
                    f"Job '{job_id}' lost ownership before promotion marker."
                )

            # 2. Confirm candidate and previous champion identities are persisted
            persisted_job = get_job(self.database_path, job_id=job_id)
            if persisted_job is None:
                raise JobNotFoundError(f"Job '{job_id}' does not exist in database.")
            if persisted_job.get("candidate_version") != str(candidate_version):
                raise RuntimeError(
                    f"Candidate version '{candidate_version}' not durably persisted "
                    f"for job '{job_id}'."
                )
            if prev_champion_version is not None and persisted_job.get(
                "previous_champion_version"
            ) != str(prev_champion_version):
                raise RuntimeError(
                    f"Previous champion version '{prev_champion_version}' not durably "
                    f"persisted for job '{job_id}'."
                )

            # 3. Atomically set promotion_started_at, require 1 row, commit transaction
            mark_promotion_started(
                self.database_path,
                job_id=job_id,
                worker_id=worker_id,
                lease_token=lease_token,
                candidate_version=candidate_version,
                candidate_run_id=candidate_run_id,
                previous_champion_version=prev_champion_version,
            )

        # 4. Invoke training pipeline with callback and job_id correlation
        metrics, decision = run_training_and_promote(
            df=df,
            model_path=self.model_path,
            config=self.training_config,
            benchmark_df=self.benchmark_df,
            benchmark_version=self.benchmark_version,
            candidate_callback=on_candidate_registered,
            job_id=job_id,
            audit_db_path=self.database_path,
        )

        # 5. Post-promotion validation & fencing checks
        if context.is_cancelled:
            raise JobOwnershipError(
                f"Job '{job_id}' execution cancelled or lease lost during promotion."
            )

        validate_job_ownership(
            self.database_path,
            job_id=job_id,
            worker_id=worker_id,
            lease_token=lease_token,
        )

        cand_ver = str(
            decision.candidate_version
            or candidate_metadata.get("candidate_version")
            or ""
        )
        cand_run = decision.candidate_run_id or candidate_metadata.get(
            "candidate_run_id"
        )
        prev_champ = decision.champion_version or candidate_metadata.get(
            "previous_champion_version"
        )

        payload_updates: dict[str, Any] = {
            "metrics": metrics,
            "gate_outcomes": decision.gate_outcomes,
        }

        # Retrieve promotion audit record correlated with exact job_id
        try:
            audit_record = get_promotion_audit_by_job_id(
                job_id,
                database_path=self.database_path,
            )
        except Exception as audit_err:
            logger.error(
                "promotion_audit_retrieval_failed",
                extra={"job_id": job_id, "error": str(audit_err)},
            )
            return JobExecutionResult(
                status="reconciliation_required",
                failure_reason=f"Promotion audit query error: {audit_err}",
                promotion_approved=False,
                promotion_reason="audit_query_failed",
                promotion_audit_id=decision.audit_id,
                candidate_version=cand_ver,
                candidate_run_id=cand_run,
                previous_champion_version=prev_champ,
                payload_updates=payload_updates,
            )

        effective_audit_id = decision.audit_id or (
            audit_record["audit_id"] if audit_record is not None else None
        )

        # Verify current remote MLflow alias state
        client = self._get_client()
        observed_alias_version: str | None = None
        try:
            current_champ_ver = client.get_model_version_by_alias(
                self.model_name,
                self.alias,
            )
            observed_alias_version = str(current_champ_ver.version)
        except MlflowException as exc:
            missing_codes = {
                ErrorCode.Name(ErrorCode.RESOURCE_DOES_NOT_EXIST),
                str(ErrorCode.RESOURCE_DOES_NOT_EXIST),
                ErrorCode.RESOURCE_DOES_NOT_EXIST,
                "404",
            }
            if exc.error_code not in missing_codes:
                raise

        # Case 1: Ambiguous MLflow outcome (mutation failed or state unverified)
        if decision.reason in ("alias_state_unverified", "alias_mutation_failed"):
            logger.error(
                "ambiguous_mlflow_promotion_outcome",
                extra={
                    "job_id": job_id,
                    "reason": decision.reason,
                    "observed_alias": observed_alias_version,
                    "candidate_version": cand_ver,
                },
            )
            return JobExecutionResult(
                status="reconciliation_required",
                failure_reason=(
                    f"Ambiguous MLflow promotion outcome: {decision.reason} "
                    f"(alias points to version '{observed_alias_version}')."
                ),
                promotion_approved=False,
                promotion_reason=decision.reason,
                promotion_audit_id=effective_audit_id,
                candidate_version=cand_ver,
                candidate_run_id=cand_run,
                previous_champion_version=prev_champ,
                payload_updates=payload_updates,
            )

        # Case 2: Approved promotion
        if decision.approved:
            # Check A: Audit record must exist and indicate approval
            if audit_record is None or not audit_record.get("approved"):
                logger.error(
                    "promotion_approved_but_audit_missing_or_unapproved",
                    extra={
                        "job_id": job_id,
                        "candidate_version": cand_ver,
                        "audit_present": audit_record is not None,
                    },
                )
                return JobExecutionResult(
                    status="reconciliation_required",
                    failure_reason=(
                        "Promotion approved by gate but exact job audit record "
                        "is missing or invalid."
                    ),
                    promotion_approved=False,
                    promotion_reason="audit_missing_or_unapproved",
                    promotion_audit_id=effective_audit_id,
                    candidate_version=cand_ver,
                    candidate_run_id=cand_run,
                    previous_champion_version=prev_champ,
                    payload_updates=payload_updates,
                )

            # Check B: Remote champion alias must point to the candidate version
            if observed_alias_version != cand_ver:
                logger.error(
                    "promotion_approved_but_alias_mismatch",
                    extra={
                        "job_id": job_id,
                        "candidate_version": cand_ver,
                        "observed_alias": observed_alias_version,
                    },
                )
                return JobExecutionResult(
                    status="reconciliation_required",
                    failure_reason=(
                        f"Promotion approved by gate but champion alias points to "
                        f"version '{observed_alias_version}' instead of candidate "
                        f"version '{cand_ver}'."
                    ),
                    promotion_approved=False,
                    promotion_reason="alias_mismatch_after_approval",
                    promotion_audit_id=effective_audit_id,
                    candidate_version=cand_ver,
                    candidate_run_id=cand_run,
                    previous_champion_version=prev_champ,
                    payload_updates=payload_updates,
                )

            # Succeeded outcome
            return JobExecutionResult(
                status="succeeded",
                promotion_approved=True,
                promotion_reason=decision.reason,
                promotion_audit_id=effective_audit_id,
                candidate_version=cand_ver,
                candidate_run_id=cand_run,
                previous_champion_version=prev_champ,
                payload_updates=payload_updates,
            )

        # Case 3: Rejected promotion
        # Check: Verify alias did not unexpectedly change to candidate
        if observed_alias_version == cand_ver and cand_ver and cand_ver != prev_champ:
            logger.error(
                "promotion_rejected_but_alias_points_to_candidate",
                extra={
                    "job_id": job_id,
                    "candidate_version": cand_ver,
                    "decision_reason": decision.reason,
                },
            )
            return JobExecutionResult(
                status="reconciliation_required",
                failure_reason=(
                    f"Promotion was rejected with reason '{decision.reason}' but "
                    f"champion alias unexpectedly points to candidate version "
                    f"'{cand_ver}'."
                ),
                promotion_approved=False,
                promotion_reason="alias_mutated_on_rejection",
                promotion_audit_id=effective_audit_id,
                candidate_version=cand_ver,
                candidate_run_id=cand_run,
                previous_champion_version=prev_champ,
                payload_updates=payload_updates,
            )

        # Check: Verify exact job rejection audit record exists and indicates rejection
        if audit_record is None or audit_record.get("approved"):
            logger.error(
                "promotion_rejected_but_audit_missing_or_approved",
                extra={
                    "job_id": job_id,
                    "candidate_version": cand_ver,
                    "audit_present": audit_record is not None,
                },
            )
            return JobExecutionResult(
                status="reconciliation_required",
                failure_reason=(
                    "Promotion rejected by gate but exact job audit record "
                    "is missing or invalid."
                ),
                promotion_approved=False,
                promotion_reason="audit_missing_or_inconsistent",
                promotion_audit_id=effective_audit_id,
                candidate_version=cand_ver,
                candidate_run_id=cand_run,
                previous_champion_version=prev_champ,
                payload_updates=payload_updates,
            )

        # Clean rejection
        return JobExecutionResult(
            status="rejected",
            failure_reason=decision.reason,
            promotion_approved=False,
            promotion_reason=decision.reason,
            promotion_audit_id=effective_audit_id,
            candidate_version=cand_ver,
            candidate_run_id=cand_run,
            previous_champion_version=prev_champ,
            payload_updates=payload_updates,
        )


TrainingJobHandler = ControlledTrainingJobHandler
