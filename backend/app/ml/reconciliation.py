"""Operator reconciliation workflow for jobs quarantined in reconciliation_required.

Provides read-only evidence inspection, exclusive maintenance locking,
safe MLflow alias reconciliation with explicit operator authorization, and
atomic database finalization in accordance with safety invariants.
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode
from mlflow.tracking import MlflowClient

from backend.app.config import settings
from backend.app.ml.audit import (
    AuditDatabaseError,
    get_promotion_audit_by_job_id,
)
from backend.app.ml.jobs import JobNotFoundError, get_job
from backend.app.ml.lock import (
    MaintenanceLockConflictError,
    operator_maintenance_lock,
    resolve_shared_lock_path,
)
from backend.app.ml.migrations import (
    VALID_RESOLUTIONS,
    finalize_operator_resolution_atomic,
    validate_operator_resolution_inputs,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EvidenceVerificationError",
    "JobNotQuarantinedError",
    "MLflowOperationError",
    "ReconciliationError",
    "ReconciliationEvidence",
    "SafetyInvariantViolationError",
    "inspect_quarantined_job",
    "main",
    "reconcile_job",
]


class ReconciliationError(RuntimeError):
    """Base exception for operator reconciliation errors."""


class JobNotQuarantinedError(ReconciliationError):
    """Raised when target job is not in active reconciliation_required status."""


class EvidenceVerificationError(ReconciliationError):
    """Raised when job or audit evidence is missing, duplicate, or contradictory."""


class SafetyInvariantViolationError(ReconciliationError):
    """Raised when resolution violates operator safety invariants."""


class MLflowOperationError(ReconciliationError):
    """Raised when MLflow registry interaction or verification fails."""


def _is_resource_does_not_exist(exc: BaseException) -> bool:
    """Check if an exception indicates an MLflow resource does not exist (404)."""
    if not isinstance(exc, MlflowException):
        return False
    missing_codes = {
        ErrorCode.Name(ErrorCode.RESOURCE_DOES_NOT_EXIST),
        str(ErrorCode.RESOURCE_DOES_NOT_EXIST),
        ErrorCode.RESOURCE_DOES_NOT_EXIST,
        "404",
        404,
    }
    error_code = getattr(exc, "error_code", None)
    if error_code in missing_codes:
        return True
    return str(error_code).upper() == "RESOURCE_DOES_NOT_EXIST"


@dataclass(frozen=True)
class ReconciliationEvidence:
    """Read-only evidence collected for a quarantined job."""

    job_id: str
    job_status: str
    is_active: int | None
    candidate_version: str | None
    candidate_run_id: str | None
    previous_champion_version: str | None
    promotion_started_at: str | None
    audit_record: dict[str, Any] | None
    audit_error: str | None
    observed_alias_version: str | None
    observed_alias_run_id: str | None
    model_name: str
    alias: str
    alias_error: str | None
    candidate_model_found: bool
    recommended_action: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert evidence structure to dictionary."""
        return asdict(self)


def inspect_quarantined_job(
    job_id: str,
    database_path: str | Path | None = None,
    model_name: str | None = None,
    alias: str | None = None,
    client: MlflowClient | None = None,
) -> ReconciliationEvidence:
    """Read-only inspection of a quarantined job and its external evidence.

    Does not acquire execution locks, mutate database state, or modify MLflow aliases.
    Reports missing, duplicate, contradictory, and unverifiable evidence explicitly.

    Args:
        job_id: Identifier of the job to inspect.
        database_path: Optional path to SQLite audit database.
        model_name: Optional MLflow registered model name.
        alias: Optional MLflow alias name.
        client: Optional MlflowClient instance.

    Returns:
        ReconciliationEvidence containing verified findings and recommended action.
    """
    db_path = Path(
        database_path if database_path is not None else settings.lifecycle_audit_db_path
    )
    eff_model = model_name or settings.model_name
    eff_alias = alias or settings.model_alias

    job = get_job(database_path=db_path, job_id=job_id)
    if job is None:
        raise JobNotFoundError(f"Job '{job_id}' does not exist in database.")

    cand_ver = (
        str(job["candidate_version"])
        if job.get("candidate_version") is not None
        else None
    )
    cand_run = (
        str(job["candidate_run_id"])
        if job.get("candidate_run_id") is not None
        else None
    )
    prev_champ = (
        str(job["previous_champion_version"])
        if job.get("previous_champion_version") is not None
        else None
    )
    promo_started = job.get("promotion_started_at")
    job_status = job.get("status", "unknown")
    is_active = job.get("is_active")

    # Audit inspection
    audit_record: dict[str, Any] | None = None
    audit_error: str | None = None
    try:
        audit_record = get_promotion_audit_by_job_id(job_id, database_path=db_path)
    except AuditDatabaseError as exc:
        audit_error = str(exc)
    except Exception as exc:
        audit_error = f"Unexpected error reading audit: {exc}"

    details: dict[str, Any] = {}
    audit_run = (
        str(audit_record["candidate_run_id"])
        if audit_record and audit_record.get("candidate_run_id") is not None
        else None
    )
    if cand_run and audit_run and cand_run != audit_run:
        details["job_audit_run_id_mismatch"] = (
            f"Job run_id '{cand_run}' != Audit run_id '{audit_run}'"
        )

    # MLflow inspection
    mlflow_client = client or MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    observed_alias_ver: str | None = None
    observed_alias_run: str | None = None
    alias_error: str | None = None
    candidate_found = False

    try:
        ver_obj = mlflow_client.get_model_version_by_alias(eff_model, eff_alias)
        observed_alias_ver = str(ver_obj.version)
        observed_alias_run = getattr(ver_obj, "run_id", None)
    except Exception as exc:
        if _is_resource_does_not_exist(exc):
            observed_alias_ver = None
            details["alias_absence_verified"] = True
        else:
            alias_error = f"MLflow alias query failed: {exc}"

    # Inspect candidate version in MLflow if specified
    if cand_ver is not None and alias_error is None:
        try:
            cand_obj = mlflow_client.get_model_version(eff_model, cand_ver)
            candidate_found = True
            cand_run_id = getattr(cand_obj, "run_id", None)
            details["candidate_registered_run_id"] = cand_run_id
            if cand_run and cand_run_id != cand_run:
                details["run_id_mismatch"] = (
                    f"Job run_id '{cand_run}' != MLflow run_id '{cand_run_id}'"
                )
            if audit_run and cand_run_id and cand_run_id != audit_run:
                details["audit_mlflow_run_id_mismatch"] = (
                    f"Audit run_id '{audit_run}' != MLflow run_id '{cand_run_id}'"
                )
        except Exception as exc:
            if not _is_resource_does_not_exist(exc):
                details["candidate_lookup_error"] = str(exc)

    if observed_alias_ver == cand_ver:
        expected_run = cand_run or audit_run
        if expected_run and observed_alias_run and observed_alias_run != expected_run:
            details["alias_run_id_mismatch"] = (
                f"Expected run_id '{expected_run}' != "
                f"Alias run_id '{observed_alias_run}'"
            )
        cand_mlflow_run = details.get("candidate_registered_run_id")
        if (
            cand_mlflow_run
            and observed_alias_run
            and observed_alias_run != cand_mlflow_run
        ):
            details["alias_candidate_run_id_mismatch"] = (
                f"MLflow candidate run_id '{cand_mlflow_run}' != "
                f"Alias run_id '{observed_alias_run}'"
            )

    # Formulate recommended action
    has_identity_mismatch = any(
        k in details
        for k in (
            "run_id_mismatch",
            "alias_run_id_mismatch",
            "audit_mlflow_run_id_mismatch",
            "job_audit_run_id_mismatch",
            "alias_candidate_run_id_mismatch",
        )
    )
    recommended = "cannot_resolve_unverifiable"
    if job_status != "reconciliation_required" or is_active != 1:
        recommended = "not_quarantined"
    elif audit_error is not None:
        recommended = "quarantine_due_to_audit_error"
    elif alias_error is not None:
        recommended = "quarantine_due_to_mlflow_error"
    elif has_identity_mismatch:
        recommended = "quarantine_due_to_identity_mismatch"
    elif audit_record is None:
        recommended = "quarantine_due_to_missing_audit"
    elif audit_record.get("approved") is True:
        if observed_alias_ver == cand_ver and cand_ver is not None:
            recommended = "ready_for_success_resolution"
        elif observed_alias_ver == prev_champ:
            recommended = "requires_alias_update_for_success"
        else:
            recommended = "quarantine_unexpected_alias_version"
    else:  # audit_record.get("approved") is False
        if observed_alias_ver == prev_champ:
            recommended = "ready_for_failure_resolution"
        elif observed_alias_ver == cand_ver and cand_ver is not None:
            if prev_champ is not None:
                recommended = "requires_rollback_for_failure"
            else:
                recommended = "requires_bootstrap_deletion_for_failure"
        elif observed_alias_ver is None and prev_champ is None:
            recommended = "ready_for_failure_resolution_bootstrap_absent"
        else:
            recommended = "quarantine_unexpected_alias_version"

    return ReconciliationEvidence(
        job_id=job_id,
        job_status=job_status,
        is_active=is_active,
        candidate_version=cand_ver,
        candidate_run_id=cand_run,
        previous_champion_version=prev_champ,
        promotion_started_at=promo_started,
        audit_record=audit_record,
        audit_error=audit_error,
        observed_alias_version=observed_alias_ver,
        observed_alias_run_id=observed_alias_run,
        model_name=eff_model,
        alias=eff_alias,
        alias_error=alias_error,
        candidate_model_found=candidate_found,
        recommended_action=recommended,
        details=details,
    )


def reconcile_job(
    job_id: str,
    selected_resolution: str,
    operator: str,
    reason: str,
    *,
    database_path: str | Path | None = None,
    lock_path: str | Path | None = None,
    model_name: str | None = None,
    alias: str | None = None,
    update_alias: bool = False,
    rollback_alias: bool = False,
    delete_alias: bool = False,
    force_bootstrap_removal: bool = False,
    client: MlflowClient | None = None,
) -> dict[str, Any]:
    """Execute fail-closed operator reconciliation of a quarantined job.

    Acquires exclusive maintenance lock before inspecting or mutating state.
    Re-reads all evidence under lock, enforces strict safety invariants,
    executes verified MLflow alias operations, and atomically finalizes the database.

    Args:
        job_id: Identifier of the quarantined job.
        selected_resolution: Must be 'resolved_succeeded' or 'resolved_failed'.
        operator: Non-empty identifier of the operating person or role.
        reason: Non-empty rationale for the resolution decision.
        database_path: Optional path to SQLite audit database.
        lock_path: Optional path to worker/maintenance process lock file.
        model_name: Optional MLflow registered model name.
        alias: Optional MLflow alias name.
        update_alias: Explicit authorization to update alias to candidate for success.
        rollback_alias: Explicit authorization to restore champion for failure.
        delete_alias: Explicit authorization to delete alias in bootstrap failure.
        force_bootstrap_removal: Explicit secondary confirmation for deletion.
        client: Optional MlflowClient instance.

    Returns:
        Finalized job record dictionary with updated terminal status.
    """
    if not operator or not operator.strip():
        raise ValueError("operator must be a non-empty string.")
    if not reason or not reason.strip():
        raise ValueError("reason must be a non-empty string.")

    validate_operator_resolution_inputs(selected_resolution, 0)

    db_path = Path(
        database_path if database_path is not None else settings.lifecycle_audit_db_path
    )
    shared_lock_path = resolve_shared_lock_path(lock_path)
    eff_model = model_name or settings.model_name
    eff_alias = alias or settings.model_alias
    mlflow_client = client or MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    logger.info(
        "operator_reconciliation_attempt job_id=%s resolution=%s operator=%s",
        job_id,
        selected_resolution,
        operator,
    )

    # Acquire exclusive maintenance lock. Fails immediately if worker is active.
    with operator_maintenance_lock(shared_lock_path):
        # 1. Freshly inspect job state under lock
        job = get_job(database_path=db_path, job_id=job_id)
        if job is None:
            raise JobNotFoundError(f"Job '{job_id}' does not exist.")

        status = job.get("status")
        is_active = job.get("is_active")
        if status != "reconciliation_required" or is_active != 1:
            raise JobNotQuarantinedError(
                f"Job '{job_id}' is not in active 'reconciliation_required' status "
                f"(current status='{status}', is_active={is_active}). "
                "Reconciliation refused."
            )

        cand_ver = (
            str(job["candidate_version"])
            if job.get("candidate_version") is not None
            else None
        )
        cand_run = (
            str(job["candidate_run_id"])
            if job.get("candidate_run_id") is not None
            else None
        )
        prev_champ = (
            str(job["previous_champion_version"])
            if job.get("previous_champion_version") is not None
            else None
        )

        # 2. Re-read exact promotion audit for this job_id
        try:
            audit_record = get_promotion_audit_by_job_id(job_id, database_path=db_path)
        except AuditDatabaseError as exc:
            raise EvidenceVerificationError(
                f"Duplicate or corrupted promotion audit records for job '{job_id}': "
                f"{exc}. Job remains quarantined."
            ) from exc

        if audit_record is not None:
            if audit_record.get("job_id") != job_id:
                raise EvidenceVerificationError(
                    f"Audit record job_id mismatch: expected '{job_id}', "
                    f"got '{audit_record.get('job_id')}'. Job remains quarantined."
                )
            if audit_record.get("model_name") != eff_model:
                raise EvidenceVerificationError(
                    f"Audit record model_name mismatch: expected '{eff_model}', "
                    f"got '{audit_record.get('model_name')}'. Job remains quarantined."
                )
            if cand_ver and audit_record.get("candidate_version") is not None:
                if str(audit_record["candidate_version"]) != cand_ver:
                    raise EvidenceVerificationError(
                        f"Audit candidate version "
                        f"'{audit_record['candidate_version']}' does not match "
                        f"job candidate version '{cand_ver}'. "
                        "Job remains quarantined."
                    )
            audit_run = (
                str(audit_record["candidate_run_id"])
                if audit_record.get("candidate_run_id") is not None
                else None
            )
            if cand_run and audit_run and audit_run != cand_run:
                raise EvidenceVerificationError(
                    f"Audit candidate run ID '{audit_run}' "
                    f"does not match job candidate run ID '{cand_run}'. "
                    "Job remains quarantined."
                )
        else:
            audit_run = None

        # 3. Query current remote MLflow alias state
        alias_obj = None
        try:
            alias_obj = mlflow_client.get_model_version_by_alias(eff_model, eff_alias)
            observed_alias_ver: str | None = str(alias_obj.version)
        except Exception as exc:
            if _is_resource_does_not_exist(exc):
                observed_alias_ver = None
            else:
                raise MLflowOperationError(
                    f"Cannot verify remote MLflow alias '{eff_alias}' for model "
                    f"'{eff_model}': {exc}. Job remains quarantined."
                ) from exc

        # 4. Strict candidate & alias identity validation before any mutation
        cand_obj = None
        cand_obj_run = None
        if cand_ver is not None:
            try:
                cand_obj = mlflow_client.get_model_version(eff_model, cand_ver)
                cand_model_name = getattr(cand_obj, "name", None)
                if cand_model_name and cand_model_name != eff_model:
                    raise EvidenceVerificationError(
                        f"Candidate model name mismatch in MLflow: registered "
                        f"'{cand_model_name}' != expected '{eff_model}'. "
                        "Job remains quarantined."
                    )
                cand_obj_run = getattr(cand_obj, "run_id", None)
                if cand_run and cand_obj_run != cand_run:
                    raise EvidenceVerificationError(
                        f"Candidate model run_id mismatch in MLflow: registered "
                        f"'{cand_obj_run}' != job '{cand_run}'. "
                        "Job remains quarantined."
                    )
                if audit_run and cand_obj_run and cand_obj_run != audit_run:
                    raise EvidenceVerificationError(
                        f"Candidate model run_id mismatch in MLflow: registered "
                        f"'{cand_obj_run}' != audit '{audit_run}'. "
                        "Job remains quarantined."
                    )
                if not cand_run and not audit_run:
                    logger.warning(
                        "candidate_run_id_missing_from_evidence job_id=%s version=%s. "
                        "Limitation: relying on model and version without run ID.",
                        job_id,
                        cand_ver,
                    )
            except Exception as exc:
                if isinstance(
                    exc, (EvidenceVerificationError, SafetyInvariantViolationError)
                ):
                    raise
                raise MLflowOperationError(
                    f"Failed to verify candidate version '{cand_ver}' in MLflow: "
                    f"{exc}. Job remains quarantined."
                ) from exc

        # Verify live alias identity if it points to candidate version
        if observed_alias_ver is not None and observed_alias_ver == cand_ver:
            if alias_obj is not None:
                alias_model_name = getattr(alias_obj, "name", None)
                if alias_model_name and alias_model_name != eff_model:
                    raise EvidenceVerificationError(
                        f"Live alias '{eff_alias}' model name mismatch: "
                        f"'{alias_model_name}' != expected '{eff_model}'. "
                        "Job remains quarantined."
                    )
                alias_obj_run = getattr(alias_obj, "run_id", None)
                expected_run = cand_run or audit_run
                if expected_run and alias_obj_run and alias_obj_run != expected_run:
                    raise EvidenceVerificationError(
                        f"Live alias '{eff_alias}' points to version "
                        f"'{observed_alias_ver}' with run_id '{alias_obj_run}', "
                        f"expected candidate run_id '{expected_run}'. "
                        "Job remains quarantined."
                    )
                if cand_obj_run and alias_obj_run and alias_obj_run != cand_obj_run:
                    raise EvidenceVerificationError(
                        f"Live alias '{eff_alias}' points to version "
                        f"'{observed_alias_ver}' with run_id '{alias_obj_run}', "
                        f"conflicts with candidate model run_id '{cand_obj_run}'. "
                        "Job remains quarantined."
                    )

        # 5. Process resolution according to strict invariants
        target_champ_ver: str | None = None
        rollback_performed: int = 0
        action_name: str

        if selected_resolution == "resolved_succeeded":
            if audit_record is None or not audit_record.get("approved"):
                audit_reason = audit_record.get("reason") if audit_record else "missing"
                raise EvidenceVerificationError(
                    f"Cannot resolve job '{job_id}' as succeeded: promotion audit "
                    f"is not approved (reason: {audit_reason}). "
                    "Job remains quarantined."
                )

            if cand_ver is None:
                raise EvidenceVerificationError(
                    f"Cannot resolve job '{job_id}' as succeeded: candidate_version "
                    "is missing from job record. Job remains quarantined."
                )

            if observed_alias_ver == cand_ver:
                action_name = "resolved_succeeded_alias_verified"
                target_champ_ver = cand_ver
                rollback_performed = 0
            elif observed_alias_ver == prev_champ:
                if not update_alias:
                    raise SafetyInvariantViolationError(
                        f"Cannot resolve job '{job_id}' as succeeded: alias "
                        f"'{eff_alias}' currently points to previous champion "
                        f"'{prev_champ}' instead of candidate '{cand_ver}'. "
                        "Explicit '--update-alias' required."
                    )
                # Mutate alias with explicit authorization
                try:
                    mlflow_client.set_registered_model_alias(
                        eff_model, eff_alias, cand_ver
                    )
                except Exception as exc:
                    raise MLflowOperationError(
                        f"Failed to update alias '{eff_alias}' to candidate "
                        f"'{cand_ver}': {exc}. Job remains quarantined."
                    ) from exc

                # Verify remote alias state immediately after mutation
                try:
                    updated_obj = mlflow_client.get_model_version_by_alias(
                        eff_model, eff_alias
                    )
                    if str(updated_obj.version) != cand_ver:
                        raise MLflowOperationError(
                            f"Alias update verification failed: alias points to "
                            f"'{updated_obj.version}', expected '{cand_ver}'. "
                            "Job remains quarantined."
                        )
                except Exception as exc:
                    if isinstance(exc, MLflowOperationError):
                        raise
                    raise MLflowOperationError(
                        f"Failed to verify alias after update: {exc}. "
                        "Job remains quarantined."
                    ) from exc

                action_name = "resolved_succeeded_alias_updated"
                target_champ_ver = cand_ver
                rollback_performed = 0
            else:
                raise SafetyInvariantViolationError(
                    f"Cannot resolve job '{job_id}' as succeeded: alias points to "
                    f"unexpected version '{observed_alias_ver}' (candidate is "
                    f"'{cand_ver}', previous champion is '{prev_champ}'). "
                    "Job remains quarantined."
                )

        elif selected_resolution == "resolved_failed":
            if observed_alias_ver == prev_champ:
                if prev_champ is None:
                    action_name = "bootstrap_alias_already_absent"
                    target_champ_ver = None
                    rollback_performed = 0
                else:
                    action_name = "resolved_failed_alias_unchanged"
                    target_champ_ver = prev_champ
                    rollback_performed = 0
            elif observed_alias_ver == cand_ver and cand_ver is not None:
                if prev_champ is not None:
                    # Rollback to known previous champion
                    if not rollback_alias:
                        raise SafetyInvariantViolationError(
                            f"Cannot resolve job '{job_id}' as failed: alias points "
                            f"to candidate '{cand_ver}'. Explicit '--rollback-alias' "
                            f"required to restore previous champion '{prev_champ}'."
                        )
                    # Verify previous champion model identity and version in MLflow
                    try:
                        champ_obj = mlflow_client.get_model_version(
                            eff_model, prev_champ
                        )
                        champ_name = getattr(champ_obj, "name", None)
                        if champ_name and champ_name != eff_model:
                            raise EvidenceVerificationError(
                                f"Previous champion model name mismatch in MLflow: "
                                f"'{champ_name}' != expected '{eff_model}'. "
                                "Job remains quarantined."
                            )
                        if str(champ_obj.version) != str(prev_champ):
                            raise EvidenceVerificationError(
                                f"Previous champion version mismatch in MLflow: "
                                f"'{champ_obj.version}' != expected '{prev_champ}'. "
                                "Job remains quarantined."
                            )
                    except Exception as exc:
                        if isinstance(
                            exc,
                            (EvidenceVerificationError, SafetyInvariantViolationError),
                        ):
                            raise
                        raise MLflowOperationError(
                            f"Cannot verify previous champion version '{prev_champ}' "
                            f"in MLflow: {exc}. Job remains quarantined."
                        ) from exc

                    try:
                        mlflow_client.set_registered_model_alias(
                            eff_model, eff_alias, prev_champ
                        )
                    except Exception as exc:
                        raise MLflowOperationError(
                            f"Failed to rollback alias '{eff_alias}' to champion "
                            f"'{prev_champ}': {exc}. Job remains quarantined."
                        ) from exc

                    # Verify rollback immediately
                    try:
                        rb_obj = mlflow_client.get_model_version_by_alias(
                            eff_model, eff_alias
                        )
                        if str(rb_obj.version) != prev_champ:
                            raise MLflowOperationError(
                                f"Rollback verification failed: alias points to "
                                f"'{rb_obj.version}', expected '{prev_champ}'. "
                                "Job remains quarantined."
                            )
                    except Exception as exc:
                        if isinstance(exc, MLflowOperationError):
                            raise
                        raise MLflowOperationError(
                            f"Failed to verify alias rollback: {exc}. "
                            "Job remains quarantined."
                        ) from exc

                    action_name = "resolved_failed_rollback_performed"
                    target_champ_ver = prev_champ
                    rollback_performed = 1
                else:
                    # Bootstrap case: no previous champion exists
                    if not (delete_alias and force_bootstrap_removal):
                        raise SafetyInvariantViolationError(
                            f"Cannot resolve job '{job_id}' as failed: job has no "
                            f"previous champion and alias points to candidate "
                            f"'{cand_ver}'. Deletion requires explicit "
                            "'--delete-alias' and '--force-bootstrap-removal'."
                        )

                    try:
                        mlflow_client.delete_registered_model_alias(
                            eff_model, eff_alias
                        )
                    except Exception as exc:
                        raise MLflowOperationError(
                            f"Failed to delete registered model alias "
                            f"'{eff_alias}': {exc}. Job remains quarantined."
                        ) from exc

                    # Verify genuine absence via RESOURCE_DOES_NOT_EXIST
                    try:
                        remaining_obj = mlflow_client.get_model_version_by_alias(
                            eff_model, eff_alias
                        )
                        raise MLflowOperationError(
                            f"Alias deletion verification failed: alias "
                            f"'{eff_alias}' still points to version "
                            f"'{remaining_obj.version}'. Job remains quarantined."
                        )
                    except Exception as exc:
                        if isinstance(exc, MLflowOperationError):
                            raise
                        if not _is_resource_does_not_exist(exc):
                            raise MLflowOperationError(
                                f"Unexpected error verifying alias deletion: "
                                f"{exc}. Job remains quarantined."
                            ) from exc

                    action_name = "bootstrap_alias_deleted"
                    target_champ_ver = None
                    rollback_performed = 0
            elif observed_alias_ver is None and prev_champ is None:
                # Bootstrap case where alias was already absent
                action_name = "bootstrap_alias_already_absent"
                target_champ_ver = None
                rollback_performed = 0
            else:
                raise SafetyInvariantViolationError(
                    f"Cannot resolve job '{job_id}' as failed: alias points to "
                    f"unexpected version '{observed_alias_ver}' (candidate is "
                    f"'{cand_ver}', previous champion is '{prev_champ}'). "
                    "Job remains quarantined."
                )
        else:
            raise ValueError(f"Unsupported selected_resolution: {selected_resolution}")

        # 6. Atomic database finalization
        resolution_id = f"res-{uuid4()}"
        evidence_payload = {
            "action": action_name,
            "job_id": job_id,
            "operator": operator.strip(),
            "candidate_version": cand_ver,
            "candidate_run_id": cand_run,
            "previous_champion_version": prev_champ,
            "observed_champion_version": observed_alias_ver,
            "target_champion_version": target_champ_ver,
            "rollback_performed": rollback_performed,
            "audit_id": audit_record.get("audit_id") if audit_record else None,
            "audit_approved": audit_record.get("approved") if audit_record else None,
            "resolved_at": datetime.now(UTC).isoformat(),
        }

        # Validate inputs strictly prior to database transaction
        validate_operator_resolution_inputs(selected_resolution, rollback_performed)

        finalize_operator_resolution_atomic(
            database_path=db_path,
            resolution_id=resolution_id,
            job_id=job_id,
            operator=operator.strip(),
            selected_resolution=selected_resolution,
            candidate_version=cand_ver,
            observed_champion_version=observed_alias_ver,
            target_champion_version=target_champ_ver,
            rollback_performed=rollback_performed,
            resolution_reason=reason.strip(),
            evidence_payload=evidence_payload,
        )

        finalized_job = get_job(database_path=db_path, job_id=job_id)
        if finalized_job is None:
            raise JobNotFoundError(f"Finalized job '{job_id}' could not be fetched.")

        logger.info(
            "operator_reconciliation_completed job_id=%s resolution=%s action=%s",
            job_id,
            selected_resolution,
            action_name,
        )
        return finalized_job


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for operator reconciliation."""
    parser = argparse.ArgumentParser(
        prog="modelforge-reconcile",
        description="Operator reconciliation tool for quarantined retraining jobs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # inspect subcommand
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect quarantined job evidence without locking or mutating state.",
    )
    inspect_parser.add_argument("--job-id", required=True, help="Job identifier")
    inspect_parser.add_argument(
        "--db-path", default=None, help="Path to audit SQLite database"
    )
    inspect_parser.add_argument(
        "--model-name", default=None, help="MLflow registered model name"
    )
    inspect_parser.add_argument(
        "--alias", default=None, help="MLflow registered model alias"
    )

    # resolve subcommand
    resolve_parser = subparsers.add_parser(
        "resolve",
        help="Execute operator reconciliation under exclusive maintenance lock.",
    )
    resolve_parser.add_argument("--job-id", required=True, help="Job identifier")
    resolve_parser.add_argument(
        "--resolution",
        required=True,
        choices=sorted(VALID_RESOLUTIONS),
        help="Resolution outcome ('resolved_succeeded' or 'resolved_failed')",
    )
    resolve_parser.add_argument(
        "--operator", required=True, help="Identifier of operating human/role"
    )
    resolve_parser.add_argument(
        "--reason", required=True, help="Non-empty rationale for resolution"
    )
    resolve_parser.add_argument(
        "--db-path", default=None, help="Path to audit SQLite database"
    )
    resolve_parser.add_argument(
        "--lock-path", default=None, help="Path to process execution lock file"
    )
    resolve_parser.add_argument(
        "--model-name", default=None, help="MLflow registered model name"
    )
    resolve_parser.add_argument(
        "--alias", default=None, help="MLflow registered model alias"
    )
    resolve_parser.add_argument(
        "--update-alias",
        action="store_true",
        help="Explicitly authorize alias update to candidate version for success",
    )
    resolve_parser.add_argument(
        "--rollback-alias",
        action="store_true",
        help="Explicitly authorize restoring previous champion alias for failure",
    )
    resolve_parser.add_argument(
        "--delete-alias",
        action="store_true",
        help="Explicitly authorize alias deletion in bootstrap failure",
    )
    resolve_parser.add_argument(
        "--force-bootstrap-removal",
        action="store_true",
        help="Explicit confirmation required to remove alias with no champion",
    )
    resolve_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirmation flag required to execute reconciliation",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for operator reconciliation."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        try:
            evidence = inspect_quarantined_job(
                job_id=args.job_id,
                database_path=args.db_path,
                model_name=args.model_name,
                alias=args.alias,
            )
            print(json.dumps(evidence.to_dict(), indent=2))
            return 0
        except JobNotFoundError as exc:
            sys.stderr.write(f"Error: {exc}\n")
            return 1
        except Exception as exc:
            sys.stderr.write(f"Unexpected inspection error: {exc}\n")
            return 2

    elif args.command == "resolve":
        if not args.confirm:
            sys.stderr.write(
                f"Aborted: Reconciliation of job '{args.job_id}' "
                f"requires '--confirm' flag.\n"
                f"Planned resolution: {args.resolution}\n"
                f"Operator: {args.operator}\n"
                f"Reason: {args.reason}\n"
            )
            return 1

        try:
            finalized = reconcile_job(
                job_id=args.job_id,
                selected_resolution=args.resolution,
                operator=args.operator,
                reason=args.reason,
                database_path=args.db_path,
                lock_path=args.lock_path,
                model_name=args.model_name,
                alias=args.alias,
                update_alias=args.update_alias,
                rollback_alias=args.rollback_alias,
                delete_alias=args.delete_alias,
                force_bootstrap_removal=args.force_bootstrap_removal,
            )
            print(
                f"Successfully resolved job '{args.job_id}' "
                f"to status '{finalized['status']}'."
            )
            return 0
        except MaintenanceLockConflictError as exc:
            sys.stderr.write(f"Maintenance lock error: {exc}\n")
            return 1
        except (ReconciliationError, ValueError) as exc:
            sys.stderr.write(f"Reconciliation refused: {exc}\n")
            return 1
        except Exception as exc:
            sys.stderr.write(f"Fatal reconciliation error: {exc}\n")
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
