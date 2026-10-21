from dataclasses import dataclass

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode

from backend.app.config import settings


@dataclass(frozen=True)
class PromotionDecision:
    """Result of evaluating whether a candidate model should be promoted."""

    approved: bool
    reason: str
    candidate_f1: float
    champion_f1: float | None


def evaluate_promotion(
    candidate_f1: float,
    champion_f1: float | None,
    *,
    minimum_f1: float = 0.0,
) -> PromotionDecision:
    """Approve promotion only when the candidate meets the quality rules."""
    for name, value in (
        ("candidate_f1", candidate_f1),
        ("minimum_f1", minimum_f1),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")

    if champion_f1 is not None and not 0.0 <= champion_f1 <= 1.0:
        raise ValueError("champion_f1 must be between 0 and 1.")

    if candidate_f1 < minimum_f1:
        return PromotionDecision(
            approved=False,
            reason="candidate_below_minimum_f1",
            candidate_f1=candidate_f1,
            champion_f1=champion_f1,
        )

    if champion_f1 is not None and candidate_f1 < champion_f1:
        return PromotionDecision(
            approved=False,
            reason="candidate_f1_below_champion",
            candidate_f1=candidate_f1,
            champion_f1=champion_f1,
        )

    return PromotionDecision(
        approved=True,
        reason="candidate_meets_quality_gate",
        candidate_f1=candidate_f1,
        champion_f1=champion_f1,
    )


def promote_model(
    candidate_version: int,
    candidate_f1: float,
    *,
    model_name: str | None = None,
    alias: str | None = None,
    minimum_f1: float | None = None,
) -> PromotionDecision:
    """Promote a registered model version only if it passes the quality gate."""
    model_name = model_name or settings.model_name
    alias = alias or settings.model_alias
    minimum_f1 = settings.promotion_minimum_f1 if minimum_f1 is None else minimum_f1

    client = MlflowClient()

    try:
        champion = client.get_model_version_by_alias(model_name, alias)
    except MlflowException as exc:
        missing_alias_code = ErrorCode.Name(ErrorCode.RESOURCE_DOES_NOT_EXIST)
        if exc.error_code != missing_alias_code:
            raise
        champion_f1 = None
    else:
        run = client.get_run(champion.run_id)
        champion_f1 = run.data.metrics.get("f1_score")
        if champion_f1 is None:
            raise ValueError(
                f"Champion version {champion.version} has no recorded f1_score."
            )

    decision = evaluate_promotion(
        candidate_f1,
        champion_f1,
        minimum_f1=minimum_f1,
    )

    if decision.approved:
        client.set_registered_model_alias(
            model_name,
            alias,
            str(candidate_version),
        )

    return decision
