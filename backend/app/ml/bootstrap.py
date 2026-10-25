import logging
from dataclasses import dataclass
from pathlib import Path

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode

from backend.app.config import settings
from backend.app.data.pipeline import load_and_validate_churn_data
from backend.app.ml.tracking import setup_mlflow
from backend.app.ml.workflow import run_training_and_promote

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path("data/raw/churn.csv")
DEFAULT_MODEL_PATH = Path("models/churn_model.joblib")


@dataclass(frozen=True)
class BootstrapResult:
    """Outcome of a bootstrap execution."""

    status: str  # "bootstrapped", "already_bootstrapped", "failed"
    model_name: str
    alias: str
    version: str | None
    metrics: dict[str, float] | None
    decision_reason: str | None
    message: str


def get_champion_version(
    model_name: str | None = None,
    alias: str | None = None,
) -> str | None:
    """Return the model version assigned to the alias, or None if absent."""
    model_name = model_name or settings.model_name
    alias = alias or settings.model_alias

    client = MlflowClient()
    try:
        champion = client.get_model_version_by_alias(model_name, alias)
        return str(champion.version)
    except MlflowException as exc:
        missing_code = ErrorCode.Name(ErrorCode.RESOURCE_DOES_NOT_EXIST)
        if exc.error_code == missing_code:
            return None
        raise


def bootstrap_model(
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    model_name: str | None = None,
    alias: str | None = None,
    force: bool = False,
) -> BootstrapResult:
    """Bootstrap MLflow tracking, initial model, and champion alias idempotently."""
    model_name = model_name or settings.model_name
    alias = alias or settings.model_alias
    data_path = Path(data_path)
    model_path = Path(model_path)

    setup_mlflow()

    existing_version = get_champion_version(model_name, alias)

    if existing_version is not None and not force:
        logger.info(
            "Champion alias already assigned model_name=%s alias=%s version=%s",
            model_name,
            alias,
            existing_version,
        )
        return BootstrapResult(
            status="already_bootstrapped",
            model_name=model_name,
            alias=alias,
            version=existing_version,
            metrics=None,
            decision_reason="already_assigned",
            message=(
                f"Model '{model_name}' already has '{alias}' assigned to "
                f"version {existing_version}. Bootstrap skipped."
            ),
        )

    logger.info(
        "Bootstrapping model model_name=%s alias=%s force=%s",
        model_name,
        alias,
        force,
    )

    df = load_and_validate_churn_data(data_path)
    metrics, decision = run_training_and_promote(df, model_path)

    if not decision.approved:
        logger.warning(
            "Bootstrap candidate rejected by quality gate reason=%s",
            decision.reason,
        )
        return BootstrapResult(
            status="failed",
            model_name=model_name,
            alias=alias,
            version=existing_version,
            metrics=metrics,
            decision_reason=decision.reason,
            message=(
                f"Bootstrap candidate model failed quality gate: {decision.reason}."
            ),
        )

    current_version = get_champion_version(model_name, alias)

    logger.info(
        "Bootstrap succeeded model_name=%s alias=%s version=%s",
        model_name,
        alias,
        current_version,
    )

    return BootstrapResult(
        status="bootstrapped",
        model_name=model_name,
        alias=alias,
        version=current_version,
        metrics=metrics,
        decision_reason=decision.reason,
        message=(
            f"Successfully bootstrapped model '{model_name}' version {current_version} "
            f"with alias '{alias}'."
        ),
    )
