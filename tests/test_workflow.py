from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.app.data.pipeline import load_and_validate_churn_data
from backend.app.ml.promotion import PromotionDecision
from backend.app.ml.workflow import run_training, run_training_and_promote

DATA_PATH = Path("data/raw/churn.csv")


def test_run_training(tmp_path: Path):
    df = load_and_validate_churn_data(DATA_PATH)
    model_path = tmp_path / "churn_model.joblib"

    with (
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=nullcontext(),
        ),
        patch("backend.app.ml.workflow.mlflow.set_tag"),
        patch("backend.app.ml.workflow.mlflow.log_param"),
        patch("backend.app.ml.workflow.mlflow.log_metric"),
        patch("backend.app.ml.workflow.mlflow.log_artifact"),
    ):
        metrics = run_training(df, model_path)

    assert model_path.exists()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0


def test_run_training_and_promote_promotes_registered_candidate(
    tmp_path: Path,
):
    df = load_and_validate_churn_data(DATA_PATH)
    model_path = tmp_path / "candidate.joblib"

    decision = PromotionDecision(
        approved=True,
        reason="candidate_meets_quality_gate",
        candidate_f1=0.85,
        champion_f1=0.80,
    )
    model_info = type("ModelInfoStub", (), {"registered_model_version": 7})()

    with (
        patch(
            "backend.app.ml.workflow.mlflow.sklearn.log_model",
            return_value=model_info,
        ) as log_model,
        patch(
            "backend.app.ml.workflow.promote_model",
            return_value=decision,
        ) as promote,
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=nullcontext(),
        ),
        patch("backend.app.ml.workflow.mlflow.set_tag"),
        patch("backend.app.ml.workflow.mlflow.log_param"),
        patch("backend.app.ml.workflow.mlflow.log_metric"),
        patch("backend.app.ml.workflow.mlflow.log_artifact"),
    ):
        metrics, result = run_training_and_promote(df, model_path)

    assert model_path.exists()
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert result is decision
    log_model.assert_called_once()
    promote.assert_called_once()
    assert promote.call_args.kwargs["candidate_version"] == 7
    assert promote.call_args.kwargs["candidate_f1"] == metrics["f1_score"]


def test_run_training_and_promote_fails_if_version_is_missing(
    tmp_path: Path,
):
    df = load_and_validate_churn_data(DATA_PATH)
    model_path = tmp_path / "candidate.joblib"
    model_info = type("ModelInfoStub", (), {"registered_model_version": None})()

    with (
        patch(
            "backend.app.ml.workflow.mlflow.sklearn.log_model",
            return_value=model_info,
        ),
        patch("backend.app.ml.workflow.promote_model") as promote,
        pytest.raises(RuntimeError, match="did not return a registered model version"),
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=nullcontext(),
        ),
        patch("backend.app.ml.workflow.mlflow.set_tag"),
        patch("backend.app.ml.workflow.mlflow.log_param"),
        patch("backend.app.ml.workflow.mlflow.log_metric"),
        patch("backend.app.ml.workflow.mlflow.log_artifact"),
    ):
        run_training_and_promote(df, model_path)

    promote.assert_not_called()


def test_run_training_and_promote_returns_rejection_without_changing_alias(
    tmp_path: Path,
):
    df = load_and_validate_churn_data(DATA_PATH)
    model_path = tmp_path / "rejected_candidate.joblib"
    decision = PromotionDecision(
        approved=False,
        reason="candidate_f1_below_champion",
        candidate_f1=0.75,
        champion_f1=0.80,
    )
    model_info = type("ModelInfoStub", (), {"registered_model_version": 8})()

    with (
        patch(
            "backend.app.ml.workflow.mlflow.sklearn.log_model",
            return_value=model_info,
        ),
        patch(
            "backend.app.ml.workflow.promote_model",
            return_value=decision,
        ) as promote,
        patch("backend.app.ml.workflow.setup_mlflow"),
        patch(
            "backend.app.ml.workflow.mlflow.start_run",
            return_value=nullcontext(),
        ),
        patch("backend.app.ml.workflow.mlflow.set_tag"),
        patch("backend.app.ml.workflow.mlflow.log_param"),
        patch("backend.app.ml.workflow.mlflow.log_metric"),
        patch("backend.app.ml.workflow.mlflow.log_artifact"),
    ):
        metrics, result = run_training_and_promote(df, model_path)

    assert model_path.exists()
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert result.approved is False
    assert result.reason == "candidate_f1_below_champion"
    promote.assert_called_once()
