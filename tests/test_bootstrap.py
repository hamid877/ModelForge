import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode

from backend.app.ml.bootstrap import (
    BootstrapResult,
    bootstrap_model,
    get_champion_version,
)
from backend.app.ml.promotion import PromotionDecision
from scripts.bootstrap import main as cli_main


def test_get_champion_version_returns_version_when_exists():
    champion = type("ModelVersion", (), {"version": "3"})()
    with patch("backend.app.ml.bootstrap.MlflowClient") as mock_client:
        client = mock_client.return_value
        client.get_model_version_by_alias.return_value = champion

        version = get_champion_version("churn-classifier", "champion")

    assert version == "3"
    client.get_model_version_by_alias.assert_called_once_with(
        "churn-classifier", "champion"
    )


def test_get_champion_version_returns_none_when_alias_or_model_missing():
    with patch("backend.app.ml.bootstrap.MlflowClient") as mock_client:
        client = mock_client.return_value
        client.get_model_version_by_alias.side_effect = MlflowException(
            "Resource does not exist",
            error_code=ErrorCode.RESOURCE_DOES_NOT_EXIST,
        )

        version = get_champion_version("churn-classifier", "champion")

    assert version is None


def test_bootstrap_fresh_setup_missing_model(tmp_path: Path):
    """When MLflow has no model or alias, bootstrap trains, registers, and promotes."""
    decision = PromotionDecision(
        approved=True,
        reason="candidate_meets_quality_gate",
        candidate_f1=0.88,
        champion_f1=None,
    )
    metrics = {"accuracy": 0.90, "f1_score": 0.88}

    with (
        patch("backend.app.ml.bootstrap.setup_mlflow"),
        patch(
            "backend.app.ml.bootstrap.get_champion_version",
            side_effect=[None, "1"],
        ),
        patch("backend.app.ml.bootstrap.load_and_validate_churn_data") as mock_load,
        patch(
            "backend.app.ml.bootstrap.run_training_and_promote",
            return_value=(metrics, decision),
        ) as mock_train,
    ):
        mock_load.return_value = MagicMock()
        result = bootstrap_model(
            data_path=tmp_path / "churn.csv",
            model_path=tmp_path / "model.joblib",
        )

    assert isinstance(result, BootstrapResult)
    assert result.status == "bootstrapped"
    assert result.version == "1"
    assert result.metrics == metrics
    assert result.decision_reason == "candidate_meets_quality_gate"
    mock_load.assert_called_once_with(tmp_path / "churn.csv")
    mock_train.assert_called_once()


def test_bootstrap_missing_alias_on_existing_model(tmp_path: Path):
    """When model versions exist but alias is missing, bootstrap assigns champion."""
    decision = PromotionDecision(
        approved=True,
        reason="candidate_meets_quality_gate",
        candidate_f1=0.85,
        champion_f1=None,
    )
    metrics = {"accuracy": 0.87, "f1_score": 0.85}

    with (
        patch("backend.app.ml.bootstrap.setup_mlflow"),
        patch(
            "backend.app.ml.bootstrap.get_champion_version",
            side_effect=[None, "2"],
        ),
        patch("backend.app.ml.bootstrap.load_and_validate_churn_data") as mock_load,
        patch(
            "backend.app.ml.bootstrap.run_training_and_promote",
            return_value=(metrics, decision),
        ),
    ):
        mock_load.return_value = MagicMock()
        result = bootstrap_model(
            data_path=tmp_path / "churn.csv",
            model_path=tmp_path / "model.joblib",
        )

    assert result.status == "bootstrapped"
    assert result.version == "2"
    assert result.metrics == metrics


def test_bootstrap_idempotent_when_champion_exists(tmp_path: Path):
    """When champion alias already exists, bootstrap does not retrain."""
    with (
        patch("backend.app.ml.bootstrap.setup_mlflow"),
        patch(
            "backend.app.ml.bootstrap.get_champion_version",
            return_value="3",
        ),
        patch("backend.app.ml.bootstrap.load_and_validate_churn_data") as mock_load,
        patch("backend.app.ml.bootstrap.run_training_and_promote") as mock_train,
    ):
        result = bootstrap_model(
            data_path=tmp_path / "churn.csv",
            model_path=tmp_path / "model.joblib",
            force=False,
        )

    assert result.status == "already_bootstrapped"
    assert result.version == "3"
    assert result.decision_reason == "already_assigned"
    mock_load.assert_not_called()
    mock_train.assert_not_called()


def test_bootstrap_forced_rerun_promotes_winning_candidate(tmp_path: Path):
    """With force=True, training runs even if champion exists and promotes."""
    decision = PromotionDecision(
        approved=True,
        reason="candidate_meets_quality_gate",
        candidate_f1=0.92,
        champion_f1=0.85,
    )
    metrics = {"accuracy": 0.93, "f1_score": 0.92}

    with (
        patch("backend.app.ml.bootstrap.setup_mlflow"),
        patch(
            "backend.app.ml.bootstrap.get_champion_version",
            side_effect=["3", "4"],
        ),
        patch("backend.app.ml.bootstrap.load_and_validate_churn_data") as mock_load,
        patch(
            "backend.app.ml.bootstrap.run_training_and_promote",
            return_value=(metrics, decision),
        ) as mock_train,
    ):
        mock_load.return_value = MagicMock()
        result = bootstrap_model(
            data_path=tmp_path / "churn.csv",
            model_path=tmp_path / "model.joblib",
            force=True,
        )

    assert result.status == "bootstrapped"
    assert result.version == "4"
    assert result.metrics == metrics
    mock_load.assert_called_once()
    mock_train.assert_called_once()


def test_bootstrap_rejected_candidate_does_not_promote_or_overwrite_champion(
    tmp_path: Path,
):
    """Rejected candidate under force=True fails and preserves champion."""
    decision = PromotionDecision(
        approved=False,
        reason="candidate_f1_below_champion",
        candidate_f1=0.75,
        champion_f1=0.85,
    )
    metrics = {"accuracy": 0.80, "f1_score": 0.75}

    with (
        patch("backend.app.ml.bootstrap.setup_mlflow"),
        patch(
            "backend.app.ml.bootstrap.get_champion_version",
            return_value="3",
        ),
        patch("backend.app.ml.bootstrap.load_and_validate_churn_data") as mock_load,
        patch(
            "backend.app.ml.bootstrap.run_training_and_promote",
            return_value=(metrics, decision),
        ),
    ):
        mock_load.return_value = MagicMock()
        result = bootstrap_model(
            data_path=tmp_path / "churn.csv",
            model_path=tmp_path / "model.joblib",
            force=True,
        )

    assert result.status == "failed"
    assert result.version == "3"
    assert result.decision_reason == "candidate_f1_below_champion"
    assert "failed quality gate" in result.message


def test_bootstrap_initial_candidate_below_minimum_f1_fails(tmp_path: Path):
    """Initial candidate failing minimum threshold is not promoted."""
    decision = PromotionDecision(
        approved=False,
        reason="candidate_below_minimum_f1",
        candidate_f1=0.55,
        champion_f1=None,
    )
    metrics = {"accuracy": 0.60, "f1_score": 0.55}

    with (
        patch("backend.app.ml.bootstrap.setup_mlflow"),
        patch(
            "backend.app.ml.bootstrap.get_champion_version",
            return_value=None,
        ),
        patch("backend.app.ml.bootstrap.load_and_validate_churn_data") as mock_load,
        patch(
            "backend.app.ml.bootstrap.run_training_and_promote",
            return_value=(metrics, decision),
        ),
    ):
        mock_load.return_value = MagicMock()
        result = bootstrap_model(
            data_path=tmp_path / "churn.csv",
            model_path=tmp_path / "model.joblib",
        )

    assert result.status == "failed"
    assert result.version is None
    assert result.decision_reason == "candidate_below_minimum_f1"


def test_cli_bootstrap_success(capsys):
    result = BootstrapResult(
        status="bootstrapped",
        model_name="churn-classifier",
        alias="champion",
        version="1",
        metrics={"accuracy": 0.95, "f1_score": 0.94},
        decision_reason="candidate_meets_quality_gate",
        message="Model bootstrapped",
    )

    with patch(
        "scripts.bootstrap.bootstrap_model",
        return_value=result,
    ) as mock_bootstrap:
        exit_code = cli_main(["--force"])

    assert exit_code == 0
    mock_bootstrap.assert_called_once()
    assert mock_bootstrap.call_args.kwargs["force"] is True
    out = capsys.readouterr().out
    assert "Bootstrap status: bootstrapped" in out
    assert "F1 score: 0.9400" in out


def test_cli_bootstrap_already_bootstrapped(capsys):
    result = BootstrapResult(
        status="already_bootstrapped",
        model_name="churn-classifier",
        alias="champion",
        version="2",
        metrics=None,
        decision_reason="already_assigned",
        message="Already bootstrapped",
    )

    with patch(
        "scripts.bootstrap.bootstrap_model",
        return_value=result,
    ):
        exit_code = cli_main([])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Bootstrap status: already_bootstrapped" in out


def test_cli_bootstrap_failure(capsys):
    result = BootstrapResult(
        status="failed",
        model_name="churn-classifier",
        alias="champion",
        version=None,
        metrics={"accuracy": 0.50, "f1_score": 0.40},
        decision_reason="candidate_below_minimum_f1",
        message="Promotion failed",
    )

    with patch(
        "scripts.bootstrap.bootstrap_model",
        return_value=result,
    ):
        exit_code = cli_main([])

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "Bootstrap status: failed" in out


def test_cli_bootstrap_unexpected_exception(capsys):
    with patch(
        "scripts.bootstrap.bootstrap_model",
        side_effect=RuntimeError("MLflow connection timed out"),
    ):
        exit_code = cli_main([])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Bootstrap error: MLflow connection timed out" in err


def test_cli_bootstrap_script_direct_execution():
    """Verify scripts/bootstrap.py can be invoked directly without PYTHONPATH."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "scripts/bootstrap.py", "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "Bootstrap ModelForge MLflow tracking" in result.stdout
