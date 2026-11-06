from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from backend.app.config import settings
from backend.app.ml.db import db_connection, immediate_transaction
from backend.app.ml.drift_alerts import get_alert_actions, list_drift_alerts
from backend.app.ml.jobs import (
    claim_queued_job,
    create_queued_job,
    transition_job_status,
)
from backend.app.ml.migrations import migrate_audit_database_idempotent
from backend.app.ml.retraining import (
    assess_retraining_need,
    evaluate_retraining_policy,
)


@pytest.fixture(autouse=True)
def isolate_retraining_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Ensure all retraining policy tests use isolated temporary databases."""
    test_db = tmp_path / "isolated_retraining_default.db"
    monkeypatch.setattr(settings, "lifecycle_audit_db_path", str(test_db))
    monkeypatch.setattr(settings, "promotion_audit_db_path", str(test_db))
    return test_db


@pytest.fixture
def reference_data():
    return pd.DataFrame(
        {
            "age": list(range(20, 40)),
            "tenure": list(range(1, 21)),
            "monthly_charges": [50.0 + i for i in range(20)],
            "contract_type": ["monthly", "annual"] * 10,
            "support_tickets": list(range(20)),
        }
    )


def test_stable_data_does_not_recommend_retraining(reference_data):
    result = assess_retraining_need(reference_data, reference_data.copy())

    assert result["status"] == "stable"
    assert result["retraining_recommended"] is False
    assert result["drift_report"]["drift_detected"] is False


def test_detected_drift_recommends_retraining(reference_data):
    current = reference_data.copy()
    current["age"] += 100

    result = assess_retraining_need(reference_data, current)

    assert result["status"] == "retraining_recommended"
    assert result["retraining_recommended"] is True
    assert result["drift_report"]["drift_detected"] is True


def test_insufficient_data_does_not_recommend_retraining(reference_data):
    small_data = reference_data.iloc[:5]

    result = assess_retraining_need(small_data, small_data.copy())

    assert result["status"] == "insufficient_data"
    assert result["retraining_recommended"] is False


def test_invalid_data_preserves_validation_error(reference_data):
    current = reference_data.drop(columns=["age"])

    with pytest.raises(ValueError, match="missing required features"):
        assess_retraining_need(reference_data, current)


def test_custom_drift_settings_are_passed_to_detector(
    reference_data,
):
    report = {
        "status": "stable",
        "drift_detected": False,
        "reference_samples": 20,
        "current_samples": 20,
        "features": {},
    }

    current = reference_data.copy()

    with patch(
        "backend.app.ml.retraining.detect_data_drift",
        return_value=report,
    ) as detector:
        result = assess_retraining_need(reference_data, current)

    detector.assert_called_once()
    args, kwargs = detector.call_args

    pd.testing.assert_frame_equal(args[0], reference_data)
    pd.testing.assert_frame_equal(args[1], current)
    assert kwargs == {
        "min_samples": 10,
        "p_value_threshold": 0.05,
    }
    assert result["reason"] == "no_feature_drift_detected"
    assert result["drift_report"] is report


def test_drift_recommendation_includes_reason(reference_data):
    current = reference_data.copy()
    current["age"] += 100

    result = assess_retraining_need(reference_data, current)

    assert result["retraining_recommended"] is True
    assert result["reason"] == "feature_drift_detected"


def test_retrain_model_without_promotion_calls_run_training(reference_data, tmp_path):
    """Retraining candidate is decoupled from promotion."""
    from backend.app.ml.retraining import retrain_model_without_promotion

    current = reference_data.copy()
    current["churn"] = [0, 1] * 10
    model_path = tmp_path / "candidate.joblib"

    with patch("backend.app.ml.retraining.run_training") as mock_train:
        mock_train.return_value = {"accuracy": 0.85, "f1_score": 0.80}

        metrics = retrain_model_without_promotion(current, model_path)

    assert metrics["f1_score"] == 0.80
    mock_train.assert_called_once_with(
        current, model_path, config=None, register_model=True
    )


@pytest.fixture
def migrated_db(tmp_path):
    """Provide an isolated, migrated SQLite database in tmp_path."""
    db_path = tmp_path / "lifecycle_test.db"
    migrate_audit_database_idempotent(db_path)
    return db_path


def test_evaluate_policy_insufficient_data_performs_no_db_writes(
    reference_data, migrated_db
):
    """Small sample size yields insufficient_data without any database writes."""
    small = reference_data.iloc[:5]
    result = evaluate_retraining_policy(
        reference=reference_data,
        current=small,
        dataset_id="ds-small",
        trigger_source="drift_assess",
        database_path=migrated_db,
    )

    assert result["status"] == "insufficient_data"
    assert result["retraining_recommended"] is False
    assert result["drift_detected"] is None
    assert result["job"] is None
    assert result["active_job_id"] is None
    assert result["alert_id"] is None
    assert result["cooldown_remaining_seconds"] is None

    # Verify strictly 0 rows written across all tables
    with db_connection(migrated_db, isolation_level=None) as conn:
        assert conn.execute("SELECT count(*) FROM drift_alerts;").fetchone()[0] == 0
        assert (
            conn.execute("SELECT count(*) FROM drift_alert_actions;").fetchone()[0] == 0
        )
        assert conn.execute("SELECT count(*) FROM retraining_jobs;").fetchone()[0] == 0


def test_evaluate_policy_stable_persists_alert_and_action(reference_data, migrated_db):
    """When distributions match, stable status and action are persisted without job."""
    result = evaluate_retraining_policy(
        reference=reference_data,
        current=reference_data.copy(),
        dataset_id="ds-stable",
        trigger_source="drift_assess",
        database_path=migrated_db,
    )

    assert result["status"] == "stable"
    assert result["retraining_recommended"] is False
    assert result["drift_detected"] is False
    assert result["job"] is None
    assert result["active_job_id"] is None
    assert result["alert_id"] is not None
    assert result["cooldown_remaining_seconds"] is None

    # Verify alert and action were persisted
    alerts = list_drift_alerts(migrated_db)
    assert len(alerts) == 1
    assert alerts[0]["alert_id"] == result["alert_id"]
    assert alerts[0]["drift_detected"] is False

    actions = get_alert_actions(result["alert_id"], database_path=migrated_db)
    assert len(actions) == 1
    assert actions[0]["action_type"] == "drift_assessed_stable"
    assert actions[0]["job_id"] is None

    # No retraining jobs created
    with db_connection(migrated_db, isolation_level=None) as conn:
        assert conn.execute("SELECT count(*) FROM retraining_jobs;").fetchone()[0] == 0


def test_evaluate_policy_drift_queues_job_and_links_alert(reference_data, migrated_db):
    """Drift detected when eligible queues a job and links alert and action."""
    current = reference_data.copy()
    current["age"] += 100

    result = evaluate_retraining_policy(
        reference=reference_data,
        current=current,
        dataset_id="ds-drift-1",
        trigger_source="drift_assess",
        alert_id=None,
        database_path=migrated_db,
    )

    assert result["status"] == "queued"
    assert result["retraining_recommended"] is True
    assert result["drift_detected"] is True
    assert result["job"] is not None
    assert result["job"]["dataset_id"] == "ds-drift-1"
    assert result["job"]["status"] == "queued"
    assert result["job"]["alert_id"] == result["alert_id"]
    assert result["cooldown_remaining_seconds"] is None

    # Verify alert was saved
    alerts = list_drift_alerts(migrated_db)
    assert len(alerts) == 1
    assert alerts[0]["drift_detected"] is True

    # Verify action was linked to both alert and job
    actions = get_alert_actions(result["alert_id"], database_path=migrated_db)
    assert len(actions) == 1
    assert actions[0]["action_type"] == "triggered_retraining"
    assert actions[0]["job_id"] == result["job"]["job_id"]


def test_evaluate_policy_blocked_when_active_job_exists(reference_data, migrated_db):
    """Active job returns active_job_exists and active_job_id."""
    # Pre-create an active queued job
    preexisting = create_queued_job(
        migrated_db,
        dataset_id="ds-preexisting",
        trigger_source="manual",
    )

    current = reference_data.copy()
    current["age"] += 100

    result = evaluate_retraining_policy(
        reference=reference_data,
        current=current,
        dataset_id="ds-blocked",
        trigger_source="drift_assess",
        database_path=migrated_db,
    )

    assert result["status"] == "active_job_exists"
    assert result["retraining_recommended"] is False
    assert result["drift_detected"] is True
    assert result["job"] is None
    assert result["active_job_id"] == preexisting["job_id"]
    assert result["alert_id"] is not None
    assert result["cooldown_remaining_seconds"] is None

    # Alert action should record active_job_conflict referencing preexisting job
    actions = get_alert_actions(result["alert_id"], database_path=migrated_db)
    assert len(actions) == 1
    assert actions[0]["action_type"] == "active_job_conflict"
    assert actions[0]["job_id"] == preexisting["job_id"]


def test_evaluate_policy_cooldown_active(reference_data, migrated_db, monkeypatch):
    """Recent successful job in cooldown suppresses queuing and records action."""
    from backend.app.config import settings

    monkeypatch.setattr(settings, "retraining_cooldown_seconds", 3600)

    # Insert a job that succeeded 600 seconds ago (10 minutes ago)
    now = datetime.now(UTC)
    completed_time = (now - timedelta(seconds=600)).isoformat()

    job = create_queued_job(
        migrated_db,
        dataset_id="ds-prior-succ",
        trigger_source="manual",
    )
    claimed = claim_queued_job(migrated_db, worker_id="w1", job_id=job["job_id"])
    assert claimed is not None
    transition_job_status(
        migrated_db,
        job_id=job["job_id"],
        worker_id="w1",
        lease_token=claimed["lease_token"],
        target_status="succeeded",
    )

    with db_connection(migrated_db, isolation_level=None) as conn:
        with immediate_transaction(conn):
            conn.execute(
                "UPDATE retraining_jobs SET completed_at = ? WHERE job_id = ?;",
                (completed_time, job["job_id"]),
            )

    current = reference_data.copy()
    current["age"] += 100

    result = evaluate_retraining_policy(
        reference=reference_data,
        current=current,
        dataset_id="ds-cooldown-check",
        trigger_source="drift_assess",
        database_path=migrated_db,
    )

    assert result["status"] == "cooldown_active"
    assert result["retraining_recommended"] is False
    assert result["drift_detected"] is True
    assert result["job"] is None
    assert result["alert_id"] is not None
    assert result["cooldown_remaining_seconds"] is not None
    assert 2950 <= result["cooldown_remaining_seconds"] <= 3050

    # Verify action recorded as cooldown_active
    actions = get_alert_actions(result["alert_id"], database_path=migrated_db)
    assert len(actions) == 1
    assert actions[0]["action_type"] == "cooldown_active"


def test_evaluate_policy_failed_and_rejected_jobs_do_not_extend_cooldown(
    reference_data, migrated_db, monkeypatch
):
    """Failed and rejected jobs must not extend cooldown; only succeeded jobs do."""
    from backend.app.config import settings

    monkeypatch.setattr(settings, "retraining_cooldown_seconds", 3600)

    now = datetime.now(UTC)

    # 1. Job that succeeded 2 hours ago (7200s ago, > 3600s cooldown)
    succ_job = create_queued_job(
        migrated_db, dataset_id="ds-succ", trigger_source="manual"
    )
    c1 = claim_queued_job(migrated_db, worker_id="w1", job_id=succ_job["job_id"])
    assert c1 is not None
    transition_job_status(
        migrated_db,
        job_id=succ_job["job_id"],
        worker_id="w1",
        lease_token=c1["lease_token"],
        target_status="succeeded",
    )
    succ_completed = (now - timedelta(seconds=7200)).isoformat()

    # 2. Job that failed 15 minutes ago (900s ago)
    fail_job = create_queued_job(
        migrated_db, dataset_id="ds-fail", trigger_source="manual"
    )
    c2 = claim_queued_job(migrated_db, worker_id="w2", job_id=fail_job["job_id"])
    assert c2 is not None
    transition_job_status(
        migrated_db,
        job_id=fail_job["job_id"],
        worker_id="w2",
        lease_token=c2["lease_token"],
        target_status="failed",
        failure_reason="Handler crashed",
    )
    fail_completed = (now - timedelta(seconds=900)).isoformat()

    # 3. Job that was rejected 5 minutes ago (300s ago)
    rej_job = create_queued_job(
        migrated_db, dataset_id="ds-rej", trigger_source="manual"
    )
    c3 = claim_queued_job(migrated_db, worker_id="w3", job_id=rej_job["job_id"])
    assert c3 is not None
    transition_job_status(
        migrated_db,
        job_id=rej_job["job_id"],
        worker_id="w3",
        lease_token=c3["lease_token"],
        target_status="rejected",
        promotion_approved=False,
        promotion_reason="Quality gate rejected",
    )
    rej_completed = (now - timedelta(seconds=300)).isoformat()

    # Set historical timestamps
    with db_connection(migrated_db, isolation_level=None) as conn:
        with immediate_transaction(conn):
            conn.execute(
                "UPDATE retraining_jobs SET completed_at = ? WHERE job_id = ?;",
                (succ_completed, succ_job["job_id"]),
            )
            conn.execute(
                "UPDATE retraining_jobs SET completed_at = ? WHERE job_id = ?;",
                (fail_completed, fail_job["job_id"]),
            )
            conn.execute(
                "UPDATE retraining_jobs SET completed_at = ? WHERE job_id = ?;",
                (rej_completed, rej_job["job_id"]),
            )

    current = reference_data.copy()
    current["age"] += 100

    # Because only succeeded jobs are considered, and the last succeeded job
    # completed 2 hours ago (> 1 hour cooldown), cooldown has elapsed!
    result = evaluate_retraining_policy(
        reference=reference_data,
        current=current,
        dataset_id="ds-after-fail-rej",
        trigger_source="drift_assess",
        database_path=migrated_db,
    )

    assert result["status"] == "queued"
    assert result["retraining_recommended"] is True
    assert result["job"] is not None
    assert result["job"]["dataset_id"] == "ds-after-fail-rej"


def test_evaluate_policy_cooldown_exact_boundaries(
    reference_data, migrated_db, monkeypatch
):
    """Verify exact cooldown boundary behavior: inside vs on/outside boundary."""
    from backend.app.config import settings

    cooldown = 100
    monkeypatch.setattr(settings, "retraining_cooldown_seconds", cooldown)

    current = reference_data.copy()
    current["age"] += 100

    # Boundary 1: elapsed == cooldown - 1s -> cooldown active
    now = datetime.now(UTC)
    job1 = create_queued_job(migrated_db, dataset_id="b1", trigger_source="m")
    c1 = claim_queued_job(migrated_db, worker_id="w1", job_id=job1["job_id"])
    assert c1 is not None
    transition_job_status(
        migrated_db,
        job_id=job1["job_id"],
        worker_id="w1",
        lease_token=c1["lease_token"],
        target_status="succeeded",
    )
    with db_connection(migrated_db, isolation_level=None) as conn:
        with immediate_transaction(conn):
            conn.execute(
                "UPDATE retraining_jobs SET completed_at = ? WHERE job_id = ?;",
                (
                    (now - timedelta(seconds=cooldown - 1)).isoformat(),
                    job1["job_id"],
                ),
            )

    res_active = evaluate_retraining_policy(
        reference=reference_data,
        current=current,
        dataset_id="ds-boundary-1",
        trigger_source="drift_assess",
        database_path=migrated_db,
    )
    assert res_active["status"] == "cooldown_active"
    assert res_active["cooldown_remaining_seconds"] == 1

    # Boundary 2: elapsed == cooldown -> cooldown elapsed, new job queued!
    with db_connection(migrated_db, isolation_level=None) as conn:
        with immediate_transaction(conn):
            conn.execute(
                "UPDATE retraining_jobs SET completed_at = ? WHERE job_id = ?;",
                ((now - timedelta(seconds=cooldown)).isoformat(), job1["job_id"]),
            )

    res_exact = evaluate_retraining_policy(
        reference=reference_data,
        current=current,
        dataset_id="ds-boundary-exact",
        trigger_source="drift_assess",
        database_path=migrated_db,
    )
    assert res_exact["status"] == "queued"
    assert res_exact["job"] is not None


def test_evaluate_policy_race_condition_conflict(reference_data, migrated_db):
    """If create_queued_job encounters a race, handle gracefully."""
    from backend.app.ml.jobs import ActiveJobExistsError

    current = reference_data.copy()
    current["age"] += 100

    with patch(
        "backend.app.ml.retraining.create_queued_job",
        side_effect=ActiveJobExistsError("Conflict!"),
    ):
        result = evaluate_retraining_policy(
            reference=reference_data,
            current=current,
            dataset_id="ds-race",
            trigger_source="drift_assess",
            database_path=migrated_db,
        )

    assert result["status"] == "active_job_exists"
    assert result["retraining_recommended"] is False
    assert result["job"] is None


def test_evaluate_policy_validation_errors(reference_data, migrated_db):
    """dataset_id and trigger_source must be non-empty strings."""
    with pytest.raises(ValueError, match="dataset_id must be a non-empty string"):
        evaluate_retraining_policy(
            reference=reference_data,
            current=reference_data,
            dataset_id="",
            trigger_source="manual",
            database_path=migrated_db,
        )

    with pytest.raises(ValueError, match="trigger_source must be a non-empty string"):
        evaluate_retraining_policy(
            reference=reference_data,
            current=reference_data,
            dataset_id="ds1",
            trigger_source="   ",
            database_path=migrated_db,
        )
