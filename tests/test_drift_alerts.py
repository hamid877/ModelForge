"""Tests for drift alert persistence, retrieval, filtering, and append-only actions."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from backend.app.ml.db import (
    db_connection,
    get_db_connection,
    immediate_transaction,
)
from backend.app.ml.drift_alerts import (
    AlertNotFoundError,
    DriftAlertError,
    DuplicateActionError,
    DuplicateAlertError,
    get_alert_actions,
    get_drift_alert,
    list_alert_actions,
    list_drift_alerts,
    persist_drift_alert,
    persist_drift_alert_with_action,
    record_alert_action,
)
from backend.app.ml.migrations import migrate_audit_database_idempotent


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Provide a fresh, migrated, isolated SQLite database path."""
    db_file = tmp_path / "drift_test.db"
    migrate_audit_database_idempotent(db_file)
    return db_file


def test_persist_and_retrieve_alert(db_path: Path):
    """Verify persisting and retrieving a drift alert with all fields intact."""
    alert = persist_drift_alert(
        db_path,
        alert_id="alert-test-001",
        detected_at="2026-05-01T12:00:00+00:00",
        drift_detected=True,
        reference_samples=100,
        current_samples=50,
        p_value_threshold=0.05,
        features={
            "age": {
                "test": "kolmogorov_smirnov",
                "statistic": 0.35,
                "p_value": 0.001,
                "drift_detected": True,
            }
        },
    )

    assert alert["alert_id"] == "alert-test-001"
    assert alert["drift_detected"] is True
    assert alert["reference_samples"] == 100
    assert alert["current_samples"] == 50
    assert alert["p_value_threshold"] == 0.05
    assert alert["features"]["age"]["statistic"] == 0.35

    # Retrieve by ID
    retrieved = get_drift_alert("alert-test-001", db_path)
    assert retrieved is not None
    assert retrieved["alert_id"] == "alert-test-001"
    assert retrieved["detected_at"] == "2026-05-01T12:00:00+00:00"
    assert retrieved["drift_detected"] is True
    assert retrieved["reference_samples"] == 100
    assert retrieved["current_samples"] == 50
    assert retrieved["p_value_threshold"] == 0.05
    assert retrieved["features"]["age"]["p_value"] == 0.001

    # Retrieve via keyword alias
    retrieved_kw = get_drift_alert(id="alert-test-001", database_path=db_path)
    assert retrieved_kw is not None
    assert retrieved_kw["alert_id"] == "alert-test-001"

    # Non-existent alert returns None
    assert get_drift_alert("non-existent-alert-id", db_path) is None


def test_persist_alert_auto_fields_and_dict_input(db_path: Path):
    """Verify default generation of alert_id and UTC timestamps and dict input."""
    # Persist via alert_data mapping without explicit alert_id or detected_at
    alert_payload = {
        "drift_detected": False,
        "reference_samples": 200,
        "current_samples": 180,
        "p_value_threshold": 0.01,
        "features": {
            "tenure": {
                "test": "ks",
                "statistic": 0.05,
                "p_value": 0.45,
                "drift_detected": False,
            }
        },
    }

    result = persist_drift_alert(db_path, alert_data=alert_payload)

    assert result["alert_id"].startswith("alert-")
    assert result["drift_detected"] is False
    assert result["reference_samples"] == 200
    assert result["current_samples"] == 180
    assert result["p_value_threshold"] == 0.01

    # Verify detected_at is valid UTC ISO string
    parsed_dt = datetime.fromisoformat(result["detected_at"])
    assert parsed_dt.tzinfo is not None

    # Verify retrieval
    retrieved = get_drift_alert(result["alert_id"], db_path)
    assert retrieved is not None
    assert retrieved["drift_detected"] is False


def test_persist_alert_with_datetime_object(db_path: Path):
    """Verify passing Python datetime objects with or without tzinfo."""
    dt_with_tz = datetime(2026, 6, 15, 14, 30, tzinfo=UTC)
    alert = persist_drift_alert(
        db_path,
        detected_at=dt_with_tz,
        drift_detected=1,
        reference_samples=30,
        current_samples=30,
        p_value_threshold=0.05,
        features={"feat": {"test": "dummy"}},
    )
    assert "2026-06-15T14:30:00" in alert["detected_at"]


def test_duplicate_alert_id_rejected(db_path: Path):
    """Verify attempting duplicate alert ID raises DuplicateAlertError."""
    persist_drift_alert(
        db_path,
        alert_id="alert-dup-001",
        drift_detected=True,
        reference_samples=50,
        current_samples=50,
        p_value_threshold=0.05,
        features={"f1": {"drift_detected": True}},
    )

    with pytest.raises(DuplicateAlertError, match="already exists"):
        persist_drift_alert(
            db_path,
            alert_id="alert-dup-001",
            drift_detected=False,
            reference_samples=99,
            current_samples=99,
            p_value_threshold=0.01,
            features={"f1": {"drift_detected": False}},
        )

    # Verify original alert is completely unmodified
    original = get_drift_alert("alert-dup-001", db_path)
    assert original is not None
    assert original["drift_detected"] is True
    assert original["reference_samples"] == 50


def test_invalid_alert_data_rejected(db_path: Path):
    """Verify invalid inputs raise ValueError before touching SQLite."""
    valid_features = {"f": {"drift_detected": True}}

    # 1. Empty or whitespace alert_id
    for bad_id in ["", "   ", "\t\n"]:
        with pytest.raises(ValueError, match="alert_id"):
            persist_drift_alert(
                db_path,
                alert_id=bad_id,
                drift_detected=True,
                reference_samples=10,
                current_samples=10,
                features=valid_features,
            )

    # 2. Invalid drift_detected (None, string, out of bounds int)
    for bad_drift in [None, "true", "yes", 2, -1, 10]:
        with pytest.raises(ValueError, match="drift_detected"):
            persist_drift_alert(
                db_path,
                drift_detected=bad_drift,  # type: ignore
                reference_samples=10,
                current_samples=10,
                features=valid_features,
            )

    # 3. Invalid reference_samples (negative, bool, non-int)
    for bad_ref in [-1, -100, True, False, "10", 10.5]:
        with pytest.raises(ValueError, match="reference_samples"):
            persist_drift_alert(
                db_path,
                drift_detected=True,
                reference_samples=bad_ref,  # type: ignore
                current_samples=10,
                features=valid_features,
            )

    # 4. Invalid current_samples (negative, bool, non-int)
    for bad_cur in [-1, -50, True, False, "10", 10.5]:
        with pytest.raises(ValueError, match="current_samples"):
            persist_drift_alert(
                db_path,
                drift_detected=True,
                reference_samples=10,
                current_samples=bad_cur,  # type: ignore
                features=valid_features,
            )

    # 5. Invalid p_value_threshold (<= 0.0, > 1.0, non-numeric, bool)
    for bad_p in [0.0, -0.05, 1.0001, 2.0, "0.05", True, False]:
        with pytest.raises(ValueError, match="p_value_threshold"):
            persist_drift_alert(
                db_path,
                drift_detected=True,
                reference_samples=10,
                current_samples=10,
                p_value_threshold=bad_p,  # type: ignore
                features=valid_features,
            )

    # 6. Missing or invalid features payload
    with pytest.raises(ValueError, match="features or features_payload"):
        persist_drift_alert(
            db_path,
            drift_detected=True,
            reference_samples=10,
            current_samples=10,
            features=None,
            features_payload=None,
        )

    with pytest.raises(ValueError, match="features_payload"):
        persist_drift_alert(
            db_path,
            drift_detected=True,
            reference_samples=10,
            current_samples=10,
            features_payload="   ",
        )

    with pytest.raises(ValueError, match="valid JSON"):
        persist_drift_alert(
            db_path,
            drift_detected=True,
            reference_samples=10,
            current_samples=10,
            features_payload="invalid-json-string",
        )

    # 7. Invalid detected_at timestamp
    for bad_ts in ["not-a-timestamp", "   ", "2026-99-99"]:
        with pytest.raises(ValueError, match="[Tt]imestamp"):
            persist_drift_alert(
                db_path,
                detected_at=bad_ts,
                drift_detected=True,
                reference_samples=10,
                current_samples=10,
                features=valid_features,
            )


def test_sensitive_and_unnecessary_payload_stripped(db_path: Path):
    """Verify sensitive fields and large sample arrays are omitted from features."""
    dirty_features = {
        "numerical_stat": {
            "test": "ks",
            "statistic": 0.4,
            "p_value": 0.005,
            "drift_detected": True,
            "password": "super-secret-password",
            "api_key": "secret-api-key",
            "raw_data": list(range(100)),  # oversized raw array
        },
        "safe_stat": {
            "test": "chi2",
            "statistic": 10.5,
            "p_value": 0.02,
            "drift_detected": True,
        },
    }

    persist_drift_alert(
        db_path,
        alert_id="alert-sensitive-clean",
        drift_detected=True,
        reference_samples=100,
        current_samples=100,
        features=dirty_features,
    )

    retrieved = get_drift_alert("alert-sensitive-clean", db_path)
    assert retrieved is not None
    features = retrieved["features"]

    # Sensitive fields removed
    assert "password" not in features["numerical_stat"]
    assert "api_key" not in features["numerical_stat"]
    assert "raw_data" not in features["numerical_stat"]

    # Safe statistics preserved
    assert features["numerical_stat"]["statistic"] == 0.4
    assert features["safe_stat"]["statistic"] == 10.5


def test_listing_and_filtering_alerts(db_path: Path):
    """Verify listing alerts with filtering and pagination."""
    # Persist 4 distinct alerts across time
    persist_drift_alert(
        db_path,
        alert_id="alert-jan",
        detected_at="2026-01-15T00:00:00+00:00",
        drift_detected=False,
        reference_samples=10,
        current_samples=10,
        features={"f": {"drift_detected": False}},
    )
    persist_drift_alert(
        db_path,
        alert_id="alert-feb",
        detected_at="2026-02-15T00:00:00+00:00",
        drift_detected=True,
        reference_samples=10,
        current_samples=10,
        features={"f": {"drift_detected": True}},
    )
    persist_drift_alert(
        db_path,
        alert_id="alert-mar",
        detected_at="2026-03-15T00:00:00+00:00",
        drift_detected=True,
        reference_samples=10,
        current_samples=10,
        features={"f": {"drift_detected": True}},
    )
    persist_drift_alert(
        db_path,
        alert_id="alert-apr",
        detected_at="2026-04-15T00:00:00+00:00",
        drift_detected=False,
        reference_samples=10,
        current_samples=10,
        features={"f": {"drift_detected": False}},
    )

    # 1. Default list: all 4 records, descending by detected_at
    all_alerts = list_drift_alerts(db_path)
    assert len(all_alerts) == 4
    assert [a["alert_id"] for a in all_alerts] == [
        "alert-apr",
        "alert-mar",
        "alert-feb",
        "alert-jan",
    ]

    # 2. Ascending order
    asc_alerts = list_drift_alerts(db_path, descending=False)
    assert [a["alert_id"] for a in asc_alerts] == [
        "alert-jan",
        "alert-feb",
        "alert-mar",
        "alert-apr",
    ]

    # 3. Filter drift_detected = True
    drifted = list_drift_alerts(db_path, drift_detected=True)
    assert len(drifted) == 2
    assert [a["alert_id"] for a in drifted] == ["alert-mar", "alert-feb"]

    # 4. Filter drift_detected = False
    stable = list_drift_alerts(db_path, drift_detected=False)
    assert len(stable) == 2
    assert [a["alert_id"] for a in stable] == ["alert-apr", "alert-jan"]

    # 5. Filter by time window (Feb through Mar)
    windowed = list_drift_alerts(
        db_path,
        start_time="2026-02-01T00:00:00+00:00",
        end_time="2026-03-31T23:59:59+00:00",
    )
    assert len(windowed) == 2
    assert [a["alert_id"] for a in windowed] == ["alert-mar", "alert-feb"]

    # 6. Pagination (limit and offset)
    paged = list_drift_alerts(db_path, limit=2, offset=1)
    assert len(paged) == 2
    assert [a["alert_id"] for a in paged] == ["alert-mar", "alert-feb"]

    # 7. Pagination validation errors
    for bad_limit in [0, -1, True, "5"]:
        with pytest.raises(ValueError, match="limit"):
            list_drift_alerts(db_path, limit=bad_limit)  # type: ignore

    for bad_offset in [-1, -5, True, "0"]:
        with pytest.raises(ValueError, match="offset"):
            list_drift_alerts(db_path, offset=bad_offset)  # type: ignore


def test_successful_action_persistence_and_retrieval(db_path: Path):
    """Verify recording and retrieving actions for an existing drift alert."""
    persist_drift_alert(
        db_path,
        alert_id="alert-act-001",
        drift_detected=True,
        reference_samples=100,
        current_samples=50,
        features={"feat": {"drift_detected": True}},
    )

    # 1. Record first action: acknowledged
    act1 = record_alert_action(
        db_path,
        alert_id="alert-act-001",
        action_type="acknowledged",
        actor="operator_alice",
        timestamp="2026-05-01T10:00:00+00:00",
        notes="Reviewed drift metric and confirmed shift.",
    )
    assert act1["action_id"].startswith("act-")
    assert act1["alert_id"] == "alert-act-001"
    assert act1["action_type"] == "acknowledged"
    assert act1["actor"] == "operator_alice"
    assert act1["notes"] == "Reviewed drift metric and confirmed shift."

    # 2. Record second action: triggered_retraining
    act2 = record_alert_action(
        db_path,
        alert_id="alert-act-001",
        action_type="triggered_retraining",
        actor="automated_policy",
        timestamp="2026-05-01T10:05:00+00:00",
        notes="Retraining dispatched.",
    )
    assert act2["action_type"] == "triggered_retraining"

    # 3. Retrieve actions (default chronological ASC)
    actions = get_alert_actions("alert-act-001", db_path)
    assert len(actions) == 2
    assert actions[0]["action_type"] == "acknowledged"
    assert actions[1]["action_type"] == "triggered_retraining"

    # 4. Retrieve actions descending
    actions_desc = get_alert_actions("alert-act-001", db_path, descending=True)
    assert actions_desc[0]["action_type"] == "triggered_retraining"
    assert actions_desc[1]["action_type"] == "acknowledged"

    # 5. Check alias list_alert_actions
    alias_actions = list_alert_actions("alert-act-001", db_path)
    assert len(alias_actions) == 2


def test_missing_alert_references_rejected(db_path: Path):
    """Verify actions referencing non-existent alerts are rejected."""
    # Attempt to record an action for a non-existent alert
    with pytest.raises(AlertNotFoundError, match="does not exist"):
        record_alert_action(
            db_path,
            alert_id="alert-does-not-exist",
            action_type="investigate",
            actor="tester",
        )

    # Direct foreign key check in SQLite
    with get_db_connection(db_path) as conn:
        with pytest.raises(
            sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"
        ):
            conn.execute(
                """
                INSERT INTO drift_alert_actions (
                    action_id, alert_id, timestamp, action_type, actor
                ) VALUES ('act-dangling', 'alert-missing', datetime('now'), 't', 'a');
                """
            )

    # get_alert_actions with require_alert_exists=True raises AlertNotFoundError
    with pytest.raises(AlertNotFoundError, match="does not exist"):
        get_alert_actions("alert-missing", db_path, require_alert_exists=True)

    # get_alert_actions default returns empty list for unknown alert
    assert get_alert_actions("alert-missing", db_path) == []


def test_invalid_action_inputs_rejected(db_path: Path):
    """Verify invalid action inputs raise ValueError before database interaction."""
    # 1. Invalid alert_id
    for bad_id in ["", "   ", None, 123]:
        with pytest.raises(ValueError, match="alert_id"):
            record_alert_action(
                db_path,
                alert_id=bad_id,  # type: ignore
                action_type="ack",
                actor="alice",
            )

    # 2. Invalid action_type
    for bad_type in ["", "   ", None, 123]:
        with pytest.raises(ValueError, match="action_type"):
            record_alert_action(
                db_path,
                alert_id="alert-1",
                action_type=bad_type,  # type: ignore
                actor="alice",
            )

    # 3. Invalid actor
    for bad_actor in ["", "   ", None, 123]:
        with pytest.raises(ValueError, match="actor"):
            record_alert_action(
                db_path,
                alert_id="alert-1",
                action_type="ack",
                actor=bad_actor,  # type: ignore
            )

    # 4. Invalid action_id
    for bad_aid in ["", "   ", 123]:
        with pytest.raises(ValueError, match="action_id"):
            record_alert_action(
                db_path,
                alert_id="alert-1",
                action_type="ack",
                actor="alice",
                action_id=bad_aid,  # type: ignore
            )

    # 5. Invalid job_id
    with pytest.raises(ValueError, match="job_id"):
        record_alert_action(
            db_path,
            alert_id="alert-1",
            action_type="ack",
            actor="alice",
            job_id=123,  # type: ignore
        )

    # 6. Invalid notes
    with pytest.raises(ValueError, match="notes"):
        record_alert_action(
            db_path,
            alert_id="alert-1",
            action_type="ack",
            actor="alice",
            notes=123,  # type: ignore
        )


def test_duplicate_action_id_rejected(db_path: Path):
    """Verify duplicate action_id raises DuplicateActionError."""
    persist_drift_alert(
        db_path,
        alert_id="alert-act-dup",
        drift_detected=True,
        reference_samples=10,
        current_samples=10,
        features={"f": {"drift_detected": True}},
    )

    record_alert_action(
        db_path,
        alert_id="alert-act-dup",
        action_id="act-fixed-001",
        action_type="initial",
        actor="system",
    )

    with pytest.raises(DuplicateActionError, match="already exists"):
        record_alert_action(
            db_path,
            alert_id="alert-act-dup",
            action_id="act-fixed-001",
            action_type="duplicate",
            actor="system",
        )


def test_alert_immutability_enforced(db_path: Path):
    """Verify triggers prevent UPDATE and DELETE on drift_alerts."""
    persist_drift_alert(
        db_path,
        alert_id="alert-immutable",
        drift_detected=True,
        reference_samples=100,
        current_samples=50,
        features={"f": {"drift_detected": True}},
    )

    with get_db_connection(db_path) as conn:
        # UPDATE should abort via trigger
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            conn.execute(
                "UPDATE drift_alerts SET drift_detected = 0 "
                "WHERE alert_id = 'alert-immutable';"
            )

        # DELETE should abort via trigger
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            conn.execute("DELETE FROM drift_alerts WHERE alert_id = 'alert-immutable';")

    # Verify original state persists untouched
    alert = get_drift_alert("alert-immutable", db_path)
    assert alert is not None
    assert alert["drift_detected"] is True


def test_action_append_only_enforced(db_path: Path):
    """Verify triggers prevent UPDATE and DELETE on drift_alert_actions."""
    persist_drift_alert(
        db_path,
        alert_id="alert-append-only",
        drift_detected=True,
        reference_samples=100,
        current_samples=50,
        features={"f": {"drift_detected": True}},
    )

    action = record_alert_action(
        db_path,
        alert_id="alert-append-only",
        action_type="ack",
        actor="operator",
    )
    action_id = action["action_id"]

    with get_db_connection(db_path) as conn:
        # UPDATE should abort via trigger
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute(
                "UPDATE drift_alert_actions SET actor = 'rogue' "
                f"WHERE action_id = '{action_id}';"
            )

        # DELETE should abort via trigger
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute(
                f"DELETE FROM drift_alert_actions WHERE action_id = '{action_id}';"
            )

    # Verify action remains intact
    actions = get_alert_actions("alert-append-only", db_path)
    assert len(actions) == 1
    assert actions[0]["actor"] == "operator"


def test_transaction_rollback_on_failure(db_path: Path):
    """Verify failure during composed alert/action rolls back all changes atomically."""
    # Attempt atomic persist of alert + action referencing a non-existent retraining job
    with pytest.raises(
        (sqlite3.IntegrityError, DriftAlertError), match="retraining job"
    ):
        persist_drift_alert_with_action(
            db_path,
            alert_id="alert-rollback-test",
            drift_detected=True,
            reference_samples=10,
            current_samples=10,
            features={"f": {"drift_detected": True}},
            action_type="triggered_retraining",
            actor="orchestrator",
            job_id="job-non-existent-999",
        )

    # Verify alert was rolled back and never committed
    assert get_drift_alert("alert-rollback-test", db_path) is None

    # Verify drift_alerts and drift_alert_actions tables are completely empty
    with get_db_connection(db_path) as conn:
        alert_count = conn.execute("SELECT COUNT(*) FROM drift_alerts;").fetchone()[0]
        action_count = conn.execute(
            "SELECT COUNT(*) FROM drift_alert_actions;"
        ).fetchone()[0]
        assert alert_count == 0
        assert action_count == 0

    # Also test manual immediate_transaction rollback
    with db_connection(db_path, isolation_level=None) as conn:
        with pytest.raises(RuntimeError, match="simulated failure"):
            with immediate_transaction(conn):
                persist_drift_alert(
                    conn=conn,
                    alert_id="alert-manual-rollback",
                    drift_detected=True,
                    reference_samples=10,
                    current_samples=10,
                    features={"f": {"drift_detected": True}},
                )
                raise RuntimeError("simulated failure")

    assert get_drift_alert("alert-manual-rollback", db_path) is None


def test_database_isolation(tmp_path: Path):
    """Verify independent databases in tmp_path remain strictly isolated."""
    db1 = tmp_path / "db1.db"
    db2 = tmp_path / "db2.db"

    migrate_audit_database_idempotent(db1)
    migrate_audit_database_idempotent(db2)

    persist_drift_alert(
        db1,
        alert_id="alert-in-db1",
        drift_detected=True,
        reference_samples=20,
        current_samples=20,
        features={"f": {"drift_detected": True}},
    )

    # db1 has the alert, db2 does not
    assert get_drift_alert("alert-in-db1", db1) is not None
    assert get_drift_alert("alert-in-db1", db2) is None
    assert len(list_drift_alerts(db1)) == 1
    assert len(list_drift_alerts(db2)) == 0
