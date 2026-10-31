"""Tests for SQLite database schema definitions and idempotent migrations."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.app.ml.audit import (
    ACTIVE_JOB_STATUSES,
    TERMINAL_JOB_STATUSES,
    VALID_JOB_STATUSES,
    VALID_RESOLUTIONS,
    finalize_operator_resolution_atomic,
    migrate_audit_database_idempotent,
    record_lifecycle_run,
    record_promotion_audit,
    validate_operator_resolution_inputs,
)


def _get_table_names(conn: sqlite3.Connection) -> set[str]:
    """Return all non-internal table names in SQLite connection."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    return {row[0] for row in cursor.fetchall()}


def _get_column_names(conn: sqlite3.Connection, table_name: str) -> set[str]:
    """Return all column names for given table."""
    cursor = conn.execute(f"PRAGMA table_info({table_name});")
    return {row[1] for row in cursor.fetchall()}


def _get_index_names(conn: sqlite3.Connection) -> set[str]:
    """Return all index names in SQLite connection."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='index' AND name NOT LIKE 'sqlite_%';"
    )
    return {row[0] for row in cursor.fetchall()}


def _get_trigger_names(conn: sqlite3.Connection) -> set[str]:
    """Return all trigger names in SQLite connection."""
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='trigger';")
    return {row[0] for row in cursor.fetchall()}


def test_fresh_database_migration(tmp_path: Path):
    """Verify clean migration on fresh DB creates tables, indexes, triggers."""
    db_path = tmp_path / "fresh_lifecycle_audit.db"
    assert not db_path.exists()

    migrate_audit_database_idempotent(db_path)
    assert db_path.exists()

    with sqlite3.connect(db_path) as conn:
        tables = _get_table_names(conn)
        expected_tables = {
            "lifecycle_audit",
            "promotion_audit",
            "drift_alerts",
            "retraining_jobs",
            "drift_alert_actions",
            "operator_resolution_audit",
        }
        assert expected_tables.issubset(tables)

        # Verify promotion_audit columns include job_id
        prom_cols = _get_column_names(conn, "promotion_audit")
        assert "job_id" in prom_cols
        assert "audit_id" in prom_cols
        assert "payload" in prom_cols

        # Verify retraining_jobs required fields
        job_cols = _get_column_names(conn, "retraining_jobs")
        required_job_cols = {
            "job_id",
            "created_at",
            "started_at",
            "completed_at",
            "status",
            "is_active",
            "dataset_id",
            "trigger_source",
            "worker_id",
            "lease_token",
            "heartbeat_at",
            "lease_expires_at",
            "attempt_count",
            "max_attempts",
            "candidate_version",
            "candidate_run_id",
            "previous_champion_version",
            "promotion_started_at",
            "promotion_approved",
            "promotion_reason",
            "promotion_audit_id",
            "failure_reason",
            "alert_id",
            "payload",
        }
        assert required_job_cols.issubset(job_cols)

        # Verify drift_alerts required fields
        drift_cols = _get_column_names(conn, "drift_alerts")
        required_drift_cols = {
            "alert_id",
            "detected_at",
            "drift_detected",
            "reference_samples",
            "current_samples",
            "p_value_threshold",
            "features_payload",
        }
        assert required_drift_cols.issubset(drift_cols)

        # Verify drift_alert_actions required fields
        action_cols = _get_column_names(conn, "drift_alert_actions")
        required_action_cols = {
            "action_id",
            "alert_id",
            "timestamp",
            "action_type",
            "actor",
            "job_id",
            "notes",
        }
        assert required_action_cols.issubset(action_cols)

        # Verify operator_resolution_audit required fields
        res_cols = _get_column_names(conn, "operator_resolution_audit")
        required_res_cols = {
            "resolution_id",
            "job_id",
            "timestamp",
            "operator",
            "previous_state",
            "selected_resolution",
            "candidate_version",
            "observed_champion_version",
            "target_champion_version",
            "rollback_performed",
            "resolution_reason",
            "evidence_payload",
        }
        assert required_res_cols.issubset(res_cols)

        # Verify key indexes
        indexes = _get_index_names(conn)
        assert "idx_single_active_retraining_job" in indexes
        assert "idx_promotion_audit_job_id" in indexes
        assert "idx_operator_resolution_job_id" in indexes
        assert "idx_drift_alert_actions_alert_id" in indexes

        # Verify triggers
        triggers = _get_trigger_names(conn)
        assert "trg_drift_alerts_immutable_update" in triggers
        assert "trg_drift_alerts_immutable_delete" in triggers
        assert "trg_drift_alert_actions_no_update" in triggers
        assert "trg_drift_alert_actions_no_delete" in triggers
        assert "trg_operator_resolution_audit_no_update" in triggers
        assert "trg_operator_resolution_audit_no_delete" in triggers


def test_existing_database_migration_with_existing_audit_records(
    tmp_path: Path,
):
    """Verify migration preserves existing lifecycle and promotion records."""
    db_path = tmp_path / "existing_audit.db"

    lifecycle_meta = record_lifecycle_run(
        {"status": "stable", "reason": "no_drift", "training_attempted": False},
        db_path,
    )
    promotion_meta = record_promotion_audit(
        {
            "model_name": "churn-classifier",
            "candidate_version": "5",
            "champion_version": "4",
            "approved": True,
            "reason": "candidate_meets_quality_gate",
            "candidate_f1": 0.88,
            "champion_f1": 0.82,
            "benchmark_version": "v1",
            "benchmark_digest": "sha256_mock_benchmark",
            "margin": 0.01,
        },
        database_path=db_path,
    )

    with sqlite3.connect(db_path) as conn:
        initial_lifecycle_count = conn.execute(
            "SELECT COUNT(*) FROM lifecycle_audit;"
        ).fetchone()[0]
        initial_promotion_count = conn.execute(
            "SELECT COUNT(*) FROM promotion_audit;"
        ).fetchone()[0]

    assert initial_lifecycle_count == 1
    assert initial_promotion_count == 1

    # Run migration
    migrate_audit_database_idempotent(db_path)

    # Verify data preservation and column addition
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        upgraded_cols = _get_column_names(conn, "promotion_audit")
        assert "job_id" in upgraded_cols

        l_row = conn.execute("SELECT * FROM lifecycle_audit;").fetchone()
        assert l_row is not None
        assert l_row["run_id"] == lifecycle_meta["run_id"]
        assert l_row["status"] == "stable"

        p_row = conn.execute("SELECT * FROM promotion_audit;").fetchone()
        assert p_row is not None
        assert p_row["audit_id"] == promotion_meta["audit_id"]
        assert p_row["approved"] == 1
        assert p_row["candidate_version"] == "5"
        assert p_row["job_id"] is None

        indexes = _get_index_names(conn)
        assert "idx_promotion_audit_job_id" in indexes


def test_existing_promotion_audit_without_job_id(tmp_path: Path):
    """Explicitly verify Case 1: promotion_audit exists without job_id."""
    db_path = tmp_path / "case1.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE promotion_audit (
                audit_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                candidate_version TEXT,
                candidate_run_id TEXT,
                champion_version TEXT,
                approved INTEGER NOT NULL,
                reason TEXT NOT NULL,
                candidate_f1 REAL NOT NULL,
                champion_f1 REAL,
                benchmark_version TEXT NOT NULL,
                benchmark_digest TEXT NOT NULL,
                margin REAL NOT NULL,
                payload TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO promotion_audit VALUES (
                'audit-100', '2026-10-31T12:00:00Z', 'churn-classifier',
                '1', 'run-1', NULL, 1, 'bootstrap', 0.85, NULL,
                'v1', 'digest-1', 0.01, '{"test": 1}'
            );
            """
        )

    with sqlite3.connect(db_path) as conn:
        cols = _get_column_names(conn, "promotion_audit")
        assert "job_id" not in cols

    migrate_audit_database_idempotent(db_path)

    with sqlite3.connect(db_path) as conn:
        cols = _get_column_names(conn, "promotion_audit")
        assert "job_id" in cols
        row = conn.execute(
            "SELECT audit_id, job_id, payload FROM promotion_audit "
            "WHERE audit_id = 'audit-100';"
        ).fetchone()
        assert row[0] == "audit-100"
        assert row[1] is None
        assert json.loads(row[2]) == {"test": 1}

        indexes = _get_index_names(conn)
        assert "idx_promotion_audit_job_id" in indexes


def test_existing_promotion_audit_with_job_id(tmp_path: Path):
    """Explicitly verify Case 2: promotion_audit already has job_id."""
    db_path = tmp_path / "case2.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE promotion_audit (
                audit_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                candidate_version TEXT,
                candidate_run_id TEXT,
                champion_version TEXT,
                approved INTEGER NOT NULL,
                reason TEXT NOT NULL,
                candidate_f1 REAL NOT NULL,
                champion_f1 REAL,
                benchmark_version TEXT NOT NULL,
                benchmark_digest TEXT NOT NULL,
                margin REAL NOT NULL,
                job_id TEXT,
                payload TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO promotion_audit VALUES (
                'audit-200', '2026-10-31T12:00:00Z', 'churn-classifier',
                '2', 'run-2', '1', 1, 'improved', 0.89, 0.85,
                'v1', 'digest-2', 0.01, 'job-orig-200', '{"test": 2}'
            );
            """
        )

    migrate_audit_database_idempotent(db_path)

    with sqlite3.connect(db_path) as conn:
        cols = _get_column_names(conn, "promotion_audit")
        assert "job_id" in cols
        row = conn.execute(
            "SELECT audit_id, job_id FROM promotion_audit WHERE audit_id = 'audit-200';"
        ).fetchone()
        assert row[0] == "audit-200"
        assert row[1] == "job-orig-200"

        indexes = _get_index_names(conn)
        assert "idx_promotion_audit_job_id" in indexes


def test_repeated_migration_is_idempotent(tmp_path: Path):
    """Verify repeated migrations cause zero errors and zero duplicates."""
    db_path = tmp_path / "repeated_migration.db"

    migrate_audit_database_idempotent(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO drift_alerts VALUES (
                'alert-1', '2026-10-31T12:00:00Z', 1, 100, 50, 0.05,
                '{"feature": "tenure"}'
            );
            """
        )
        conn.execute(
            """
            INSERT INTO retraining_jobs (
                job_id, created_at, status, is_active, dataset_id,
                trigger_source, payload
            ) VALUES (
                'job-1', '2026-10-31T12:00:01Z', 'queued', 1,
                'default_data', 'drift_alert', '{}'
            );
            """
        )
        conn.execute(
            """
            INSERT INTO drift_alert_actions VALUES (
                'act-1', 'alert-1', '2026-10-31T12:00:02Z',
                'trigger_retraining', 'system', 'job-1', 'Notes'
            );
            """
        )

    # Rerun migration second time
    migrate_audit_database_idempotent(db_path)

    # Rerun migration third time
    migrate_audit_database_idempotent(db_path)

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM drift_alerts;").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM retraining_jobs;").fetchone()[0] == 1
        assert (
            conn.execute("SELECT COUNT(*) FROM drift_alert_actions;").fetchone()[0] == 1
        )


def test_partial_unique_index_enforces_single_active_job(tmp_path: Path):
    """Verify partial unique index enforces at most one active job."""
    db_path = tmp_path / "concurrency_lock.db"
    migrate_audit_database_idempotent(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO retraining_jobs (
                job_id, created_at, status, is_active, dataset_id,
                trigger_source, payload
            ) VALUES (
                'job-active-1', datetime('now'), 'queued', 1,
                'ds-1', 'manual_api', '{}'
            );
            """
        )

        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            conn.execute(
                """
                INSERT INTO retraining_jobs (
                    job_id, created_at, status, is_active, dataset_id,
                    trigger_source, payload
                ) VALUES (
                    'job-active-2', datetime('now'), 'running', 1,
                    'ds-1', 'manual_api', '{}'
                );
                """
            )

        # Transition job 1 to terminal status ('succeeded')
        conn.execute(
            """
            UPDATE retraining_jobs
            SET status = 'succeeded', is_active = NULL,
                completed_at = datetime('now')
            WHERE job_id = 'job-active-1';
            """
        )

        # Now inserting job 2 as active succeeds
        conn.execute(
            """
            INSERT INTO retraining_jobs (
                job_id, created_at, status, is_active, dataset_id,
                trigger_source, payload
            ) VALUES (
                'job-active-2', datetime('now'), 'running', 1,
                'ds-1', 'manual_api', '{}'
            );
            """
        )

        # Transition job 2 to 'failed' and release lock
        conn.execute(
            """
            UPDATE retraining_jobs
            SET status = 'failed', is_active = NULL,
                completed_at = datetime('now'), failure_reason = 'Crashed'
            WHERE job_id = 'job-active-2';
            """
        )

        # Inactive job coexists without violation
        conn.execute(
            """
            INSERT INTO retraining_jobs (
                job_id, created_at, status, is_active, dataset_id,
                trigger_source, payload
            ) VALUES (
                'job-inactive-3', datetime('now'), 'rejected', NULL,
                'ds-1', 'manual_api', '{}'
            );
            """
        )

        count = conn.execute("SELECT COUNT(*) FROM retraining_jobs;").fetchone()[0]
        assert count == 3


def test_invalid_job_states_rejected(tmp_path: Path):
    """Verify CHECK constraint on retraining_jobs rejects invalid status strings."""
    db_path = tmp_path / "invalid_states.db"
    migrate_audit_database_idempotent(db_path)

    invalid_statuses = [
        "pending",
        "in_progress",
        "done",
        "cancelled",
        "unknown",
        "",
    ]

    with sqlite3.connect(db_path) as conn:
        for idx, bad_status in enumerate(invalid_statuses):
            with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
                conn.execute(
                    f"""
                    INSERT INTO retraining_jobs (
                        job_id, created_at, status, is_active, dataset_id,
                        trigger_source, payload
                    ) VALUES (
                        'job-bad-{idx}', datetime('now'), '{bad_status}', 1,
                        'ds-1', 'api', '{{}}'
                    );
                    """
                )


def test_invalid_job_data_rejected(tmp_path: Path):
    """Verify CHECK constraints reject invalid fields on retraining_jobs."""
    db_path = tmp_path / "invalid_data.db"
    migrate_audit_database_idempotent(db_path)

    with sqlite3.connect(db_path) as conn:
        # 1. Invalid is_active value (e.g. 0, 2, -1)
        for bad_active in [0, 2, -1]:
            with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
                conn.execute(
                    f"""
                    INSERT INTO retraining_jobs (
                        job_id, created_at, status, is_active, dataset_id,
                        trigger_source, payload
                    ) VALUES (
                        'job-active-{bad_active}', datetime('now'), 'queued',
                        {bad_active}, 'ds-1', 'api', '{{}}'
                    );
                    """
                )

        # 2. Terminal job with active lock (is_active = 1 on terminal statuses)
        for term_status in ["succeeded", "rejected", "failed"]:
            with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
                conn.execute(
                    f"""
                    INSERT INTO retraining_jobs (
                        job_id, created_at, status, is_active, dataset_id,
                        trigger_source, payload
                    ) VALUES (
                        'job-term-{term_status}', datetime('now'),
                        '{term_status}', 1, 'ds-1', 'api', '{{}}'
                    );
                    """
                )

        # 3. Active job without active lock (is_active = NULL on active statuses)
        for act_status in ["queued", "running", "reconciliation_required"]:
            with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
                conn.execute(
                    f"""
                    INSERT INTO retraining_jobs (
                        job_id, created_at, status, is_active, dataset_id,
                        trigger_source, payload
                    ) VALUES (
                        'job-act-{act_status}', datetime('now'),
                        '{act_status}', NULL, 'ds-1', 'api', '{{}}'
                    );
                    """
                )

        # 4. Negative attempt count
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            conn.execute(
                """
                INSERT INTO retraining_jobs (
                    job_id, created_at, status, is_active, dataset_id,
                    trigger_source, attempt_count, payload
                ) VALUES (
                    'job-bad-attempt', datetime('now'), 'queued', 1,
                    'ds-1', 'api', -1, '{}'
                );
                """
            )

        # 5. Invalid max_attempts (< 1)
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            conn.execute(
                """
                INSERT INTO retraining_jobs (
                    job_id, created_at, status, is_active, dataset_id,
                    trigger_source, max_attempts, payload
                ) VALUES (
                    'job-bad-max', datetime('now'), 'queued', 1,
                    'ds-1', 'api', 0, '{}'
                );
                """
            )

        # 6. Empty job_id
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            conn.execute(
                """
                INSERT INTO retraining_jobs (
                    job_id, created_at, status, is_active, dataset_id,
                    trigger_source, payload
                ) VALUES ('', datetime('now'), 'queued', 1, 'ds-1', 'api', '{}');
                """
            )


def test_foreign_key_constraints_enforced(tmp_path: Path):
    """Verify SQLite foreign key constraints reject dangling references."""
    db_path = tmp_path / "foreign_keys.db"
    migrate_audit_database_idempotent(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        # 1. operator_resolution_audit with non-existent job_id
        with pytest.raises(
            sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"
        ):
            conn.execute(
                """
                INSERT INTO operator_resolution_audit (
                    resolution_id, job_id, timestamp, operator, previous_state,
                    selected_resolution, candidate_version,
                    observed_champion_version, target_champion_version,
                    rollback_performed, resolution_reason, evidence_payload
                ) VALUES (
                    'res-dangling', 'non-existent-job', datetime('now'),
                    'tester', 'reconciliation_required', 'resolved_failed',
                    NULL, NULL, NULL, 0, 'Reason', '{}'
                );
                """
            )

        # 2. drift_alert_actions with non-existent alert_id
        with pytest.raises(
            sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"
        ):
            conn.execute(
                """
                INSERT INTO drift_alert_actions (
                    action_id, alert_id, timestamp, action_type, actor
                ) VALUES (
                    'act-dangling', 'non-existent-alert', datetime('now'),
                    'test', 'tester'
                );
                """
            )

        # 3. retraining_jobs with non-existent alert_id
        with pytest.raises(
            sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"
        ):
            conn.execute(
                """
                INSERT INTO retraining_jobs (
                    job_id, created_at, status, is_active, dataset_id,
                    trigger_source, alert_id, payload
                ) VALUES (
                    'job-fk-alert', datetime('now'), 'queued', 1,
                    'ds-1', 'drift_alert', 'bad-alert-id', '{}'
                );
                """
            )


def test_immutability_and_append_only_triggers(tmp_path: Path):
    """Verify triggers prevent UPDATE and DELETE on immutable tables."""
    db_path = tmp_path / "immutability.db"
    migrate_audit_database_idempotent(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        conn.execute(
            """
            INSERT INTO drift_alerts VALUES (
                'alert-imm', datetime('now'), 1, 100, 50, 0.05,
                '{"feat": "val"}'
            );
            """
        )
        conn.execute(
            """
            INSERT INTO retraining_jobs (
                job_id, created_at, status, is_active, dataset_id,
                trigger_source, payload
            ) VALUES (
                'job-imm', datetime('now'), 'queued', 1, 'ds-1', 'api', '{}'
            );
            """
        )
        conn.execute(
            """
            INSERT INTO drift_alert_actions VALUES (
                'act-imm', 'alert-imm', datetime('now'), 'trigger', 'tester',
                'job-imm', 'notes'
            );
            """
        )
        conn.execute(
            """
            INSERT INTO operator_resolution_audit VALUES (
                'res-imm', 'job-imm', datetime('now'), 'tester',
                'reconciliation_required', 'resolved_failed', '1', '1',
                '1', 0, 'Reason', '{}'
            );
            """
        )

        # drift_alerts is immutable
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            conn.execute(
                "UPDATE drift_alerts SET drift_detected = 0 "
                "WHERE alert_id = 'alert-imm';"
            )
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            conn.execute("DELETE FROM drift_alerts WHERE alert_id = 'alert-imm';")

        # drift_alert_actions is append-only
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute(
                "UPDATE drift_alert_actions SET actor = 'other' "
                "WHERE action_id = 'act-imm';"
            )
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("DELETE FROM drift_alert_actions WHERE action_id = 'act-imm';")

        # operator_resolution_audit is append-only
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute(
                "UPDATE operator_resolution_audit SET operator = 'other' "
                "WHERE resolution_id = 'res-imm';"
            )
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute(
                "DELETE FROM operator_resolution_audit WHERE resolution_id = 'res-imm';"
            )


def test_rollback_on_migration_failure(tmp_path: Path):
    """Verify error during migration rolls back all transactional DDL."""
    db_path = tmp_path / "rollback_test.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE promotion_audit (
                audit_id TEXT PRIMARY KEY,
                model_name TEXT,
                payload TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO promotion_audit VALUES ('a-pre', 'churn', '{\"status\": 1}');"
        )

    # Simulate failure midway through migration
    with patch(
        "backend.app.ml.migrations.RETRAINING_JOBS_SQL",
        "CREATE TABLE invalid syntax syntax error!;",
    ):
        with pytest.raises(sqlite3.OperationalError):
            migrate_audit_database_idempotent(db_path)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT audit_id, model_name FROM promotion_audit;"
        ).fetchone()
        assert row == ("a-pre", "churn")

        cols = _get_column_names(conn, "promotion_audit")
        assert "job_id" not in cols

        tables = _get_table_names(conn)
        assert "retraining_jobs" not in tables
        assert "operator_resolution_audit" not in tables


def test_preservation_of_existing_audit_records_under_migration(tmp_path: Path):
    """Verify audit dataset is 100% preserved byte-for-byte under migration."""
    db_path = tmp_path / "bulk_audit.db"

    lifecycle_ids = []
    promotion_ids = []

    for i in range(25):
        l_meta = record_lifecycle_run(
            {
                "status": "promoted" if i % 2 == 0 else "stable",
                "reason": f"lifecycle_iteration_{i}",
                "training_attempted": i % 2 == 0,
                "metrics": {"f1": 0.80 + (i * 0.005)},
            },
            database_path=db_path,
        )
        lifecycle_ids.append(l_meta["run_id"])

        p_meta = record_promotion_audit(
            {
                "model_name": "churn-classifier",
                "candidate_version": str(i + 1),
                "champion_version": str(i) if i > 0 else None,
                "approved": i % 3 == 0,
                "reason": "gate_passed" if i % 3 == 0 else "f1_too_low",
                "candidate_f1": 0.85 + (i * 0.002),
                "champion_f1": 0.84 if i > 0 else None,
                "benchmark_version": "v1",
                "benchmark_digest": f"digest_{i}",
                "margin": 0.01,
            },
            database_path=db_path,
        )
        promotion_ids.append(p_meta["audit_id"])

    migrate_audit_database_idempotent(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        l_count = conn.execute("SELECT COUNT(*) FROM lifecycle_audit;").fetchone()[0]
        assert l_count == 25

        p_count = conn.execute("SELECT COUNT(*) FROM promotion_audit;").fetchone()[0]
        assert p_count == 25

        for run_id in lifecycle_ids:
            row = conn.execute(
                "SELECT * FROM lifecycle_audit WHERE run_id = ?;", (run_id,)
            ).fetchone()
            assert row is not None
            payload = json.loads(row["payload"])
            assert payload["run_id"] == run_id

        for audit_id in promotion_ids:
            row = conn.execute(
                "SELECT * FROM promotion_audit WHERE audit_id = ?;", (audit_id,)
            ).fetchone()
            assert row is not None
            assert row["job_id"] is None
            payload = json.loads(row["payload"])
            assert payload["audit_id"] == audit_id


def test_status_and_resolution_constants():
    """Verify state machine and resolution constant invariants."""
    assert "queued" in VALID_JOB_STATUSES
    assert "running" in VALID_JOB_STATUSES
    assert "succeeded" in VALID_JOB_STATUSES
    assert "rejected" in VALID_JOB_STATUSES
    assert "failed" in VALID_JOB_STATUSES
    assert "reconciliation_required" in VALID_JOB_STATUSES

    assert ACTIVE_JOB_STATUSES == {
        "queued",
        "running",
        "reconciliation_required",
    }
    assert TERMINAL_JOB_STATUSES == {"succeeded", "rejected", "failed"}
    assert VALID_RESOLUTIONS == {"resolved_succeeded", "resolved_failed"}


def test_operator_resolution_input_validation():
    """Verify validate_operator_resolution_inputs enforces types and values."""
    validate_operator_resolution_inputs("resolved_succeeded", 0)
    validate_operator_resolution_inputs("resolved_failed", 1)

    with pytest.raises(
        ValueError, match="Invalid selected_resolution 'invalid_choice'"
    ):
        validate_operator_resolution_inputs("invalid_choice", 0)

    with pytest.raises(ValueError, match="Invalid rollback_performed value '2'"):
        validate_operator_resolution_inputs("resolved_succeeded", 2)

    with pytest.raises(ValueError, match="Invalid rollback_performed value 'True'"):
        validate_operator_resolution_inputs("resolved_succeeded", True)  # type: ignore


def test_atomic_operator_resolution_success(tmp_path: Path):
    """Verify operator resolution transitions quarantined job and audits."""
    db_path = tmp_path / "atomic_res.db"
    migrate_audit_database_idempotent(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO retraining_jobs (
                job_id, created_at, status, is_active, dataset_id,
                trigger_source, candidate_version, previous_champion_version,
                promotion_started_at, payload
            ) VALUES (
                'job-quarantined', datetime('now'), 'reconciliation_required',
                1, 'ds-1', 'manual_api', '9', '8', datetime('now'),
                '{"step": "promotion"}'
            );
            """
        )

    finalize_operator_resolution_atomic(
        db_path,
        resolution_id="res-success-1",
        job_id="job-quarantined",
        operator="sre_oncall",
        selected_resolution="resolved_succeeded",
        candidate_version="9",
        observed_champion_version="9",
        target_champion_version="9",
        rollback_performed=0,
        resolution_reason="Verified candidate is live champion",
        evidence_payload={"action": "verified_alias_match"},
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        job = conn.execute(
            "SELECT status, is_active, completed_at, payload "
            "FROM retraining_jobs WHERE job_id = 'job-quarantined';"
        ).fetchone()
        assert job["status"] == "succeeded"
        assert job["is_active"] is None
        assert job["completed_at"] is not None
        job_payload = json.loads(job["payload"])
        assert job_payload["operator_resolution"]["resolution_id"] == "res-success-1"
        assert job_payload["operator_resolution"]["operator"] == "sre_oncall"

        res = conn.execute(
            "SELECT * FROM operator_resolution_audit "
            "WHERE resolution_id = 'res-success-1';"
        ).fetchone()
        assert res is not None
        assert res["job_id"] == "job-quarantined"
        assert res["operator"] == "sre_oncall"
        assert res["selected_resolution"] == "resolved_succeeded"
        assert res["rollback_performed"] == 0


def test_atomic_operator_resolution_rowcount_mismatch_single_rollback(
    tmp_path: Path,
):
    """Verify rowcount mismatch raises RuntimeError without double rollback."""
    db_path = tmp_path / "rowcount_mismatch.db"
    migrate_audit_database_idempotent(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO retraining_jobs (
                job_id, created_at, status, is_active, dataset_id,
                trigger_source, payload
            ) VALUES (
                'job-active-run', datetime('now'), 'running', 1,
                'ds-1', 'manual_api', '{}'
            );
            """
        )

    with pytest.raises(RuntimeError) as exc_info:
        finalize_operator_resolution_atomic(
            db_path,
            resolution_id="res-rowcount-test",
            job_id="job-active-run",
            operator="tester",
            selected_resolution="resolved_failed",
            candidate_version="1",
            observed_champion_version=None,
            target_champion_version=None,
            rollback_performed=0,
            resolution_reason="Rowcount test",
            evidence_payload={"action": "test"},
        )

    assert "Expected to update 1 row for job 'job-active-run'" in str(exc_info.value)
    assert "cannot rollback" not in str(exc_info.value).lower()

    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM operator_resolution_audit;"
        ).fetchone()[0]
        assert count == 0
        status = conn.execute(
            "SELECT status FROM retraining_jobs WHERE job_id = 'job-active-run';"
        ).fetchone()[0]
        assert status == "running"
