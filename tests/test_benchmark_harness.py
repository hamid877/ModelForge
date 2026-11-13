"""Tests for M11 benchmark harness and metrics collection.

Validates that the benchmark harness correctly initializes isolated databases,
executes workload scenarios without corrupting database integrity, properly
distinguishes SQLITE_BUSY lock errors from logical ownership-fencing rejections,
and correctly records metrics.
"""

from benchmarks.m11.harness import (
    _check_correctness,
    _classify_error,
    _create_temp_db,
    _initialize_db,
    _seed_baseline_data,
    _verify_journal_mode,
    capture_environment,
    run_w6_threads,
)


def test_harness_temp_db_isolation():
    """Verify temp database is created, migrated, seeded, and verified intact."""
    db_path = _create_temp_db()
    try:
        assert db_path.exists()
        _initialize_db(db_path)
        assert _verify_journal_mode(db_path) == "delete"

        seed = _seed_baseline_data(db_path)
        assert len(seed["alert_ids"]) == 5
        assert len(seed["lifecycle_run_ids"]) == 10
        assert len(seed["audit_ids"]) == 5

        correctness = _check_correctness(db_path)
        assert correctness["ok"] is True
    finally:
        db_path.unlink(missing_ok=True)


def test_harness_error_classification():
    """Verify lock contention errors are separated from ownership fencing."""
    import sqlite3

    from backend.app.ml.audit import AuditDatabaseError
    from backend.app.ml.jobs import ActiveJobExistsError, JobOwnershipError

    # Direct sqlite3 lock error
    lock_err = sqlite3.OperationalError("database is locked")
    assert _classify_error(lock_err) == "SQLITE_BUSY"

    # Wrapped lock error via AuditDatabaseError
    wrapped_err = AuditDatabaseError("Failed to read promotion audit database.")
    wrapped_err.__cause__ = sqlite3.OperationalError("database is locked")
    assert _classify_error(wrapped_err) == "SQLITE_BUSY"

    # Ownership fencing errors
    ownership_err = JobOwnershipError("Worker is not active owner")
    assert _classify_error(ownership_err) == "ownership"

    active_job_err = ActiveJobExistsError("Active job already exists")
    assert _classify_error(active_job_err) == "ownership"

    # Other operational errors
    syntax_err = sqlite3.OperationalError("near 'FOO': syntax error")
    assert _classify_error(syntax_err) == "other"


def test_harness_environment_capture():
    """Verify environment capture includes all required reproducibility fields."""
    env = capture_environment()
    assert "timestamp_utc" in env
    assert "python_version" in env
    assert "sqlite_version" in env
    assert env["journal_mode"] == "DELETE"
    assert env["busy_timeout_ms"] == 5000
    assert env["thread_concurrency_levels"] == [1, 5, 10, 25]
    assert env["process_concurrency_levels"] == [1, 2, 4]


def test_harness_quick_scenario_execution():
    """Verify a scenario runs end-to-end, records metrics, and leaves DB intact."""
    db_path = _create_temp_db()
    try:
        _initialize_db(db_path)
        metrics = run_w6_threads(
            n=2,
            db_path=db_path,
            warmup_s=0.5,
            measure_s=1.0,
            repetition=1,
        )
        assert metrics.scenario == "W6"
        assert metrics.concurrency == 2
        assert metrics.concurrency_type == "thread"
        assert metrics.total_ops > 0
        assert metrics.ops_per_second > 0
        assert metrics.integrity_ok is True
    finally:
        db_path.unlink(missing_ok=True)
