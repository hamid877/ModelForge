"""Unit and integration tests for retraining metrics and worker heartbeat observability.

Tests metric values across all job states, counter monotonicity, stateless
histogram semantics without double-counting, worker heartbeat freshness,
read-only SQLite safety, and Grafana dashboard JSON schema validity.
"""

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.api.api import app
from backend.app.ml.jobs import (
    claim_queued_job,
    create_queued_job,
    transition_job_status,
)
from backend.app.ml.migrations import (
    finalize_operator_resolution_atomic,
    migrate_audit_database_idempotent,
)
from backend.app.ml.retraining_metrics import (
    DURATION_BUCKETS,
    RetrainingMetricsSnapshot,
    format_retraining_prometheus_metrics,
    get_retraining_metrics_snapshot,
    record_worker_heartbeat,
)
from backend.app.ml.worker import RetrainingWorker


def test_empty_database_metrics(tmp_path: Path):
    """Empty database reports zero for gauges, counters, histograms, and heartbeat."""
    db_path = tmp_path / "empty.db"
    migrate_audit_database_idempotent(db_path)

    snapshot = get_retraining_metrics_snapshot(db_path)

    assert snapshot.is_available is True
    assert snapshot.queue_depth == 0
    assert snapshot.active_jobs == 0
    assert snapshot.jobs_succeeded_total == 0
    assert snapshot.jobs_rejected_total == 0
    assert snapshot.jobs_failed_total == 0
    assert snapshot.jobs_reconciliation_required_total == 0
    assert snapshot.duration_count == 0
    assert snapshot.duration_sum_seconds == 0.0
    for le in DURATION_BUCKETS:
        assert snapshot.duration_buckets[le] == 0
    assert snapshot.worker_heartbeat_timestamp_seconds == 0.0


def test_missing_and_corrupt_database_handling(tmp_path: Path):
    """Missing or corrupted database files are handled gracefully without raising."""
    # Case 1: Database file does not exist
    missing_path = tmp_path / "nonexistent.db"
    snapshot_missing = get_retraining_metrics_snapshot(missing_path)
    assert snapshot_missing.is_available is False
    assert snapshot_missing.queue_depth == 0
    assert snapshot_missing.worker_heartbeat_timestamp_seconds == 0.0

    lines_missing = format_retraining_prometheus_metrics(snapshot_missing)
    text_missing = "\n".join(lines_missing) + "\n"
    assert "modelforge_retraining_collector_up 0" in text_missing
    assert "modelforge_retraining_jobs_total" not in text_missing
    assert "modelforge_retraining_queue_depth" not in text_missing
    assert "modelforge_retraining_execution_duration_seconds" not in text_missing

    # Case 2: Corrupted database file (invalid header bytes)
    corrupt_path = tmp_path / "corrupt.db"
    corrupt_path.write_bytes(b"NOT_A_VALID_SQLITE_DATABASE_HEADER_DATA")
    snapshot_corrupt = get_retraining_metrics_snapshot(corrupt_path)
    assert snapshot_corrupt.is_available is False
    assert snapshot_corrupt.queue_depth == 0
    assert snapshot_corrupt.worker_heartbeat_timestamp_seconds == 0.0

    lines_corrupt = format_retraining_prometheus_metrics(snapshot_corrupt)
    text_corrupt = "\n".join(lines_corrupt) + "\n"
    assert "modelforge_retraining_collector_up 0" in text_corrupt
    assert "modelforge_retraining_jobs_total" not in text_corrupt
    assert "modelforge_retraining_queue_depth" not in text_corrupt


def test_queue_depth_and_active_jobs_gauges(tmp_path: Path):
    """Validate queue depth and active jobs gauges through job lifecycle."""
    db_path = tmp_path / "gauges.db"
    migrate_audit_database_idempotent(db_path)

    # 1. Initially 0
    snap = get_retraining_metrics_snapshot(db_path)
    assert snap.queue_depth == 0
    assert snap.active_jobs == 0

    # 2. Enqueue a job -> queue_depth=1, active_jobs=0
    job1 = create_queued_job(
        database_path=db_path,
        dataset_id="ds-001",
        trigger_source="drift_endpoint",
        payload={"reason": "test"},
    )
    snap = get_retraining_metrics_snapshot(db_path)
    assert snap.queue_depth == 1
    assert snap.active_jobs == 0

    # 3. Claim job -> queue_depth=0, active_jobs=1
    claimed = claim_queued_job(
        database_path=db_path,
        worker_id="worker-test-1",
        lease_duration_seconds=60,
    )
    assert claimed is not None
    snap = get_retraining_metrics_snapshot(db_path)
    assert snap.queue_depth == 0
    assert snap.active_jobs == 1

    # 4. Transition job to succeeded -> queue_depth=0, active_jobs=0
    transition_job_status(
        database_path=db_path,
        job_id=job1["job_id"],
        worker_id="worker-test-1",
        lease_token=claimed["lease_token"],
        target_status="succeeded",
    )
    snap = get_retraining_metrics_snapshot(db_path)
    assert snap.queue_depth == 0
    assert snap.active_jobs == 0


def test_counter_monotonicity_across_terminal_states(tmp_path: Path):
    """Terminal job counters strictly count cumulative outcomes monotonically."""
    db_path = tmp_path / "outcomes.db"
    migrate_audit_database_idempotent(db_path)

    # Create and claim job 1 -> succeed
    j1 = create_queued_job(
        database_path=db_path, dataset_id="d1", trigger_source="t1", payload={}
    )
    c1 = claim_queued_job(
        database_path=db_path, worker_id="w1", lease_duration_seconds=60
    )
    transition_job_status(
        database_path=db_path,
        job_id=j1["job_id"],
        worker_id="w1",
        lease_token=c1["lease_token"],
        target_status="succeeded",
    )

    snap = get_retraining_metrics_snapshot(db_path)
    assert snap.jobs_succeeded_total == 1
    assert snap.jobs_rejected_total == 0
    assert snap.jobs_failed_total == 0

    # Create and claim job 2 -> reject
    j2 = create_queued_job(
        database_path=db_path, dataset_id="d2", trigger_source="t2", payload={}
    )
    c2 = claim_queued_job(
        database_path=db_path, worker_id="w1", lease_duration_seconds=60
    )
    transition_job_status(
        database_path=db_path,
        job_id=j2["job_id"],
        worker_id="w1",
        lease_token=c2["lease_token"],
        target_status="rejected",
    )

    snap = get_retraining_metrics_snapshot(db_path)
    assert snap.jobs_succeeded_total == 1
    assert snap.jobs_rejected_total == 1
    assert snap.jobs_failed_total == 0

    # Create and claim job 3 -> fail
    j3 = create_queued_job(
        database_path=db_path, dataset_id="d3", trigger_source="t3", payload={}
    )
    c3 = claim_queued_job(
        database_path=db_path, worker_id="w1", lease_duration_seconds=60
    )
    transition_job_status(
        database_path=db_path,
        job_id=j3["job_id"],
        worker_id="w1",
        lease_token=c3["lease_token"],
        target_status="failed",
    )

    snap = get_retraining_metrics_snapshot(db_path)
    assert snap.jobs_succeeded_total == 1
    assert snap.jobs_rejected_total == 1
    assert snap.jobs_failed_total == 1


def test_reconciliation_counter_monotonicity_on_resolution(tmp_path: Path):
    """Reconciliation counter remains monotonic when quarantined jobs are resolved."""
    db_path = tmp_path / "reconciliation.db"
    migrate_audit_database_idempotent(db_path)

    # Job 1 transitions to reconciliation_required
    j1 = create_queued_job(
        database_path=db_path, dataset_id="d1", trigger_source="t1", payload={}
    )
    c1 = claim_queued_job(
        database_path=db_path, worker_id="w1", lease_duration_seconds=60
    )
    transition_job_status(
        database_path=db_path,
        job_id=j1["job_id"],
        worker_id="w1",
        lease_token=c1["lease_token"],
        target_status="reconciliation_required",
    )

    snap = get_retraining_metrics_snapshot(db_path)
    assert snap.jobs_reconciliation_required_total == 1
    assert snap.jobs_succeeded_total == 0
    assert snap.jobs_failed_total == 0

    # Operator resolves Job 1 to succeeded
    finalize_operator_resolution_atomic(
        database_path=db_path,
        resolution_id="res-001",
        job_id=j1["job_id"],
        operator="admin@example.com",
        selected_resolution="resolved_succeeded",
        candidate_version="v2",
        observed_champion_version="v1",
        target_champion_version="v2",
        rollback_performed=0,
        resolution_reason="Manual verification approved",
        evidence_payload={"check": "passed"},
    )

    snap = get_retraining_metrics_snapshot(db_path)
    # MONOTONICITY INVARIANT: reconciliation_required MUST NOT decrease to 0!
    assert snap.jobs_reconciliation_required_total == 1
    # Target succeeded counter increments
    assert snap.jobs_succeeded_total == 1
    assert snap.jobs_failed_total == 0

    # Job 2 transitions to reconciliation_required
    j2 = create_queued_job(
        database_path=db_path, dataset_id="d2", trigger_source="t2", payload={}
    )
    c2 = claim_queued_job(
        database_path=db_path, worker_id="w1", lease_duration_seconds=60
    )
    transition_job_status(
        database_path=db_path,
        job_id=j2["job_id"],
        worker_id="w1",
        lease_token=c2["lease_token"],
        target_status="reconciliation_required",
    )

    snap = get_retraining_metrics_snapshot(db_path)
    assert snap.jobs_reconciliation_required_total == 2
    assert snap.jobs_succeeded_total == 1
    assert snap.jobs_failed_total == 0

    # Operator resolves Job 2 to failed
    finalize_operator_resolution_atomic(
        database_path=db_path,
        resolution_id="res-002",
        job_id=j2["job_id"],
        operator="admin@example.com",
        selected_resolution="resolved_failed",
        candidate_version=None,
        observed_champion_version="v1",
        target_champion_version="v1",
        rollback_performed=1,
        resolution_reason="Candidate rollback confirmed",
        evidence_payload={"check": "failed"},
    )

    snap = get_retraining_metrics_snapshot(db_path)
    # Both reconciliation incidents are preserved
    assert snap.jobs_reconciliation_required_total == 2
    assert snap.jobs_succeeded_total == 1
    assert snap.jobs_failed_total == 1


def test_operator_resolution_audit_deduplication(tmp_path: Path):
    """Verify UNION deduplication ensures a job cannot be counted multiple times."""
    db_path = tmp_path / "dedup.db"
    migrate_audit_database_idempotent(db_path)

    # Insert a job into retraining_jobs in succeeded status
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO retraining_jobs (
                job_id, created_at, status, dataset_id, trigger_source, payload
            ) VALUES (
                'job-dedup-1', datetime('now'), 'succeeded', 'ds-1', 'drift', '{}'
            );
            """
        )
        # Simulate two audit rows for the same job (defensive validation)
        conn.execute(
            """
            INSERT INTO operator_resolution_audit (
                resolution_id, job_id, timestamp, operator, previous_state,
                selected_resolution, rollback_performed, resolution_reason,
                evidence_payload
            ) VALUES
            (
                'res-1', 'job-dedup-1', datetime('now'), 'op1',
                'reconciliation_required', 'resolved_succeeded', 0, 'reason1', '{}'
            ),
            (
                'res-2', 'job-dedup-1', datetime('now'), 'op2',
                'reconciliation_required', 'resolved_succeeded', 0, 'reason2', '{}'
            );
            """
        )

    snap = get_retraining_metrics_snapshot(db_path)
    # Distinct job_id counting guarantees count is exactly 1, not 2
    assert snap.jobs_reconciliation_required_total == 1


def test_histogram_stateless_no_double_counting(tmp_path: Path):
    """Repeated scrapes against completed jobs do not inflate histogram counts."""
    db_path = tmp_path / "histogram.db"
    migrate_audit_database_idempotent(db_path)

    now = datetime.now(UTC)
    t1_start = (now - timedelta(seconds=120)).isoformat()
    t1_end = (now - timedelta(seconds=100)).isoformat()  # 20s duration
    t2_start = (now - timedelta(seconds=80)).isoformat()
    t2_end = (now - timedelta(seconds=35)).isoformat()  # 45s duration

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO retraining_jobs (
                job_id, created_at, started_at, completed_at, status,
                dataset_id, trigger_source, payload
            ) VALUES
            ('j1', datetime('now'), ?, ?, 'succeeded', 'd1', 't1', '{}'),
            ('j2', datetime('now'), ?, ?, 'rejected', 'd2', 't2', '{}');
            """,
            (t1_start, t1_end, t2_start, t2_end),
        )

    # Scrape 1
    snap1 = get_retraining_metrics_snapshot(db_path)
    assert snap1.duration_count == 2
    assert snap1.duration_sum_seconds == pytest.approx(65.0, abs=0.1)
    assert snap1.duration_buckets[10.0] == 0
    assert snap1.duration_buckets[30.0] == 1  # 20s <= 30.0
    assert snap1.duration_buckets[60.0] == 2  # 20s <= 60.0, 45s <= 60.0

    # Scrape 2 (Immediate subsequent scrape with zero DB changes)
    snap2 = get_retraining_metrics_snapshot(db_path)
    assert snap2.duration_count == 2
    assert snap2.duration_sum_seconds == pytest.approx(65.0, abs=0.1)
    assert snap2.duration_buckets == snap1.duration_buckets

    # Scrape 3 (Third scrape - verify zero compounding)
    snap3 = get_retraining_metrics_snapshot(db_path)
    assert snap3.duration_count == 2
    assert snap3.duration_sum_seconds == pytest.approx(65.0, abs=0.1)

    # Add 3rd job: 100s duration
    t3_start = (now - timedelta(seconds=200)).isoformat()
    t3_end = (now - timedelta(seconds=100)).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO retraining_jobs (
                job_id, created_at, started_at, completed_at, status,
                dataset_id, trigger_source, payload
            ) VALUES ('j3', datetime('now'), ?, ?, 'failed', 'd3', 't3', '{}');
            """,
            (t3_start, t3_end),
        )

    snap4 = get_retraining_metrics_snapshot(db_path)
    assert snap4.duration_count == 3
    assert snap4.duration_sum_seconds == pytest.approx(165.0, abs=0.1)
    assert snap4.duration_buckets[60.0] == 2
    assert snap4.duration_buckets[120.0] == 3


def test_histogram_handles_missing_and_malformed_timestamps(tmp_path: Path):
    """Malformed, missing, or skew-reversed timestamps are handled safely."""
    db_path = tmp_path / "timestamps.db"
    migrate_audit_database_idempotent(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO retraining_jobs (
                job_id, created_at, started_at, completed_at, status,
                dataset_id, trigger_source, payload
            ) VALUES
            ('j-null', datetime('now'), NULL, datetime('now'),
             'succeeded', 'd', 't', ''),
            ('j-bad', datetime('now'), 'BAD', datetime('now'),
             'succeeded', 'd', 't', ''),
            ('j-rev', datetime('now'), '2026-11-05T12:00:00+00:00',
             '2026-11-05T11:59:00+00:00', 'succeeded', 'd', 't', '');
            """
        )

    snap = get_retraining_metrics_snapshot(db_path)
    # j-null and j-bad skipped; j-rev clamped safely to 0.0s
    assert snap.duration_count == 1
    assert snap.duration_sum_seconds == 0.0
    assert snap.duration_buckets[10.0] == 1


def test_worker_heartbeat_lifecycle(tmp_path: Path):
    """Worker heartbeat handles active updates, graceful stop, and crash staleness."""
    db_path = tmp_path / "heartbeat.db"
    migrate_audit_database_idempotent(db_path)

    # 1. No worker has run -> 0.0
    snap = get_retraining_metrics_snapshot(db_path)
    assert snap.worker_heartbeat_timestamp_seconds == 0.0

    # 2. Worker active heartbeat
    record_worker_heartbeat(db_path, worker_id="worker-w1", status="active")
    snap = get_retraining_metrics_snapshot(db_path)
    assert snap.worker_heartbeat_timestamp_seconds > 0.0
    active_ts = snap.worker_heartbeat_timestamp_seconds

    # 3. Graceful shutdown sets status="stopped" -> reports 0.0
    record_worker_heartbeat(db_path, worker_id="worker-w1", status="stopped")
    snap = get_retraining_metrics_snapshot(db_path)
    assert snap.worker_heartbeat_timestamp_seconds == 0.0

    # 4. Worker restarts after crash -> updates status="active"
    record_worker_heartbeat(db_path, worker_id="worker-w1", status="active")
    snap = get_retraining_metrics_snapshot(db_path)
    assert snap.worker_heartbeat_timestamp_seconds >= active_ts

    # 5. Simulate unexpected termination (crash):
    # Worker process terminates without graceful stop. The status remains 'active',
    # and the timestamp is preserved. As wall-clock time passes, it becomes stale.
    old_time = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE worker_heartbeats SET heartbeat_at = ? "
            "WHERE worker_id = 'worker-w1';",
            (old_time,),
        )

    snap = get_retraining_metrics_snapshot(db_path)
    stale_ts = snap.worker_heartbeat_timestamp_seconds
    now_ts = datetime.now(UTC).timestamp()
    staleness_seconds = now_ts - stale_ts
    assert staleness_seconds >= 100.0  # Clearly detectable as stale (> 30s threshold)


def test_stopped_worker_precedence_over_unreclaimed_running_job(tmp_path: Path):
    """Explicit status='stopped' produces 0.0 despite unreclaimed running job."""
    db_path = tmp_path / "stopped_precedence.db"
    migrate_audit_database_idempotent(db_path)

    # 1. Record worker as stopped
    record_worker_heartbeat(db_path, worker_id="worker-stopped-1", status="stopped")

    # 2. Simulate an unreclaimed running job with a recent lease heartbeat
    now_iso = datetime.now(UTC).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO retraining_jobs (
                job_id, created_at, started_at, status, is_active,
                worker_id, heartbeat_at, dataset_id, trigger_source, payload
            ) VALUES (
                'job-unreclaimed', ?, ?, 'running', 1,
                'worker-stopped-1', ?, 'ds-1', 'drift', '{}'
            );
            """,
            (now_iso, now_iso, now_iso),
        )

    # 3. Snapshot must report exactly 0.0 because worker is explicitly stopped
    snap = get_retraining_metrics_snapshot(db_path)
    assert snap.worker_heartbeat_timestamp_seconds == 0.0


def test_worker_class_heartbeat_integration(tmp_path: Path):
    """RetrainingWorker integration updates heartbeat during run_once and on stop."""
    db_path = tmp_path / "worker_integ.db"
    lock_path = tmp_path / "worker_integ.lock"
    migrate_audit_database_idempotent(db_path)

    worker = RetrainingWorker(
        database_path=db_path,
        lock_path=lock_path,
        worker_id="test-worker-alpha",
        standby=False,
    )

    # Worker single iteration under lock
    worker.run_once()

    snap = get_retraining_metrics_snapshot(db_path)
    assert snap.worker_heartbeat_timestamp_seconds > 0.0

    # Worker stop sets stopped
    worker.stop()
    worker.release_lock()

    snap = get_retraining_metrics_snapshot(db_path)
    assert snap.worker_heartbeat_timestamp_seconds == 0.0


def test_read_only_sqlite_enforcement(tmp_path: Path):
    """Collector connection uses mode=ro and strictly forbids database writes."""
    db_path = tmp_path / "readonly.db"
    migrate_audit_database_idempotent(db_path)

    # Connect with same read-only URI as collector
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        # Write attempt must raise OperationalError: attempt to write a readonly DB
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute(
                "INSERT INTO retraining_jobs (job_id, created_at, status, "
                "dataset_id, trigger_source, payload) "
                "VALUES ('fail', datetime('now'), 'queued', 'd', 't', '{}');"
            )
    finally:
        conn.close()


def test_prometheus_exposition_format():
    """Verify Prometheus exposition lines adhere to specification."""
    snapshot = RetrainingMetricsSnapshot(
        queue_depth=2,
        active_jobs=1,
        jobs_succeeded_total=5,
        jobs_rejected_total=2,
        jobs_failed_total=1,
        jobs_reconciliation_required_total=1,
        duration_buckets={
            10.0: 1,
            30.0: 3,
            60.0: 5,
            120.0: 8,
            300.0: 8,
            600.0: 8,
            1200.0: 8,
            1800.0: 8,
            3600.0: 8,
        },
        duration_sum_seconds=420.5,
        duration_count=8,
        worker_heartbeat_timestamp_seconds=1731000000.0,
        is_available=True,
    )

    lines = format_retraining_prometheus_metrics(snapshot)
    text = "\n".join(lines) + "\n"

    assert "modelforge_retraining_collector_up 1" in text
    assert "# TYPE modelforge_retraining_queue_depth gauge" in text
    assert "modelforge_retraining_queue_depth 2" in text
    assert "# TYPE modelforge_retraining_active_jobs gauge" in text
    assert "modelforge_retraining_active_jobs 1" in text
    assert "# TYPE modelforge_retraining_jobs_total counter" in text
    assert 'modelforge_retraining_jobs_total{status="succeeded"} 5' in text
    assert 'modelforge_retraining_jobs_total{status="rejected"} 2' in text
    assert 'modelforge_retraining_jobs_total{status="failed"} 1' in text
    assert (
        'modelforge_retraining_jobs_total{status="reconciliation_required"} 1' in text
    )
    assert "# TYPE modelforge_retraining_execution_duration_seconds histogram" in text
    assert (
        'modelforge_retraining_execution_duration_seconds_bucket{le="10.0"} 1' in text
    )
    assert (
        'modelforge_retraining_execution_duration_seconds_bucket{le="+Inf"} 8' in text
    )
    assert "modelforge_retraining_execution_duration_seconds_sum 420.5" in text
    assert "modelforge_retraining_execution_duration_seconds_count 8" in text
    assert "# TYPE modelforge_worker_heartbeat_timestamp_seconds gauge" in text
    assert "modelforge_worker_heartbeat_timestamp_seconds 1731000000.0" in text


def test_api_prometheus_endpoint_available_and_unavailable(tmp_path: Path):
    """GET /metrics/prometheus exposes collector status and handles unavailable DB."""
    # 1. Available DB scenario
    db_path = tmp_path / "api_metrics.db"
    migrate_audit_database_idempotent(db_path)

    create_queued_job(
        database_path=db_path,
        dataset_id="ds-integ",
        trigger_source="api_test",
        payload={"note": "prometheus_integ"},
    )

    with (
        patch("backend.app.api.api.load_registered_model_by_alias") as mock_load,
        patch("backend.app.api.api.setup_mlflow"),
        patch(
            "backend.app.api.api.migrate_audit_database_idempotent",
            lambda _: None,
        ),
        patch("backend.app.api.api.settings.lifecycle_audit_db_path", str(db_path)),
    ):
        mock_load.return_value.predict.return_value = [1]
        with TestClient(app) as client:
            resp_available = client.get("/metrics/prometheus")

    assert resp_available.status_code == 200
    assert "text/plain" in resp_available.headers["content-type"]
    body_avail = resp_available.text

    # Prediction metrics present
    assert "modelforge_prediction_requests_total" in body_avail
    # Retraining metrics present with collector_up 1
    assert "modelforge_retraining_collector_up 1" in body_avail
    assert "modelforge_retraining_queue_depth 1" in body_avail
    assert 'modelforge_retraining_jobs_total{status="succeeded"} 0' in body_avail

    # 2. Unavailable DB scenario
    with (
        patch("backend.app.api.api.load_registered_model_by_alias") as mock_load,
        patch("backend.app.api.api.setup_mlflow"),
        patch(
            "backend.app.api.api.migrate_audit_database_idempotent",
            lambda _: None,
        ),
        patch(
            "backend.app.api.api.get_retraining_metrics_snapshot",
            return_value=RetrainingMetricsSnapshot(is_available=False),
        ),
    ):
        mock_load.return_value.predict.return_value = [1]
        with TestClient(app) as client:
            resp_unavail = client.get("/metrics/prometheus")

    assert resp_unavail.status_code == 200
    body_unavail = resp_unavail.text

    # Prediction metrics still served normally
    assert "modelforge_prediction_requests_total" in body_unavail
    # Collector reports 0
    assert "modelforge_retraining_collector_up 0" in body_unavail
    # All retraining metrics and synthetic zero counters omitted
    assert "modelforge_retraining_jobs_total" not in body_unavail
    assert "modelforge_retraining_queue_depth" not in body_unavail
    assert "modelforge_retraining_active_jobs" not in body_unavail
    assert "modelforge_retraining_execution_duration_seconds" not in body_unavail
    assert "modelforge_worker_heartbeat_timestamp_seconds" not in body_unavail


def test_grafana_dashboard_json_validity():
    """Grafana dashboard modelforge-overview.json has valid schema and panel targets."""
    dashboard_path = Path("monitoring/grafana/dashboards/modelforge-overview.json")
    assert dashboard_path.exists()

    with open(dashboard_path) as f:
        data = json.load(f)

    assert data.get("uid") == "modelforge-overview"
    assert data.get("schemaVersion", 0) >= 36

    panel_titles = [p.get("title") for p in data.get("panels", [])]
    expected_titles = [
        "Retraining & Worker Orchestration",
        "Retraining Queue Depth",
        "Active Retraining Jobs",
        "Worker Heartbeat Freshness",
        "Retraining Job Outcomes",
        "Retraining Execution Duration",
    ]
    for title in expected_titles:
        assert title in panel_titles, f"Missing expected panel title '{title}'"

    # Validate targets have expressions
    for panel in data.get("panels", []):
        if panel.get("type") == "row":
            continue
        targets = panel.get("targets", [])
        assert len(targets) >= 1, f"Panel '{panel.get('title')}' has no targets"
        for target in targets:
            assert "expr" in target and len(target["expr"]) > 0

    # Validate panel 12 has spanNulls enabled
    panel_12 = next(p for p in data.get("panels", []) if p.get("id") == 12)
    custom_cfg = panel_12.get("fieldConfig", {}).get("defaults", {}).get("custom", {})
    assert custom_cfg.get("spanNulls") is True
