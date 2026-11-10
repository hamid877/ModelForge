"""Tests for Retraining Job Status API endpoints (Phase 6D).

Verifies GET /retraining/jobs/active and GET /retraining/jobs/{job_id}
for read-only behavior, status codes, response envelope, field exclusions,
and database failure handling.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from pydantic import SecretStr
from starlette.testclient import TestClient

import backend.app.api.api as api_module
from backend.app.api.api import app
from backend.app.config import ClientApiKeyConfig
from backend.app.ml.db import db_connection
from backend.app.ml.drift_alerts import persist_drift_alert
from backend.app.ml.jobs import (
    claim_queued_job,
    create_queued_job,
    mark_promotion_started,
    quarantine_expired_job,
    transition_job_status,
)
from backend.app.ml.migrations import migrate_audit_database_idempotent

TEST_RETRAINING_TOKEN = "test-jobs-retraining-token-001"

TEST_JOBS_API_KEYS = (
    ClientApiKeyConfig(
        client_id="test-jobs-retraining-client",
        token=SecretStr(TEST_RETRAINING_TOKEN),
        roles=("retraining",),
    ),
)

RETRAINING_AUTH_HEADERS = {"Authorization": f"Bearer {TEST_RETRAINING_TOKEN}"}


@pytest.fixture(autouse=True)
def configure_test_api_keys(monkeypatch):
    """Configure synthetic test-only API keys for job endpoint tests."""
    monkeypatch.setattr(
        "backend.app.api.security.settings.api_keys",
        TEST_JOBS_API_KEYS,
    )


@pytest.fixture(autouse=True)
def mock_mlflow_setup(monkeypatch):
    monkeypatch.setattr(api_module, "setup_mlflow", lambda: None)


@pytest.fixture(autouse=True)
def mock_migration_setup(monkeypatch):
    monkeypatch.setattr(api_module, "migrate_audit_database_idempotent", lambda _: None)


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary lifecycle database with initialized schema."""
    db_file = tmp_path / "lifecycle_test.db"
    migrate_audit_database_idempotent(db_file)
    return db_file


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_get_active_job_when_none_exists(mock_load_model, test_db):
    """GET /retraining/jobs/active returns active=false when empty."""
    mock_load_model.return_value.predict.return_value = [0]

    with (
        patch("backend.app.api.api.settings.lifecycle_audit_db_path", str(test_db)),
        TestClient(app) as client,
    ):
        response = client.get(
            "/retraining/jobs/active",
            headers=RETRAINING_AUTH_HEADERS,
        )

    assert response.status_code == 200
    data = response.json()
    assert data == {"active": False, "job": None}


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_get_active_job_when_only_terminal_jobs_exist(mock_load_model, test_db):
    """GET /retraining/jobs/active returns active=false for terminal jobs."""
    mock_load_model.return_value.predict.return_value = [0]

    job = create_queued_job(
        test_db,
        dataset_id="ds-term-1",
        trigger_source="manual",
    )
    claimed = claim_queued_job(test_db, worker_id="worker-term-1")
    transition_job_status(
        test_db,
        job_id=job["job_id"],
        target_status="succeeded",
        worker_id="worker-term-1",
        lease_token=claimed["lease_token"],
    )

    with (
        patch("backend.app.api.api.settings.lifecycle_audit_db_path", str(test_db)),
        TestClient(app) as client,
    ):
        response = client.get(
            "/retraining/jobs/active",
            headers=RETRAINING_AUTH_HEADERS,
        )

    assert response.status_code == 200
    data = response.json()
    assert data == {"active": False, "job": None}


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_get_active_job_queued(mock_load_model, test_db):
    """GET /retraining/jobs/active returns active=true and queued job details."""
    mock_load_model.return_value.predict.return_value = [0]

    alert = persist_drift_alert(
        database_path=test_db,
        drift_detected=True,
        reference_samples=100,
        current_samples=50,
        p_value_threshold=0.05,
        features={"feature1": {"drift_detected": True, "p_value": 0.01}},
    )

    created = create_queued_job(
        test_db,
        dataset_id="ds-active-queued",
        trigger_source="drift_assess",
        alert_id=alert["alert_id"],
    )

    with (
        patch("backend.app.api.api.settings.lifecycle_audit_db_path", str(test_db)),
        TestClient(app) as client,
    ):
        response = client.get(
            "/retraining/jobs/active",
            headers=RETRAINING_AUTH_HEADERS,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["active"] is True
    job_data = data["job"]
    assert job_data["job_id"] == created["job_id"]
    assert job_data["status"] == "queued"
    assert job_data["is_active"] is True
    assert job_data["dataset_id"] == "ds-active-queued"
    assert job_data["trigger_source"] == "drift_assess"
    assert job_data["alert_id"] == alert["alert_id"]
    assert job_data["attempt_count"] == 0
    assert job_data["max_attempts"] == 3


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_get_active_job_running(mock_load_model, test_db):
    """GET /retraining/jobs/active returns running job details."""
    mock_load_model.return_value.predict.return_value = [0]

    created = create_queued_job(
        test_db,
        dataset_id="ds-active-running",
        trigger_source="drift_assess",
    )
    claim_queued_job(test_db, worker_id="worker-run-1")

    with (
        patch("backend.app.api.api.settings.lifecycle_audit_db_path", str(test_db)),
        TestClient(app) as client,
    ):
        response = client.get(
            "/retraining/jobs/active",
            headers=RETRAINING_AUTH_HEADERS,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["active"] is True
    job_data = data["job"]
    assert job_data["job_id"] == created["job_id"]
    assert job_data["status"] == "running"
    assert job_data["is_active"] is True
    assert job_data["started_at"] is not None


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_get_active_job_reconciliation_required(mock_load_model, test_db):
    """GET /retraining/jobs/active returns job in reconciliation_required."""
    mock_load_model.return_value.predict.return_value = [0]

    created = create_queued_job(
        test_db,
        dataset_id="ds-active-quarantine",
        trigger_source="manual",
    )
    claimed = claim_queued_job(test_db, worker_id="worker-recon-1")
    mark_promotion_started(
        test_db,
        job_id=created["job_id"],
        worker_id="worker-recon-1",
        lease_token=claimed["lease_token"],
    )
    future_time = datetime.now(UTC) + timedelta(hours=1)
    quarantine_expired_job(
        test_db,
        job_id=created["job_id"],
        reason="lease_expired_after_promotion_marker",
        now=future_time,
    )

    with (
        patch("backend.app.api.api.settings.lifecycle_audit_db_path", str(test_db)),
        TestClient(app) as client,
    ):
        response = client.get(
            "/retraining/jobs/active",
            headers=RETRAINING_AUTH_HEADERS,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["active"] is True
    job_data = data["job"]
    assert job_data["job_id"] == created["job_id"]
    assert job_data["status"] == "reconciliation_required"
    assert job_data["is_active"] is True
    assert job_data["failure_reason"] == "lease_expired_after_promotion_marker"


@pytest.mark.parametrize(
    "target_status",
    [
        "queued",
        "running",
        "succeeded",
        "rejected",
        "failed",
        "reconciliation_required",
    ],
)
@patch("backend.app.api.api.load_registered_model_by_alias")
def test_get_job_by_id_all_six_states(mock_load_model, test_db, target_status):
    """GET /retraining/jobs/{job_id} retrieves jobs across all 6 possible statuses."""
    mock_load_model.return_value.predict.return_value = [0]

    job = create_queued_job(
        test_db,
        dataset_id=f"ds-{target_status}",
        trigger_source="test_runner",
    )
    jid = job["job_id"]

    if target_status != "queued":
        claimed = claim_queued_job(test_db, worker_id="worker-test")
        token = claimed["lease_token"]

        if target_status in ("succeeded", "rejected", "failed"):
            transition_job_status(
                test_db,
                job_id=jid,
                target_status=target_status,
                worker_id="worker-test",
                lease_token=token,
            )
        elif target_status == "reconciliation_required":
            mark_promotion_started(
                test_db,
                job_id=jid,
                worker_id="worker-test",
                lease_token=token,
            )
            future_time = datetime.now(UTC) + timedelta(hours=1)
            quarantine_expired_job(
                test_db,
                job_id=jid,
                reason="test_quarantine",
                now=future_time,
            )

    with (
        patch("backend.app.api.api.settings.lifecycle_audit_db_path", str(test_db)),
        TestClient(app) as client,
    ):
        response = client.get(
            f"/retraining/jobs/{jid}",
            headers=RETRAINING_AUTH_HEADERS,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == jid
    assert data["status"] == target_status
    if target_status in ("queued", "running", "reconciliation_required"):
        assert data["is_active"] is True
    else:
        assert data["is_active"] is False


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_get_job_by_id_unknown_returns_404(mock_load_model, test_db):
    """GET /retraining/jobs/{job_id} returns 404 when not found."""
    mock_load_model.return_value.predict.return_value = [0]

    with (
        patch("backend.app.api.api.settings.lifecycle_audit_db_path", str(test_db)),
        TestClient(app) as client,
    ):
        response = client.get(
            "/retraining/jobs/job-non-existent-999",
            headers=RETRAINING_AUTH_HEADERS,
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Retraining job 'job-non-existent-999' not found."
    }


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_field_exclusions_enforced(mock_load_model, test_db):
    """Ensure excluded fields are absent from both endpoints."""
    mock_load_model.return_value.predict.return_value = [0]

    created = create_queued_job(
        test_db,
        dataset_id="ds-exclusions",
        trigger_source="manual",
        payload={"sensitive_env": "secret_data"},
    )
    claim_queued_job(test_db, worker_id="worker-exclusion-test")

    with (
        patch("backend.app.api.api.settings.lifecycle_audit_db_path", str(test_db)),
        TestClient(app) as client,
    ):
        resp_active = client.get(
            "/retraining/jobs/active",
            headers=RETRAINING_AUTH_HEADERS,
        )
        resp_by_id = client.get(
            f"/retraining/jobs/{created['job_id']}",
            headers=RETRAINING_AUTH_HEADERS,
        )

    assert resp_active.status_code == 200
    assert resp_by_id.status_code == 200

    excluded_fields = [
        "lease_token",
        "worker_id",
        "heartbeat_at",
        "lease_expires_at",
        "payload",
    ]

    # Verify active endpoint JSON keys and raw text
    active_job = resp_active.json()["job"]
    for field in excluded_fields:
        assert field not in active_job
        assert field not in resp_active.text

    # Verify job-by-id endpoint JSON keys and raw text
    job_detail = resp_by_id.json()
    for field in excluded_fields:
        assert field not in job_detail
        assert field not in resp_by_id.text


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_endpoints_are_strictly_read_only(mock_load_model, test_db):
    """Repeated calls to status endpoints must not mutate database records."""
    mock_load_model.return_value.predict.return_value = [0]

    created = create_queued_job(
        test_db,
        dataset_id="ds-readonly-check",
        trigger_source="manual",
    )
    claim_queued_job(test_db, worker_id="worker-ro")
    jid = created["job_id"]

    # Take snapshot of DB row
    with db_connection(test_db) as conn:
        cursor = conn.execute("SELECT * FROM retraining_jobs WHERE job_id = ?;", (jid,))
        snapshot_before = cursor.fetchone()

    with (
        patch("backend.app.api.api.settings.lifecycle_audit_db_path", str(test_db)),
        TestClient(app) as client,
    ):
        for _ in range(5):
            res_act = client.get(
                "/retraining/jobs/active",
                headers=RETRAINING_AUTH_HEADERS,
            )
            assert res_act.status_code == 200
            res_id = client.get(
                f"/retraining/jobs/{jid}",
                headers=RETRAINING_AUTH_HEADERS,
            )
            assert res_id.status_code == 200

    # Verify DB row is completely unchanged
    with db_connection(test_db) as conn:
        cursor = conn.execute("SELECT * FROM retraining_jobs WHERE job_id = ?;", (jid,))
        snapshot_after = cursor.fetchone()

    assert snapshot_before == snapshot_after


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_database_failure_returns_503(mock_load_model):
    """Database connectivity failure returns HTTP 503 on both endpoints."""
    mock_load_model.return_value.predict.return_value = [0]

    with (
        patch(
            "backend.app.api.api.get_active_job",
            side_effect=RuntimeError("DB disconnected"),
        ),
        patch(
            "backend.app.api.api.get_job",
            side_effect=RuntimeError("DB disconnected"),
        ),
        TestClient(app) as client,
    ):
        res_active = client.get(
            "/retraining/jobs/active",
            headers=RETRAINING_AUTH_HEADERS,
        )
        res_id = client.get(
            "/retraining/jobs/job-any-id",
            headers=RETRAINING_AUTH_HEADERS,
        )

    assert res_active.status_code == 503
    assert res_active.json() == {"detail": "The retraining job service is unavailable."}

    assert res_id.status_code == 503
    assert res_id.json() == {"detail": "The retraining job service is unavailable."}


@patch("backend.app.api.api.load_registered_model_by_alias")
def test_existing_endpoints_remain_unaffected(mock_load_model, test_db):
    """Verify health, ready, and metrics endpoints function without interference."""
    mock_load_model.return_value.predict.return_value = [0]

    with (
        patch("backend.app.api.api.settings.lifecycle_audit_db_path", str(test_db)),
        TestClient(app) as client,
    ):
        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json() == {"status": "healthy"}

        res_ready = client.get("/ready")
        assert res_ready.status_code == 200
        assert res_ready.json() == {"status": "ready"}
