import json
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from backend.app.api import api as api_module
from backend.app.api.security import (
    AuthenticatedClient,
    SlidingWindowRateLimiter,
    authenticate_token,
    get_current_client,
    require_role,
)
from backend.app.config import ClientApiKeyConfig, Settings
from backend.app.ml.audit import record_promotion_audit
from backend.app.ml.jobs import create_queued_job
from backend.app.ml.migrations import migrate_audit_database_idempotent

# Synthetic test-only credentials
TEST_PREDICT_TOKEN = "test-predict-token-value-12345"
TEST_ADMIN_TOKEN = "test-admin-token-value-99999"
TEST_AUDIT_TOKEN = "test-audit-token-value-54321"

TEST_KEYS = (
    ClientApiKeyConfig(
        client_id="test-predict-client",
        token=SecretStr(TEST_PREDICT_TOKEN),
        roles=("predict",),
    ),
    ClientApiKeyConfig(
        client_id="test-admin-client",
        token=SecretStr(TEST_ADMIN_TOKEN),
        roles=("admin",),
    ),
    ClientApiKeyConfig(
        client_id="test-audit-client",
        token=SecretStr(TEST_AUDIT_TOKEN),
        roles=("audit",),
    ),
)


@pytest.fixture
def auth_test_app(monkeypatch):
    """Create a test FastAPI application with isolated synthetic credentials."""
    monkeypatch.setattr(
        "backend.app.api.security.settings.api_keys",
        TEST_KEYS,
    )

    test_app = FastAPI()

    @test_app.get("/protected")
    def protected_route(client: AuthenticatedClient = Depends(get_current_client)):
        return {
            "client_id": client.client_id,
            "roles": sorted(client.roles),
            "is_admin": client.is_admin,
        }

    return test_app


# 1. Valid Bearer token authentication
def test_valid_bearer_token_authentication(auth_test_app):
    """Verify successful authentication using standard Authorization: Bearer <token>."""
    with TestClient(auth_test_app) as client:
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {TEST_PREDICT_TOKEN}"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "client_id": "test-predict-client",
        "roles": ["predict"],
        "is_admin": False,
    }


# 2. Valid X-API-Key authentication
def test_valid_x_api_key_authentication(auth_test_app):
    """Verify successful authentication using X-API-Key: <token>."""
    with TestClient(auth_test_app) as client:
        response = client.get(
            "/protected",
            headers={"X-API-Key": TEST_PREDICT_TOKEN},
        )

    assert response.status_code == 200
    assert response.json() == {
        "client_id": "test-predict-client",
        "roles": ["predict"],
        "is_admin": False,
    }


# 3. Missing credentials
def test_missing_credentials_returns_401(auth_test_app):
    """Verify request with no auth headers returns 401 with WWW-Authenticate."""
    with TestClient(auth_test_app) as client:
        response = client.get("/protected")

    assert response.status_code == 401
    assert response.headers.get("WWW-Authenticate") == "Bearer"
    assert response.json()["detail"] == "Missing authentication credentials."


# 4. Invalid credentials
def test_invalid_credentials_returns_401(auth_test_app):
    """Verify unrecognized token returns 401 with sanitized error message."""
    invalid_token = "attacker-supplied-unknown-token"
    with TestClient(auth_test_app) as client:
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {invalid_token}"},
        )

    assert response.status_code == 401
    assert response.headers.get("WWW-Authenticate") == "Bearer"
    assert response.json()["detail"] == "Invalid authentication credentials."
    # Verify submitted token is never echoed in error response
    assert invalid_token not in response.text


# 5. Malformed Bearer header
@pytest.mark.parametrize(
    "malformed_header",
    [
        "Basic dXNlcjpwYXNz",
        "Bearer",
        "Bearer ",
        "NotBearer token123",
        "Token xyz",
        "",
    ],
)
def test_malformed_bearer_header_returns_401(auth_test_app, malformed_header):
    """Verify malformed headers return 401 with WWW-Authenticate header."""
    with TestClient(auth_test_app) as client:
        response = client.get(
            "/protected",
            headers={"Authorization": malformed_header},
        )

    assert response.status_code == 401
    assert response.headers.get("WWW-Authenticate") == "Bearer"
    assert response.json()["detail"] == "Invalid authentication credentials."


# 6. Both headers supplied with matching credentials
def test_both_headers_matching_credentials(auth_test_app):
    """Verify supplying both headers with matching tokens succeeds cleanly."""
    with TestClient(auth_test_app) as client:
        response = client.get(
            "/protected",
            headers={
                "Authorization": f"Bearer {TEST_ADMIN_TOKEN}",
                "X-API-Key": TEST_ADMIN_TOKEN,
            },
        )

    assert response.status_code == 200
    assert response.json()["client_id"] == "test-admin-client"
    assert response.json()["is_admin"] is True


# 7. Both headers supplied with conflicting credentials
def test_both_headers_conflicting_credentials_returns_401(auth_test_app):
    """Verify supplying both headers with conflicting credentials returns 401."""
    with TestClient(auth_test_app) as client:
        response = client.get(
            "/protected",
            headers={
                "Authorization": f"Bearer {TEST_PREDICT_TOKEN}",
                "X-API-Key": TEST_ADMIN_TOKEN,
            },
        )

    assert response.status_code == 401
    assert response.headers.get("WWW-Authenticate") == "Bearer"
    assert (
        response.json()["detail"] == "Conflicting authentication credentials supplied."
    )


# 8. Duplicate client IDs rejected during configuration
def test_configuration_rejects_duplicate_client_ids(monkeypatch):
    """Verify configuration validation rejects duplicate client IDs."""
    payload = (
        '[{"client_id": "duplicate-id", "token": "token-1", "roles": ["predict"]}, '
        '{"client_id": "duplicate-id", "token": "token-2", "roles": ["audit"]}]'
    )
    monkeypatch.setenv("MODELFORGE_API_KEYS", payload)
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)
    assert "Duplicate client_id" in str(exc_info.value)


# 9. Invalid role names rejected
@pytest.mark.parametrize(
    "invalid_role",
    ["superuser", "root", "guest", "predict_write", "analytics", ""],
)
def test_configuration_rejects_invalid_roles(monkeypatch, invalid_role):
    """Verify configuration validation rejects any role outside the approved set."""
    payload = (
        f'[{{"client_id": "c1", "token": "token-1", "roles": ["{invalid_role}"]}}]'
    )
    monkeypatch.setenv("MODELFORGE_API_KEYS", payload)
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)
    assert "Invalid role" in str(exc_info.value) or "roles" in str(exc_info.value)


# 10. Missing credential configuration handled safely
def test_missing_credential_configuration_handled_safely(monkeypatch):
    """Verify unconfigured credentials default to empty and fail closed with 401."""
    monkeypatch.delenv("MODELFORGE_API_KEYS", raising=False)
    config = Settings(_env_file=None)
    assert config.api_keys == ()

    # An app with empty api_keys must safely reject authentication without 500 errors
    test_app = FastAPI()

    @test_app.get("/protected")
    def protected(client: AuthenticatedClient = Depends(get_current_client)):
        return {"status": "ok"}

    monkeypatch.setattr(
        "backend.app.api.security.settings.api_keys",
        (),
    )

    with TestClient(test_app) as client:
        # Request with a token
        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer some-token"},
        )
        assert response.status_code == 401
        assert response.headers.get("WWW-Authenticate") == "Bearer"
        assert response.json()["detail"] == "Invalid authentication credentials."

        # Request without a token
        response_missing = client.get("/protected")
        assert response_missing.status_code == 401
        assert response_missing.headers.get("WWW-Authenticate") == "Bearer"
        assert (
            response_missing.json()["detail"] == "Missing authentication credentials."
        )


# 11. Secret values are not exposed in representations or error messages
def test_secret_values_never_exposed(monkeypatch):
    """Verify secret values are masked in representations and error messages."""
    sensitive_token = "super-secret-sensitive-token-987654321"
    config = ClientApiKeyConfig(
        client_id="sensitive-client",
        token=SecretStr(sensitive_token),
        roles=("predict",),
    )

    # Object representations must mask the secret
    assert sensitive_token not in repr(config)
    assert sensitive_token not in repr(config.token)
    assert "**********" in repr(config)
    assert "**********" in repr(config.token)

    # Validation errors on token must not contain the sensitive token value
    with pytest.raises(ValidationError) as exc_info:
        ClientApiKeyConfig(
            client_id="c1",
            token=SecretStr("   "),  # whitespace token
            roles=("predict",),
        )
    assert sensitive_token not in str(exc_info.value)


# 12. Authentication errors do not echo submitted tokens
def test_auth_errors_never_echo_submitted_token(auth_test_app):
    """Verify malformed and invalid token strings are never echoed in HTTP responses."""
    attacker_token = "malicious-canary-token-string-xyz123"

    with TestClient(auth_test_app) as client:
        # Invalid Bearer
        r1 = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {attacker_token}"},
        )
        assert attacker_token not in r1.text

        # Invalid X-API-Key
        r2 = client.get(
            "/protected",
            headers={"X-API-Key": attacker_token},
        )
        assert attacker_token not in r2.text

        # Conflicting credentials
        r3 = client.get(
            "/protected",
            headers={
                "Authorization": f"Bearer {attacker_token}",
                "X-API-Key": "different-token",
            },
        )
        assert attacker_token not in r3.text
        assert "different-token" not in r3.text


# Additional tests: AuthenticatedClient role behaviors and immutability
def test_authenticated_client_has_role_and_admin_expansion():
    """Verify role checks and admin wildcard permissions."""
    predict_client = AuthenticatedClient(
        client_id="pred-1",
        roles=frozenset({"predict"}),
    )
    assert predict_client.has_role("predict") is True
    assert predict_client.has_role("monitoring") is False
    assert predict_client.has_role("retraining") is False
    assert predict_client.has_role("audit") is False
    assert predict_client.is_admin is False

    admin_client = AuthenticatedClient(
        client_id="admin-1",
        roles=frozenset({"admin"}),
    )
    # Admin must grant access to all protected roles
    assert admin_client.has_role("predict") is True
    assert admin_client.has_role("monitoring") is True
    assert admin_client.has_role("retraining") is True
    assert admin_client.has_role("audit") is True
    assert admin_client.has_role("admin") is True
    assert admin_client.is_admin is True


def test_authenticated_client_immutability():
    """Verify AuthenticatedClient instances are strictly immutable."""
    client = AuthenticatedClient(
        client_id="immutable-client",
        roles=frozenset({"predict"}),
    )
    with pytest.raises(FrozenInstanceError):
        client.client_id = "modified-id"  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        client.roles = frozenset({"admin"})  # type: ignore[misc]


def test_constant_time_token_comparison_used():
    """Verify token lookup uses constant-time comparison via compare_digest."""
    with patch(
        "backend.app.api.security.secrets.compare_digest",
        wraps=secrets.compare_digest,
    ) as mock_compare:
        authenticate_token(
            TEST_PREDICT_TOKEN,
            configured_keys=TEST_KEYS,
        )
        assert mock_compare.called


# ---------------------------------------------------------------------------
# Phase 2: RBAC Dependency & Endpoint Protection Tests
# ---------------------------------------------------------------------------

TEST_MONITORING_TOKEN = "test-monitoring-token-value-24680"
TEST_RETRAINING_TOKEN = "test-retraining-token-value-13579"

RBAC_TEST_KEYS = (
    ClientApiKeyConfig(
        client_id="rbac-predict-client",
        token=SecretStr(TEST_PREDICT_TOKEN),
        roles=("predict",),
    ),
    ClientApiKeyConfig(
        client_id="rbac-monitoring-client",
        token=SecretStr(TEST_MONITORING_TOKEN),
        roles=("monitoring",),
    ),
    ClientApiKeyConfig(
        client_id="rbac-audit-client",
        token=SecretStr(TEST_AUDIT_TOKEN),
        roles=("audit",),
    ),
    ClientApiKeyConfig(
        client_id="rbac-retraining-client",
        token=SecretStr(TEST_RETRAINING_TOKEN),
        roles=("retraining",),
    ),
    ClientApiKeyConfig(
        client_id="rbac-admin-client",
        token=SecretStr(TEST_ADMIN_TOKEN),
        roles=("admin",),
    ),
)

ROLE_TOKENS = {
    "predict": TEST_PREDICT_TOKEN,
    "monitoring": TEST_MONITORING_TOKEN,
    "audit": TEST_AUDIT_TOKEN,
    "retraining": TEST_RETRAINING_TOKEN,
    "admin": TEST_ADMIN_TOKEN,
}


def _sample_customer_record() -> dict[str, object]:
    return {
        "age": 35,
        "tenure": 24,
        "monthly_charges": 79.99,
        "contract_type": "month-to-month",
        "support_tickets": 2,
    }


def _sample_drift_records() -> list[dict[str, object]]:
    return [
        {
            "age": 20 + i,
            "tenure": 1 + i,
            "monthly_charges": 50.0 + i,
            "contract_type": "monthly" if i % 2 == 0 else "annual",
            "support_tickets": i,
        }
        for i in range(10)
    ]


@pytest.fixture
def rbac_env(tmp_path, monkeypatch):
    """Set up isolated app dependencies, DB, reference CSV, and synthetic RBAC keys."""
    db_path = tmp_path / "rbac_lifecycle.db"
    migrate_audit_database_idempotent(db_path)

    record_promotion_audit(
        {
            "audit_id": "rbac-audit-1",
            "timestamp": "2026-11-10T10:00:00Z",
            "model_name": "churn-classifier",
            "approved": True,
            "reason": "candidate_meets_quality_gate",
            "candidate_f1": 0.85,
        },
        database_path=db_path,
    )
    job = create_queued_job(
        db_path,
        dataset_id="rbac-ds-1",
        trigger_source="drift_assess",
    )

    reference = pd.DataFrame(
        {
            "age": list(range(20, 40)),
            "tenure": list(range(1, 21)),
            "monthly_charges": [50.0 + i for i in range(20)],
            "contract_type": ["monthly", "annual"] * 10,
            "support_tickets": list(range(20)),
        }
    )
    ref_path = tmp_path / "rbac_reference.csv"
    reference.to_csv(ref_path, index=False)

    monkeypatch.setattr("backend.app.api.security.settings.api_keys", RBAC_TEST_KEYS)
    monkeypatch.setattr(api_module, "setup_mlflow", lambda: None)
    monkeypatch.setattr(api_module, "migrate_audit_database_idempotent", lambda _: None)
    monkeypatch.setattr(api_module.settings, "lifecycle_audit_db_path", str(db_path))
    monkeypatch.setattr(api_module.settings, "promotion_audit_db_path", str(db_path))
    monkeypatch.setattr(api_module.settings, "drift_reference_path", str(ref_path))
    monkeypatch.setattr(api_module.settings, "drift_min_samples", 10)

    # Clear cached reference state so each test reads its isolated CSV
    api_module.app.state.drift_reference = None
    api_module.app.state.drift_reference_path = None

    with patch("backend.app.api.api.load_registered_model_by_alias") as mock_load:
        mock_load.return_value.predict.return_value = [1]
        with TestClient(api_module.app) as client:
            yield {
                "client": client,
                "job_id": job["job_id"],
                "audit_id": "rbac-audit-1",
            }


def _protected_endpoints(job_id: str) -> list[tuple[str, str, dict | None, str]]:
    """Return (method, path, json_body, required_role) for all protected routes."""
    return [
        ("POST", "/predict", _sample_customer_record(), "predict"),
        (
            "POST",
            "/monitoring/drift",
            {"records": _sample_drift_records()},
            "monitoring",
        ),
        (
            "POST",
            "/monitoring/drift/assess",
            {"records": _sample_drift_records(), "dataset_id": "rbac-assess-ds"},
            "monitoring",
        ),
        ("GET", "/audit/promotions", None, "audit"),
        ("GET", "/retraining/jobs/active", None, "retraining"),
        ("GET", f"/retraining/jobs/{job_id}", None, "retraining"),
    ]


def test_require_role_unit_behavior(monkeypatch, caplog):
    """Verify require_role enforces role membership, allows admin, and sanitizes 403."""
    monkeypatch.setattr("backend.app.api.security.settings.api_keys", RBAC_TEST_KEYS)

    unit_app = FastAPI()

    @unit_app.get("/only-predict")
    def only_predict(
        client: AuthenticatedClient = Depends(require_role("predict")),
    ):
        return {"client_id": client.client_id}

    with TestClient(unit_app) as client:
        # Unauthenticated -> 401
        r_unauth = client.get("/only-predict")
        assert r_unauth.status_code == 401
        assert r_unauth.headers.get("WWW-Authenticate") == "Bearer"

        # Wrong role -> 403 sanitized
        with caplog.at_level("WARNING", logger="backend.app.api.security"):
            r_forbidden = client.get(
                "/only-predict",
                headers={"Authorization": f"Bearer {TEST_AUDIT_TOKEN}"},
            )
        assert r_forbidden.status_code == 403
        assert r_forbidden.json() == {"detail": "Insufficient permissions."}
        assert "predict" not in r_forbidden.text
        assert "audit" not in r_forbidden.text
        assert TEST_AUDIT_TOKEN not in r_forbidden.text
        assert "authz_failed" in caplog.text
        assert TEST_AUDIT_TOKEN not in caplog.text

        # Correct role -> 200
        r_ok = client.get(
            "/only-predict",
            headers={"Authorization": f"Bearer {TEST_PREDICT_TOKEN}"},
        )
        assert r_ok.status_code == 200
        assert r_ok.json() == {"client_id": "rbac-predict-client"}

        # Admin role -> 200
        r_admin = client.get(
            "/only-predict",
            headers={"X-API-Key": TEST_ADMIN_TOKEN},
        )
        assert r_admin.status_code == 200
        assert r_admin.json() == {"client_id": "rbac-admin-client"}


def test_require_role_rejects_empty_role_name():
    """Verify require_role fails fast on blank role names."""
    with pytest.raises(ValueError, match="role_name must not be empty"):
        require_role("   ")


def test_unauthenticated_requests_to_protected_endpoints_return_401(rbac_env):
    """Every protected endpoint rejects unauthenticated requests with 401."""
    client = rbac_env["client"]
    for method, path, body, _ in _protected_endpoints(rbac_env["job_id"]):
        response = client.request(method, path, json=body)
        assert response.status_code == 401, (
            f"{method} {path} returned {response.status_code}"
        )
        assert response.headers.get("WWW-Authenticate") == "Bearer"
        assert response.json() == {"detail": "Missing authentication credentials."}


def test_invalid_credentials_on_protected_endpoints_return_401(rbac_env):
    """Every protected endpoint rejects invalid credentials with sanitized 401."""
    client = rbac_env["client"]
    bogus_token = "invalid-rbac-token-canary-98765"
    for method, path, body, _ in _protected_endpoints(rbac_env["job_id"]):
        response = client.request(
            method,
            path,
            json=body,
            headers={"Authorization": f"Bearer {bogus_token}"},
        )
        assert response.status_code == 401, (
            f"{method} {path} returned {response.status_code}"
        )
        assert response.headers.get("WWW-Authenticate") == "Bearer"
        assert response.json() == {"detail": "Invalid authentication credentials."}
        assert bogus_token not in response.text


def test_role_authorization_matrix_enforced_across_all_endpoints(rbac_env):
    """Verify each role accesses only its endpoint group and admin accesses all."""
    client = rbac_env["client"]
    endpoints = _protected_endpoints(rbac_env["job_id"])

    for role_name, token in ROLE_TOKENS.items():
        for method, path, body, required_role in endpoints:
            response = client.request(
                method,
                path,
                json=body,
                headers={"Authorization": f"Bearer {token}"},
            )
            if role_name == "admin" or role_name == required_role:
                assert response.status_code == 200, (
                    f"Expected 200 for role={role_name} on {method} {path}, "
                    f"got {response.status_code}: {response.text}"
                )
            else:
                assert response.status_code == 403, (
                    f"Expected 403 for role={role_name} on {method} {path}, "
                    f"got {response.status_code}: {response.text}"
                )
                assert response.json() == {"detail": "Insufficient permissions."}
                assert token not in response.text
                assert required_role not in response.text
                assert role_name not in response.text


def test_x_api_key_accepted_for_authorized_protected_endpoints(rbac_env):
    """Verify X-API-Key header works seamlessly with RBAC on protected endpoints."""
    client = rbac_env["client"]
    for method, path, body, required_role in _protected_endpoints(rbac_env["job_id"]):
        token = ROLE_TOKENS[required_role]
        response = client.request(
            method,
            path,
            json=body,
            headers={"X-API-Key": token},
        )
        assert response.status_code == 200, f"{method} {path} failed with X-API-Key"


def test_authorized_endpoints_preserve_existing_behavior(rbac_env):
    """Verify endpoint payloads and behaviors remain unchanged after authorization."""
    client = rbac_env["client"]
    job_id = rbac_env["job_id"]

    # 1. POST /predict
    r_predict = client.post(
        "/predict",
        headers={"Authorization": f"Bearer {TEST_PREDICT_TOKEN}"},
        json=_sample_customer_record(),
    )
    assert r_predict.status_code == 200
    assert r_predict.json() == {"churn_prediction": 1}

    # 2. POST /monitoring/drift
    r_drift = client.post(
        "/monitoring/drift",
        headers={"Authorization": f"Bearer {TEST_MONITORING_TOKEN}"},
        json={"records": _sample_drift_records()},
    )
    assert r_drift.status_code == 200
    drift_data = r_drift.json()
    assert drift_data["status"] in {"stable", "drift_detected"}
    assert "features" in drift_data

    # 3. POST /monitoring/drift/assess
    r_assess = client.post(
        "/monitoring/drift/assess",
        headers={"Authorization": f"Bearer {TEST_MONITORING_TOKEN}"},
        json={"records": _sample_drift_records(), "dataset_id": "rbac-behavior-ds"},
    )
    assert r_assess.status_code == 200
    assess_data = r_assess.json()
    assert assess_data["status"] in {"stable", "queued", "active_job_exists"}
    assert "drift_report" in assess_data

    # 4. GET /audit/promotions
    r_audit = client.get(
        "/audit/promotions",
        headers={"Authorization": f"Bearer {TEST_AUDIT_TOKEN}"},
    )
    assert r_audit.status_code == 200
    audit_data = r_audit.json()
    assert len(audit_data) == 1
    assert audit_data[0]["audit_id"] == rbac_env["audit_id"]

    # 5. GET /retraining/jobs/active
    r_active = client.get(
        "/retraining/jobs/active",
        headers={"Authorization": f"Bearer {TEST_RETRAINING_TOKEN}"},
    )
    assert r_active.status_code == 200
    active_data = r_active.json()
    assert active_data["active"] is True
    assert active_data["job"]["job_id"] == job_id

    # 6. GET /retraining/jobs/{job_id}
    r_job = client.get(
        f"/retraining/jobs/{job_id}",
        headers={"Authorization": f"Bearer {TEST_RETRAINING_TOKEN}"},
    )
    assert r_job.status_code == 200
    assert r_job.json()["job_id"] == job_id
    assert r_job.json()["status"] == "queued"


@pytest.mark.parametrize(
    "public_path",
    ["/health", "/ready", "/metrics", "/metrics/prometheus"],
)
def test_public_endpoints_remain_accessible_without_credentials(rbac_env, public_path):
    """Public operational endpoints remain accessible with no credentials."""
    client = rbac_env["client"]
    response = client.get(public_path)
    assert response.status_code == 200


def test_openapi_security_schemes_and_route_requirements():
    """Verify OpenAPI advertises auth on protected routes only and leaks no secrets."""
    schema = api_module.app.openapi()
    schemes = schema.get("components", {}).get("securitySchemes", {})
    assert "HTTPBearer" in schemes
    assert "APIKeyHeader" in schemes
    assert schemes["HTTPBearer"]["type"] == "http"
    assert schemes["HTTPBearer"]["scheme"] == "bearer"
    assert schemes["APIKeyHeader"]["type"] == "apiKey"
    assert schemes["APIKeyHeader"]["in"] == "header"
    assert schemes["APIKeyHeader"]["name"] == "X-API-Key"

    paths = schema.get("paths", {})
    expected_security = [{"HTTPBearer": []}, {"APIKeyHeader": []}]

    protected_routes = [
        ("/predict", "post"),
        ("/monitoring/drift", "post"),
        ("/monitoring/drift/assess", "post"),
        ("/audit/promotions", "get"),
        ("/retraining/jobs/active", "get"),
        ("/retraining/jobs/{job_id}", "get"),
    ]
    for path, method in protected_routes:
        op = paths[path][method]
        assert op.get("security") == expected_security, (
            f"Expected security requirement on {method.upper()} {path}"
        )

    public_routes = [
        ("/health", "get"),
        ("/ready", "get"),
        ("/metrics", "get"),
    ]
    for path, method in public_routes:
        op = paths[path][method]
        assert not op.get("security"), (
            f"Public route {method.upper()} {path} must not require security"
        )

    # /metrics/prometheus is excluded from schema
    assert "/metrics/prometheus" not in paths

    # Ensure no synthetic/test/dev tokens appear in OpenAPI schema
    schema_text = json.dumps(schema)
    for token in ROLE_TOKENS.values():
        assert token not in schema_text
    assert "dev-predict-token" not in schema_text
    assert "dev-admin-token" not in schema_text


# ---------------------------------------------------------------------------
# Phase 3: Rate Limiting Unit & Integration Tests
# ---------------------------------------------------------------------------


class FakeClock:
    """Deterministic controllable clock for sliding-window rate-limiter tests."""

    def __init__(self, initial: float = 1000.0) -> None:
        self._now = float(initial)
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> float:
        with self._lock:
            self._now += seconds
            return self._now

    def set(self, timestamp: float) -> float:
        with self._lock:
            self._now = float(timestamp)
            return self._now


@pytest.fixture
def rate_limit_env(rbac_env, monkeypatch):
    """Isolate rate-limit settings and inject a deterministic FakeClock per test."""
    clock = FakeClock(initial=1000.0)
    monkeypatch.setattr(api_module.settings, "rate_limit_enabled", True)
    monkeypatch.setattr(api_module.settings, "rate_limit_requests_per_minute", 3)
    monkeypatch.setattr(api_module.settings, "rate_limit_window_seconds", 60)

    limiter = SlidingWindowRateLimiter(
        max_requests=3,
        window_seconds=60,
        clock=clock,
    )
    api_module.app.state.rate_limiter = limiter

    return {
        **rbac_env,
        "clock": clock,
        "limiter": limiter,
    }


def test_rate_limiter_rejects_invalid_constructor_args():
    """Verify SlidingWindowRateLimiter validates limits, windows, and client IDs."""
    with pytest.raises(ValueError, match="max_requests"):
        SlidingWindowRateLimiter(max_requests=0, window_seconds=60)
    with pytest.raises(ValueError, match="max_requests"):
        SlidingWindowRateLimiter(max_requests=-3, window_seconds=60)
    with pytest.raises(ValueError, match="max_requests"):
        SlidingWindowRateLimiter(max_requests=True, window_seconds=60)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="window_seconds"):
        SlidingWindowRateLimiter(max_requests=5, window_seconds=0)
    with pytest.raises(ValueError, match="window_seconds"):
        SlidingWindowRateLimiter(max_requests=5, window_seconds=-10)
    with pytest.raises(ValueError, match="window_seconds"):
        SlidingWindowRateLimiter(max_requests=5, window_seconds=float("inf"))

    with pytest.raises(ValueError, match="cleanup_interval_seconds"):
        SlidingWindowRateLimiter(
            max_requests=5,
            window_seconds=60,
            cleanup_interval_seconds=0,
        )

    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    with pytest.raises(ValueError, match="client_id"):
        limiter.check_rate_limit("   ")


def test_rate_limiter_allows_within_limit_and_blocks_exceeding():
    """Verify requests within limit succeed and the next request returns blocked."""
    clock = FakeClock(100.0)
    limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60, clock=clock)

    for expected_remaining in (2, 1, 0):
        decision = limiter.check_rate_limit("client-a")
        assert decision.allowed is True
        assert decision.retry_after_seconds == 0
        assert decision.remaining == expected_remaining
        assert decision.limit == 3

    blocked = limiter.check_rate_limit("client-a")
    assert blocked.allowed is False
    assert blocked.retry_after_seconds == 60
    assert blocked.remaining == 0
    assert blocked.limit == 3


def test_rate_limiter_retry_after_boundary_and_oldest_expiration():
    """Verify exact Retry-After calculation from oldest blocking entry & boundaries."""
    clock = FakeClock(100.0)
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60, clock=clock)

    # First accepted at t=100.0, second at t=125.0
    assert limiter.check_rate_limit("client-a").allowed is True
    clock.set(125.0)
    assert limiter.check_rate_limit("client-a").allowed is True

    # Blocked at t=130.0: oldest is t=100.0, expires at t=160.0 -> Retry-After = 30
    clock.set(130.0)
    blocked_1 = limiter.check_rate_limit("client-a")
    assert blocked_1.allowed is False
    assert blocked_1.retry_after_seconds == 30

    # Just before expiration at t=159.999 -> still blocked, Retry-After >= 1 (never 0)
    clock.set(159.999)
    blocked_2 = limiter.check_rate_limit("client-a")
    assert blocked_2.allowed is False
    assert isinstance(blocked_2.retry_after_seconds, int)
    assert blocked_2.retry_after_seconds == 1

    # Exactly at expiration t=160.0 -> oldest (100.0) expires, request succeeds!
    clock.set(160.0)
    allowed_after = limiter.check_rate_limit("client-a")
    assert allowed_after.allowed is True
    assert allowed_after.remaining == 0

    # Immediate next request at t=160.0 is blocked by t=125.0 (expires at t=185.0)
    blocked_3 = limiter.check_rate_limit("client-a")
    assert blocked_3.allowed is False
    assert blocked_3.retry_after_seconds == 25


def test_rate_limiter_prunes_expired_entries_and_cleans_inactive_clients():
    """Verify expired timestamps are pruned and inactive clients are removed."""
    clock = FakeClock(0.0)
    limiter = SlidingWindowRateLimiter(
        max_requests=3,
        window_seconds=60,
        clock=clock,
        cleanup_interval_seconds=60,
    )

    # Client A makes 2 requests at t=0 and t=10; Client B makes 1 request at t=50
    limiter.check_rate_limit("client-a")
    clock.set(10.0)
    limiter.check_rate_limit("client-a")
    assert limiter.client_request_count("client-a") == 2

    clock.set(50.0)
    limiter.check_rate_limit("client-b")
    assert limiter.tracked_client_count() == 2

    # Advance to t=70: client-a's entries (t=0, t=10) expired; client-b (t=50) active
    clock.set(70.0)
    # Periodic sweep runs on next check_rate_limit (since 70 - 0 >= 60)
    limiter.check_rate_limit("client-b")

    assert limiter.client_request_count("client-a") == 0
    assert limiter.tracked_client_count() == 1
    assert limiter.client_request_count("client-b") == 2

    # Advance past client-b's window and run explicit cleanup
    clock.set(140.0)
    removed = limiter.cleanup()
    assert removed == 1
    assert limiter.tracked_client_count() == 0


def test_rate_limiter_thread_safety_under_concurrency():
    """Concurrent requests cannot exceed the configured limit due to race conditions."""
    clock = FakeClock(500.0)
    max_requests = 5
    total_threads = 25
    limiter = SlidingWindowRateLimiter(
        max_requests=max_requests,
        window_seconds=60,
        clock=clock,
    )
    barrier = threading.Barrier(total_threads)

    def _worker() -> bool:
        barrier.wait(timeout=5.0)
        return limiter.check_rate_limit("concurrent-client").allowed

    with ThreadPoolExecutor(max_workers=total_threads) as pool:
        results = list(pool.map(lambda _: _worker(), range(total_threads)))

    assert sum(1 for allowed in results if allowed) == max_requests
    assert sum(1 for allowed in results if not allowed) == (
        total_threads - max_requests
    )
    assert limiter.client_request_count("concurrent-client") == max_requests


def test_api_requests_within_limit_succeed_and_exceeding_returns_429(rate_limit_env):
    """Requests within limit succeed; exceeding returns 429 with Retry-After."""
    client = rate_limit_env["client"]
    headers = {"Authorization": f"Bearer {TEST_PREDICT_TOKEN}"}
    payload = _sample_customer_record()

    for _ in range(3):
        resp = client.post("/predict", headers=headers, json=payload)
        assert resp.status_code == 200

    blocked = client.post("/predict", headers=headers, json=payload)
    assert blocked.status_code == 429
    assert blocked.json() == {"detail": "Rate limit exceeded. Try again later."}
    retry_after = blocked.headers.get("Retry-After")
    assert retry_after is not None
    assert retry_after.isdigit()
    assert int(retry_after) == 60


def test_api_requests_succeed_after_window_expires(rate_limit_env):
    """Requests succeed again once the oldest timestamp expires from the window."""
    client = rate_limit_env["client"]
    clock: FakeClock = rate_limit_env["clock"]
    headers = {"Authorization": f"Bearer {TEST_AUDIT_TOKEN}"}

    for i in range(3):
        clock.set(1000.0 + i * 10.0)
        resp = client.get("/audit/promotions", headers=headers)
        assert resp.status_code == 200

    # At t=1025.0, oldest request (t=1000.0) expires at t=1060.0 -> Retry-After=35
    clock.set(1025.0)
    blocked = client.get("/audit/promotions", headers=headers)
    assert blocked.status_code == 429
    assert blocked.headers.get("Retry-After") == "35"

    # At t=1059.5, still blocked with Retry-After=1
    clock.set(1059.5)
    still_blocked = client.get("/audit/promotions", headers=headers)
    assert still_blocked.status_code == 429
    assert still_blocked.headers.get("Retry-After") == "1"

    # At t=1060.0, t=1000.0 expires and request succeeds
    clock.set(1060.0)
    recovered = client.get("/audit/promotions", headers=headers)
    assert recovered.status_code == 200


def test_api_different_clients_have_independent_quotas(rate_limit_env):
    """Exhausting one client's quota does not affect another authenticated client."""
    client = rate_limit_env["client"]

    # Exhaust predict client's quota
    for _ in range(3):
        r = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {TEST_PREDICT_TOKEN}"},
            json=_sample_customer_record(),
        )
        assert r.status_code == 200

    r_blocked = client.post(
        "/predict",
        headers={"Authorization": f"Bearer {TEST_PREDICT_TOKEN}"},
        json=_sample_customer_record(),
    )
    assert r_blocked.status_code == 429

    # Audit and retraining clients still have full independent quotas
    for _ in range(3):
        r_audit = client.get(
            "/audit/promotions",
            headers={"Authorization": f"Bearer {TEST_AUDIT_TOKEN}"},
        )
        assert r_audit.status_code == 200

        r_retrain = client.get(
            "/retraining/jobs/active",
            headers={"Authorization": f"Bearer {TEST_RETRAINING_TOKEN}"},
        )
        assert r_retrain.status_code == 200


def test_api_same_client_shares_quota_across_protected_endpoints(rate_limit_env):
    """Multiple requests from the same client across endpoints share a single quota."""
    client = rate_limit_env["client"]
    headers = {"Authorization": f"Bearer {TEST_MONITORING_TOKEN}"}

    r1 = client.post(
        "/monitoring/drift",
        headers=headers,
        json={"records": _sample_drift_records()},
    )
    assert r1.status_code == 200

    r2 = client.post(
        "/monitoring/drift/assess",
        headers=headers,
        json={"records": _sample_drift_records(), "dataset_id": "rl-shared-ds-1"},
    )
    assert r2.status_code == 200

    r3 = client.post(
        "/monitoring/drift",
        headers=headers,
        json={"records": _sample_drift_records()},
    )
    assert r3.status_code == 200

    # 4th request across either monitoring endpoint hits the shared client quota
    r4 = client.post(
        "/monitoring/drift/assess",
        headers=headers,
        json={"records": _sample_drift_records(), "dataset_id": "rl-shared-ds-2"},
    )
    assert r4.status_code == 429


def test_api_rate_limiting_enforced_across_all_six_protected_routes(rate_limit_env):
    """Verify all six protected endpoints enforce per-client rate limiting."""
    client = rate_limit_env["client"]
    job_id = rate_limit_env["job_id"]
    clock: FakeClock = rate_limit_env["clock"]

    for method, path, body, role_name in _protected_endpoints(job_id):
        # Advance window cleanly for each route check
        clock.advance(120.0)
        headers = {"Authorization": f"Bearer {ROLE_TOKENS[role_name]}"}
        for _ in range(3):
            r = client.request(method, path, headers=headers, json=body)
            assert r.status_code == 200, f"{method} {path} returned {r.status_code}"

        r_exceeded = client.request(method, path, headers=headers, json=body)
        assert r_exceeded.status_code == 429, (
            f"Expected 429 on {method} {path}, got {r_exceeded.status_code}"
        )
        assert r_exceeded.json() == {"detail": "Rate limit exceeded. Try again later."}
        assert int(r_exceeded.headers["Retry-After"]) >= 1


def test_api_disabled_rate_limiting_does_not_reject(rate_limit_env, monkeypatch):
    """Verify disabling rate limiting allows requests past the configured limit."""
    monkeypatch.setattr(api_module.settings, "rate_limit_enabled", False)
    client = rate_limit_env["client"]
    headers = {"Authorization": f"Bearer {TEST_PREDICT_TOKEN}"}

    for _ in range(10):
        resp = client.post("/predict", headers=headers, json=_sample_customer_record())
        assert resp.status_code == 200


@pytest.mark.parametrize(
    "public_path",
    ["/health", "/ready", "/metrics", "/metrics/prometheus"],
)
def test_api_public_endpoints_are_never_rate_limited(rate_limit_env, public_path):
    """Verify public operational endpoints bypass rate limiting entirely."""
    client = rate_limit_env["client"]
    for _ in range(15):
        resp = client.get(public_path)
        assert resp.status_code == 200


def test_api_unauthorized_requests_do_not_consume_quotas(rate_limit_env):
    """Unauthenticated and invalid-credential requests (401) do not consume quotas."""
    client = rate_limit_env["client"]
    limiter: SlidingWindowRateLimiter = rate_limit_env["limiter"]

    # Flood with missing and invalid credentials
    for _ in range(10):
        r_missing = client.post("/predict", json=_sample_customer_record())
        assert r_missing.status_code == 401

        r_invalid = client.post(
            "/predict",
            headers={"Authorization": "Bearer bogus-token-not-valid"},
            json=_sample_customer_record(),
        )
        assert r_invalid.status_code == 401

    assert limiter.tracked_client_count() == 0

    # Valid client still gets all 3 requests
    for _ in range(3):
        r_ok = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {TEST_PREDICT_TOKEN}"},
            json=_sample_customer_record(),
        )
        assert r_ok.status_code == 200


def test_api_insufficient_role_requests_do_not_consume_quotas(rate_limit_env):
    """Authenticated clients rejected with 403 Forbidden do not consume quota."""
    client = rate_limit_env["client"]
    limiter: SlidingWindowRateLimiter = rate_limit_env["limiter"]

    # predict client tries /audit/promotions (403 Forbidden) repeatedly
    for _ in range(10):
        r_forbidden = client.get(
            "/audit/promotions",
            headers={"Authorization": f"Bearer {TEST_PREDICT_TOKEN}"},
        )
        assert r_forbidden.status_code == 403

    assert limiter.client_request_count("rbac-predict-client") == 0

    # predict client still has its full quota on /predict
    for _ in range(3):
        r_ok = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {TEST_PREDICT_TOKEN}"},
            json=_sample_customer_record(),
        )
        assert r_ok.status_code == 200


def test_api_admin_requests_use_admin_client_quota(rate_limit_env):
    """Admin requests share the admin client's quota without affecting other clients."""
    client = rate_limit_env["client"]
    limiter: SlidingWindowRateLimiter = rate_limit_env["limiter"]
    admin_headers = {"Authorization": f"Bearer {TEST_ADMIN_TOKEN}"}

    assert (
        client.post(
            "/predict",
            headers=admin_headers,
            json=_sample_customer_record(),
        ).status_code
        == 200
    )
    assert client.get("/audit/promotions", headers=admin_headers).status_code == 200
    assert (
        client.get("/retraining/jobs/active", headers=admin_headers).status_code == 200
    )

    assert limiter.client_request_count("rbac-admin-client") == 3
    assert limiter.client_request_count("rbac-predict-client") == 0

    # 4th admin request across any protected route returns 429
    r_admin_429 = client.get(
        f"/retraining/jobs/{rate_limit_env['job_id']}",
        headers=admin_headers,
    )
    assert r_admin_429.status_code == 429

    # Non-admin predict client remains unaffected
    r_predict_ok = client.post(
        "/predict",
        headers={"Authorization": f"Bearer {TEST_PREDICT_TOKEN}"},
        json=_sample_customer_record(),
    )
    assert r_predict_ok.status_code == 200


def test_api_rate_limiter_never_leaks_credentials_or_internal_state(
    rate_limit_env, caplog
):
    """429 responses and logs must never leak tokens or internal rate-limit state."""
    client = rate_limit_env["client"]
    headers = {"Authorization": f"Bearer {TEST_PREDICT_TOKEN}"}

    for _ in range(3):
        assert (
            client.post(
                "/predict",
                headers=headers,
                json=_sample_customer_record(),
            ).status_code
            == 200
        )

    with caplog.at_level("WARNING", logger="backend.app.api.security"):
        blocked = client.post(
            "/predict",
            headers=headers,
            json=_sample_customer_record(),
        )

    assert blocked.status_code == 429
    assert blocked.json() == {"detail": "Rate limit exceeded. Try again later."}
    assert TEST_PREDICT_TOKEN not in blocked.text
    assert "rbac-predict-client" not in blocked.text
    assert "deque" not in blocked.text
    assert "rate_limit_exceeded" in caplog.text
    assert TEST_PREDICT_TOKEN not in caplog.text


def test_api_concurrent_requests_respect_rate_limit(rate_limit_env):
    """Concurrent HTTP requests through FastAPI cannot bypass the rate limit."""
    client = rate_limit_env["client"]
    headers = {"Authorization": f"Bearer {TEST_AUDIT_TOKEN}"}
    total_requests = 12
    barrier = threading.Barrier(total_requests)

    def _call() -> int:
        barrier.wait(timeout=5.0)
        return client.get("/audit/promotions", headers=headers).status_code

    with ThreadPoolExecutor(max_workers=total_requests) as pool:
        statuses = list(pool.map(lambda _: _call(), range(total_requests)))

    assert statuses.count(200) == 3
    assert statuses.count(429) == 9
