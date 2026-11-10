import json
import secrets
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
