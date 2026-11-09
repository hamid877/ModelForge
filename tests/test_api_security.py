import secrets
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from backend.app.api.security import (
    AuthenticatedClient,
    authenticate_token,
    get_current_client,
)
from backend.app.config import ClientApiKeyConfig, Settings

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
