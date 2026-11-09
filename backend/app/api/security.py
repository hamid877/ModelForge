import logging
import secrets
from dataclasses import dataclass

from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKeyHeader

from backend.app.config import ClientApiKeyConfig, settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AuthenticatedClient:
    """Immutable representation of an authenticated API client."""

    client_id: str
    roles: frozenset[str]

    def has_role(self, role: str) -> bool:
        """Return True if the client has the specified role or has admin access."""
        return "admin" in self.roles or role in self.roles

    @property
    def is_admin(self) -> bool:
        """Return True if the client has the admin role."""
        return "admin" in self.roles


http_bearer = HTTPBearer(
    auto_error=False,
    scheme_name="HTTPBearer",
    description="Bearer token authentication via 'Authorization: Bearer <token>'.",
)

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    scheme_name="APIKeyHeader",
    description="API key authentication via 'X-API-Key: <token>' header.",
)


def authenticate_token(
    token: str,
    configured_keys: tuple[ClientApiKeyConfig, ...] | None = None,
) -> AuthenticatedClient | None:
    """Authenticate a token against configured credentials in constant time.

    Returns an AuthenticatedClient if valid, or None if invalid or unconfigured.
    Never logs or leaks secret values.
    """
    keys = settings.api_keys if configured_keys is None else configured_keys
    if not keys:
        return None

    clean_token = token.strip()
    if not clean_token:
        return None

    matched: ClientApiKeyConfig | None = None
    for client_config in keys:
        secret_val = client_config.token.get_secret_value()
        if secrets.compare_digest(clean_token, secret_val):
            matched = client_config

    if matched is None:
        return None

    return AuthenticatedClient(
        client_id=matched.client_id,
        roles=frozenset(matched.roles),
    )


def get_current_client(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Security(http_bearer),
    api_key: str | None = Security(api_key_header),
) -> AuthenticatedClient:
    """FastAPI dependency to authenticate the current client.

    Supports both 'Authorization: Bearer <token>' and 'X-API-Key: <token>' headers.
    Rejects missing, malformed, invalid, or conflicting credentials with HTTP 401.
    Always includes 'WWW-Authenticate: Bearer' on 401 responses.
    Never echoes supplied credentials in error messages or logs.
    """
    client_ip = request.client.host if request.client else "unknown"
    raw_auth = request.headers.get("Authorization")
    raw_api_key = request.headers.get("X-API-Key")

    if raw_auth is None and raw_api_key is None:
        logger.warning(
            "auth_failed reason=missing_credentials client_ip=%s path=%s",
            client_ip,
            request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    bearer_token: str | None = None
    if raw_auth is not None:
        if bearer is None or not bearer.credentials.strip():
            logger.warning(
                "auth_failed reason=malformed_auth_header client_ip=%s path=%s",
                client_ip,
                request.url.path,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        bearer_token = bearer.credentials.strip()

    api_key_token: str | None = None
    if raw_api_key is not None:
        clean_key = raw_api_key.strip()
        if not clean_key:
            logger.warning(
                "auth_failed reason=empty_api_key_header client_ip=%s path=%s",
                client_ip,
                request.url.path,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        api_key_token = clean_key

    if bearer_token is not None and api_key_token is not None:
        if not secrets.compare_digest(bearer_token, api_key_token):
            logger.warning(
                "auth_failed reason=conflicting_credentials client_ip=%s path=%s",
                client_ip,
                request.url.path,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Conflicting authentication credentials supplied.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = bearer_token
    else:
        token = bearer_token if bearer_token is not None else api_key_token

    assert token is not None
    authenticated_client = authenticate_token(token)
    if authenticated_client is None:
        logger.warning(
            "auth_failed reason=invalid_credentials client_ip=%s path=%s",
            client_ip,
            request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info(
        "auth_succeeded client_id=%s client_ip=%s path=%s",
        authenticated_client.client_id,
        client_ip,
        request.url.path,
    )
    return authenticated_client
