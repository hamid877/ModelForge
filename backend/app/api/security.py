import logging
import math
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security, status
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


def require_role(role_name: str) -> Callable[..., AuthenticatedClient]:
    """Return a FastAPI dependency enforcing the required role or admin access."""
    clean_role = role_name.strip()
    if not clean_role:
        raise ValueError("role_name must not be empty.")

    def _require_role(
        request: Request,
        client: Annotated[AuthenticatedClient, Depends(get_current_client)],
    ) -> AuthenticatedClient:
        if not client.has_role(clean_role):
            client_ip = request.client.host if request.client else "unknown"
            logger.warning(
                "authz_failed client_id=%s required_role=%s client_ip=%s path=%s",
                client.client_id,
                clean_role,
                client_ip,
                request.url.path,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions.",
            )
        return client

    return _require_role


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Result of evaluating a client's sliding-window rate-limit quota."""

    allowed: bool
    retry_after_seconds: int
    remaining: int
    limit: int


class SlidingWindowRateLimiter:
    """Thread-safe, process-local per-client sliding-window rate limiter.

    Tracks accepted request timestamps per authenticated ``client_id`` in
    ``collections.deque`` instances protected by a ``threading.Lock``.
    Inactive client entries are pruned on access and during periodic sweeps
    to prevent unbounded memory growth.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        cleanup_interval_seconds: float | None = None,
    ) -> None:
        if (
            isinstance(max_requests, bool)
            or not isinstance(max_requests, int)
            or max_requests < 1
        ):
            raise ValueError("max_requests must be a positive integer.")
        if (
            isinstance(window_seconds, bool)
            or not isinstance(window_seconds, (int, float))
            or not math.isfinite(window_seconds)
            or window_seconds <= 0
        ):
            raise ValueError("window_seconds must be a finite positive number.")
        if cleanup_interval_seconds is not None and (
            isinstance(cleanup_interval_seconds, bool)
            or not isinstance(cleanup_interval_seconds, (int, float))
            or not math.isfinite(cleanup_interval_seconds)
            or cleanup_interval_seconds <= 0
        ):
            raise ValueError(
                "cleanup_interval_seconds must be a finite positive number."
            )

        self._max_requests = max_requests
        self._window_seconds = float(window_seconds)
        self._clock = clock
        self._cleanup_interval = (
            float(cleanup_interval_seconds)
            if cleanup_interval_seconds is not None
            else float(window_seconds)
        )
        self._lock = threading.Lock()
        self._clients: dict[str, deque[float]] = {}
        self._last_cleanup: float | None = None

    @property
    def max_requests(self) -> int:
        """Maximum accepted requests allowed within a single sliding window."""
        return self._max_requests

    @property
    def window_seconds(self) -> float:
        """Sliding window duration in seconds."""
        return self._window_seconds

    def _prune_expired_locked(self, timestamps: deque[float], now: float) -> None:
        while timestamps and timestamps[0] + self._window_seconds <= now:
            timestamps.popleft()

    def _cleanup_inactive_locked(self, now: float) -> int:
        stale_clients: list[str] = []
        for client_id, timestamps in self._clients.items():
            self._prune_expired_locked(timestamps, now)
            if not timestamps:
                stale_clients.append(client_id)

        for client_id in stale_clients:
            del self._clients[client_id]

        self._last_cleanup = now
        return len(stale_clients)

    def check_rate_limit(self, client_id: str) -> RateLimitDecision:
        """Evaluate and record a request for ``client_id`` under the sliding window.

        Rejected requests do not consume quota. Returns a ``RateLimitDecision``
        with a positive integer ``retry_after_seconds`` when blocked.
        """
        clean_client_id = client_id.strip() if isinstance(client_id, str) else ""
        if not clean_client_id:
            raise ValueError("client_id must not be empty.")

        with self._lock:
            now = self._clock()
            if self._last_cleanup is None:
                self._last_cleanup = now
            elif now - self._last_cleanup >= self._cleanup_interval:
                self._cleanup_inactive_locked(now)

            timestamps = self._clients.get(clean_client_id)
            if timestamps is not None:
                self._prune_expired_locked(timestamps, now)
                if not timestamps:
                    del self._clients[clean_client_id]
                    timestamps = None

            if timestamps is not None and len(timestamps) >= self._max_requests:
                oldest_blocking = timestamps[len(timestamps) - self._max_requests]
                remaining_seconds = (oldest_blocking + self._window_seconds) - now
                retry_after = max(1, math.ceil(remaining_seconds))
                return RateLimitDecision(
                    allowed=False,
                    retry_after_seconds=retry_after,
                    remaining=0,
                    limit=self._max_requests,
                )

            if timestamps is None:
                timestamps = deque()
                self._clients[clean_client_id] = timestamps

            timestamps.append(now)
            return RateLimitDecision(
                allowed=True,
                retry_after_seconds=0,
                remaining=self._max_requests - len(timestamps),
                limit=self._max_requests,
            )

    def cleanup(self) -> int:
        """Prune expired entries across all tracked clients and remove empty ones."""
        with self._lock:
            return self._cleanup_inactive_locked(self._clock())

    def tracked_client_count(self) -> int:
        """Return the number of client entries currently held in memory."""
        with self._lock:
            return len(self._clients)

    def client_request_count(self, client_id: str) -> int:
        """Return the number of stored timestamps for ``client_id``."""
        with self._lock:
            timestamps = self._clients.get(client_id)
            return len(timestamps) if timestamps is not None else 0


_rate_limiter_init_lock = threading.Lock()


def get_rate_limiter(request: Request) -> SlidingWindowRateLimiter:
    """Return the process-local rate limiter attached to the FastAPI application."""
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is not None:
        return limiter

    with _rate_limiter_init_lock:
        limiter = getattr(request.app.state, "rate_limiter", None)
        if limiter is None:
            limiter = SlidingWindowRateLimiter(
                max_requests=settings.rate_limit_requests_per_minute,
                window_seconds=settings.rate_limit_window_seconds,
            )
            request.app.state.rate_limiter = limiter
        return limiter


def enforce_rate_limit(
    request: Request,
    client: Annotated[AuthenticatedClient, Depends(get_current_client)],
) -> AuthenticatedClient:
    """FastAPI dependency enforcing per-client sliding-window rate limits.

    Intended to execute after authentication and RBAC role enforcement so that
    unauthenticated (401) and unauthorized (403) requests never consume quota.
    """
    if not settings.rate_limit_enabled:
        return client

    limiter = get_rate_limiter(request)
    decision = limiter.check_rate_limit(client.client_id)
    if not decision.allowed:
        client_ip = request.client.host if request.client else "unknown"
        logger.warning(
            "rate_limit_exceeded client_id=%s client_ip=%s path=%s retry_after=%s",
            client.client_id,
            client_ip,
            request.url.path,
            decision.retry_after_seconds,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": str(decision.retry_after_seconds)},
        )

    return client
