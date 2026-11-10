# ModelForge API Security

This document describes the authentication, role-based access control (RBAC), and rate-limiting mechanisms protecting the ModelForge FastAPI service (`backend/app/api/api.py`, `backend/app/api/security.py`, and `backend/app/config.py`).

> **Scope Note**: Milestone 10 Phase 1 (Authentication Foundation), Phase 2 (RBAC and Endpoint Protection), and Phase 3 (Rate Limiting) are implemented. Phase 4 (Security Hardening and Final Acceptance Verification) is pending.

---

## Authentication

Protected endpoints authenticate clients via `get_current_client` in `backend/app/api/security.py`:

- **Supported Headers**:
  - `Authorization: Bearer <token>` (`HTTPBearer` OpenAPI scheme)
  - `X-API-Key: <token>` (`APIKeyHeader` OpenAPI scheme)
- **Dual-Header Handling**: Supplying both headers with identical tokens is permitted; supplying conflicting tokens is rejected with HTTP `401 Unauthorized`.
- **Credential Validation**: Tokens are matched against `settings.api_keys` (`MODELFORGE_API_KEYS`) using constant-time comparison (`secrets.compare_digest`). No default credentials exist; unconfigured `MODELFORGE_API_KEYS` fails closed.
- **Error Sanitization**: Authentication failures return HTTP `401 Unauthorized` with `WWW-Authenticate: Bearer` and sanitized detail messages without echoing supplied tokens or internal configuration.

---

## Role-Based Access Control (RBAC)

Authorization is enforced per route using `require_role(role_name: str)` in `backend/app/api/security.py`, which depends on `get_current_client` and evaluates `AuthenticatedClient.has_role(role_name)`.

### Defined Roles

Configured client roles in `backend/app/config.py` (`ALLOWED_API_ROLES`) are restricted to:

- `predict`
- `monitoring`
- `audit`
- `retraining`
- `admin`

### Admin Wildcard Behavior

Clients assigned the `admin` role satisfy every `require_role(...)` check (`"admin" in self.roles or role in self.roles` in `AuthenticatedClient.has_role`) and are permitted across all protected endpoint groups.

### Protected Endpoints

| Endpoint | Required Role | Access Behavior |
|---|---|---|
| `POST /predict` | `predict` | Requires `predict` or `admin` |
| `POST /monitoring/drift` | `monitoring` | Requires `monitoring` or `admin` |
| `POST /monitoring/drift/assess` | `monitoring` | Requires `monitoring` or `admin` |
| `GET /audit/promotions` | `audit` | Requires `audit` or `admin` |
| `GET /retraining/jobs/active` | `retraining` | Requires `retraining` or `admin` |
| `GET /retraining/jobs/{job_id}` | `retraining` | Requires `retraining` or `admin` |

Authenticated clients lacking the required role (and without `admin`) receive HTTP `403 Forbidden` with the sanitized body `{"detail": "Insufficient permissions."}`.

### Public Endpoints

The following operational health and observability endpoints remain public, require no credentials, and are completely exempt from rate limiting:

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /metrics/prometheus` (excluded from the OpenAPI schema via `include_in_schema=False`)

---

## Rate Limiting

All six protected endpoints enforce per-client rate limiting via `enforce_rate_limit` and `SlidingWindowRateLimiter` in `backend/app/api/security.py`. Rate limiting executes **after** authentication (`get_current_client`) and authorization (`require_role`), ensuring that unauthenticated (`401`) and unauthorized (`403`) requests never consume an authenticated client's quota.

### Sliding-Window Algorithm

- **Data Structure**: Each active `client_id` maps to a `collections.deque` storing monotonic timestamps of accepted requests.
- **Synchronization**: Shared in-memory state is protected by a `threading.Lock` acquired only for the fast in-memory check-and-record step; the lock is released before executing any endpoint handler.
- **Expiration & Cleanup**: Expired timestamps (`timestamp + window_seconds <= now`) are pruned on each request for that client, and periodic sweeps remove inactive client entries with empty deques so memory usage remains bounded.
- **Rejected Requests**: Requests denied with HTTP `429` are not recorded in the client's window and therefore do not extend the block duration.

### Configuration & Defaults

Rate limiting is configured via Pydantic `Settings` (`backend/app/config.py`):

| Environment Variable | Default | Validation | Description |
|---|---|---|---|
| `MODELFORGE_RATE_LIMIT_ENABLED` | `true` | Boolean | Enables or explicitly disables rate-limit enforcement. |
| `MODELFORGE_RATE_LIMIT_REQUESTS_PER_MINUTE` | `60` | Integer `>= 1` | Maximum accepted requests per client within the configured sliding window. |
| `MODELFORGE_RATE_LIMIT_WINDOW_SECONDS` | `60` | Integer `>= 1` | Sliding window duration in seconds. |

Invalid values (such as `0`, negative numbers, or non-numeric strings) fail fast with a Pydantic `ValidationError` and never silently disable rate limiting. Rate limiting may only be disabled explicitly by setting `MODELFORGE_RATE_LIMIT_ENABLED=false` for controlled local testing or development.

### Per-Client Quota Behavior

- **Quota Key**: Quotas are keyed by authenticated `client_id` rather than client IP address.
- **Independent Quotas**: Different `client_id` values maintain independent sliding-window quotas.
- **Shared Across Routes**: Multiple requests from the same `client_id` across any protected endpoints share that client's single quota.
- **Admin Quota**: Requests authenticated with an `admin`-role client consume that admin client's own `client_id` quota.
- **Public Exemptions**: `/health`, `/ready`, `/metrics`, and `/metrics/prometheus` are never rate-limited.

### HTTP 429 Response & `Retry-After` Semantics

When a client exceeds its configured quota, the API returns HTTP `429 Too Many Requests` with a sanitized JSON body and an integer `Retry-After` header:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 42
Content-Type: application/json

{"detail": "Rate limit exceeded. Try again later."}
```

- `Retry-After` is always a positive integer (`>= 1`) representing the number of seconds until the oldest blocking request in the window expires (`ceil((oldest_blocking_timestamp + window_seconds) - now)`).
- No internal window state, remaining counts, or client credentials are exposed in the error response.

### Process-Local Limitation & Multi-Worker Implications

- **Process-Local State**: `SlidingWindowRateLimiter` stores request timestamps in process memory on `app.state.rate_limiter` (with no Redis or SQLite writes).
- **Single-Worker Accuracy**: In the default single-process local development setup (`uvicorn ... --workers 1`), limits are enforced consistently across all incoming requests.
- **Multi-Worker / Distributed Deployments**: Increasing the Uvicorn worker count (`--workers N`) or running multiple API replicas creates independent in-memory rate-limiter instances per process. Under round-robin or load-balanced traffic, a client can effectively send up to `N * rate_limit_requests_per_minute` requests across the cluster. Globally consistent rate limiting across multiple workers or replicas requires a shared external store (such as Redis).

