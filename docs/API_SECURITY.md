# ModelForge API Security

This document describes the authentication and role-based access control (RBAC) mechanisms protecting the ModelForge FastAPI service (`backend/app/api/api.py`, `backend/app/api/security.py`, and `backend/app/config.py`).

> **Scope Note**: Milestone 10 Phase 1 (Authentication Foundation) and Phase 2 (RBAC and Endpoint Protection) are implemented. Phase 3 (Rate Limiting) is not yet implemented.

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

The following operational health and observability endpoints remain public and require no credentials:

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /metrics/prometheus` (excluded from the OpenAPI schema via `include_in_schema=False`)

