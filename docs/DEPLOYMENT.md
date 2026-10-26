# ModelForge — Deployment Guide

> **Status: local / development-class deployment only.**  
> ModelForge has been validated on a single-machine Docker Compose stack.  
> It has **not** been hardened, load-tested, or audited for production internet
> exposure. Do not serve it publicly without an independent security review.

---

## Contents

1. [Service Architecture](#1-service-architecture)
2. [Docker Networking and Port Binding](#2-docker-networking-and-port-binding)
3. [Environment Variables and Secrets](#3-environment-variables-and-secrets)
4. [Persistent Volumes](#4-persistent-volumes)
5. [MLflow Artifacts and Model Registry](#5-mlflow-artifacts-and-model-registry)
6. [Deployment Procedure](#6-deployment-procedure)
7. [Security Considerations](#7-security-considerations)
8. [Known Limitations](#8-known-limitations)

---

## 1. Service Architecture

ModelForge is a four-service Docker Compose stack. All services are defined in
`docker-compose.yml` at the repository root.

```
┌──────────────────────────────────────────────────────────────┐
│  Host machine                                                │
│                                                              │
│  127.0.0.1:5001 ──► mlflow (modelforge-api:dev)             │
│                        │  SQLite DB: ./mlflow-data/mlflow.db │
│                        │  Artifact store: ./mlruns/          │
│                                                              │
│  127.0.0.1:8000 ──► api (modelforge-api:dev)                │
│                        │  reads ./mlruns/ (artifact store)   │
│                        │  reads ./data/   (CSV reference)    │
│                                                              │
│  127.0.0.1:9090 ──► prometheus (prom/prometheus:v3.5.0)     │
│                        │  scrapes api:8000/metrics/prometheus│
│                        │  volume: prometheus_data (named)    │
│                                                              │
│  127.0.0.1:3000 ──► grafana (grafana/grafana:12.2.0)        │
│                        │  reads prometheus:9090              │
│                        │  volume: grafana_data (named)       │
└──────────────────────────────────────────────────────────────┘
```

### Service images

| Service | Image | Source |
|---|---|---|
| `mlflow` | `modelforge-api:dev` | Built locally from `Dockerfile` |
| `api` | `modelforge-api:dev` | Same image; different `command` entrypoint |
| `prometheus` | `prom/prometheus:v3.5.0` | Docker Hub upstream |
| `grafana` | `grafana/grafana:12.2.0` | Docker Hub upstream |

Both `mlflow` and `api` use the same image. The MLflow server is launched by
overriding `command:` in Compose; the API uses the image's default `CMD`.

### Startup dependency chain

Compose enforces a strict health-check dependency order:

```
mlflow (healthy) → api (healthy) → prometheus (healthy) → grafana
```

If any service fails its health check, downstream services do not start.

### Container user

Both `mlflow` and `api` containers run as UID/GID `1000:1000` (the `modelforge`
system user created in the Dockerfile). Ensure that host directories mounted
into these containers (`./mlflow-data/`, `./mlruns/`, `./data/`) are readable
and writable by UID 1000.

---

## 2. Docker Networking and Port Binding

### Internal (container-to-container) networking

Docker Compose creates an implicit bridge network. Services communicate using
their Compose service names as hostnames:

| From | To | Address used |
|---|---|---|
| `api` | `mlflow` | `http://mlflow:5000` |
| `prometheus` | `api` | `http://api:8000` |
| `grafana` | `prometheus` | `http://prometheus:9090` |

### Host port bindings

All host ports are explicitly bound to the loopback address `127.0.0.1`. They
are **not** accessible from other network interfaces.

| Host binding | Container port | Service |
|---|---|---|
| `127.0.0.1:5001` | `5000` | MLflow tracking UI |
| `127.0.0.1:8000` | `8000` | FastAPI prediction API |
| `127.0.0.1:9090` | `9090` | Prometheus |
| `127.0.0.1:3000` | `3000` | Grafana |

> If you need to expose a service beyond `localhost`, you must modify the
> `ports:` binding in `docker-compose.yml` and apply the security hardening
> described in [Section 7](#7-security-considerations) first.

### MLflow `--allowed-hosts`

The MLflow container is started with the following `--allowed-hosts` argument:

```
mlflow:5000,localhost:5001,localhost:5000,127.0.0.1:*
```

This restricts which `Host` headers the MLflow server will accept, mitigating
host-header injection. Adjust this value if you change the port mapping.

---

## 3. Environment Variables and Secrets

### `.env` file (Grafana credentials only)

The stack reads a `.env` file at the repository root. Only two variables are
required:

| Variable | Description |
|---|---|
| `GRAFANA_ADMIN_USER` | Grafana administrator username |
| `GRAFANA_ADMIN_PASSWORD` | Grafana administrator password |

Docker Compose will **abort startup with an error** if either variable is unset
or the `.env` file is missing. The error message will identify the unset
variable.

Do not commit `.env`. Copy `.env.example` instead:

```bash
cp .env.example .env
# then edit .env
```

### API environment variables

The `api` service receives the following environment variables directly from
`docker-compose.yml`:

| Variable | Value in Compose | Default in `config.py` | Description |
|---|---|---|---|
| `MODELFORGE_MODEL_NAME` | `churn-classifier` | `churn-classifier` | MLflow registered model name to load |
| `MODELFORGE_MODEL_ALIAS` | `champion` | `champion` | Model alias to resolve at startup |
| `MODELFORGE_MLFLOW_TRACKING_URI` | `http://mlflow:5000` | `http://127.0.0.1:5001` | MLflow server URL (uses internal Docker hostname) |

### Full settings reference (`config.py`)

All settings use the `MODELFORGE_` prefix and are loaded by `pydantic-settings`:

| Setting | Env var | Default | Description |
|---|---|---|---|
| `model_name` | `MODELFORGE_MODEL_NAME` | `churn-classifier` | Registered model name |
| `model_alias` | `MODELFORGE_MODEL_ALIAS` | `champion` | Alias resolved at startup |
| `mlflow_tracking_uri` | `MODELFORGE_MLFLOW_TRACKING_URI` | `http://127.0.0.1:5001` | MLflow server URI |
| `promotion_minimum_f1` | `MODELFORGE_PROMOTION_MINIMUM_F1` | `0.70` | Minimum F1 score for promotion |
| `drift_reference_path` | `MODELFORGE_DRIFT_REFERENCE_PATH` | `data/raw/churn.csv` | Reference CSV for drift detection |
| `drift_min_samples` | `MODELFORGE_DRIFT_MIN_SAMPLES` | `10` | Minimum batch size for drift detection |
| `drift_p_value_threshold` | `MODELFORGE_DRIFT_P_VALUE_THRESHOLD` | `0.05` | p-value threshold for KS / chi-squared tests |
| `lifecycle_audit_db_path` | `MODELFORGE_LIFECYCLE_AUDIT_DB_PATH` | `data/lifecycle_audit.db` | SQLite path for lifecycle audit events |

> **Changing `promotion_minimum_f1`**: This threshold is enforced by the
> bootstrap and promotion code. Lowering it reduces quality assurance; raising
> it may cause bootstrap to fail on the provided dataset if the model cannot
> exceed the new threshold.

---

## 4. Persistent Volumes

ModelForge uses a mix of **bind mounts** (host paths) and **named volumes**
(Docker-managed).

### Bind mounts

| Host path | Service | Content | Gitignored? |
|---|---|---|---|
| `./mlflow-data/` | `mlflow` | SQLite database (`mlflow.db`) | Yes |
| `./mlruns/` | `mlflow`, `api` | MLflow artifact store | Yes |
| `./data/` | `api` | Training data and audit DB | Partially* |
| `./monitoring/prometheus/prometheus.yml` | `prometheus` | Scrape config (read-only) | No |
| `./monitoring/grafana/provisioning/` | `grafana` | Dashboard provisioning (read-only) | No |
| `./monitoring/grafana/dashboards/` | `grafana` | Dashboard JSON files (read-only) | No |

\* `data/raw/churn.csv` is tracked in Git; `data/lifecycle_audit.db` and
`data/processed/` are gitignored.

### Named volumes

| Volume | Service | Content | Survives `down`? | Removed by `down -v`? |
|---|---|---|---|---|
| `prometheus_data` | `prometheus` | TSDB time-series data | Yes | Yes |
| `grafana_data` | `grafana` | Grafana state, user settings | Yes | Yes |

### Data that must not be committed

The following paths are listed in `.gitignore` and must never be committed:

- `mlflow.db` (root-level SQLite, legacy)
- `mlruns/` — MLflow artifact store
- `.docker-data/`
- `data/raw/*` (except `churn.csv`)
- `data/lifecycle_audit.db`
- `models/*.joblib`, `models/*.pkl`
- `backups/`
- `.env`, `.env.*`

---

## 5. MLflow Artifacts and Model Registry

### Backend store

MLflow uses a SQLite backend store at `/mlflow-data/mlflow.db` inside the
container, mounted from `./mlflow-data/mlflow.db` on the host. This stores all
run metadata, parameters, metrics, and registry entries.

SQLite is appropriate for local development and single-node deployments. It is
not suited to concurrent multi-process writes. For multi-user or high-throughput
deployments a PostgreSQL backend is recommended, but this is not currently
configured.

### Artifact store

MLflow uses `/mlruns/` inside the container as the artifact destination
(`--artifacts-destination /mlruns`). The `--serve-artifacts` flag means the
MLflow server proxies artifact downloads, so clients only need the tracking URI.

The `api` container also mounts `./mlruns/` to support loading model artifacts
directly via the `mlflow.sklearn.load_model()` path resolver.

### Registered model and alias scheme

| Name | Value |
|---|---|
| Registered model | `churn-classifier` |
| Champion alias | `champion` |
| Model URI (used by API) | `models:/churn-classifier@champion` |

The API resolves this URI at startup via `load_registered_model_by_alias()`.
If the alias does not exist, startup fails and `/ready` returns `503`.

### Experiment

Training runs are logged under the experiment named `"ModelForge Churn Prediction"`.
This experiment is created automatically by `setup_mlflow()` on first use.

---

## 6. Deployment Procedure

This is the canonical procedure for deploying to a fresh host or after a clean
repository clone.

### Step 1 — Obtain the code

```bash
git clone https://github.com/hamid877/ModelForge.git
cd ModelForge
```

### Step 2 — Configure environment

```bash
cp .env.example .env
# Edit .env: set GRAFANA_ADMIN_USER and GRAFANA_ADMIN_PASSWORD
```

### Step 3 — Build the application image

```bash
docker build -t modelforge-api:dev .
```

### Step 4 — Install Python dependencies (for bootstrap and tests only)

```bash
uv sync
```

### Step 5 — Start MLflow

```bash
docker compose up -d mlflow
# Wait for: Status = healthy
docker compose ps mlflow
```

### Step 6 — Bootstrap the champion model

```bash
make bootstrap
```

Verify the champion alias was assigned:

```bash
# MLflow UI: http://localhost:5001 → Models → churn-classifier → Aliases
# Or inspect directly:
docker compose exec mlflow python -c "
from mlflow import MlflowClient
c = MlflowClient('http://localhost:5000')
v = c.get_model_version_by_alias('churn-classifier', 'champion')
print('Champion version:', v.version)
"
```

### Step 7 — Start the remaining services

```bash
make up
```

### Step 8 — Verify health

```bash
curl http://127.0.0.1:8000/health   # {"status":"healthy"}
curl http://127.0.0.1:8000/ready    # {"status":"ready"}
```

### Updating the deployment

After a code change:

1. Rebuild the image: `docker build -t modelforge-api:dev .`
2. Restart affected services: `docker compose up -d --no-deps api` (or `mlflow`)
3. Verify: `curl http://127.0.0.1:8000/ready`

---

## 7. Security Considerations

The following risks are present in the current configuration. Addressing them
requires changes outside the scope of these milestones.

### All ports are loopback-only

All four services are bound to `127.0.0.1`. Do not change the port bindings to
`0.0.0.0` without placing an authenticated reverse proxy (e.g. nginx + TLS +
HTTP Basic/OAuth) in front of every service.

### No authentication on the API

The FastAPI application (`/predict`, `/monitoring/drift`, `/metrics`) has no
authentication layer. Any process that can reach port 8000 can submit predictions
and trigger drift analysis.

### No authentication on Prometheus

Prometheus (`/`, `/metrics`, `/api/v1/query`) has no access control. It is
loopback-only, but there is no additional protection.

### No TLS

All inter-service communication is plain HTTP, including the MLflow tracking
connection from the API container to the MLflow container. This is acceptable on
a loopback-only single-host deployment but is inappropriate for any network path
that traverses shared infrastructure.

### Grafana credentials in `.env`

Grafana admin credentials are stored in `.env` on disk. For any shared or
persistent environment, store secrets in a secrets manager and inject them at
runtime rather than via a plaintext file.

### SQLite is not suitable for multi-process MLflow

The SQLite backend does not support concurrent writers. Running more than one
MLflow server process against the same `mlflow.db` will cause database
corruption.

### Container user is UID 1000

Both `mlflow` and `api` containers run as UID 1000. If the host running Docker
has a different UID for the owner of the bind-mount directories, volume
permission errors will occur.

---

## 8. Known Limitations

| Limitation | Impact | Workaround / Path forward |
|---|---|---|
| **SQLite backend** | Not suitable for concurrent MLflow writes or multi-node deployments | Migrate to PostgreSQL backend |
| **Single-node stack** | No horizontal scaling, no load balancing | Use Kubernetes + multi-replica deployment |
| **No TLS** | Data in transit is unencrypted | Add a TLS-terminating reverse proxy |
| **No API authentication** | All endpoints are open | Add authentication middleware |
| **Loopback-only binding** | Services not reachable remotely | Intentional for local dev; requires proxy for remote access |
| **`modelforge-api:dev` image** | Must be built locally; not published to a registry | Publish to a container registry (e.g. GHCR) and update Compose |
| **`StarletteDeprecationWarning` in tests** | Test suite emits one warning | Upgrade to `httpx2` (Milestone 5) |
| **No secrets management** | Credentials are in `.env` on disk | Integrate Docker Secrets or a vault |
| **Prometheus metrics are in-process** | Metrics reset on each API restart | Accept limitation for current scale |
| **No retraining trigger** | Model is not automatically retrained when drift is detected | Drift endpoint reports drift; retraining must be triggered manually |
