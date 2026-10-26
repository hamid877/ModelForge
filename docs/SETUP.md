# ModelForge — Setup Guide

> **Scope: local development only.**  
> This guide configures ModelForge on a single developer workstation.  
> Production deployment considerations are in [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone and Environment Setup](#2-clone-and-environment-setup)
3. [Install Python Dependencies](#3-install-python-dependencies)
4. [Build the Docker Image](#4-build-the-docker-image)
5. [Start MLflow](#5-start-mlflow)
6. [Bootstrap the Champion Model](#6-bootstrap-the-champion-model)
7. [Launch the Full Stack](#7-launch-the-full-stack)
8. [Health and Readiness Checks](#8-health-and-readiness-checks)
9. [Running Tests and Linters](#9-running-tests-and-linters)
10. [Shutdown](#10-shutdown)

---

## 1. Prerequisites

| Requirement | Minimum version | Notes |
|---|---|---|
| Python | 3.12 | Pinned in `.python-version` and `pyproject.toml` |
| [`uv`](https://docs.astral.sh/uv/) | latest stable | Manages the virtual environment and lockfile |
| Docker Engine | 24+ | Runs all services via Compose |
| Docker Compose | v2 (`docker compose`) | V2 syntax is used throughout; `docker-compose` (V1) is **not** supported |

Verify your setup:

```bash
python --version          # must be >= 3.12
uv --version
docker compose version    # must start with "Docker Compose version v2"
```

---

## 2. Clone and Environment Setup

```bash
git clone https://github.com/hamid877/ModelForge.git
cd ModelForge
```

Copy the environment template and set your Grafana credentials:

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder password:

```
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=replace-with-a-strong-unique-password
```

> **Never commit `.env`.**  
> It is listed in `.gitignore` and must stay local.  
> Only `.env.example` (which contains no real credentials) is tracked in Git.

### What `.env` controls

The only variables read from `.env` at runtime are the two Grafana credentials
injected by Docker Compose:

| Variable | Used by | Compose ref |
|---|---|---|
| `GRAFANA_ADMIN_USER` | Grafana | `GF_SECURITY_ADMIN_USER` |
| `GRAFANA_ADMIN_PASSWORD` | Grafana | `GF_SECURITY_ADMIN_PASSWORD` |

All other ModelForge settings are either built into the image or passed directly
in `docker-compose.yml` as `environment:` keys. See [DEPLOYMENT.md](DEPLOYMENT.md)
for the full environment-variable reference.

---

## 3. Install Python Dependencies

`uv sync` installs both production and development dependencies (pytest, ruff, httpx)
into a local `.venv`:

```bash
uv sync
```

This command is idempotent. The lockfile (`uv.lock`) guarantees reproducible
installs across machines.

> The virtual environment is created at `.venv/` and is **not** used inside
> Docker containers. Container builds run `uv sync --frozen --no-dev` inside the
> image layer.

---

## 4. Build the Docker Image

The `mlflow` and `api` services both use the same image tag `modelforge-api:dev`,
which you must build locally before running any service:

```bash
docker build -t modelforge-api:dev .
```

`prometheus` and `grafana` use upstream images pulled automatically by Docker
Compose (`prom/prometheus:v3.5.0`, `grafana/grafana:12.2.0`).

---

## 5. Start MLflow

The API and bootstrap script both require MLflow to be reachable before they run.
Start the MLflow service in isolation first:

```bash
docker compose up -d mlflow
```

Wait for the health check to pass (up to ~2 minutes on a fresh machine):

```bash
docker compose ps mlflow
# Expected: Status = healthy
```

You can also verify directly:

```bash
curl http://127.0.0.1:5001/health
```

> **Port mapping**: MLflow listens inside the container on port `5000`.  
> It is exposed on the host as `127.0.0.1:5001` (loopback-bound).
> The MLflow tracking UI is at http://localhost:5001.

### MLflow storage

MLflow uses two host-mounted directories:

| Host path | Container path | Content |
|---|---|---|
| `./mlflow-data/` | `/mlflow-data/` | SQLite database (`mlflow.db`) |
| `./mlruns/` | `/mlruns/` | Artifact store (logged model artifacts) |

Both directories are created automatically on first run and are listed in
`.gitignore` — do not commit them.

---

## 6. Bootstrap the Champion Model

The API startup sequence loads the champion model from the MLflow registry.
**If no `champion` alias exists, the API will fail its readiness check** and
return `503 not_ready`. The bootstrap workflow initialises this alias
idempotently.

### What bootstrap does

1. Connects to MLflow and checks whether the `champion` alias already exists on
   the `churn-classifier` registered model.
2. If the alias exists and `--force` is not passed, it exits immediately with
   `status=already_bootstrapped` (safe to re-run).
3. Otherwise it loads `data/raw/churn.csv`, trains a logistic-regression
   pipeline, and runs quality-gate evaluation:
   - Candidate F1 must be >= 0.70 (`promotion_minimum_f1`).
   - Candidate F1 must be >= the existing champion's F1 (if one exists).
4. If the candidate passes, it registers the model as `churn-classifier` and
   assigns the `champion` alias.

### Running bootstrap

```bash
# Recommended: via Makefile
make bootstrap

# Equivalent direct invocation
uv run python scripts/bootstrap.py
```

#### Bootstrap CLI flags

| Flag | Default | Description |
|---|---|---|
| `--data-path PATH` | `data/raw/churn.csv` | Training data CSV |
| `--model-path PATH` | `models/churn_model.joblib` | Local artifact path for the serialised model |
| `--model-name NAME` | `churn-classifier` (from settings) | MLflow registered model name |
| `--alias ALIAS` | `champion` (from settings) | MLflow model alias to assign |
| `--force` | off | Re-train and promote even when a champion already exists (quality gate still applies) |

#### Expected output — first run

```
INFO: Bootstrapping model model_name=churn-classifier alias=champion force=False
Bootstrap status: bootstrapped
Successfully bootstrapped model 'churn-classifier' version 1 with alias 'champion'.
Accuracy: 0.8150
F1 score: 0.7200
```

#### Expected output — subsequent runs (idempotent)

```
INFO: Champion alias already assigned model_name=churn-classifier alias=champion version=1
Bootstrap status: already_bootstrapped
Model 'churn-classifier' already has 'champion' assigned to version 1. Bootstrap skipped.
```

#### Bootstrap exit codes

| Code | Meaning |
|---|---|
| `0` | Success or idempotent skip |
| `1` | Failure (quality gate rejected, data missing, MLflow unreachable) |

> Bootstrap connects to the tracking server at `MODELFORGE_MLFLOW_TRACKING_URI`
> (default: `http://127.0.0.1:5001`). MLflow must be running before you
> invoke bootstrap.

---

## 7. Launch the Full Stack

Once the champion model exists in the registry, start all remaining services:

```bash
make up
# equivalent: docker compose up -d
```

Compose starts services in dependency order:

```
mlflow  (already healthy)
  └── api           (waits for mlflow: healthy)
        └── prometheus  (waits for api: healthy)
              └── grafana     (waits for prometheus: healthy)
```

The full stack may take 2–3 minutes for all health checks to clear on a
cold start.

### Local endpoints

| Service | URL | Notes |
|---|---|---|
| **API** | http://localhost:8000 | FastAPI application |
| **API docs** | http://localhost:8000/docs | Swagger UI |
| **MLflow UI** | http://localhost:5001 | Experiment tracking and model registry |
| **Prometheus** | http://localhost:9090 | Metrics query interface |
| **Grafana** | http://localhost:3000 | Pre-provisioned dashboards (login with `.env` credentials) |

All ports are bound to `127.0.0.1` (loopback only). They are **not** reachable
from other machines on the same network.

---

## 8. Health and Readiness Checks

### API liveness — `/health`

Returns HTTP 200 as soon as the FastAPI process is up, regardless of model state:

```bash
curl http://127.0.0.1:8000/health
# {"status":"healthy"}
```

### API readiness — `/ready`

Returns HTTP 200 only after the champion model has been successfully loaded into
memory. Returns HTTP 503 if the model failed to load:

```bash
curl http://127.0.0.1:8000/ready
# {"status":"ready"}
```

> If `/ready` returns `503`, the most common cause is a missing or mis-named
> `champion` alias. Re-run `make bootstrap`, then restart the API container:
> `docker compose restart api`.

### Docker Compose service status

```bash
docker compose ps
```

All services should show `healthy`. If any shows `unhealthy` or `starting`,
check the logs:

```bash
docker compose logs mlflow
docker compose logs api
docker compose logs prometheus
docker compose logs grafana
```

---

## 9. Running Tests and Linters

Tests are fully isolated — no live MLflow server or Docker connection is required.

### Test suite

```bash
make test
# equivalent: uv run pytest tests/ -q
```

Expected: All tests pass cleanly with 0 warnings in approximately 5 seconds.

### Linting

```bash
make lint
# equivalent: uv run ruff check .
```

### Format check

```bash
make format
# equivalent: uv run ruff format --check .
```

### CI behaviour

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs automatically on
every push to `main`, `develop`, and `feature/**` branches, and on all pull
requests targeting `main` or `develop`. It runs the same three commands plus
`--tb=short` on pytest.

---

## 10. Shutdown

Stop all running containers (named volumes and bind-mount data are preserved):

```bash
make down
# equivalent: docker compose down
```

To also delete the named volumes (`prometheus_data`, `grafana_data`):

```bash
docker compose down -v
```

> **Host-mounted directories** (`./mlflow-data/`, `./mlruns/`) are bind mounts
> and are **not** removed by `docker compose down -v`. Delete them manually for
> a fully clean state:
>
> ```bash
> rm -rf ./mlflow-data ./mlruns
> ```
