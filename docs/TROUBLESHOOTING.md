# ModelForge — Troubleshooting Guide

Diagnostic commands, error messages, root causes, and fixes for the most
common issues encountered when setting up and operating ModelForge.

---

## Contents

1. [Service Startup Failures](#1-service-startup-failures)
2. [Missing Champion / Alias Errors](#2-missing-champion-alias-errors)
3. [MLflow Connection Problems](#3-mlflow-connection-problems)
4. [Artifact Persistence Issues](#4-artifact-persistence-issues)
5. [Docker and Image Problems](#5-docker-and-image-problems)
6. [Configuration Problems](#6-configuration-problems)
7. [Test Failures](#7-test-failures)
8. [CI Failures](#8-ci-failures)

---

## 1. Service Startup Failures

### API container fails health check and restarts

**Symptom**

```
docker compose ps
# api  ...  unhealthy  Restarting
```

**Diagnosis**

```bash
docker compose logs api --tail=50
```

**Common causes and fixes**

| Log message | Cause | Fix |
|---|---|---|
| `model_loading_failed` | No `champion` alias in registry | Run `make bootstrap` while MLflow is healthy |
| `Connection refused http://mlflow:5000` | MLflow not yet healthy | Wait for `docker compose ps mlflow` → `healthy`, then `docker compose restart api` |
| `RESOURCE_DOES_NOT_EXIST` | Model `churn-classifier` not registered | Run `make bootstrap` |

---

### MLflow container fails health check

**Symptom**

```
docker compose ps mlflow
# mlflow  ...  unhealthy
```

**Diagnosis**

```bash
docker compose logs mlflow --tail=50
```

**Common causes and fixes**

| Symptom | Cause | Fix |
|---|---|---|
| `Permission denied /mlflow-data/mlflow.db` | Host directory not writable by UID 1000 | `chmod -R a+rwX ./mlflow-data ./mlruns` |
| `unable to open database file` | `./mlflow-data/` directory missing | `mkdir -p mlflow-data mlruns` then restart |
| Port 5001 already in use | Another process on the host | `lsof -i :5001` — stop the conflicting process |

---

### Grafana or Prometheus will not start

**Symptom**

```
grafana  ...  Created   (not starting)
```

**Cause**  
Compose enforces a strict dependency chain. Grafana waits for Prometheus, which
waits for the API, which waits for MLflow. If any upstream service is unhealthy,
downstream services are held back.

**Fix**  
Resolve upstream failures first (MLflow → API → Prometheus), then Grafana will
start automatically.

---

### `Set GRAFANA_ADMIN_USER in .env` error at Compose startup

**Symptom**

```
invalid interpolation format for services.grafana.environment.GF_SECURITY_ADMIN_USER:
required variable GRAFANA_ADMIN_USER is missing a value: Set GRAFANA_ADMIN_USER in .env
```

**Cause**  
The `.env` file is missing or does not define both required variables.

**Fix**

```bash
cp .env.example .env
# Edit .env — set GRAFANA_ADMIN_USER and GRAFANA_ADMIN_PASSWORD
```

---

## 2. Missing Champion / Alias Errors

### `/ready` returns `503 not_ready`

**Diagnosis**

```bash
curl http://127.0.0.1:8000/ready
# {"status":"not_ready"}

docker compose logs api --tail=30
# look for: model_loading_failed
```

**Cause**  
The `champion` alias on the `churn-classifier` registered model does not exist,
or the model could not be downloaded from the artifact store.

**Fix**

1. Ensure MLflow is healthy:
   ```bash
   curl http://127.0.0.1:5001/health
   ```

2. Run bootstrap (safe to re-run):
   ```bash
   make bootstrap
   ```

3. Restart the API:
   ```bash
   docker compose restart api
   ```

4. Check readiness again:
   ```bash
   curl http://127.0.0.1:8000/ready
   ```

---

### Bootstrap exits with `status: failed` — quality gate rejected

**Symptom**

```
Bootstrap status: failed
Bootstrap candidate model failed quality gate: candidate_below_minimum_f1.
```

**Cause**  
The trained model's F1 score fell below the `promotion_minimum_f1` threshold
(default: 0.70), or below the existing champion's F1 score.

**Diagnosis**

```bash
uv run python scripts/bootstrap.py 2>&1 | tail -20
```

**Fixes**

- Verify `data/raw/churn.csv` exists and is not corrupt:
  ```bash
  python -c "import pandas as pd; df = pd.read_csv('data/raw/churn.csv'); print(df.shape)"
  ```
- If the dataset is the expected one, the model should pass the default 0.70
  threshold. A failure here usually means the CSV was replaced with an
  incompatible file.
- Do **not** lower `MODELFORGE_PROMOTION_MINIMUM_F1` to bypass the gate without
  understanding why the model is underperforming.

---

### Bootstrap reports `already_bootstrapped` but `/ready` still returns 503

**Cause**  
The alias is assigned in the registry, but the API container was started before
the alias existed (or the model artifact is not accessible from the container).

**Fix**

```bash
docker compose restart api
# wait ~30 s
curl http://127.0.0.1:8000/ready
```

---

## 3. MLflow Connection Problems

### Bootstrap or API cannot reach MLflow

**Symptom** (bootstrap)

```
Bootstrap error: HTTPConnectionPool(host='127.0.0.1', port=5001): Max retries exceeded
```

**Symptom** (API logs)

```
model_loading_failed ... Connection refused
```

**Diagnosis**

```bash
# From the host:
curl http://127.0.0.1:5001/health

# Check if the container is running:
docker compose ps mlflow
```

**Fix**

```bash
docker compose up -d mlflow
# wait for healthy
docker compose ps mlflow
```

---

### API container cannot reach MLflow at `http://mlflow:5000`

**Symptom**

API logs show `Connection refused` or `Name or service not known` for
`http://mlflow:5000`.

**Cause**  
The API container is on a different Docker network from the MLflow container,
or the MLflow service is not yet healthy.

**Diagnosis**

```bash
docker compose ps mlflow
docker network ls
docker inspect modelforge_default   # (or your network name)
```

**Fix**

Ensure both services are in the same Compose project:

```bash
docker compose down
docker compose up -d mlflow
# wait for healthy
docker compose up -d api
```

---

### `--allowed-hosts` rejection

**Symptom**

MLflow returns `403 Forbidden` or `400 Bad Request` with a host-header message.

**Cause**  
The `Host` header used to reach MLflow is not in the `--allowed-hosts` list.

**Current allowed hosts**:
```
mlflow:5000,localhost:5001,localhost:5000,127.0.0.1:*
```

**Fix**  
If you have changed the port mapping in `docker-compose.yml`, update the
`--allowed-hosts` argument in the `mlflow` service `command:` block to match.

---

## 4. Artifact Persistence Issues

### MLflow artifacts not persisting after `docker compose down`

**Cause**  
`mlruns/` and `mlflow-data/` are bind mounts. They survive `docker compose down`
(and even `docker compose down -v`). However, if you ran a full teardown with
`rm -rf mlflow-data mlruns`, all run metadata and artifacts are gone.

**Fix**: Re-run bootstrap:

```bash
make bootstrap
```

---

### Artifacts not accessible to the API container

**Symptom**

API logs: `No such file or directory` or `mlflow.exceptions.MlflowException:
Could not find artifact`.

**Cause**  
The `./mlruns/` directory on the host is not mounted into the API container, or
was mounted with wrong permissions.

**Diagnosis**

```bash
docker compose exec api ls /mlruns
```

**Fix**  
Verify the `api` service has a volume mount for `./mlruns:/mlruns` in
`docker-compose.yml`. If you changed it, restore it and `docker compose up -d`.

---

### Lifecycle audit database not created

**Symptom**

```
OperationalError: unable to open database file: data/lifecycle_audit.db
```

**Cause**  
The `./data/` directory is not mounted into the container, or the path is not
writable by UID 1000.

**Fix**

```bash
mkdir -p data
chmod a+rwX data
docker compose restart api
```

---

## 5. Docker and Image Problems

### `docker compose up` fails: `modelforge-api:dev` not found

**Symptom**

```
Error response from daemon: No such image: modelforge-api:dev
```

**Cause**  
The image has not been built yet.

**Fix**

```bash
docker build -t modelforge-api:dev .
```

---

### Image uses stale code after editing source files

**Symptom**  
Changes to backend code are not reflected in the running container.

**Cause**  
Docker does not automatically rebuild the image.

**Fix**

```bash
docker build -t modelforge-api:dev .
docker compose up -d --no-deps api   # restart only the api
# or restart mlflow too if you changed shared code:
docker compose up -d --no-deps mlflow api
```

---

### Volume permission errors (`Permission denied`)

**Symptom**

MLflow or API logs show `PermissionError` or `Permission denied` for files in
`/mlflow-data/`, `/mlruns/`, or `/app/data/`.

**Cause**  
The host directories are owned by a user with UID != 1000, or have restricted
permissions.

**Fix**

```bash
# Make directories writable by UID 1000:
sudo chown -R 1000:1000 mlflow-data mlruns data
# or (less preferred, broader):
chmod -R a+rwX mlflow-data mlruns data
```

---

### Port already in use

**Symptom**

```
Error response from daemon: driver failed programming external connectivity on endpoint ...:
Bind for 127.0.0.1:5001 failed: port is already allocated
```

**Diagnosis**

```bash
lsof -i :5001    # MLflow
lsof -i :8000    # API
lsof -i :9090    # Prometheus
lsof -i :3000    # Grafana
```

**Fix**  
Stop the conflicting process, or change the host-side port in
`docker-compose.yml` (e.g. `"127.0.0.1:5002:5000"`) and update any references.

---

## 6. Configuration Problems

### Settings not applied

**Symptom**  
A `MODELFORGE_*` environment variable is set but the application ignores it.

**Cause**  
Variables in the system environment are only read when the process starts.
Variables set after container start are not picked up until the service is
restarted.

**Fix**

```bash
docker compose up -d api   # picks up current environment
```

Also verify you set the variable with the correct `MODELFORGE_` prefix and the
correct spelling (see the settings table in [DEPLOYMENT.md](DEPLOYMENT.md)).

---

### Wrong MLflow URI in bootstrap

**Symptom**  
Bootstrap prints connection errors but the MLflow UI is accessible at
http://localhost:5001.

**Cause**  
Bootstrap runs on the **host** (via `uv run`), not inside Docker. It uses the
default URI `http://127.0.0.1:5001`. The API container uses `http://mlflow:5000`
(Docker internal hostname).

**Fix**  
Ensure the MLflow container is running and the port mapping `127.0.0.1:5001:5000`
is in place. Do not change `MODELFORGE_MLFLOW_TRACKING_URI` on the host when
running bootstrap; the default is correct for host-side execution.

---

### `data/raw/churn.csv` not found

**Symptom**

```
Bootstrap error: [Errno 2] No such file or directory: 'data/raw/churn.csv'
```

**Cause**  
The file exists in the Git repository (`data/raw/churn.csv` is explicitly
excluded from `.gitignore` via `!data/raw/churn.csv`) but may not be present
after a partial clone or if it was manually deleted.

**Fix**

```bash
git checkout -- data/raw/churn.csv
# or:
ls -la data/raw/
```

---

## 7. Test Failures

### Tests cannot import `backend` module

**Symptom**

```
ModuleNotFoundError: No module named 'backend'
```

**Cause**  
`pytest` is not being run through `uv run`, or the virtual environment is not
active.

**Fix**

```bash
uv run pytest tests/ -q
# Do NOT use: python -m pytest tests/ (unless .venv is active)
```

`pyproject.toml` configures `pythonpath = ["."]` so the repository root is on
`sys.path` when tests are run via `uv`.

---

### Tests fail with `connection refused` to MLflow

**Cause**  
Tests should not require a live MLflow server. If tests are attempting real
network connections, they may be misconfigured or a test fixture is broken.

**Fix**  
All ModelForge tests mock MLflow interactions. Run the suite with:

```bash
uv run pytest tests/ -q --tb=short
```

If a specific test is failing with network errors, inspect its fixtures for
missing mocks.

---

### `StarletteDeprecationWarning` in test output

**Symptom**

```
StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated;
install `httpx2` instead.
```

**Cause**  
This is a deprecation warning from the FastAPI test client against the installed
version of `httpx`. It is **not** a test failure.

**Status**: Tracked as Milestone 5. No action required.

---

### Specific test fails intermittently (timeout)

**Cause**  
`pytest-timeout` is configured with a 60-second global timeout in `pyproject.toml`.
Slow machines or resource contention may trigger this.

**Fix**

```bash
uv run pytest tests/ -q --timeout=120
```

---

## 8. CI Failures

### CI fails on `uv sync`

**Symptom** (GitHub Actions log)

```
error: No solution found when resolving dependencies
```

**Cause**  
A dependency update in `pyproject.toml` has not been reflected in `uv.lock`.

**Fix** (locally)

```bash
uv lock
git add uv.lock
git commit -m "chore: update uv.lock"
```

---

### CI fails on `ruff check .` or `ruff format --check .`

**Fix** (locally)

```bash
# Fix linting issues:
uv run ruff check . --fix

# Fix formatting:
uv run ruff format .

# Then commit the changes.
```

---

### CI fails on `pytest` but tests pass locally

**Common causes**

| Cause | Fix |
|---|---|
| Local `.env` or `mlflow.db` influencing test state | Tests should use mocks; check for real file I/O in tests |
| Different Python version | CI uses Python 3.12 (see `ci.yml`). Ensure local Python is also 3.12 |
| `uv.lock` out of sync | Run `uv lock` and commit the updated lockfile |

---

## Quick Diagnostic Commands

```bash
# Overall service status
docker compose ps

# Logs for a specific service
docker compose logs <service> --tail=50

# API health and readiness
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready

# MLflow health
curl http://127.0.0.1:5001/health

# Prometheus health
curl http://127.0.0.1:9090/-/healthy

# Grafana health
curl http://127.0.0.1:3000/api/health

# Check host port conflicts
lsof -i :5001 -i :8000 -i :9090 -i :3000

# Inspect host-mount directory permissions
ls -lan mlflow-data mlruns data

# Verify champion alias in MLflow (host-side)
uv run python -c "
from mlflow import MlflowClient
c = MlflowClient('http://127.0.0.1:5001')
try:
    v = c.get_model_version_by_alias('churn-classifier', 'champion')
    print('Champion version:', v.version, '| run_id:', v.run_id)
except Exception as e:
    print('No champion:', e)
"

# Run the full test suite with verbose output
uv run pytest tests/ -v --tb=short

# Check linting and formatting
uv run ruff check .
uv run ruff format --check .
```
