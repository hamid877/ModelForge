# ModelForge

ModelForge is an end-to-end machine learning platform for customer churn prediction, experiment tracking, automated model lifecycle management, and production serving.

## Core Features

- **Data Pipeline**: Automated data loading, feature validation, and preprocessing.
- **Experiment Tracking**: MLflow tracking with automatic artifact logging and versioning.
- **Model Registry & Promotion**: Quality gate evaluation (minimum F1 threshold & champion comparison) with alias management (`champion`).
- **REST Serving**: High-performance FastAPI service with per-request latency tracking and automated model loading on startup.
- **Observability**: Prometheus metrics exposition and pre-provisioned Grafana dashboards.
- **Drift Detection**: Kolmogorov-Smirnov test (numerical) and Chi-squared contingency test (categorical) with automated retraining evaluation.

---

## Quickstart: Fresh-Clone Setup

Follow these steps to set up and run ModelForge on a fresh machine or clean clone.

### 1. Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose

### 2. Environment Configuration

Clone the repository and create your local environment file:

```bash
git clone https://github.com/hamid877/ModelForge.git
cd ModelForge
cp .env.example .env
```

Set the required Grafana credentials in `.env`:
```env
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your_secure_password
```

### 3. Install Dependencies

Sync virtual environment and lockfile dependencies:

```bash
uv sync
```

### 4. Start MLflow & Bootstrap the Champion Model

ModelForge requires an active champion model in the MLflow Model Registry for the API to pass readiness checks (`/ready`).

Start the MLflow server:

```bash
docker compose up -d mlflow
```

Run the idempotent bootstrap workflow:

```bash
make bootstrap
# or directly:
uv run python scripts/bootstrap.py
```

The bootstrap script will:
1. Connect to MLflow and check if a champion model exists.
2. Train a baseline churn classification model on `data/raw/churn.csv`.
3. Evaluate the model against the quality gate (F1 >= 0.70).
4. Register the model under `churn-classifier` and assign the `champion` alias.

*(Note: Re-running `make bootstrap` is safe and idempotent; it detects existing champions and skips retraining unless `--force` is provided.)*

### 5. Launch the Full Stack

Start the remaining services (API, Prometheus, Grafana):

```bash
make up
# or: docker compose up -d
```

Verify service status:

```bash
curl http://127.0.0.1:8000/health
# {"status":"healthy"}

curl http://127.0.0.1:8000/ready
# {"status":"ready"}
```

Access local endpoints:
- **API Service**: [http://localhost:8000](http://localhost:8000) (Swagger Docs at `/docs`)
- **MLflow Tracking**: [http://localhost:5001](http://localhost:5001)
- **Prometheus**: [http://localhost:9090](http://localhost:9090)
- **Grafana**: [http://localhost:3000](http://localhost:3000)

---

## Developer Commands

ModelForge includes a `Makefile` with common tasks:

| Command | Action |
|---|---|
| `make bootstrap` | Train, register, and assign initial champion model in MLflow |
| `make test` | Run complete pytest test suite |
| `make lint` | Run Ruff linter checks |
| `make format` | Check code formatting compliance |
| `make up` | Start Docker Compose stack in detached mode |
| `make down` | Stop and tear down Docker Compose services |

---

## Testing & Quality

Run tests and linters locally:

```bash
# Run tests
uv run pytest tests/ -q

# Check linting and formatting
uv run ruff check .
uv run ruff format --check .
```

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for milestone progress, architecture decisions, and upcoming deliverables.
