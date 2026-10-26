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

## Quickstart

See **[docs/SETUP.md](docs/SETUP.md)** for the complete step-by-step guide covering:

1. Prerequisites (Python 3.12, `uv`, Docker Compose v2)
2. Environment configuration (`.env`)
3. Installing dependencies (`uv sync`)
4. Building the Docker image
5. Starting MLflow and bootstrapping the champion model
6. Launching the full stack
7. Health and readiness checks
8. Running tests and linters
9. Shutdown

---

## Documentation

| Guide | Contents |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | Local development setup, bootstrap workflow, tests |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Service architecture, ports, env vars, volumes, security |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common errors, diagnostic commands, fixes |

---

## Developer Commands

| Command | Action |
|---|---|
| `make bootstrap` | Train, register, and assign initial champion model in MLflow |
| `make test` | Run complete pytest test suite |
| `make lint` | Run Ruff linter checks |
| `make format` | Check code formatting compliance |
| `make up` | Start Docker Compose stack in detached mode |
| `make down` | Stop and tear down Docker Compose services |

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for milestone progress, architecture decisions, and upcoming deliverables.
