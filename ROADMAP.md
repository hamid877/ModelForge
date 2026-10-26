# ModelForge Engineering Roadmap

This document outlines the milestones established during the AI Engineering audit and handover. It tracks the progress of hardening, reproducibility, deployment stability, and documentation across the ModelForge platform.

---

## Handover Milestones

| Milestone | Title | Status | Description |
|---|---|---|---|
| **Milestone 1** | Restore Full CI Suite | Completed | Restored comprehensive 98+ test pytest execution in GitHub Actions CI with timeout protection, removing diagnostic shell loops. |
| **Milestone 2** | Fix Docker Compose Portability | Completed | Removed host-specific paths (`/home/hamid/...`) from `docker-compose.yml`, standardizing volume mounts on portable `./mlruns` and `./mlflow-data`. |
| **Milestone 3** | Fresh-Clone Bootstrap Workflow | Completed | Created idempotent bootstrap workflow (`scripts/bootstrap.py` and `backend/app/ml/bootstrap.py`) to initialize MLflow, train/register initial candidate, and assign the `champion` alias. |
| **Milestone 4** | Platform Documentation & Guides | Completed | Authored `docs/SETUP.md`, `docs/DEPLOYMENT.md`, and `docs/TROUBLESHOOTING.md`; condensed README to link to guides; updated ROADMAP.md. All commands, paths, and env vars verified against source. |
| **Milestone 5** | Resolve Starlette Deprecation Warning | Pending | Upgrade HTTP client tooling (`httpx` / `httpx2`) to eliminate StarletteDeprecationWarning on test execution. |
| **Milestone 6** | Final Engineering Audit & Verification | Pending | Perform complete end-to-end clean-environment deployment audit and generate final sign-off report. |

---

## Detailed Milestone Scope: Milestone 3 (Fresh-Clone Bootstrap)

### Objective
Ensure any developer or CI environment can clone a fresh copy of ModelForge, initialize MLflow tracking, train and register an initial churn prediction model, and assign the `champion` alias so that the FastAPI service starts cleanly and passes readiness checks (`/ready`).

### Key Deliverables
- **Core Bootstrap Engine (`backend/app/ml/bootstrap.py`)**:
  - Reuses existing MLflow experiment setup (`setup_mlflow`), data validation (`load_and_validate_churn_data`), training (`train_churn_model`), and promotion (`run_training_and_promote`, `promote_model`).
  - Safe alias lookup via `get_champion_version`, safely handling missing models and missing aliases (`RESOURCE_DOES_NOT_EXIST`).
  - **Strict Idempotency**: Detects existing champion models and skips retraining unless `--force` is specified.
  - **Quality Gate Preservation**: Rejects candidates that fall below minimum F1 threshold (0.70) or below an active champion's F1 score, preventing degraded candidates from overwriting active champions even when `--force` is requested.
- **CLI Utility (`scripts/bootstrap.py`)**:
  - Command-line interface with `--data-path`, `--model-path`, `--model-name`, `--alias`, and `--force` options.
  - Standard exit codes (0 for success or idempotent skip, 1 on failure).
- **Developer Makefile (`Makefile`)**:
  - Provides ergonomic commands: `make bootstrap`, `make test`, `make lint`, `make format`, `make up`, `make down`.
- **Test Coverage (`tests/test_bootstrap.py`)**:
  - Isolated unit tests covering fresh setup, missing aliases, idempotent reruns, forced promotions, quality gate rejections, and CLI parsing/execution.

---

## Detailed Milestone Scope: Milestone 4 (Platform Documentation & Guides)

### Objective
Provide accurate, verifiable documentation for setup, deployment, and troubleshooting of ModelForge. All commands, paths, environment variables, endpoints, and Makefile targets were cross-referenced against the source code before being documented. The README was condensed to link to the guides rather than duplicating them.

### Key Deliverables
- **`docs/SETUP.md`**:
  - Prerequisites (Python 3.12, `uv`, Docker Compose v2), environment configuration, `uv sync`, Docker image build, MLflow startup, bootstrap workflow, full-stack launch, health/readiness checks, test and lint commands, and shutdown.
  - Clearly scoped to local development only.
- **`docs/DEPLOYMENT.md`**:
  - Service architecture diagram, Docker internal networking (container hostnames), all host port bindings (`127.0.0.1`-only), full environment variable reference (`MODELFORGE_*` prefix via `pydantic-settings`), bind mount and named volume inventory, MLflow backend store and artifact store details, registered model and alias scheme, canonical deployment procedure, security considerations, and known limitations.
  - Explicitly notes that the current stack is not production-hardened.
- **`docs/TROUBLESHOOTING.md`**:
  - Eight diagnostic sections: service startup failures, missing champion/alias errors, MLflow connection problems, artifact persistence issues, Docker and image problems, configuration problems, test failures, and CI failures.
  - Each entry includes symptom, diagnosis command, root cause, and concrete fix.
  - Quick-reference section of copy-paste diagnostic commands at the end.
- **Updated `README.md`**:
  - Condensed to feature overview, quickstart pointer, documentation table, developer command table, and roadmap link.
  - Duplicate content from SETUP.md removed.
- **Updated `ROADMAP.md`**:
  - Milestone 4 row updated from Pending to Completed with accurate description.
  - This detailed scope section added.
