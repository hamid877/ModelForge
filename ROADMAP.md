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
| **Milestone 5** | Resolve Starlette Deprecation Warning | Completed | Migrated dev HTTP client tooling to `httpx2>=2.16.0` to eliminate StarletteDeprecationWarning during pytest execution. |
| **Milestone 6** | Final Engineering Audit & Verification | Completed | Performed evidence-based end-to-end audit, clean-clone reproducibility in isolated worktree, service persistence validation, CLI bootstrap fix, and produced final engineering sign-off report. |

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

---

## Detailed Milestone Scope: Milestone 5 (Resolve Starlette Deprecation Warning)

### Objective
Eliminate the `StarletteDeprecationWarning` emitted during pytest execution by migrating the development HTTP client dependency to `httpx2`, ensuring full compatibility with Starlette 1.7.0 while maintaining zero test warnings and clean test passes.

### Key Deliverables
- **Dev Dependency Migration (`pyproject.toml`, `uv.lock`)**:
  - Replaced `httpx>=0.28.1` with `httpx2>=2.16.0` under `[dependency-groups] dev`.
  - Regenerated `uv.lock` via `uv sync`, cleanly swapping `httpx`/`httpcore` for `httpx2`/`httpcore2`/`truststore`.
- **Zero-Warning Test Suite Verification**:
  - Verified 110/110 tests pass cleanly with 0 warnings on `uv run pytest tests/ -q --tb=short` and `uv run pytest`.
  - Confirmed `uv run ruff check .` and `uv run ruff format --check .` pass.
  - Confirmed `git diff --check` passes cleanly without formatting or whitespace issues.

---

## Detailed Milestone Scope: Milestone 6 (Final Engineering Audit & Sign-off)

### Objective
Perform an evidence-based, end-to-end engineering audit from a clean environment, validate full stack container orchestration, MLflow tracking and promotion, real-time prediction serving, Prometheus and Grafana observability pipelines, verify state persistence across restarts, address identified edge-case defects, and generate the final sign-off report.

### Key Deliverables
- **Isolated Clean-Environment Validation**:
  - Created an isolated worktree at `HEAD`, installed frozen dependencies in a fresh virtual environment, verified all tests and linters pass cleanly, and cleaned up without forced removal.
- **Direct CLI Bootstrap Remediation (`scripts/bootstrap.py`, `tests/test_bootstrap.py`)**:
  - Resolved `sys.path` issue causing `ModuleNotFoundError: No module named 'backend'` during direct CLI or `make bootstrap` invocations.
  - Added regression test `test_cli_bootstrap_script_direct_execution` verifying direct invocation without `PYTHONPATH`.
- **Documentation & Hygiene Updates (`docs/SETUP.md`, `.gitignore`)**:
  - Removed obsolete Starlette deprecation warning notes in `docs/SETUP.md`.
  - Explicitly added host bind-mount directory `mlflow-data/` to `.gitignore`.
- **Full-Stack Orchestration & Observability Audit**:
  - Validated health check dependency chain across `mlflow`, `api`, `prometheus`, and `grafana`.
  - Confirmed active `champion` alias on version 5 of registered model `churn-classifier`.
  - Verified real predictions on both classes (churn and non-churn) matching the strict API schema.
  - Verified Prometheus target health (`up`), metric ingestion (`modelforge_prediction_requests_total`), Grafana health (`ok`), datasource provisioning, and dashboard panels.
- **Service Persistence & Resilience**:
  - Stopped and restarted the Docker Compose stack without `-v`, confirming complete recovery of healthy status and preservation of model artifacts, SQLite database, and dashboard state.
- **Final Audit Report (`docs/FINAL_ENGINEERING_AUDIT.md`)**:
  - Compiled exhaustive audit evidence, verification matrix, findings breakdown, and final PASS verdict.
