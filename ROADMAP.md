# ModelForge Engineering Roadmap

This document outlines the milestones established during the AI Engineering audit and handover. It tracks the progress of hardening, reproducibility, deployment stability, and documentation across the ModelForge platform.

---

## Completed Engineering Milestones

| Milestone | Title | Status | Description |
|---|---|---|---|
| **Milestone 1** | Restore Full CI Suite | Completed | Restored comprehensive 98+ test pytest execution in GitHub Actions CI with timeout protection, removing diagnostic shell loops. |
| **Milestone 2** | Fix Docker Compose Portability | Completed | Removed host-specific paths (`/home/hamid/...`) from `docker-compose.yml`, standardizing volume mounts on portable `./mlruns` and `./mlflow-data`. |
| **Milestone 3** | Fresh-Clone Bootstrap Workflow | Completed | Created idempotent bootstrap workflow (`scripts/bootstrap.py` and `backend/app/ml/bootstrap.py`) to initialize MLflow, train/register initial candidate, and assign the `champion` alias. |
| **Milestone 4** | Platform Documentation & Guides | Completed | Authored `docs/SETUP.md`, `docs/DEPLOYMENT.md`, and `docs/TROUBLESHOOTING.md`; condensed README to link to guides; updated ROADMAP.md. All commands, paths, and env vars verified against source. |
| **Milestone 5** | Resolve Starlette Deprecation Warning | Completed | Migrated dev HTTP client tooling to `httpx2>=2.16.0` to eliminate StarletteDeprecationWarning during pytest execution. |
| **Milestone 6** | Final Engineering Audit & Verification | Completed | Performed evidence-based end-to-end audit, clean-clone reproducibility in isolated worktree, service persistence validation, CLI bootstrap fix, and produced final engineering sign-off report. |
| **Milestone 7** | Promotion Traceability & Auditability | Completed | Implemented dataset provenance tracking, promotion evaluation traceability, SQLite and MLflow decision audit records, and fail-closed safety gates. |
| **Milestone 8** | Read-Only Promotion Audit API | Completed | Implemented read-only HTTP endpoint (`GET /audit/promotions`) with strict Pydantic schemas, isolated SQLite query execution, defensive error handling, and comprehensive test coverage. |
| **Milestone 9** | Controlled Drift Response and Retraining Orchestration | Completed | Bridge drift detection and model retraining with automated trigger policy, concurrency locking, cooldown guardrails, asynchronous worker orchestration, and Prometheus/Grafana observability. All Phases 6A–6G.1 and live Docker Compose validation complete. |

---

## In Progress Engineering Milestones

*No engineering milestones are currently in progress. Milestones 1–9 are complete; Milestones 10–11 are proposed.*

---

## Proposed Future Milestones

> **Note**: The following proposed milestones represent planned future engineering work and are not yet implemented. All proposed work is subject to prioritization, scoping, and empirical evaluation. ModelForge remains a local development stack and does not claim production readiness.

| Milestone | Title | Status | Description |
|---|---|---|---|
| **Milestone 10** | API Authentication, Authorization and Rate Limiting | Proposed | Introduce lightweight authentication middleware, role-based route protection, and token-bucket rate limiting for prediction, monitoring, and audit endpoints. |
| **Milestone 11** | Persistence and Concurrency Evaluation (with Justified PostgreSQL Migration) | Proposed | Benchmark SQLite concurrency and lock contention under multi-worker loads; execute a structured migration to PostgreSQL only if justified by empirical performance data. |

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

---

## Detailed Milestone Scope: Milestone 7 / Phase 4 (Promotion Traceability & Auditability)

### Objective
Establish end-to-end auditability and cryptographic traceability across the ModelForge training and promotion lifecycle. Ensure dataset provenance, evaluation metrics on shared benchmarks, gate decisions (both approved and rejected), and model lineage are systematically persisted to SQLite and MLflow without sensitive row-level data leakage, while keeping all promotion safety gates strictly intact and fail-closed.

### Key Deliverables
- **Dataset Provenance Module (`backend/app/ml/provenance.py`)**:
  - Extracted deterministic SHA-256 digest (`compute_dataframe_sha256`), total row count, feature column names and dtypes, and target column distribution.
  - Captured train/test partition metadata (`train_rows`, `test_rows`, `test_size`, `random_state`, `stratified`, `train_sha256`, `test_sha256`).
  - Strict privacy enforcement: row-level data is never logged.
  - Gap documentation: when callers do not perform a train/test split, actual state is reported with explicit gap documentation (`split_performed: false`) rather than fabricating split metrics.
- **Evaluation Traceability & Decision Audit (`backend/app/ml/audit.py`, `backend/app/ml/promotion.py`)**:
  - Structured promotion audit schema in SQLite (`promotion_audit` table):
    - Columns: `audit_id`, `timestamp`, `model_name`, `candidate_version`, `candidate_run_id`, `champion_version`, `approved`, `reason`, `candidate_f1`, `champion_f1`, `benchmark_version`, `benchmark_digest`, `margin`, `payload`.
    - Full payload records individual gate outcomes (`benchmark_integrity`, `separation_and_overlap`, `candidate_evaluation`, `champion_evaluation`, `quality_gate`), comprehensive candidate and champion metrics (accuracy, precision, recall, f1_score, roc_auc), and dataset provenance summary.
  - Audits **both approved promotions and rejected attempts** across all failure modes (overlap, digest mismatch, missing benchmark, below minimum F1, below champion, insufficient improvement, or logging failures).
- **Consistent MLflow Lineage (`backend/app/ml/tracking.py`, `backend/app/ml/workflow.py`)**:
  - Attached consistent provenance tags (`dataset_sha256`), parameters (`dataset_rows`, `dataset_features`, `split_train_rows`, `split_test_rows`), and artifact (`dataset_provenance.json`) to training runs.
  - Attached promotion traceability tags (`promotion_approved`, `promotion_reason`, `promotion_benchmark_version`, `promotion_benchmark_digest`, `promotion_candidate_version`, `promotion_champion_version`), metrics (`promotion_candidate_f1`, `promotion_champion_f1`), and artifact (`promotion_decision.json`) to candidate runs.
- **Fail-Closed Promotion Safety**:
  - Live champion alias is never modified unless all gates pass **AND** audit recording succeeds.
  - In the event of audit database I/O failures or MLflow tracking errors during promotion, promotion aborts safely with `reason="audit_logging_failed"` and `approved=False`. The existing champion alias is completely preserved.
- **Explicit Synthetic Benchmark Limitation**:
  - Explicitly documents and verifies that `data/benchmark/churn_benchmark.csv` is a 20-row synthetic benchmark designed for evaluation mechanics, gate verification, and integrity checks. It does not establish statistical power or real-world performance sufficiency.

---

## Detailed Milestone Scope: Milestone 8 (Read-Only Promotion Audit API)

### Objective
Expose persisted model promotion audit records through a dedicated, read-only HTTP endpoint in FastAPI (`GET /audit/promotions`) to support operational visibility, governance auditing, and pipeline traceability without introducing mutating endpoints or external system coupling.

### Key Deliverables
- **Endpoint Specification (`backend/app/api/api.py`)**:
  - Route: `GET /audit/promotions`
  - Query Parameter: `limit` (integer, default `10`, validated bounds `1 <= limit <= 100`).
  - Returns HTTP 422 Unprocessable Entity for invalid parameters (`limit=0`, negative values, values exceeding 100, non-integer types).
- **Strict Response Schema (`backend/app/api/schemas.py`)**:
  - Pydantic schema: `PromotionAuditRecord`.
  - Structured fields: `audit_id` (str), `timestamp` (str), `model_name` (str), `candidate_version` (str | None), `candidate_run_id` (str | None), `champion_version` (str | None), `approved` (bool), `reason` (str), `candidate_f1` (float), `champion_f1` (float | None), `benchmark_version` (str), `benchmark_digest` (str), `margin` (float), and `payload` (dict[str, Any]).
- **SQLite Integration & Query Isolation (`backend/app/ml/audit.py`)**:
  - Read function: `get_promotion_audit_records(database_path=None, limit=100)`.
  - Queries `settings.promotion_audit_db_path` (`promotion_audit` table) ordered by `timestamp DESC LIMIT ?` (newest first).
  - **Isolation**: Direct SQLite read query; zero communication with MLflow tracking server and zero mutations of registered model aliases (`champion`).
  - **Graceful Empty States**: Returns HTTP 200 OK with empty array (`[]`) when database file is absent, `promotion_audit` table is uninitialized, or table contains zero records.
- **Defensive Error Handling**:
  - Catches `sqlite3.OperationalError`, `sqlite3.DatabaseError`, `json.JSONDecodeError`, and `OSError`, wrapping them in `AuditDatabaseError`.
  - Logs diagnostic details server-side with `logger.exception()`.
  - Returns sanitized HTTP 503 Service Unavailable (`{"detail": "The promotion audit service is unavailable."}`) preventing database path or implementation leakage.
- **Comprehensive Test Coverage (`tests/test_api.py`, `tests/test_api_schemas.py`, `tests/test_audit.py`, `tests/test_config.py`)**:
  - Verified empty list on missing database (`test_audit_promotions_missing_database_returns_empty_list`).
  - Verified empty list on uninitialized table (`test_audit_promotions_missing_table_returns_empty_list`).
  - Verified populated schema parsing and payload serialization (`test_audit_promotions_populated_database_returns_correct_fields_and_payload`).
  - Verified descending timestamp ordering (`test_audit_promotions_ordered_newest_first`).
  - Verified pagination limit bounds and validation (`test_audit_promotions_limit_restricts_count`, `test_audit_promotions_invalid_limit_returns_422`).
  - Verified configured database path resolution (`test_audit_promotions_uses_configured_db_path`, `test_promotion_audit_settings`).
  - Verified MLflow isolation and alias immutability (`test_audit_promotions_does_not_call_mlflow_or_mutate_aliases`).
  - Verified corrupted database handling and 503 response (`test_audit_promotions_corrupted_database_returns_503`).

---

## Detailed Milestone Scope: Milestone 9 — Controlled Drift Response and Retraining Orchestration

### Status: Completed (Phases 6A–6G.1 Implemented & Tested; Live Docker Compose Validation Complete)

#### Objective
Bridge drift detection and model retraining with automated trigger policies, durable SQLite state machines, concurrency locking, cooldown guardrails, resilient background worker orchestration, fail-closed promotion integration, and end-to-end Prometheus/Grafana observability.

#### Phase Breakdown & Implementation Progress
- **Phase 6A — Drift Alert Persistence**: Durable SQLite persistence of drift alerts and severity tracking. *(Implemented & Tested)*
- **Phase 6B — Drift Policy & Retraining Trigger Integration**: Automated evaluation of drift metrics against configurable trigger thresholds with cooldown enforcement and idempotency guards. *(Implemented & Tested)*
- **Phase 6C — Retraining Job Status API**: Read-only job status query endpoints (`GET /retrain/jobs/{job_id}`) with structured Pydantic schemas. *(Implemented & Tested)*
- **Phase 6D — Worker Robustness & Test Isolation**: Exclusive worker locking, defensive timeout handling, and test environment isolation. *(Implemented & Tested)*
- **Phase 6E/6F — Worker Integration & Reconciliation**: Heartbeat leasing, crash recovery, operator-guided resolution audits (`operator_resolution_audit`), and quarantine state handling (`reconciliation_required`). *(Implemented & Tested)*
- **Phase 6G — Observability & Retraining Metrics Pipeline**: API-side read-only SQLite metrics collector exposing queue depth, active jobs, monotonic terminal job counters, execution duration histograms, and worker heartbeat timestamps to Prometheus, visualized in Grafana. *(COMPLETE)*
- **Phase 6G.1 — Final Metrics Corrections**: Graceful shutdown heartbeat precedence (`status="stopped"` -> `0.0`), collector availability gauge (`modelforge_retraining_collector_up`) with synthetic zero omission during DB outages, and Grafana duration panel idle-period presentation (`custom.spanNulls: true`). *(COMPLETE)*

#### Phase 6G & 6G.1 Key Deliverables
- **Read-Only SQLite Metrics Collector (`backend/app/ml/retraining_metrics.py`)**:
  - Connects using `file:{db_path}?mode=ro` (zero write transactions, zero schema migrations during scrape).
  - Collects queue depth (`modelforge_retraining_queue_depth`) and active jobs (`modelforge_retraining_active_jobs`).
  - Emits monotonic cumulative counters (`modelforge_retraining_jobs_total{status="succeeded|rejected|failed|reconciliation_required"}`) using `COUNT(DISTINCT job_id)` across resolution audits to prevent double-counting.
  - Generates cumulative duration histograms (`modelforge_retraining_execution_duration_seconds`) statelessly via SQL bucket aggregation over completed jobs.
  - Emits worker heartbeat timestamp (`modelforge_worker_heartbeat_timestamp_seconds`), correctly preserving crash staleness while enforcing explicit `0.0` upon graceful stop.
  - Emits collector availability gauge (`modelforge_retraining_collector_up`), omitting retraining series on database errors to prevent artificial Prometheus counter resets while keeping prediction metrics online.
- **FastAPI Prometheus Integration (`backend/app/api/api.py`)**:
  - Appends formatted retraining metrics into the `/metrics` Prometheus exposition endpoint without impacting prediction latency.
- **Grafana Dashboard Panel Row (`monitoring/grafana/dashboards/modelforge-overview.json`)**:
  - Added "Retraining & Worker Orchestration" panel row containing Queue Depth, Active Jobs, Job Outcomes, Retraining Execution Duration (with `spanNulls: true`), and Worker Liveness & Heartbeat panels.
- **Verification Status**:
  - **Unit & Integration Tests**: 15/15 focused retraining metrics tests passed (`tests/test_retraining_metrics.py`), 55/55 API tests passed (`tests/test_api.py`), 446/446 full test suite passed.
  - **Lint & Format**: Clean passes on `ruff check` and `ruff format`.
  - **Persistent Integrity**: `data/lifecycle_audit.db` and `mlflow-data/mlflow.db` SHA-256 digests remained identical before and after test execution.
  - **Live Docker Compose Validation**: **COMPLETE**
    - Shared Docker image rebuilt successfully with the correct Compose build configuration.
    - `backend.app.ml.worker` and `backend.app.ml.retraining_metrics` exist in the image.
    - Worker module import succeeds.
    - Dedicated worker container starts and remains running.
    - Worker process is active.
    - `data/worker.lock` exists.
    - `worker_heartbeats` contains a persisted active worker record.
    - User confirmed heartbeat, Prometheus metrics, and Grafana dashboard live checks all work.
    - *(No independent verification claimed beyond these recorded results.)*

---

## Detailed Proposed Milestone Scopes: Milestones 10–11

> **Status: Proposed / Unscheduled.**<br>
> The following scopes describe proposed technical enhancements. None of these milestones are completed, and their implementation remains subject to engineering prioritization, capacity, and empirical validation. ModelForge remains a local development stack and does not claim production readiness.

### Proposed Milestone 10: API Authentication, Authorization and Rate Limiting

#### Objective
Implement a standards-compliant authentication, role-based authorization, and rate-limiting middleware layer to secure ModelForge endpoints against unauthorized usage and abusive query bursts.

#### Proposed Scope
- **Lightweight Authentication**: Header-based API key or Bearer token validation middleware integrated into FastAPI via dependency injection or middleware.
- **Role-Based Route Access (RBAC)**: Differentiate access privileges between prediction consumers (`/predict`), monitoring/operational services (`/monitoring/drift`), and audit/compliance callers (`/audit/promotions`).
- **Open Health Scrapes**: Maintain zero-credential access for orchestrator health checks (`/health`, `/ready`) and Prometheus scraping (`/metrics/prometheus`).
- **Rate Limiting & Burst Protection**: Token-bucket or sliding-window rate limiting per API key/client identifier with configurable limits and standard `Retry-After` headers.
- **Audit & Security Logging**: Structured logging of authentication failures and rate-limit violations with masked credentials.

#### Measurable Acceptance Criteria
1. **Authentication enforcement**: Requests without valid tokens to `/predict`, `/monitoring/drift`, and `/audit/promotions` return HTTP 401 Unauthorized; requests with insufficient role permissions return HTTP 403 Forbidden.
2. **Health availability**: `/health`, `/ready`, and `/metrics/prometheus` respond successfully (HTTP 200) without authentication credentials.
3. **Rate limit behavior**: Traffic exceeding configured requests-per-minute limits returns HTTP 429 Too Many Requests with a valid integer `Retry-After` header.
4. **Latency overhead**: Authentication and rate limiting middleware adds $\le 2.0\text{ms}$ P95 latency overhead to the prediction path.
5. **Test coverage**: Unit and integration tests covering missing credentials, invalid credentials, role authorization matrix, and rate-limit exhaustion.

---

### Proposed Milestone 11: Persistence and Concurrency Evaluation (with Justified PostgreSQL Migration)

#### Objective
Empirically evaluate SQLite concurrency limits, lock contention, and transaction throughput under multi-client loads across MLflow tracking and audit databases, executing a migration to PostgreSQL only if substantiated by empirical workload evidence.

#### Proposed Scope
- **Empirical Concurrency Benchmarking**: Build automated load-test harness simulating concurrent MLflow parameter/metric logging, candidate promotion attempts, and audit queries across 1, 5, 10, and 25 concurrent worker threads/processes.
- **Decision Threshold Matrix**: Establish explicit empirical criteria (e.g. SQLite lock contention error rate $> 0.1\%$ or P95 write latency $> 500\text{ms}$ under expected workload) that must be met before initiating database migration.
- **Pragmas & Optimization Evaluation**: Evaluate SQLite tuning (WAL mode, busy timeout adjustments) before declaring migration necessary.
- **Database Abstraction & Migration Tooling (Conditional)**: If and only if the decision threshold is breached, introduce SQLAlchemy/Alembic migrations, PostgreSQL container provisioning in Compose, and connection pooling.
- **Dual-Store Data Parity & Migration Script**: If migrated, provide migration tooling to transfer existing `promotion_audit` and `lifecycle_audit` records from SQLite to PostgreSQL with row count and cryptographic digest verification.

#### Measurable Acceptance Criteria
1. **Published benchmark report**: Comprehensive benchmark dataset documenting transaction rates, P50/P95/P99 latency, and `sqlite3.OperationalError: database is locked` frequency across worker concurrency levels.
2. **Justified decision record**: An architectural decision record (ADR) documenting whether SQLite optimizations suffice or if PostgreSQL migration is empirically justified.
3. **Zero regression under migration (if triggered)**: 100% of existing tests pass against the new database configuration; schema and data parity verified between SQLite and PostgreSQL audit records.
4. **Concurrency resilience**: Under target multi-worker concurrency, database transaction error rate is 0.0% with P95 transaction latency $< 50\text{ms}$.
5. **Container portability**: If PostgreSQL is introduced, `docker-compose.yml` starts the database with named volume persistence, health checks, and non-root service credentials without breaking single-command startup.
