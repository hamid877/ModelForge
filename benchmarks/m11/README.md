# ModelForge M11 — SQLite Concurrency Benchmark Harness

This directory contains the reproducible benchmark harness for evaluating SQLite concurrency limits, transaction throughput, lock contention, and latency under multi-worker loads across the ModelForge platform.

## Architecture and Design Principles

1. **Clean Test Isolation**: Every benchmark repetition creates a brand-new temporary database file via `tempfile.mkstemp()`. Production development databases (`data/lifecycle_audit.db`, `mlflow-data/mlflow.db`) are never accessed or modified.
2. **Production Schema Idempotency**: Schemas are initialized using the exact production migration function (`migrate_audit_database_idempotent`), preserving all table constraints (`CHECK`), primary keys, foreign keys, indexes, and SQLite triggers.
3. **Deterministic Seeding**: Before each benchmark run, deterministic baseline records (drift alerts, lifecycle audits, promotion audits) are inserted into the database.
4. **DELETE Journal Mode Baseline**: In accordance with M11 Phase 2 requirements, the database operates in standard `DELETE` journal mode with `busy_timeout = 5000ms`. WAL mode and PRAGMA tuning are deferred to subsequent phases.
5. **Separation of Errors**: Logical rejections (such as expected ownership-fencing rejections in `heartbeat_job` or `create_queued_job`) are classified as `ownership` and reported separately from true `SQLITE_BUSY` (`sqlite3.OperationalError: database is locked`) errors.
6. **Automatic Integrity Checks**: Every repetition runs `PRAGMA integrity_check`, `PRAGMA foreign_key_check`, trigger existence validation, and partial index existence validation.
7. **Causal Error Trapping**: Unwinds Python exception chains (`__cause__` and `__context__`) so that lock errors wrapped by domain exceptions (e.g. `AuditDatabaseError`) are properly recorded as `SQLITE_BUSY`.

---

## Supported Concurrency Levels

- **Thread Concurrency**: 1, 5, 10, 25 worker threads
- **Process Concurrency**: 1, 2, 4 independent OS processes via Python `multiprocessing`
- **Repetitions**: Minimum 3 independent repetitions per combination

---

## Workload Scenarios

| Scenario | Type | Description | Primary Production Module |
|---|---|---|---|
| **W1** | Thread | **Job claim contention**: Multiple worker threads race to claim queued jobs (`claim_queued_job`) and transition them to completion (`transition_job_status`). | `backend.app.ml.jobs` |
| **W2** | Thread | **Heartbeat write contention**: Multiple workers concurrently write active heartbeats (`record_worker_heartbeat`) under `BEGIN IMMEDIATE` transaction contention, while job lease heartbeats (`heartbeat_job`) and fencing checks are exercised. | `backend.app.ml.retraining_metrics`, `backend.app.ml.jobs` |
| **W3** | Thread | **Concurrent drift assessments**: Threads concurrently insert immutable drift alerts (`record_drift_alert`) under `BEGIN IMMEDIATE` transactions. | `backend.app.ml.drift_alerts` |
| **W4** | Thread | **Combined write-heavy workload**: Threads interleave drift alert insertions, lifecycle run records (`record_lifecycle_run`), and promotion audits (`record_promotion_audit`), stressing mixed autocommit and immediate transactions. | `backend.app.ml.drift_alerts`, `backend.app.ml.audit` |
| **W5** | Thread | **Read-heavy metrics queries during writes**: Reader threads query promotion audits (`get_promotion_audit_records`) and drift alerts (`get_drift_alert`) while concurrent writer threads record lifecycle updates, exposing read-write lock contention under `DELETE` journal mode. | `backend.app.ml.audit`, `backend.app.ml.drift_alerts` |
| **W6** | Thread | **Promotion audit writes**: Concurrent threads write promotion audit entries (`record_promotion_audit`) in autocommit mode. | `backend.app.ml.audit` |
| **P1** | Process | **Cross-process contention**: 1, 2, 4 independent OS processes concurrently claim, heartbeat, and finalize retraining jobs, replicating multi-container API and worker process contention. | `backend.app.ml.jobs` |

---

## Execution Profiles

- **`quick`**: 2-second warm-up, 5-second measurement window. Designed for development, regression testing, and rapid iteration (~10-12 minutes for the complete 81-run matrix).
- **`full`**: 2-second warm-up, 20-second measurement window. Designed for high-precision evaluation profiles (~35-40 minutes).

---

## Exact Execution Commands

### 1. Run Complete Baseline Benchmark (Quick Profile)
```bash
.venv/bin/python benchmarks/m11/run_benchmark.py --profile quick
```

### 2. Run Complete Baseline Benchmark (Full Evaluation Profile)
```bash
.venv/bin/python benchmarks/m11/run_benchmark.py --profile full
```

### 3. Run a Single Scenario (e.g. W1 at 10 Threads)
```bash
.venv/bin/python benchmarks/m11/run_benchmark.py --scenario W1 --threads 10 --repetitions 3
```

### 4. Run Process Contention (P1 with 4 Processes)
```bash
.venv/bin/python benchmarks/m11/run_benchmark.py --scenario P1 --processes 4 --repetitions 3
```

### 5. Custom Output Directory
```bash
.venv/bin/python benchmarks/m11/run_benchmark.py --profile quick --output-dir benchmarks/m11/results
```

---

## Output Artifacts

Each benchmark run automatically produces three artifacts in the output directory (`benchmarks/m11/results/`):

1. **`results_<timestamp>.json`**: Machine-readable JSON output containing full environment metadata, per-repetition execution metrics (ops/s, P50, P95, P99 latencies, total operations, busy counts, ownership fencing counts, failure details), and run durations.
2. **`baseline_report_<timestamp>.md`**: Human-readable Markdown baseline report formatted with summary tables, percentile latencies, busy rates, integrity status, and reproducibility notes.
3. **`integrity_<timestamp>.json`**: Complete verification audit documenting `PRAGMA integrity_check`, `PRAGMA foreign_key_check`, trigger existence, and partial index existence for every single run.

