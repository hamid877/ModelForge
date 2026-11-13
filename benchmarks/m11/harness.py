"""M11 Phase 2 — SQLite Concurrency Benchmark Harness.

Measures lock contention, throughput, and latency of the ModelForge
SQLite persistence layer under controlled concurrent loads.

Design Principles
-----------------
- Every run uses a fresh temporary database; production data is never touched.
- Schemas are initialized via the production migration function to preserve
  all constraints, indexes, and triggers exactly as production does.
- Deterministic seed data is inserted before each scenario.
- Measurements use fixed-duration windows with an explicit warm-up phase.
- At least 3 independent repetitions are performed per scenario/concurrency
  combination.
- Both thread-level and process-level concurrency are exercised.
- Each repetition records per-operation latencies enabling P50/P95/P99 reporting.
- SQLITE_BUSY errors are tracked separately from logical/ownership rejections.
- Data integrity is verified after each run.

Transaction Semantics (verified from source inspection)
--------------------------------------------------------
- db_connection() / get_db_connection() default to isolation_level=None,
  which places sqlite3 in autocommit mode (no implicit BEGIN).
- explicit write paths use immediate_transaction() -> BEGIN IMMEDIATE + COMMIT
  for serialised, exclusive write locking.
- record_lifecycle_run() and record_promotion_audit() use db_connection()
  WITHOUT immediate_transaction(): each INSERT executes in autocommit mode
  (auto-committed on execute).  This matches the Phase 1 audit findings.
- read paths (get_promotion_audit_records etc.) open read-only connections
  with no transaction wrapper; sqlite3 autocommit mode allows concurrent reads.

Adapter Notes
-------------
- W1/W2/W4 use heartbeat_job() and claim_queued_job() directly (production code).
- W3 uses record_drift_alert() directly (production code).
- W5 uses get_promotion_audit_records() and get_drift_alert() directly.
- W6 uses record_promotion_audit() directly (production code).
- P1 uses subprocess workers that call claim_queued_job() / heartbeat_job(),
  replicating the actual worker process model.
- Synthetic multi-writer heartbeat stress is used only in W2/W4; it does NOT
  alter the production single-worker deployment invariant.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import platform
import sqlite3
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

# ---------------------------------------------------------------------------
# Project root on sys.path so production modules can be imported.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.app.ml.audit import (  # noqa: E402
    record_lifecycle_run,
    record_promotion_audit,
)
from backend.app.ml.db import db_connection  # noqa: E402
from backend.app.ml.drift_alerts import (  # noqa: E402
    DuplicateAlertError,
    get_drift_alert,
    record_drift_alert,
)
from backend.app.ml.jobs import (  # noqa: E402
    ActiveJobExistsError,
    JobOwnershipError,
    claim_queued_job,
    create_queued_job,
    heartbeat_job,
    transition_job_status,
)
from backend.app.ml.migrations import (  # noqa: E402
    migrate_audit_database_idempotent,
)
from backend.app.ml.retraining_metrics import (  # noqa: E402
    record_worker_heartbeat,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOURNAL_MODE = "DELETE"  # production default; do NOT change for baseline
BUSY_TIMEOUT_MS = 5000  # production default

# Concurrency levels to exercise
THREAD_CONCURRENCY_LEVELS = (1, 5, 10, 25)
PROCESS_CONCURRENCY_LEVELS = (1, 2, 4)

# Timing parameters
WARMUP_SECONDS = 2.0
MEASUREMENT_SECONDS_QUICK = 5.0  # development / quick profile
MEASUREMENT_SECONDS_FULL = 20.0  # evaluation profile
REPETITIONS = 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OperationResult:
    """Single-operation timing and outcome record."""

    scenario: str
    success: bool
    latency_s: float
    error_type: str | None = None  # "SQLITE_BUSY", "ownership", "integrity", "other"


@dataclass
class RunMetrics:
    """Aggregated metrics for one scenario × concurrency × repetition."""

    scenario: str
    concurrency_type: str  # "thread" or "process"
    concurrency: int
    repetition: int
    duration_s: float
    total_ops: int
    succeeded: int
    failed: int
    busy_errors: int
    ownership_rejections: int  # expected fencing failures — NOT counted as errors
    other_failures: int
    ops_per_second: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    integrity_ok: bool
    integrity_detail: str
    latencies_ms: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "latencies_ms"}
        return d


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _create_temp_db() -> Path:
    """Create a fresh temporary database file."""
    fd, path_str = tempfile.mkstemp(suffix=".db", prefix="modelforge_bench_")
    os.close(fd)
    return Path(path_str)


def _initialize_db(db_path: Path) -> None:
    """Apply the production migration to a fresh database."""
    migrate_audit_database_idempotent(db_path)


def _verify_journal_mode(db_path: Path) -> str:
    """Return the current journal mode of the database."""
    with db_connection(db_path, isolation_level=None) as conn:
        row = conn.execute("PRAGMA journal_mode;").fetchone()
        return row[0] if row else "unknown"


def _check_integrity(db_path: Path) -> tuple[bool, str]:
    """Run SQLite integrity_check on the database. Returns (ok, detail)."""
    try:
        with db_connection(db_path, isolation_level=None) as conn:
            rows = conn.execute("PRAGMA integrity_check(10);").fetchall()
        messages = [r[0] for r in rows]
        ok = messages == ["ok"]
        return ok, "; ".join(messages)
    except Exception as exc:
        return False, f"exception: {exc}"


def _seed_baseline_data(db_path: Path) -> dict[str, Any]:
    """Insert deterministic baseline data for reproducible benchmark runs.

    Returns a seed_manifest describing what was inserted.
    """
    seed: dict[str, Any] = {}

    # --- Drift alerts (5 pre-existing) ---
    alert_ids: list[str] = []
    for i in range(5):
        aid = f"seed-alert-{i:04d}"
        alert_ids.append(aid)
        try:
            record_drift_alert(
                alert_id=aid,
                drift_detected=(i % 2 == 0),
                reference_samples=1000 + i * 10,
                current_samples=950 + i * 10,
                p_value_threshold=0.05,
                features={"feature_count": 8, "seed_index": i},
                database_path=db_path,
            )
        except DuplicateAlertError:
            pass  # idempotent
    seed["alert_ids"] = alert_ids

    # --- Lifecycle runs (10 pre-existing) ---
    run_ids: list[str] = []
    for i in range(10):
        result = record_lifecycle_run(
            record={
                "status": "completed" if i % 3 != 2 else "failed",
                "reason": "seed",
                "training_attempted": True,
            },
            database_path=db_path,
        )
        run_ids.append(result["run_id"])
    seed["lifecycle_run_ids"] = run_ids

    # --- Promotion audits (5 pre-existing) ---
    audit_ids: list[str] = []
    for i in range(5):
        result = record_promotion_audit(
            record={
                "model_name": "churn-classifier",
                "candidate_version": str(i + 1),
                "candidate_run_id": f"seed-run-{i:04d}",
                "champion_version": str(i) if i > 0 else None,
                "approved": i % 2 == 0,
                "reason": "seed" if i % 2 == 0 else "below_threshold",
                "candidate_f1": 0.75 + i * 0.01,
                "champion_f1": 0.74 + i * 0.01 if i > 0 else None,
                "benchmark_version": "v1",
                "benchmark_digest": "deadbeef" * 8,
                "margin": 0.01,
            },
            database_path=db_path,
        )
        audit_ids.append(result["audit_id"])
    seed["audit_ids"] = audit_ids

    return seed


# ---------------------------------------------------------------------------
# Latency aggregation
# ---------------------------------------------------------------------------


def _compute_percentiles(latencies_s: list[float]) -> tuple[float, float, float]:
    """Return (P50, P95, P99) in milliseconds from a list of second-latencies."""
    if not latencies_s:
        return 0.0, 0.0, 0.0
    ms = sorted(lat * 1000.0 for lat in latencies_s)
    n = len(ms)
    if n == 1:
        return ms[0], ms[0], ms[0]

    def _pct(p: float) -> float:
        idx = p / 100.0 * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return ms[lo] + (ms[hi] - ms[lo]) * (idx - lo)

    return _pct(50), _pct(95), _pct(99)


def _aggregate(
    results: list[OperationResult],
    scenario: str,
    concurrency_type: str,
    concurrency: int,
    repetition: int,
    duration_s: float,
    db_path: Path,
) -> RunMetrics:
    succeeded = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    busy = sum(1 for r in results if r.error_type == "SQLITE_BUSY")
    ownership = sum(1 for r in results if r.error_type == "ownership")
    other = sum(1 for r in results if r.error_type == "other")
    latencies = [r.latency_s for r in results if r.success]
    p50, p95, p99 = _compute_percentiles(latencies)
    total = len(results)
    ops_s = total / duration_s if duration_s > 0 else 0.0
    integrity_ok, integrity_detail = _check_integrity(db_path)
    return RunMetrics(
        scenario=scenario,
        concurrency_type=concurrency_type,
        concurrency=concurrency,
        repetition=repetition,
        duration_s=duration_s,
        total_ops=total,
        succeeded=succeeded,
        failed=failed,
        busy_errors=busy,
        ownership_rejections=ownership,
        other_failures=other,
        ops_per_second=ops_s,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        integrity_ok=integrity_ok,
        integrity_detail=integrity_detail,
        latencies_ms=[lat * 1000.0 for lat in latencies],
    )


# ---------------------------------------------------------------------------
# Error classification helper
# ---------------------------------------------------------------------------


def _classify_error(exc: Exception) -> str:
    """Classify an exception into an error category, inspecting causal chains."""
    curr: BaseException | None = exc
    is_busy = False
    while curr is not None:
        curr_str = str(curr).lower()
        if (
            "database is locked" in curr_str
            or "busy" in curr_str
            or "locked" in curr_str
        ):
            is_busy = True
            break
        if isinstance(curr, sqlite3.OperationalError) and (
            "locked" in curr_str or "busy" in curr_str
        ):
            is_busy = True
            break
        curr = curr.__cause__ or curr.__context__

    if is_busy:
        return "SQLITE_BUSY"

    if isinstance(exc, (JobOwnershipError, ActiveJobExistsError)):
        return "ownership"
    if isinstance(exc, sqlite3.IntegrityError):
        return "integrity"
    return "other"


# ---------------------------------------------------------------------------
# W1: Job claim contention
# ---------------------------------------------------------------------------


def _w1_worker_thread(
    db_path: Path,
    stop_event: threading.Event,
    results: list[OperationResult],
    warmup_event: threading.Event,
    worker_idx: int,
) -> None:
    """Thread worker for W1: attempts to claim and immediately release queued jobs."""
    worker_id = f"w1-worker-{worker_idx}-{uuid4().hex[:8]}"

    while not stop_event.is_set():
        # First attempt: claim any queued job
        try:
            t0 = time.perf_counter()
            claimed = claim_queued_job(database_path=db_path, worker_id=worker_id)
            dt = time.perf_counter() - t0
            if claimed is not None:
                if warmup_event.is_set():
                    results.append(OperationResult("W1-claim", True, dt))
                try:
                    transition_job_status(
                        database_path=db_path,
                        job_id=claimed["job_id"],
                        worker_id=claimed["worker_id"],
                        lease_token=claimed["lease_token"],
                        target_status="failed",
                        failure_reason="benchmark-w1-cleanup",
                    )
                except Exception:
                    pass
                continue
        except sqlite3.OperationalError as exc:
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(
                    OperationResult("W1-claim", False, dt, _classify_error(exc))
                )
        except Exception as exc:
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(
                    OperationResult("W1-claim", False, dt, _classify_error(exc))
                )

        # If no queued job was claimed, attempt to enqueue one
        job_id = f"job-w1-{uuid4()}"
        try:
            t0 = time.perf_counter()
            create_queued_job(
                database_path=db_path,
                dataset_id="bench-ds-w1",
                trigger_source="benchmark-w1",
                job_id=job_id,
            )
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(OperationResult("W1-create", True, dt))
        except ActiveJobExistsError:
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(OperationResult("W1-create", False, dt, "ownership"))
        except sqlite3.OperationalError as exc:
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(
                    OperationResult("W1-create", False, dt, _classify_error(exc))
                )
        except Exception as exc:
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(
                    OperationResult("W1-create", False, dt, _classify_error(exc))
                )


# ---------------------------------------------------------------------------
# W2: Heartbeat write contention
# ---------------------------------------------------------------------------


def _w2_worker_thread(
    db_path: Path,
    stop_event: threading.Event,
    results: list[OperationResult],
    warmup_event: threading.Event,
    shared_job: dict[str, Any] | None,
    worker_idx: int,
) -> None:
    """Thread worker for W2: multiple threads perform concurrent heartbeat writes.

    Each worker records its worker heartbeat (INSERT...ON CONFLICT UPDATE),
    exercising real concurrent write transaction contention. Worker 0 also
    maintains the shared job lease, while synthetic interlopers attempt job
    heartbeats exercising ownership-fencing.
    """
    if shared_job is None:
        return

    worker_id = f"worker-w2-{worker_idx}"
    job_id = shared_job["job_id"]

    while not stop_event.is_set():
        # 1. Worker heartbeat write (all workers perform this genuine write transaction)
        try:
            t0 = time.perf_counter()
            record_worker_heartbeat(
                database_path=db_path,
                worker_id=worker_id,
                status="active",
            )
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(OperationResult("W2-worker-heartbeat", True, dt))
        except sqlite3.OperationalError as exc:
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(
                    OperationResult(
                        "W2-worker-heartbeat", False, dt, _classify_error(exc)
                    )
                )
        except Exception as exc:
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(
                    OperationResult(
                        "W2-worker-heartbeat", False, dt, _classify_error(exc)
                    )
                )

        # 2. Job lease heartbeat
        try:
            t0 = time.perf_counter()
            if worker_idx == 0:
                heartbeat_job(
                    database_path=db_path,
                    job_id=job_id,
                    worker_id=shared_job["worker_id"],
                    lease_token=shared_job["lease_token"],
                    lease_duration_seconds=300,
                )
                dt = time.perf_counter() - t0
                if warmup_event.is_set():
                    results.append(OperationResult("W2-lease-heartbeat", True, dt))
            else:
                heartbeat_job(
                    database_path=db_path,
                    job_id=job_id,
                    worker_id=worker_id,
                    lease_token=f"bad-token-{worker_idx}",
                    lease_duration_seconds=300,
                )
                dt = time.perf_counter() - t0
                if warmup_event.is_set():
                    results.append(OperationResult("W2-lease-heartbeat", True, dt))
        except JobOwnershipError:
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(
                    OperationResult("W2-lease-heartbeat", False, dt, "ownership")
                )
        except sqlite3.OperationalError as exc:
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(
                    OperationResult(
                        "W2-lease-heartbeat", False, dt, _classify_error(exc)
                    )
                )
        except Exception as exc:
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(
                    OperationResult(
                        "W2-lease-heartbeat", False, dt, _classify_error(exc)
                    )
                )


# ---------------------------------------------------------------------------
# W3: Concurrent drift assessments
# ---------------------------------------------------------------------------


def _w3_worker_thread(
    db_path: Path,
    stop_event: threading.Event,
    results: list[OperationResult],
    warmup_event: threading.Event,
    worker_idx: int,
) -> None:
    """Thread worker for W3: concurrent drift alert creation."""
    while not stop_event.is_set():
        aid = f"w3-{worker_idx}-{uuid4().hex}"
        try:
            t0 = time.perf_counter()
            record_drift_alert(
                alert_id=aid,
                drift_detected=True,
                reference_samples=1000,
                current_samples=950,
                p_value_threshold=0.05,
                features={"worker": worker_idx},
                database_path=db_path,
            )
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(OperationResult("W3-drift", True, dt))
        except DuplicateAlertError:
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(OperationResult("W3-drift", False, dt, "integrity"))
        except sqlite3.OperationalError as exc:
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(
                    OperationResult("W3-drift", False, dt, _classify_error(exc))
                )
        except Exception as exc:
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(
                    OperationResult("W3-drift", False, dt, _classify_error(exc))
                )


# ---------------------------------------------------------------------------
# W4: Combined write-heavy workload
# ---------------------------------------------------------------------------


def _w4_worker_thread(
    db_path: Path,
    stop_event: threading.Event,
    results: list[OperationResult],
    warmup_event: threading.Event,
    worker_idx: int,
) -> None:
    """Thread worker for W4: interleaves drift alerts, lifecycle,
    and promotion writes.
    """
    op_cycle = worker_idx % 3
    while not stop_event.is_set():
        if op_cycle == 0:
            # Drift alert
            aid = f"w4-drift-{worker_idx}-{uuid4().hex}"
            try:
                t0 = time.perf_counter()
                record_drift_alert(
                    alert_id=aid,
                    drift_detected=(worker_idx % 2 == 0),
                    reference_samples=1000,
                    current_samples=940,
                    p_value_threshold=0.05,
                    features={"mix": "w4"},
                    database_path=db_path,
                )
                dt = time.perf_counter() - t0
                if warmup_event.is_set():
                    results.append(OperationResult("W4-drift", True, dt))
            except DuplicateAlertError:
                dt = time.perf_counter() - t0
                if warmup_event.is_set():
                    results.append(OperationResult("W4-drift", False, dt, "integrity"))
            except sqlite3.OperationalError as exc:
                dt = time.perf_counter() - t0
                if warmup_event.is_set():
                    results.append(
                        OperationResult("W4-drift", False, dt, _classify_error(exc))
                    )
            except Exception as exc:
                dt = time.perf_counter() - t0
                if warmup_event.is_set():
                    results.append(
                        OperationResult("W4-drift", False, dt, _classify_error(exc))
                    )
        elif op_cycle == 1:
            # Lifecycle run
            try:
                t0 = time.perf_counter()
                record_lifecycle_run(
                    record={"status": "completed", "reason": "w4-bench"},
                    database_path=db_path,
                )
                dt = time.perf_counter() - t0
                if warmup_event.is_set():
                    results.append(OperationResult("W4-lifecycle", True, dt))
            except sqlite3.OperationalError as exc:
                dt = time.perf_counter() - t0
                if warmup_event.is_set():
                    results.append(
                        OperationResult("W4-lifecycle", False, dt, _classify_error(exc))
                    )
            except Exception as exc:
                dt = time.perf_counter() - t0
                if warmup_event.is_set():
                    results.append(
                        OperationResult("W4-lifecycle", False, dt, _classify_error(exc))
                    )
        else:
            # Promotion audit
            try:
                t0 = time.perf_counter()
                record_promotion_audit(
                    record={
                        "model_name": "churn-classifier",
                        "candidate_version": str(worker_idx),
                        "approved": False,
                        "reason": "w4-bench",
                        "candidate_f1": 0.72,
                        "benchmark_version": "v1",
                        "benchmark_digest": "aabbccdd" * 8,
                        "margin": 0.01,
                    },
                    database_path=db_path,
                )
                dt = time.perf_counter() - t0
                if warmup_event.is_set():
                    results.append(OperationResult("W4-promotion", True, dt))
            except sqlite3.OperationalError as exc:
                dt = time.perf_counter() - t0
                if warmup_event.is_set():
                    results.append(
                        OperationResult("W4-promotion", False, dt, _classify_error(exc))
                    )
            except Exception as exc:
                dt = time.perf_counter() - t0
                if warmup_event.is_set():
                    results.append(
                        OperationResult("W4-promotion", False, dt, _classify_error(exc))
                    )
        op_cycle = (op_cycle + 1) % 3


# ---------------------------------------------------------------------------
# W5: Read-heavy metrics queries during writes
# ---------------------------------------------------------------------------


def _w5_read_thread(
    db_path: Path,
    stop_event: threading.Event,
    results: list[OperationResult],
    warmup_event: threading.Event,
    seed: dict[str, Any],
    worker_idx: int,
) -> None:
    """Reader thread for W5: runs read queries while writer threads write."""
    from backend.app.ml.audit import get_promotion_audit_records

    while not stop_event.is_set():
        # Interleave two read types
        if worker_idx % 2 == 0:
            # Promotion audit list
            try:
                t0 = time.perf_counter()
                get_promotion_audit_records(database_path=db_path, limit=20)
                dt = time.perf_counter() - t0
                if warmup_event.is_set():
                    results.append(OperationResult("W5-read-audit", True, dt))
            except Exception as exc:
                dt = time.perf_counter() - t0
                if warmup_event.is_set():
                    results.append(
                        OperationResult(
                            "W5-read-audit", False, dt, _classify_error(exc)
                        )
                    )
        else:
            # Drift alert read
            if seed.get("alert_ids"):
                aid = seed["alert_ids"][worker_idx % len(seed["alert_ids"])]
                try:
                    t0 = time.perf_counter()
                    get_drift_alert(alert_id=aid, database_path=db_path)
                    dt = time.perf_counter() - t0
                    if warmup_event.is_set():
                        results.append(OperationResult("W5-read-alert", True, dt))
                except Exception as exc:
                    dt = time.perf_counter() - t0
                    if warmup_event.is_set():
                        results.append(
                            OperationResult(
                                "W5-read-alert", False, dt, _classify_error(exc)
                            )
                        )


def _w5_write_thread(
    db_path: Path,
    stop_event: threading.Event,
    results: list[OperationResult],
    warmup_event: threading.Event,
    worker_idx: int,
) -> None:
    """Write thread for W5: concurrent writes alongside readers."""
    while not stop_event.is_set():
        try:
            t0 = time.perf_counter()
            record_lifecycle_run(
                record={"status": "completed", "reason": "w5-write"},
                database_path=db_path,
            )
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(OperationResult("W5-write", True, dt))
        except sqlite3.OperationalError as exc:
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(
                    OperationResult("W5-write", False, dt, _classify_error(exc))
                )
        except Exception as exc:
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(
                    OperationResult("W5-write", False, dt, _classify_error(exc))
                )


# ---------------------------------------------------------------------------
# W6: Promotion audit writes
# ---------------------------------------------------------------------------


def _w6_worker_thread(
    db_path: Path,
    stop_event: threading.Event,
    results: list[OperationResult],
    warmup_event: threading.Event,
    worker_idx: int,
) -> None:
    """Thread worker for W6: concurrent promotion audit write stress."""
    while not stop_event.is_set():
        try:
            t0 = time.perf_counter()
            record_promotion_audit(
                record={
                    "model_name": "churn-classifier",
                    "candidate_version": f"{worker_idx}",
                    "candidate_run_id": uuid4().hex,
                    "approved": False,
                    "reason": "w6-bench",
                    "candidate_f1": 0.71 + worker_idx * 0.001,
                    "benchmark_version": "v1",
                    "benchmark_digest": "cafebabe" * 8,
                    "margin": 0.01,
                },
                database_path=db_path,
            )
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(OperationResult("W6-promotion", True, dt))
        except sqlite3.OperationalError as exc:
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(
                    OperationResult("W6-promotion", False, dt, _classify_error(exc))
                )
        except Exception as exc:
            dt = time.perf_counter() - t0
            if warmup_event.is_set():
                results.append(
                    OperationResult("W6-promotion", False, dt, _classify_error(exc))
                )


# ---------------------------------------------------------------------------
# Generic thread runner
# ---------------------------------------------------------------------------


def _run_thread_scenario(
    worker_fn,
    n_threads: int,
    warmup_s: float,
    measure_s: float,
    fn_kwargs: dict[str, Any],
) -> tuple[list[OperationResult], float]:
    """Run *n_threads* copies of *worker_fn* with warm-up and measurement phases.

    Returns (results, actual_measurement_duration_s).
    """
    lock = threading.Lock()
    shared_results: list[OperationResult] = []

    stop_event = threading.Event()
    warmup_event = threading.Event()

    threads = []
    for idx in range(n_threads):
        kw = {
            **fn_kwargs,
            "stop_event": stop_event,
            "results": shared_results,
            "warmup_event": warmup_event,
            "worker_idx": idx,
        }
        t = threading.Thread(target=worker_fn, kwargs=kw, daemon=True)
        threads.append(t)

    for t in threads:
        t.start()

    # Warm-up phase
    time.sleep(warmup_s)
    # Clear any results accumulated during warm-up
    with lock:
        shared_results.clear()
    warmup_event.set()

    # Measurement phase
    t_start = time.perf_counter()
    time.sleep(measure_s)
    stop_event.set()
    actual_duration = time.perf_counter() - t_start

    for t in threads:
        t.join(timeout=10.0)

    return list(shared_results), actual_duration


# ---------------------------------------------------------------------------
# Scenario runners (thread-based)
# ---------------------------------------------------------------------------


def run_w1_threads(
    n: int,
    db_path: Path,
    warmup_s: float,
    measure_s: float,
    repetition: int,
) -> RunMetrics:
    results, duration = _run_thread_scenario(
        worker_fn=_w1_worker_thread,
        n_threads=n,
        warmup_s=warmup_s,
        measure_s=measure_s,
        fn_kwargs={"db_path": db_path},
    )
    return _aggregate(results, "W1", "thread", n, repetition, duration, db_path)


def run_w2_threads(
    n: int,
    db_path: Path,
    warmup_s: float,
    measure_s: float,
    repetition: int,
) -> RunMetrics:
    """W2: Heartbeat write contention. Requires a pre-existing running job."""
    # Create and claim exactly one job before starting threads
    jid = f"job-w2-{uuid4()}"
    create_queued_job(
        database_path=db_path,
        dataset_id="bench-ds-w2",
        trigger_source="benchmark-w2",
        job_id=jid,
    )
    shared_job = claim_queued_job(database_path=db_path, worker_id="w2-owner")
    if shared_job is None:
        raise RuntimeError("W2: Could not claim initial job")

    results, duration = _run_thread_scenario(
        worker_fn=_w2_worker_thread,
        n_threads=n,
        warmup_s=warmup_s,
        measure_s=measure_s,
        fn_kwargs={"db_path": db_path, "shared_job": shared_job},
    )
    return _aggregate(results, "W2", "thread", n, repetition, duration, db_path)


def run_w3_threads(
    n: int,
    db_path: Path,
    warmup_s: float,
    measure_s: float,
    repetition: int,
) -> RunMetrics:
    results, duration = _run_thread_scenario(
        worker_fn=_w3_worker_thread,
        n_threads=n,
        warmup_s=warmup_s,
        measure_s=measure_s,
        fn_kwargs={"db_path": db_path},
    )
    return _aggregate(results, "W3", "thread", n, repetition, duration, db_path)


def run_w4_threads(
    n: int,
    db_path: Path,
    warmup_s: float,
    measure_s: float,
    repetition: int,
) -> RunMetrics:
    results, duration = _run_thread_scenario(
        worker_fn=_w4_worker_thread,
        n_threads=n,
        warmup_s=warmup_s,
        measure_s=measure_s,
        fn_kwargs={"db_path": db_path},
    )
    return _aggregate(results, "W4", "thread", n, repetition, duration, db_path)


def run_w5_threads(
    n: int,
    db_path: Path,
    warmup_s: float,
    measure_s: float,
    repetition: int,
    seed: dict[str, Any],
) -> RunMetrics:
    """W5: Mixed read-heavy + write workload. n//2+1 readers, rest writers."""
    n_readers = max(1, (n + 1) // 2)
    n_writers = max(1, n - n_readers)

    results: list[OperationResult] = []
    stop_event = threading.Event()
    warmup_event = threading.Event()

    threads = []
    for idx in range(n_readers):
        t = threading.Thread(
            target=_w5_read_thread,
            kwargs={
                "db_path": db_path,
                "stop_event": stop_event,
                "results": results,
                "warmup_event": warmup_event,
                "seed": seed,
                "worker_idx": idx,
            },
            daemon=True,
        )
        threads.append(t)
    for idx in range(n_writers):
        t = threading.Thread(
            target=_w5_write_thread,
            kwargs={
                "db_path": db_path,
                "stop_event": stop_event,
                "results": results,
                "warmup_event": warmup_event,
                "worker_idx": idx,
            },
            daemon=True,
        )
        threads.append(t)

    for t in threads:
        t.start()

    time.sleep(warmup_s)
    results.clear()
    warmup_event.set()

    t_start = time.perf_counter()
    time.sleep(measure_s)
    stop_event.set()
    actual_duration = time.perf_counter() - t_start

    for t in threads:
        t.join(timeout=10.0)

    return _aggregate(results, "W5", "thread", n, repetition, actual_duration, db_path)


def run_w6_threads(
    n: int,
    db_path: Path,
    warmup_s: float,
    measure_s: float,
    repetition: int,
) -> RunMetrics:
    results, duration = _run_thread_scenario(
        worker_fn=_w6_worker_thread,
        n_threads=n,
        warmup_s=warmup_s,
        measure_s=measure_s,
        fn_kwargs={"db_path": db_path},
    )
    return _aggregate(results, "W6", "thread", n, repetition, duration, db_path)


# ---------------------------------------------------------------------------
# P1: Cross-process contention (subprocess-based)
# ---------------------------------------------------------------------------


def _p1_process_worker(
    db_path_str: str,
    result_queue: multiprocessing.Queue,
    stop_event: multiprocessing.Event,
    warmup_event: multiprocessing.Event,
    worker_idx: int,
) -> None:
    """Process worker for P1: API/worker-style cross-process contention.

    Each process independently attempts to create + claim + heartbeat + complete
    retraining jobs, simulating real API/worker cross-process behaviour.
    """
    # Need to put repo root on path in subprocess
    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from backend.app.ml.jobs import (
        ActiveJobExistsError,
        claim_queued_job,
        create_queued_job,
        transition_job_status,
    )

    db_path = Path(db_path_str)
    worker_id = f"p1-proc-{worker_idx}-{os.getpid()}"
    local_results: list[dict] = []

    while not stop_event.is_set():
        job_id = f"job-p1-{uuid4()}"
        # Phase 1: create
        try:
            t0 = time.perf_counter()
            create_queued_job(
                database_path=db_path,
                dataset_id="bench-ds-p1",
                trigger_source="benchmark-p1",
                job_id=job_id,
            )
            dt = time.perf_counter() - t0
            created = True
            create_err = None
        except ActiveJobExistsError:
            dt = time.perf_counter() - t0
            created = False
            create_err = "ownership"
        except sqlite3.OperationalError as exc:
            dt = time.perf_counter() - t0
            created = False
            create_err = "SQLITE_BUSY" if "locked" in str(exc).lower() else "other"
        except Exception:
            dt = time.perf_counter() - t0
            created = False
            create_err = "other"

        if warmup_event.is_set():
            local_results.append(
                {
                    "scenario": "P1-create",
                    "success": created,
                    "latency_s": dt,
                    "error_type": create_err,
                }
            )

        if not created:
            # Try claiming whatever is queued
            try:
                t0 = time.perf_counter()
                claimed = claim_queued_job(database_path=db_path, worker_id=worker_id)
                dt = time.perf_counter() - t0
                if warmup_event.is_set():
                    local_results.append(
                        {
                            "scenario": "P1-claim",
                            "success": claimed is not None,
                            "latency_s": dt,
                            "error_type": None,
                        }
                    )
                if claimed is not None:
                    # Short heartbeat then complete
                    try:
                        heartbeat_job(
                            database_path=db_path,
                            job_id=claimed["job_id"],
                            worker_id=claimed["worker_id"],
                            lease_token=claimed["lease_token"],
                            lease_duration_seconds=60,
                        )
                        transition_job_status(
                            database_path=db_path,
                            job_id=claimed["job_id"],
                            worker_id=claimed["worker_id"],
                            lease_token=claimed["lease_token"],
                            target_status="failed",
                            failure_reason="benchmark-p1-cleanup",
                        )
                    except Exception:
                        pass
            except Exception:
                dt = time.perf_counter() - t0
                if warmup_event.is_set():
                    local_results.append(
                        {
                            "scenario": "P1-claim",
                            "success": False,
                            "latency_s": dt,
                            "error_type": "other",
                        }
                    )

    result_queue.put(local_results)


def run_p1_processes(
    n: int,
    db_path: Path,
    warmup_s: float,
    measure_s: float,
    repetition: int,
) -> RunMetrics:
    """P1: Cross-process contention using multiprocessing."""
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()
    warmup_event = multiprocessing.Event()

    procs = []
    for idx in range(n):
        p = multiprocessing.Process(
            target=_p1_process_worker,
            args=(str(db_path), result_queue, stop_event, warmup_event, idx),
            daemon=True,
        )
        procs.append(p)

    for p in procs:
        p.start()

    time.sleep(warmup_s)
    warmup_event.set()

    t_start = time.perf_counter()
    time.sleep(measure_s)
    stop_event.set()
    actual_duration = time.perf_counter() - t_start

    for p in procs:
        p.join(timeout=15.0)
        if p.is_alive():
            p.terminate()

    # Collect results from queue
    all_raw: list[dict] = []
    while True:
        try:
            batch = result_queue.get_nowait()
            all_raw.extend(batch)
        except Exception:
            break

    results = [
        OperationResult(
            scenario=r["scenario"],
            success=r["success"],
            latency_s=r["latency_s"],
            error_type=r.get("error_type"),
        )
        for r in all_raw
    ]
    return _aggregate(results, "P1", "process", n, repetition, actual_duration, db_path)


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def capture_environment() -> dict[str, Any]:
    """Capture reproducibility metadata for the benchmark run."""
    env: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": sys.version,
        "sqlite_version": sqlite3.sqlite_version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "journal_mode": JOURNAL_MODE,
        "busy_timeout_ms": BUSY_TIMEOUT_MS,
        "warmup_seconds": WARMUP_SECONDS,
        "thread_concurrency_levels": list(THREAD_CONCURRENCY_LEVELS),
        "process_concurrency_levels": list(PROCESS_CONCURRENCY_LEVELS),
        "repetitions": REPETITIONS,
        "note": (
            "Baseline measurement: DELETE journal mode, 5000 ms busy_timeout. "
            "WAL not enabled. All databases are isolated temporaries."
        ),
    }
    return env


# ---------------------------------------------------------------------------
# Correctness gate
# ---------------------------------------------------------------------------


def _check_correctness(db_path: Path) -> dict[str, Any]:
    """Post-run correctness verification.

    Returns a dict with 'ok' (bool) and 'details'.
    Stops if any structural invariant fails.
    """
    details: list[str] = []

    # 1. SQLite integrity_check
    int_ok, int_detail = _check_integrity(db_path)
    details.append(f"integrity_check: {int_detail}")
    if not int_ok:
        return {"ok": False, "details": details, "stop": True}

    # 2. Foreign key check
    try:
        with db_connection(db_path, isolation_level=None) as conn:
            fk_rows = conn.execute("PRAGMA foreign_key_check;").fetchall()
        if fk_rows:
            details.append(f"foreign_key_check FAILED: {fk_rows[:5]}")
            return {"ok": False, "details": details, "stop": True}
        else:
            details.append("foreign_key_check: ok")
    except Exception as exc:
        details.append(f"foreign_key_check exception: {exc}")
        return {"ok": False, "details": details, "stop": True}

    # 3. Check triggers still exist
    try:
        with db_connection(db_path, isolation_level=None) as conn:
            triggers = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger';"
                ).fetchall()
            }
        expected_triggers = {
            "trg_drift_alerts_immutable_update",
            "trg_drift_alerts_immutable_delete",
            "trg_drift_alert_actions_no_update",
            "trg_drift_alert_actions_no_delete",
            "trg_operator_resolution_audit_no_update",
            "trg_operator_resolution_audit_no_delete",
        }
        missing = expected_triggers - triggers
        if missing:
            details.append(f"Missing triggers: {missing}")
            return {"ok": False, "details": details, "stop": True}
        details.append(f"triggers: all {len(expected_triggers)} present")
    except Exception as exc:
        details.append(f"trigger_check exception: {exc}")
        return {"ok": False, "details": details, "stop": True}

    # 4. Verify partial index on retraining_jobs
    try:
        with db_connection(db_path, isolation_level=None) as conn:
            indexes = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index';"
                ).fetchall()
            }
        if "idx_single_active_retraining_job" not in indexes:
            details.append("Missing idx_single_active_retraining_job")
            return {"ok": False, "details": details, "stop": True}
        details.append("partial_index: idx_single_active_retraining_job present")
    except Exception as exc:
        details.append(f"index_check exception: {exc}")
        return {"ok": False, "details": details, "stop": True}

    return {"ok": True, "details": details, "stop": False}
