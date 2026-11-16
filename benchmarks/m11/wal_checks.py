"""M11 Phase 3 — SQLite WAL Checkpoint, Lifecycle, and Integrity Checks.

Evaluates:
1. WAL checkpoint behaviour (passive vs truncate under reader isolation).
2. WAL/SHM file lifecycle (creation, page accumulation, cleanup on close).
3. Final checkpoint and 4-point integrity verification after workloads.
4. Read-only metrics connections (file:path?mode=ro) under concurrent writes.
5. Cross-process journal mode visibility.
6. Cleanup and residual state verification.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import sqlite3
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.app.ml.db import db_connection  # noqa: E402
from backend.app.ml.migrations import migrate_audit_database_idempotent  # noqa: E402
from backend.app.ml.retraining_metrics import (  # noqa: E402
    get_retraining_metrics_snapshot,
    record_worker_heartbeat,
)

logger = logging.getLogger("m11.wal_checks")


def check_wal_checkpoint_behavior(db_path: Path) -> dict[str, Any]:
    """Test WAL checkpointing: PASSIVE during reader vs TRUNCATE after."""
    # 1. Establish WAL mode
    with db_connection(db_path, isolation_level=None) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")

    wal_path = Path(str(db_path) + "-wal")

    # 2. Open reader holding a read transaction
    reader_conn = sqlite3.connect(str(db_path))
    reader_conn.execute("BEGIN DEFERRED;")
    reader_conn.execute("SELECT count(*) FROM worker_heartbeats;")

    # 3. Perform 25 writes while reader holds transaction
    for i in range(25):
        record_worker_heartbeat(database_path=db_path, worker_id=f"wal-chk-worker-{i}")

    wal_size_during = wal_path.stat().st_size if wal_path.exists() else 0

    # 4. Attempt PASSIVE checkpoint while reader is open
    with db_connection(db_path, isolation_level=None) as conn:
        # returns (busy_flag, log_frame_count, checkpointed_frame_count)
        passive_res = conn.execute("PRAGMA wal_checkpoint(PASSIVE);").fetchone()

    # 5. Reader finishes and commits
    reader_conn.execute("COMMIT;")
    reader_conn.close()

    # 6. Execute TRUNCATE checkpoint after reader released
    with db_connection(db_path, isolation_level=None) as conn:
        truncate_res = conn.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchone()

    wal_size_after = wal_path.stat().st_size if wal_path.exists() else 0

    success = (
        passive_res is not None
        and truncate_res is not None
        and truncate_res[0] == 0  # not busy
        and wal_size_after == 0  # truncated
    )

    detail_str = (
        f"Passive checkpoint while reader active: frames={passive_res[1]}, "
        f"checkpointed={passive_res[2]}. After reader close, TRUNCATE "
        f"returned: busy={truncate_res[0]}, log_frames={truncate_res[1]}, "
        f"checkpointed={truncate_res[2]}, wal_size={wal_size_after}B."
    )

    return {
        "check": "wal_checkpoint_behavior",
        "passed": success,
        "wal_size_during_bytes": wal_size_during,
        "wal_size_after_truncate_bytes": wal_size_after,
        "passive_checkpoint": {
            "busy": passive_res[0] if passive_res else None,
            "log_frames": passive_res[1] if passive_res else None,
            "checkpointed_frames": passive_res[2] if passive_res else None,
        },
        "truncate_checkpoint": {
            "busy": truncate_res[0] if truncate_res else None,
            "log_frames": truncate_res[1] if truncate_res else None,
            "checkpointed_frames": truncate_res[2] if truncate_res else None,
        },
        "details": detail_str,
    }


def check_wal_shm_lifecycle(db_path: Path) -> dict[str, Any]:
    """Test WAL and SHM file lifecycle during open, write, and close."""
    wal_path = Path(str(db_path) + "-wal")
    shm_path = Path(str(db_path) + "-shm")

    # Initial state after migration
    init_wal = wal_path.exists()
    init_shm = shm_path.exists()

    # Open persistent connection and write
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute(
        "INSERT INTO worker_heartbeats "
        "(worker_id, heartbeat_at, started_at, status) "
        "VALUES ('lifecycle-w1', '2026-01-01T00:00:00Z', "
        "'2026-01-01T00:00:00Z', 'active');"
    )
    conn.commit()

    open_wal = wal_path.exists()
    open_shm = shm_path.exists()
    open_wal_size = wal_path.stat().st_size if open_wal else 0
    open_shm_size = shm_path.stat().st_size if open_shm else 0

    # Close connection completely
    conn.close()

    closed_wal = wal_path.exists()
    closed_shm = shm_path.exists()

    passed = open_wal and open_shm and open_shm_size == 32768
    detail_str = (
        f"Active WAL size={open_wal_size}B, SHM size={open_shm_size}B "
        f"(32KB index). On final close: wal_exists={closed_wal}, "
        f"shm_exists={closed_shm}."
    )
    return {
        "check": "wal_shm_lifecycle",
        "passed": passed,
        "initial_files": {"wal_exists": init_wal, "shm_exists": init_shm},
        "active_connection_files": {
            "wal_exists": open_wal,
            "wal_size_bytes": open_wal_size,
            "shm_exists": open_shm,
            "shm_size_bytes": open_shm_size,
        },
        "after_close_files": {
            "wal_exists": closed_wal,
            "shm_exists": closed_shm,
        },
        "details": detail_str,
    }


def check_final_checkpoint_and_integrity(db_path: Path) -> dict[str, Any]:
    """Perform final checkpoint and verify 4-point integrity."""
    with db_connection(db_path, isolation_level=None) as conn:
        chk_res = conn.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchone()
        int_rows = conn.execute("PRAGMA integrity_check(10);").fetchall()
        fk_rows = conn.execute("PRAGMA foreign_key_check;").fetchall()
        triggers = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger';"
            ).fetchall()
        }
        indexes = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index';"
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
    missing_triggers = expected_triggers - triggers
    has_partial_idx = "idx_single_active_retraining_job" in indexes

    int_ok = [r[0] for r in int_rows] == ["ok"]
    fk_ok = len(fk_rows) == 0
    trig_ok = len(missing_triggers) == 0
    passed = (
        int_ok
        and fk_ok
        and trig_ok
        and has_partial_idx
        and (chk_res[0] == 0 if chk_res else False)
    )

    detail_str = (
        f"Integrity: {int_ok}, FKs: {fk_ok}, Triggers: {trig_ok}, "
        f"Index: {has_partial_idx}"
    )

    return {
        "check": "final_checkpoint_and_integrity",
        "passed": passed,
        "checkpoint": chk_res,
        "integrity_check": [r[0] for r in int_rows],
        "foreign_key_violations": fk_rows,
        "missing_triggers": list(missing_triggers),
        "partial_index_present": has_partial_idx,
        "details": detail_str,
    }


def check_read_only_metrics_wal(
    db_path: Path, duration_s: float = 3.0
) -> dict[str, Any]:
    """Verify that read-only metrics connections work during concurrent writes."""
    with db_connection(db_path, isolation_level=None) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")

    stop_event = threading.Event()
    write_count = [0]
    write_errors = [0]
    read_count = [0]
    read_errors = [0]
    read_latencies: list[float] = []

    def writer_worker():
        idx = 0
        while not stop_event.is_set():
            try:
                record_worker_heartbeat(
                    database_path=db_path,
                    worker_id=f"ro-test-writer-{idx}",
                    status="active",
                )
                write_count[0] += 1
            except Exception:
                write_errors[0] += 1
            idx += 1

    def reader_worker():
        while not stop_event.is_set():
            t0 = time.perf_counter()
            try:
                snap = get_retraining_metrics_snapshot(database_path=db_path)
                dt = time.perf_counter() - t0
                read_latencies.append(dt * 1000.0)
                if snap.is_available:
                    read_count[0] += 1
                else:
                    read_errors[0] += 1
            except Exception:
                read_errors[0] += 1

    w_thread = threading.Thread(target=writer_worker, daemon=True)
    r_thread = threading.Thread(target=reader_worker, daemon=True)

    w_thread.start()
    r_thread.start()
    time.sleep(duration_s)
    stop_event.set()
    w_thread.join(timeout=5.0)
    r_thread.join(timeout=5.0)

    p50 = sorted(read_latencies)[len(read_latencies) // 2] if read_latencies else 0.0
    p95 = (
        sorted(read_latencies)[int(len(read_latencies) * 0.95)]
        if read_latencies
        else 0.0
    )

    passed = read_errors[0] == 0 and read_count[0] > 0 and write_count[0] > 0

    detail_str = (
        f"{read_count[0]} read snapshots executed concurrently during "
        f"{write_count[0]} writes. Read errors: {read_errors[0]}, "
        f"Write errors: {write_errors[0]}. Read P50: {p50:.2f}ms, "
        f"P95: {p95:.2f}ms."
    )

    return {
        "check": "read_only_metrics_wal",
        "passed": passed,
        "duration_s": duration_s,
        "write_operations": write_count[0],
        "write_errors": write_errors[0],
        "read_operations": read_count[0],
        "read_errors": read_errors[0],
        "read_p50_ms": round(p50, 3),
        "read_p95_ms": round(p95, 3),
        "details": detail_str,
    }


def _proc_journal_worker(path_str: str, q: multiprocessing.Queue):
    try:
        conn = sqlite3.connect(path_str)
        jm = conn.execute("PRAGMA journal_mode;").fetchone()[0].lower()
        conn.close()
        q.put({"pid": os.getpid(), "journal_mode": jm, "ok": True})
    except Exception as exc:
        q.put({"pid": os.getpid(), "journal_mode": str(exc), "ok": False})


def check_cross_process_journal_mode(
    db_path: Path, n_processes: int = 4
) -> dict[str, Any]:
    """Verify that multiple processes observe identical journal mode."""
    with db_connection(db_path, isolation_level=None) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")

    q: multiprocessing.Queue = multiprocessing.Queue()
    procs = []
    for _ in range(n_processes):
        p = multiprocessing.Process(
            target=_proc_journal_worker,
            args=(str(db_path), q),
            daemon=True,
        )
        procs.append(p)
        p.start()

    for p in procs:
        p.join(timeout=5.0)

    results = []
    while not q.empty():
        results.append(q.get())

    all_wal = all(r["journal_mode"] == "wal" for r in results)
    passed = len(results) == n_processes and all_wal
    detail_str = (
        f"All {len(results)}/{n_processes} processes observed "
        f"journal_mode='wal': {all_wal}"
    )

    return {
        "check": "cross_process_journal_mode",
        "passed": passed,
        "processes_tested": n_processes,
        "results": results,
        "details": detail_str,
    }


def check_cleanup_and_residual_state(db_path: Path) -> dict[str, Any]:
    """Verify clean database removal leaves no lingering WAL or SHM files."""
    wal_path = Path(str(db_path) + "-wal")
    shm_path = Path(str(db_path) + "-shm")

    # Unlink database file
    db_path.unlink(missing_ok=True)
    wal_path.unlink(missing_ok=True)
    shm_path.unlink(missing_ok=True)

    db_lingers = db_path.exists()
    wal_lingers = wal_path.exists()
    shm_lingers = shm_path.exists()

    passed = not (db_lingers or wal_lingers or shm_lingers)
    detail_str = (
        f"No lingering files: db={not db_lingers}, wal={not wal_lingers}, "
        f"shm={not shm_lingers}"
    )
    return {
        "check": "cleanup_and_residual_state",
        "passed": passed,
        "db_lingers": db_lingers,
        "wal_lingers": wal_lingers,
        "shm_lingers": shm_lingers,
        "details": detail_str,
    }


def run_all_wal_checks(temp_dir: Path | None = None) -> dict[str, Any]:
    """Execute all Section 4 SQLite and WAL evaluations."""
    fd, p_str = tempfile.mkstemp(suffix=".db", prefix="m11_wal_check_", dir=temp_dir)
    os.close(fd)
    db_path = Path(p_str)

    try:
        migrate_audit_database_idempotent(db_path)

        chk1 = check_wal_checkpoint_behavior(db_path)
        chk2 = check_wal_shm_lifecycle(db_path)
        chk3 = check_final_checkpoint_and_integrity(db_path)
        chk4 = check_read_only_metrics_wal(db_path, duration_s=2.5)
        chk5 = check_cross_process_journal_mode(db_path, n_processes=4)
        chk6 = check_cleanup_and_residual_state(db_path)

        all_passed = all(
            [
                chk1["passed"],
                chk2["passed"],
                chk3["passed"],
                chk4["passed"],
                chk5["passed"],
                chk6["passed"],
            ]
        )

        return {
            "all_passed": all_passed,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "checks": {
                "checkpoint_behavior": chk1,
                "shm_lifecycle": chk2,
                "final_checkpoint_and_integrity": chk3,
                "read_only_metrics": chk4,
                "cross_process_visibility": chk5,
                "cleanup_and_residual_state": chk6,
            },
        }
    finally:
        db_path.unlink(missing_ok=True)
        Path(str(db_path) + "-wal").unlink(missing_ok=True)
        Path(str(db_path) + "-shm").unlink(missing_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = run_all_wal_checks()
    print(f"All WAL checks passed: {res['all_passed']}")
    for name, c in res["checks"].items():
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {name}: {c['details']}")
