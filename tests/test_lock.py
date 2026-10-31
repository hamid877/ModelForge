"""Tests for worker and operator process locking mechanisms."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend.app.ml.lock import (
    PROJECT_ROOT,
    MaintenanceLock,
    MaintenanceLockConflictError,
    WorkerLock,
    WorkerLockConflictError,
    acquire_maintenance_lock,
    acquire_worker_lock,
    operator_maintenance_lock,
    resolve_shared_lock_path,
    worker_execution_lock,
)

SUBPROCESS_LOCK_HOLD_SCRIPT = """
import sys, time
from pathlib import Path
from backend.app.ml.lock import {lock_class}

lock_path = sys.argv[1]
ready_file = Path(sys.argv[2])
stop_file = Path(sys.argv[3])

with {lock_class}(lock_path):
    ready_file.write_text("ready")
    while not stop_file.exists():
        time.sleep(0.05)
"""


def _spawn_lock_process(
    lock_class_name: str, lock_path: Path, tmp_path: Path, prefix: str
):
    """Spawn a subprocess that acquires the specified lock until stop_file exists."""
    ready_file = tmp_path / f"{prefix}_ready.txt"
    stop_file = tmp_path / f"{prefix}_stop.txt"
    script = SUBPROCESS_LOCK_HOLD_SCRIPT.format(lock_class=lock_class_name)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(lock_path),
            str(ready_file),
            str(stop_file),
        ]
    )

    # Wait for process to signal that the lock is held
    deadline = 5.0
    elapsed = 0.0
    while not ready_file.exists():
        time_step = 0.05
        proc.poll()
        if proc.returncode is not None:
            raise RuntimeError(
                f"Lock subprocess exited unexpectedly with code {proc.returncode}"
            )
        import time

        time.sleep(time_step)
        elapsed += time_step
        if elapsed >= deadline:
            proc.kill()
            raise TimeoutError("Lock subprocess failed to become ready in time.")

    return proc, stop_file


def test_lock_path_resolution_independent_of_working_directory(monkeypatch, tmp_path):
    """Verify lock path resolution is anchored to project root, immune to CWD."""
    default_resolved = resolve_shared_lock_path()
    assert default_resolved.is_absolute()
    assert default_resolved == (PROJECT_ROOT / "data/worker.lock").resolve()

    # Change working directory to tmp_path
    monkeypatch.chdir(tmp_path)
    assert Path.cwd() == tmp_path.resolve()

    # Must still resolve to the identical project root path
    resolved_after_chdir = resolve_shared_lock_path()
    assert resolved_after_chdir == default_resolved

    # Absolute custom path is respected
    custom_abs = (tmp_path / "custom.lock").resolve()
    assert resolve_shared_lock_path(custom_abs) == custom_abs


def test_worker_lock_descriptor_retained_throughout_lifetime(tmp_path):
    """Verify the lock descriptor remains open for the entire worker lifetime."""
    lock_file = tmp_path / "worker_fd.lock"
    lock = WorkerLock(lock_file)
    assert not lock.is_held
    assert lock.fd is None

    fd = lock.acquire()
    assert lock.is_held
    assert lock.fd == fd
    assert isinstance(fd, int) and fd > 0

    # Verify descriptor is valid and open by calling fstat
    stat_result = os.fstat(fd)
    assert stat_result.st_nlink > 0

    # Idempotent re-acquire returns the identical descriptor
    assert lock.acquire() == fd
    assert lock.is_held

    # Release descriptor
    lock.release()
    assert not lock.is_held
    assert lock.fd is None


def test_lock_release_does_not_unlink_or_recreate_file(tmp_path):
    """Verify release unlocks without unlinking or recreating the lock file."""
    lock_file = tmp_path / "persistent.lock"
    lock = WorkerLock(lock_file)
    lock.acquire()
    initial_stat = lock_file.stat()

    lock.release()
    assert lock_file.exists()
    after_release_stat = lock_file.stat()
    # Inode must be identical (not unlinked or recreated)
    assert initial_stat.st_ino == after_release_stat.st_ino

    # Re-acquisition reuses existing file
    maint = MaintenanceLock(lock_file)
    maint.acquire()
    reacquire_stat = lock_file.stat()
    assert initial_stat.st_ino == reacquire_stat.st_ino
    maint.release()


def test_worker_and_maintenance_lock_lifecycle_sequence(tmp_path):
    """Verify single-worker mutual exclusion sequence:

    - Worker 1 acquires the lock.
    - Worker 2 fails to acquire it.
    - Operator maintenance fails while Worker 1 is active.
    - After Worker 1 exits, maintenance can acquire the lock.
    - Worker startup fails while maintenance holds the lock.
    - Worker startup succeeds after maintenance releases it.
    """
    lock_file = tmp_path / "single_worker.lock"

    # 1. Worker 1 acquires the lock
    w1 = WorkerLock(lock_file)
    w1.acquire()
    assert w1.is_held

    # 2. Worker 2 fails to acquire it
    w2 = WorkerLock(lock_file)
    with pytest.raises(WorkerLockConflictError):
        w2.acquire()
    assert not w2.is_held

    # 3. Operator maintenance fails while Worker 1 is active
    m = MaintenanceLock(lock_file)
    with pytest.raises(MaintenanceLockConflictError):
        m.acquire()
    assert not m.is_held

    # 4. After Worker 1 exits, maintenance can acquire the lock
    w1.release()
    assert not w1.is_held

    m.acquire()
    assert m.is_held

    # 5. Worker startup fails while maintenance holds the lock
    w3 = WorkerLock(lock_file)
    with pytest.raises(WorkerLockConflictError):
        w3.acquire()
    assert not w3.is_held

    # 6. Worker startup succeeds after maintenance releases it
    m.release()
    assert not m.is_held

    w3.acquire()
    assert w3.is_held
    w3.release()


def test_cross_process_worker_exclusion_via_subprocess(tmp_path):
    """Test that a running worker subprocess prevents second worker and maintenance."""
    lock_file = tmp_path / "cross_proc_worker.lock"
    proc, stop_file = _spawn_lock_process("WorkerLock", lock_file, tmp_path, "w1")

    try:
        # Worker 2 in this process must fail
        with pytest.raises(WorkerLockConflictError):
            acquire_worker_lock(lock_file)

        # Maintenance in this process must fail
        with pytest.raises(MaintenanceLockConflictError):
            acquire_maintenance_lock(lock_file)
    finally:
        stop_file.write_text("stop")
        proc.wait(timeout=5)

    # After subprocess exits, maintenance can acquire
    with MaintenanceLock(lock_file) as maint:
        assert maint.is_held


def test_cross_process_maintenance_exclusion_via_subprocess(tmp_path):
    """Test that a running maintenance subprocess prevents worker startup."""
    lock_file = tmp_path / "cross_proc_maint.lock"
    proc, stop_file = _spawn_lock_process("MaintenanceLock", lock_file, tmp_path, "m1")

    try:
        # Worker startup must fail while maintenance process holds lock
        with pytest.raises(WorkerLockConflictError):
            acquire_worker_lock(lock_file)

        # Another maintenance process must also fail
        with pytest.raises(MaintenanceLockConflictError):
            acquire_maintenance_lock(lock_file)
    finally:
        stop_file.write_text("stop")
        proc.wait(timeout=5)

    # After maintenance process exits, worker can acquire
    with WorkerLock(lock_file) as worker:
        assert worker.is_held


def test_lock_release_after_process_termination_subprocess(tmp_path):
    """Test that terminating a lock holder process immediately releases kernel flock."""
    lock_file = tmp_path / "cross_proc_term.lock"
    proc, _ = _spawn_lock_process("WorkerLock", lock_file, tmp_path, "term_worker")

    try:
        # Confirms lock is held
        with pytest.raises(WorkerLockConflictError):
            acquire_worker_lock(lock_file)

        # Abruptly kill the worker process
        proc.kill()
        proc.wait(timeout=5)

        # Lock must be immediately acquirable by operator maintenance
        with MaintenanceLock(lock_file) as maint:
            assert maint.is_held
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)


def test_context_managers(tmp_path):
    """Test worker_execution_lock and operator_maintenance_lock context managers."""
    lock_file = tmp_path / "test_cm.lock"

    with worker_execution_lock(lock_file) as w_lock:
        assert w_lock.is_held
    assert not w_lock.is_held

    with operator_maintenance_lock(lock_file) as m_lock:
        assert m_lock.is_held
    assert not m_lock.is_held
