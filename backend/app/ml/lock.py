"""Process locking mechanisms for worker execution and operator maintenance.

Implements exclusive worker execution locking and exclusive operator maintenance locking
using unified kernel-level file descriptor locks (fcntl.flock).
"""

import errno
import fcntl
import logging
import os
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from types import TracebackType

from backend.app.config import settings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class LockError(RuntimeError):
    """Base exception for file locking errors."""


class LockConflictError(LockError):
    """Raised when a lock cannot be acquired because it is held by another process."""


class WorkerLockConflictError(LockConflictError):
    """Raised when worker startup is refused because lock is held by another process."""


class MaintenanceLockConflictError(LockConflictError):
    """Raised when maintenance lock cannot be acquired because a lock is active."""


def resolve_shared_lock_path(lock_path: str | Path | None = None) -> Path:
    """Resolve shared lock path anchored to project directory or provided path."""
    if lock_path is not None:
        configured = Path(lock_path)
    else:
        configured = Path(getattr(settings, "worker_lock_path", "data/worker.lock"))

    if configured.is_absolute():
        return configured.resolve()

    return (PROJECT_ROOT / configured).resolve()


class BaseFileLock(AbstractContextManager["BaseFileLock"]):
    """Base class for kernel-level flock-based process locks."""

    def __init__(self, lock_path: str | Path | None = None) -> None:
        self.lock_path: Path = resolve_shared_lock_path(lock_path)
        self._fd: int | None = None

    @property
    def fd(self) -> int | None:
        """Return the open file descriptor or None if not locked."""
        return self._fd

    @property
    def is_held(self) -> bool:
        """Return True if this lock instance holds an active file descriptor."""
        return self._fd is not None

    def release(self) -> None:
        """Release the file lock and close descriptor safely without unlinking."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            finally:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None

    def __enter__(self) -> "BaseFileLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()


class WorkerLock(BaseFileLock):
    """Exclusive worker execution lock (LOCK_EX).

    Enforces that at most one worker process can execute at any given time.
    Refuses worker startup if another worker is running or operator maintenance lock
    is held. The descriptor is retained for the entire worker lifetime.
    """

    def acquire(self) -> int:
        """Acquire exclusive worker execution lock non-blockingly."""
        if self._fd is not None:
            return self._fd

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            os.close(fd)
            if isinstance(exc, BlockingIOError) or exc.errno in (
                errno.EACCES,
                errno.EAGAIN,
                errno.EWOULDBLOCK,
            ):
                logger.warning(
                    "worker_startup_refused_lock_held lock_path=%s",
                    self.lock_path,
                )
                raise WorkerLockConflictError(
                    f"Cannot start worker: Lock is currently held on "
                    f"'{self.lock_path}'."
                ) from exc
            raise LockError(
                f"Failed to acquire worker lock on '{self.lock_path}': {exc}"
            ) from exc

        self._fd = fd
        return fd


class MaintenanceLock(BaseFileLock):
    """Exclusive operator maintenance lock (LOCK_EX).

    Held throughout an operator reconciliation session.
    Refuses acquisition if any worker is running or another session is active.
    """

    def acquire(self) -> int:
        """Acquire exclusive operator maintenance lock non-blockingly."""
        if self._fd is not None:
            return self._fd

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            os.close(fd)
            if isinstance(exc, BlockingIOError) or exc.errno in (
                errno.EACCES,
                errno.EAGAIN,
                errno.EWOULDBLOCK,
            ):
                logger.warning(
                    "maintenance_lock_acquisition_failed_lock_held lock_path=%s",
                    self.lock_path,
                )
                raise MaintenanceLockConflictError(
                    f"Cannot acquire maintenance lock: Lock is currently held on "
                    f"'{self.lock_path}'."
                ) from exc
            raise LockError(
                f"Failed to acquire maintenance lock on '{self.lock_path}': {exc}"
            ) from exc

        self._fd = fd
        return fd


def acquire_worker_lock(lock_path: str | Path | None = None) -> WorkerLock:
    """Acquire and return an active WorkerLock instance."""
    lock = WorkerLock(lock_path)
    lock.acquire()
    return lock


def acquire_maintenance_lock(lock_path: str | Path | None = None) -> MaintenanceLock:
    """Acquire and return an active MaintenanceLock instance."""
    lock = MaintenanceLock(lock_path)
    lock.acquire()
    return lock


@contextmanager
def worker_execution_lock(
    lock_path: str | Path | None = None,
) -> Generator[WorkerLock, None, None]:
    """Context manager for worker execution locking."""
    with WorkerLock(lock_path) as lock:
        yield lock


@contextmanager
def operator_maintenance_lock(
    lock_path: str | Path | None = None,
) -> Generator[MaintenanceLock, None, None]:
    """Context manager for operator maintenance locking."""
    with MaintenanceLock(lock_path) as lock:
        yield lock
