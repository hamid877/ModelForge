"""Background worker execution infrastructure for controlled retraining orchestration.

Manages worker lifecycle, exclusive flock process locking, atomic job polling and
claiming, periodic lease heartbeats, cooperative execution cancellation, and safe,
fenced state transitions without coupling to model training or promotion logic.
"""

import argparse
import logging
import signal
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol
from uuid import uuid4

from backend.app.config import settings
from backend.app.ml.jobs import (
    JobNotFoundError,
    JobOwnershipError,
    StaleLeaseError,
    claim_queued_job,
    get_job,
    heartbeat_job,
    recover_expired_jobs,
    requeue_job_for_retry,
    transition_job_status,
    validate_job_ownership,
)
from backend.app.ml.lock import (
    WorkerLock,
    WorkerLockConflictError,
    resolve_shared_lock_path,
)

logger = logging.getLogger(__name__)

__all__ = [
    "JobExecutionContext",
    "JobExecutionResult",
    "JobHandler",
    "NoOpJobHandler",
    "RetrainingWorker",
    "WorkerLockConflictError",
    "main",
]


@dataclass
class JobExecutionContext:
    """Execution context and cooperative cancellation handle for a claimed job."""

    job_id: str
    worker_id: str
    lease_token: str
    cancel_event: threading.Event = field(default_factory=threading.Event)

    @property
    def is_cancelled(self) -> bool:
        """Return True if cancellation has been requested."""
        return self.cancel_event.is_set()


@dataclass
class JobExecutionResult:
    """Outcome and metadata returned by a JobHandler upon execution completion."""

    status: str = "succeeded"
    failure_reason: str | None = None
    promotion_approved: bool | None = None
    promotion_reason: str | None = None
    promotion_audit_id: str | None = None
    candidate_version: str | None = None
    candidate_run_id: str | None = None
    previous_champion_version: str | None = None
    payload_updates: Mapping[str, Any] | None = None


class JobHandler(Protocol):
    """Protocol defining the injectable job execution interface."""

    def __call__(
        self,
        job: dict[str, Any],
        context: JobExecutionContext,
    ) -> JobExecutionResult:
        """Execute a claimed retraining job."""
        ...


class NoOpJobHandler:
    """Default no-op handler completing jobs successfully without side effects."""

    def __call__(
        self,
        job: dict[str, Any],
        context: JobExecutionContext,
    ) -> JobExecutionResult:
        """Return a successful execution result without performing real training."""
        return JobExecutionResult(status="succeeded")


class RetrainingWorker:
    """Dedicated background worker runtime for consuming retraining jobs."""

    def __init__(
        self,
        database_path: str | Path | None = None,
        lock_path: str | Path | None = None,
        worker_id: str | None = None,
        handler: JobHandler | None = None,
        poll_interval_seconds: float | None = None,
        heartbeat_interval_seconds: float | None = None,
        lease_duration_seconds: int | None = None,
        shutdown_timeout: float | None = None,
    ) -> None:
        self.database_path: Path = Path(
            database_path
            if database_path is not None
            else settings.lifecycle_audit_db_path
        )
        self.lock_path: Path = resolve_shared_lock_path(lock_path)
        self.worker_id: str = worker_id or f"worker-{uuid4().hex[:12]}"
        if not self.worker_id.strip():
            raise ValueError("worker_id must be a non-empty string.")

        self.handler: JobHandler = handler or NoOpJobHandler()
        self.poll_interval_seconds: float = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else getattr(settings, "retraining_poll_interval_seconds", 5.0)
        )
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0.")

        self.heartbeat_interval_seconds: float = (
            heartbeat_interval_seconds
            if heartbeat_interval_seconds is not None
            else getattr(settings, "retraining_heartbeat_interval_seconds", 15.0)
        )
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be > 0.")

        self.lease_duration_seconds: int = (
            lease_duration_seconds
            if lease_duration_seconds is not None
            else getattr(settings, "retraining_lease_duration_seconds", 60)
        )
        if self.lease_duration_seconds < 1:
            raise ValueError("lease_duration_seconds must be >= 1.")

        self.shutdown_timeout: float = (
            shutdown_timeout if shutdown_timeout is not None else 2.0
        )
        if self.shutdown_timeout < 0:
            raise ValueError("shutdown_timeout must be >= 0.")

        self._lock = WorkerLock(self.lock_path)
        self._shutdown_event = threading.Event()
        self._current_context: JobExecutionContext | None = None
        self._handler_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None

    @property
    def is_locked(self) -> bool:
        """Return True if this worker holds an open exclusive lock descriptor."""
        return self._lock.is_held

    @property
    def is_running(self) -> bool:
        """Return True if worker is active and has not received a shutdown signal."""
        return self.is_locked and not self._shutdown_event.is_set()

    @property
    def is_executing(self) -> bool:
        """Return True if a job handler or heartbeat thread is currently active."""
        handler_alive = (
            self._handler_thread is not None and self._handler_thread.is_alive()
        )
        heartbeat_alive = (
            self._heartbeat_thread is not None and self._heartbeat_thread.is_alive()
        )
        return handler_alive or heartbeat_alive

    def acquire_lock(self) -> int:
        """Acquire exclusive worker execution lock non-blockingly."""
        return self._lock.acquire()

    def release_lock(self, force: bool = False) -> None:
        """Release exclusive worker execution lock descriptor safely.

        Never releases the lock if a handler or heartbeat thread is still alive,
        unless force=True is explicitly passed.
        """
        if not force and self.is_executing:
            logger.warning(
                "refusing_to_release_lock_while_work_active",
                extra={
                    "worker_id": self.worker_id,
                    "current_status": "work_active",
                    "failure_reason": (
                        "Worker execution lock retained because a handler or "
                        "heartbeat thread is still active."
                    ),
                },
            )
            return
        self._lock.release()

    def stop(self) -> None:
        """Signal worker to stop polling and cooperatively cancel active execution."""
        self._shutdown_event.set()
        if self._current_context is not None:
            self._current_context.cancel_event.set()

    def __enter__(self) -> "RetrainingWorker":
        self.acquire_lock()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release_lock()

    def _setup_signal_handlers(self) -> None:
        """Install graceful shutdown signal handlers on the main thread."""
        try:

            def _handle_signal(signum: int, _frame: Any) -> None:
                logger.info(
                    "worker_shutdown_signal_received",
                    extra={"worker_id": self.worker_id, "signal": signum},
                )
                self.stop()

            signal.signal(signal.SIGTERM, _handle_signal)
            signal.signal(signal.SIGINT, _handle_signal)
        except (ValueError, AttributeError):
            # When running in secondary threads (e.g. unit tests), skip signal setup
            pass

    def run_once(self) -> dict[str, Any] | None:
        """Execute a single polling iteration under exclusive lock protection.

        1. Ensures lock descriptor is held.
        2. Scans and recovers any expired jobs (reclaims, fails, or quarantines).
        3. Attempts to atomically claim the oldest queued job.
        4. If claimed, executes the job with background lease heartbeats.

        Returns:
            Finalized job record if a job was claimed and completed, or None.
        """
        if not self.is_locked:
            self.acquire_lock()

        # 1. Recover expired leases adhering strictly to retry and quarantine rules
        recover_expired_jobs(self.database_path)

        # 2. Claim next available queued job
        job = claim_queued_job(
            self.database_path,
            worker_id=self.worker_id,
            lease_duration_seconds=self.lease_duration_seconds,
        )
        if job is None:
            return None

        # 3. Process claimed job
        return self.process_job(job)

    def process_job(self, job: dict[str, Any]) -> dict[str, Any] | None:
        """Process a claimed retraining job with active lease heartbeats and fencing."""
        job_id = job["job_id"]
        lease_token = job["lease_token"]
        attempt_count = job.get("attempt_count", 0)

        cancel_event = threading.Event()
        stop_heartbeat_event = threading.Event()

        context = JobExecutionContext(
            job_id=job_id,
            worker_id=self.worker_id,
            lease_token=lease_token,
            cancel_event=cancel_event,
        )
        self._current_context = context

        def _heartbeat_loop() -> None:
            while not stop_heartbeat_event.wait(
                timeout=self.heartbeat_interval_seconds
            ):
                if stop_heartbeat_event.is_set() or self._shutdown_event.is_set():
                    break
                try:
                    heartbeat_job(
                        self.database_path,
                        job_id=job_id,
                        worker_id=self.worker_id,
                        lease_token=lease_token,
                        lease_duration_seconds=self.lease_duration_seconds,
                    )
                    logger.debug(
                        "lease_heartbeat_extended",
                        extra={
                            "worker_id": self.worker_id,
                            "job_id": job_id,
                            "lease_attempt": attempt_count,
                            "heartbeat_result": "extended",
                            "current_status": "running",
                        },
                    )
                except (JobOwnershipError, JobNotFoundError, StaleLeaseError) as exc:
                    logger.warning(
                        "lease_heartbeat_failed_ownership_lost",
                        extra={
                            "worker_id": self.worker_id,
                            "job_id": job_id,
                            "lease_attempt": attempt_count,
                            "heartbeat_result": "ownership_lost",
                            "current_status": "running",
                            "failure_reason": str(exc),
                        },
                    )
                    cancel_event.set()
                    break
                except Exception as exc:
                    logger.error(
                        "lease_heartbeat_unexpected_error",
                        extra={
                            "worker_id": self.worker_id,
                            "job_id": job_id,
                            "lease_attempt": attempt_count,
                            "heartbeat_result": "error",
                            "current_status": "running",
                            "failure_reason": str(exc),
                        },
                    )

        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            name=f"heartbeat-{job_id}",
            daemon=True,
        )
        self._heartbeat_thread = heartbeat_thread
        heartbeat_thread.start()
        start_time = time.monotonic()

        handler_result: JobExecutionResult | None = None
        handler_exc: BaseException | None = None

        def _run_handler() -> None:
            nonlocal handler_result, handler_exc
            try:
                handler_result = self.handler(job, context)
            except BaseException as exc:
                handler_exc = exc

        handler_thread = threading.Thread(
            target=_run_handler,
            name=f"handler-{job_id}",
            daemon=True,
        )
        self._handler_thread = handler_thread
        handler_thread.start()

        try:
            logger.info(
                "job_execution_started",
                extra={
                    "worker_id": self.worker_id,
                    "job_id": job_id,
                    "lease_attempt": attempt_count,
                    "current_status": "running",
                },
            )
            # Wait for handler with graceful shutdown monitoring
            while handler_thread.is_alive():
                handler_thread.join(timeout=0.1)
                if self._shutdown_event.is_set():
                    shutdown_deadline = time.monotonic() + self.shutdown_timeout
                    while (
                        handler_thread.is_alive()
                        and time.monotonic() < shutdown_deadline
                    ):
                        handler_thread.join(timeout=0.1)
                    break

            execution_duration = time.monotonic() - start_time

            # If handler is still alive after shutdown timeout, preserve lock
            if handler_thread.is_alive():
                logger.warning(
                    "handler_ignored_cancellation_still_running",
                    extra={
                        "worker_id": self.worker_id,
                        "job_id": job_id,
                        "lease_attempt": attempt_count,
                        "execution_duration": execution_duration,
                        "failure_reason": (
                            "Handler did not terminate within shutdown timeout; "
                            "lock will remain held."
                        ),
                    },
                )
                return None

        finally:
            stop_heartbeat_event.set()
            heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None
            if not handler_thread.is_alive():
                self._handler_thread = None
                self._current_context = None

        # Check for unexpected exception in handler
        if handler_exc is not None:
            logger.exception(
                "job_handler_unexpected_exception",
                extra={
                    "worker_id": self.worker_id,
                    "job_id": job_id,
                    "lease_attempt": attempt_count,
                    "execution_duration": execution_duration,
                    "failure_reason": str(handler_exc),
                },
                exc_info=handler_exc,
            )

            # If cancelled due to lease loss, never mark failed or overwrite state
            if cancel_event.is_set():
                logger.warning(
                    "job_failed_after_ownership_lost_not_updating_state",
                    extra={
                        "worker_id": self.worker_id,
                        "job_id": job_id,
                        "lease_attempt": attempt_count,
                        "execution_duration": execution_duration,
                        "failure_reason": "Ownership lost; not updating state",
                    },
                )
                return None

            # Verify current lease ownership before updating DB
            try:
                validate_job_ownership(
                    self.database_path,
                    job_id=job_id,
                    worker_id=self.worker_id,
                    lease_token=lease_token,
                )
            except (JobOwnershipError, JobNotFoundError, StaleLeaseError) as own_exc:
                logger.warning(
                    "lease_validation_failed_after_exception",
                    extra={
                        "worker_id": self.worker_id,
                        "job_id": job_id,
                        "lease_attempt": attempt_count,
                        "execution_duration": execution_duration,
                        "failure_reason": str(own_exc),
                    },
                )
                return None

            job_record = get_job(self.database_path, job_id=job_id)
            if job_record is None:
                return None

            # Case 1: Post-promotion failure -> quarantine into reconciliation_required
            if job_record.get("promotion_started_at") is not None:
                fail_reason = (
                    f"Handler exception after promotion started: {handler_exc}"
                )
                try:
                    return transition_job_status(
                        self.database_path,
                        job_id=job_id,
                        worker_id=self.worker_id,
                        lease_token=lease_token,
                        target_status="reconciliation_required",
                        failure_reason=fail_reason,
                    )
                except JobOwnershipError as own_exc:
                    logger.warning(
                        "fenced_quarantine_transition_rejected_ownership_lost",
                        extra={
                            "worker_id": self.worker_id,
                            "job_id": job_id,
                            "lease_attempt": attempt_count,
                            "execution_duration": execution_duration,
                            "failure_reason": str(own_exc),
                        },
                    )
                    return None

            # Case 2: Pre-promotion failure with attempts remaining -> Requeue for retry
            curr_attempts = job_record.get("attempt_count", attempt_count)
            max_attempts = job_record.get("max_attempts", 3)

            if curr_attempts < max_attempts:
                fail_reason = (
                    f"Handler exception on attempt {curr_attempts}/{max_attempts}: "
                    f"{handler_exc}"
                )
                try:
                    requeued = requeue_job_for_retry(
                        self.database_path,
                        job_id=job_id,
                        worker_id=self.worker_id,
                        lease_token=lease_token,
                        failure_reason=fail_reason,
                    )
                    logger.warning(
                        "job_requeued_for_retry",
                        extra={
                            "worker_id": self.worker_id,
                            "job_id": job_id,
                            "lease_attempt": curr_attempts,
                            "current_status": "queued",
                            "execution_duration": execution_duration,
                            "failure_reason": fail_reason,
                        },
                    )
                    return requeued
                except JobOwnershipError as own_exc:
                    logger.warning(
                        "fenced_requeue_rejected_ownership_lost",
                        extra={
                            "worker_id": self.worker_id,
                            "job_id": job_id,
                            "lease_attempt": curr_attempts,
                            "execution_duration": execution_duration,
                            "failure_reason": str(own_exc),
                        },
                    )
                    return None

            # Case 3: Pre-promotion failure with attempts exhausted -> Terminal failed
            fail_reason = (
                f"Handler exception with attempts exhausted "
                f"({curr_attempts}/{max_attempts}): {handler_exc}"
            )
            try:
                return transition_job_status(
                    self.database_path,
                    job_id=job_id,
                    worker_id=self.worker_id,
                    lease_token=lease_token,
                    target_status="failed",
                    failure_reason=fail_reason,
                )
            except JobOwnershipError as own_exc:
                logger.warning(
                    "fenced_failure_transition_rejected_ownership_lost",
                    extra={
                        "worker_id": self.worker_id,
                        "job_id": job_id,
                        "lease_attempt": curr_attempts,
                        "execution_duration": execution_duration,
                        "failure_reason": str(own_exc),
                    },
                )
                return None

        # Handler returned result. If cancelled due to lease loss or shutdown:
        if cancel_event.is_set():
            logger.warning(
                "stale_worker_cannot_finalize_cancelled",
                extra={
                    "worker_id": self.worker_id,
                    "job_id": job_id,
                    "lease_attempt": attempt_count,
                    "current_status": "ownership_lost",
                    "execution_duration": execution_duration,
                    "failure_reason": (
                        "Job execution cancelled due to lease loss or shutdown"
                    ),
                },
            )
            return None

        if handler_result is None:
            return None

        # Verify active ownership before attempting final state transition
        try:
            validate_job_ownership(
                self.database_path,
                job_id=job_id,
                worker_id=self.worker_id,
                lease_token=lease_token,
            )
        except (JobOwnershipError, JobNotFoundError, StaleLeaseError) as own_exc:
            logger.warning(
                "stale_worker_cannot_finalize_validation_failed",
                extra={
                    "worker_id": self.worker_id,
                    "job_id": job_id,
                    "lease_attempt": attempt_count,
                    "current_status": "ownership_lost",
                    "execution_duration": execution_duration,
                    "failure_reason": str(own_exc),
                },
            )
            return None

        # Perform atomic, fenced state transition
        try:
            finalized = transition_job_status(
                self.database_path,
                job_id=job_id,
                worker_id=self.worker_id,
                lease_token=lease_token,
                target_status=handler_result.status,
                failure_reason=handler_result.failure_reason,
                promotion_approved=handler_result.promotion_approved,
                promotion_reason=handler_result.promotion_reason,
                promotion_audit_id=handler_result.promotion_audit_id,
                candidate_version=handler_result.candidate_version,
                candidate_run_id=handler_result.candidate_run_id,
                previous_champion_version=handler_result.previous_champion_version,
                payload_updates=handler_result.payload_updates,
            )
            logger.info(
                "job_execution_finalized",
                extra={
                    "worker_id": self.worker_id,
                    "job_id": job_id,
                    "lease_attempt": attempt_count,
                    "current_status": finalized["status"],
                    "execution_duration": execution_duration,
                },
            )
            return finalized
        except JobOwnershipError as own_exc:
            logger.warning(
                "stale_worker_cannot_finalize_transition_rejected",
                extra={
                    "worker_id": self.worker_id,
                    "job_id": job_id,
                    "lease_attempt": attempt_count,
                    "current_status": "ownership_lost",
                    "execution_duration": execution_duration,
                    "failure_reason": str(own_exc),
                },
            )
            return None

    def run(self) -> None:
        """Run worker polling loop with exclusive lock protection until stopped."""
        self._setup_signal_handlers()
        self.acquire_lock()
        try:
            logger.info(
                "worker_started",
                extra={"worker_id": self.worker_id, "lock_path": str(self.lock_path)},
            )
            while not self._shutdown_event.is_set():
                try:
                    self.run_once()
                except Exception as exc:
                    logger.exception(
                        "worker_poll_cycle_error",
                        extra={
                            "worker_id": self.worker_id,
                            "failure_reason": str(exc),
                        },
                    )
                if self._shutdown_event.wait(timeout=self.poll_interval_seconds):
                    break
        finally:
            self.release_lock()
            logger.info("worker_stopped", extra={"worker_id": self.worker_id})


def main(argv: list[str] | None = None) -> int:
    """Executable entry point for the retraining worker process."""
    parser = argparse.ArgumentParser(
        description="ModelForge Background Retraining Worker",
    )
    parser.add_argument("--db-path", default=None, help="Path to SQLite database")
    parser.add_argument("--lock-path", default=None, help="Path to worker lock file")
    parser.add_argument("--worker-id", default=None, help="Custom worker ID")
    parser.add_argument(
        "--poll-interval", type=float, default=None, help="Polling interval in seconds"
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=None,
        help="Heartbeat interval in seconds",
    )
    parser.add_argument(
        "--lease-duration",
        type=int,
        default=None,
        help="Lease duration in seconds",
    )
    parser.add_argument(
        "--shutdown-timeout",
        type=float,
        default=None,
        help="Graceful shutdown timeout in seconds",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Execute a single polling iteration and exit",
    )
    args = parser.parse_args(argv)

    worker = RetrainingWorker(
        database_path=args.db_path,
        lock_path=args.lock_path,
        worker_id=args.worker_id,
        poll_interval_seconds=args.poll_interval,
        heartbeat_interval_seconds=args.heartbeat_interval,
        lease_duration_seconds=args.lease_duration,
        shutdown_timeout=args.shutdown_timeout,
    )

    try:
        if args.once:
            with worker:
                worker.run_once()
        else:
            worker.run()
        return 0
    except WorkerLockConflictError as exc:
        logger.error(
            "worker_startup_failed_lock_conflict",
            extra={"worker_id": worker.worker_id, "failure_reason": str(exc)},
        )
        return 1
    except Exception as exc:
        logger.exception(
            "worker_fatal_error",
            extra={"worker_id": worker.worker_id, "failure_reason": str(exc)},
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
