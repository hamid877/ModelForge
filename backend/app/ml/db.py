"""Shared SQLite database connection helper and transaction utilities.

Provides uniform connection handling, foreign keys enforcement, and busy timeout
configuration across all ModelForge persistence components.
"""

import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from backend.app.config import settings

logger = logging.getLogger(__name__)


def get_db_connection(
    database_path: str | Path | None = None,
    *,
    timeout_ms: int | None = None,
    isolation_level: str | None = None,
    row_factory: Any = None,
) -> sqlite3.Connection:
    """Create a configured SQLite connection with foreign keys and busy timeout enabled.

    Args:
        database_path: Path to SQLite DB file. Defaults to lifecycle_audit_db_path.
        timeout_ms: Busy timeout in milliseconds. Defaults to sqlite_busy_timeout_ms.
        isolation_level: SQLite isolation level. Default None enables manual mode.
        row_factory: Optional row factory (e.g. sqlite3.Row).

    Returns:
        Configured sqlite3.Connection with foreign_keys and busy_timeout set.
    """
    path = Path(
        database_path if database_path is not None else settings.lifecycle_audit_db_path
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    timeout = (
        timeout_ms
        if timeout_ms is not None
        else getattr(settings, "sqlite_busy_timeout_ms", 5000)
    )

    conn = sqlite3.connect(str(path), isolation_level=isolation_level)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(f"PRAGMA busy_timeout = {int(timeout)};")
    if row_factory is not None:
        conn.row_factory = row_factory
    return conn


@contextmanager
def db_connection(
    database_path: str | Path | None = None,
    *,
    timeout_ms: int | None = None,
    isolation_level: str | None = None,
    row_factory: Any = None,
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for acquiring and closing a configured SQLite connection."""
    conn = get_db_connection(
        database_path=database_path,
        timeout_ms=timeout_ms,
        isolation_level=isolation_level,
        row_factory=row_factory,
    )
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def immediate_transaction(
    conn: sqlite3.Connection,
) -> Generator[sqlite3.Connection, None, None]:
    """Execute block inside BEGIN IMMEDIATE transaction with rollback on error."""
    conn.execute("BEGIN IMMEDIATE;")
    try:
        yield conn
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise
