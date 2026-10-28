import json
import sqlite3

import pytest

from backend.app.ml.audit import AuditDatabaseError, record_lifecycle_run


def test_record_lifecycle_run_persists_payload(tmp_path):
    database_path = tmp_path / "audit.db"
    record = {
        "status": "stable",
        "reason": "no_feature_drift_detected",
        "training_attempted": False,
    }

    metadata = record_lifecycle_run(record, database_path)

    assert metadata["run_id"]
    assert metadata["timestamp"]

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT run_id, timestamp, status, payload
            FROM lifecycle_audit
            """
        ).fetchone()

    assert row is not None
    assert row[0] == metadata["run_id"]
    assert row[1] == metadata["timestamp"]
    assert row[2] == "stable"

    payload = json.loads(row[3])
    assert payload["status"] == "stable"
    assert payload["reason"] == "no_feature_drift_detected"
    assert payload["training_attempted"] is False
    assert payload["run_id"] == metadata["run_id"]


def test_record_lifecycle_run_creates_parent_directory(tmp_path):
    database_path = tmp_path / "nested" / "audit.db"

    metadata = record_lifecycle_run(
        {"status": "promoted"},
        database_path,
    )

    assert database_path.exists()
    assert metadata["run_id"]


def test_each_lifecycle_record_has_unique_id(tmp_path):
    database_path = tmp_path / "audit.db"

    first = record_lifecycle_run({"status": "stable"}, database_path)
    second = record_lifecycle_run({"status": "promoted"}, database_path)

    assert first["run_id"] != second["run_id"]

    with sqlite3.connect(database_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM lifecycle_audit").fetchone()[0]

    assert count == 2


def test_get_promotion_audit_records_missing_table(tmp_path):
    database_path = tmp_path / "empty.db"
    with sqlite3.connect(database_path) as conn:
        conn.execute("CREATE TABLE other_table (id INT)")

    from backend.app.ml.audit import get_promotion_audit_records

    records = get_promotion_audit_records(database_path=database_path)
    assert records == []


def test_get_promotion_audit_records_corrupted_file(tmp_path):
    corrupt_db = tmp_path / "corrupt.db"
    corrupt_db.write_text("not a valid sqlite file")

    from backend.app.ml.audit import get_promotion_audit_records

    with pytest.raises(
        AuditDatabaseError, match="Failed to read promotion audit database"
    ):
        get_promotion_audit_records(database_path=corrupt_db)
