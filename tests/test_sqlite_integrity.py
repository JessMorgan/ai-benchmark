"""Tests for SQLite run integrity diagnostics."""
from benchmark.sqlite_integrity import check_integrity
from benchmark.sqlite_schema import connect_database


def test_healthy_database_passes_integrity_check(tmp_path):
    connection = connect_database(str(tmp_path / "run.sqlite3"))
    try:
        report = check_integrity(connection)
        assert report.ok
        assert report.sqlite_integrity == "ok"
    finally:
        connection.close()


def test_integrity_report_detects_orphan_selection(tmp_path):
    connection = connect_database(str(tmp_path / "run.sqlite3"))
    try:
        connection.execute("INSERT INTO runs(run_id, created_at, status, score_schema, storage_profile) VALUES ('run', 1, 'running', 'v1', 'compact')")
        connection.execute("INSERT INTO run_revisions(revision_id, run_id, revision_number, status, runner_mode, config_json, config_sha256, created_at) VALUES (1, 'run', 1, 'running', 'http', '{}', 'hash', 1)")
        connection.execute("UPDATE runs SET current_revision_id = 1 WHERE run_id = 'run'")
        connection.commit()
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("DROP TRIGGER validate_benchmark_selection")
        connection.execute("INSERT INTO revision_cells(revision_id, cell_id, scheduled, status, updated_at) VALUES (1, 99, 1, 'completed', 1)")
        connection.execute("INSERT INTO benchmark_selections(revision_id, cell_id, attempt_id, selected_at) VALUES (1, 99, 123, 1)")
        connection.commit()
        report = check_integrity(connection)
        assert not report.ok
        assert any(issue.category == "orphan-selection" for issue in report.issues)
    finally:
        connection.close()
