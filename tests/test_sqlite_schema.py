"""Tests for the Stage 2 SQLite schema and migration layer."""
import sqlite3
import tempfile
import unittest

from benchmark.sqlite_schema import (
    SQLITE_SCHEMA_VERSION,
    configure_connection,
    connect_database,
    foreign_keys_enabled,
    initialize_schema,
    schema_version,
    sqlite_options,
    table_names,
)


class TestSQLiteSchema(unittest.TestCase):
    def test_new_database_has_expected_version_tables_and_options(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            connection = connect_database(f"{tmpdir}/run.sqlite3")
            self.assertEqual(schema_version(connection), SQLITE_SCHEMA_VERSION)
            self.assertTrue(foreign_keys_enabled(connection))
            self.assertEqual(str(sqlite_options(connection)["journal_mode"]).lower(), "wal")
            self.assertIn("cells", table_names(connection))
            self.assertIn("judge_vote_attempts", table_names(connection))
            self.assertIn("current_judge_votes", table_names(connection))
            self.assertIn("legacy_import_records", table_names(connection))
            connection.close()

    def test_initialize_is_idempotent(self):
        connection = sqlite3.connect(":memory:")
        configure_connection(connection)
        initialize_schema(connection)
        initialize_schema(connection)
        self.assertEqual(schema_version(connection), 1)
        connection.close()

    def test_missing_version_is_migrated_forward(self):
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO schema_meta VALUES ('schema_version', '0')")
        initialize_schema(connection)
        self.assertEqual(schema_version(connection), SQLITE_SCHEMA_VERSION)
        connection.close()

    def test_newer_schema_is_rejected(self):
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO schema_meta VALUES ('schema_version', '999')")
        with self.assertRaisesRegex(RuntimeError, "newer"):
            initialize_schema(connection)
        connection.close()

    def test_revision_and_selection_integrity(self):
        connection = sqlite3.connect(":memory:")
        configure_connection(connection)
        initialize_schema(connection)
        connection.execute(
            "INSERT INTO runs VALUES ('run-a', 1, 'running', 'score-v1', 'compact', NULL)"
        )
        connection.execute(
            "INSERT INTO run_revisions VALUES (1, 'run-a', 1, 'running', NULL, NULL, 'http', NULL, '{}', 'hash', 1)"
        )
        connection.execute("UPDATE runs SET current_revision_id = 1 WHERE run_id = 'run-a'")
        connection.execute(
            "INSERT INTO target_instances VALUES "
            "(1, 'run-a', 'model-a', 'http', 'Local', 'model-a', 0, NULL, NULL, 'sig-a', 1, NULL)"
        )
        connection.execute("INSERT INTO revision_targets VALUES (1, 1, 1, 0)")
        connection.execute(
            "INSERT INTO plugin_definitions VALUES ('plugin', '1.0.0', 'Plugin', 20, 1, NULL)"
        )
        connection.execute("INSERT INTO revision_plugins VALUES (1, 'plugin', '1.0.0', 1)")
        connection.execute(
            "INSERT INTO cells VALUES (1, 'run-a', 1, 'plugin', '1.0.0', 1)"
        )
        connection.execute("INSERT INTO revision_cells VALUES (1, 1, 1, 'pending', NULL, 1)")
        connection.execute(
            "INSERT INTO benchmark_attempts "
            "(attempt_id, revision_id, cell_id, attempt_number) VALUES (1, 1, 1, 1)"
        )
        connection.execute(
            "INSERT INTO benchmark_selections VALUES (1, 1, 1, 2, 'completed')"
        )
        with self.assertRaises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO benchmark_selections VALUES (1, 1, 999, 2, 'bad')"
            )
        connection.commit()
        connection.close()

    def test_current_revision_must_belong_to_run(self):
        connection = sqlite3.connect(":memory:")
        configure_connection(connection)
        initialize_schema(connection)
        connection.execute("INSERT INTO runs VALUES ('run-a', 1, 'running', 'v1', 'compact', NULL)")
        connection.execute("INSERT INTO runs VALUES ('run-b', 1, 'running', 'v1', 'compact', NULL)")
        connection.execute(
            "INSERT INTO run_revisions VALUES (1, 'run-a', 1, 'running', NULL, NULL, 'http', NULL, '{}', 'h', 1)"
        )
        with self.assertRaisesRegex(sqlite3.IntegrityError, "current revision"):
            connection.execute("UPDATE runs SET current_revision_id = 1 WHERE run_id = 'run-b'")
        connection.close()

    def test_current_judge_vote_identity_is_validated(self):
        connection = sqlite3.connect(":memory:")
        configure_connection(connection)
        initialize_schema(connection)
        connection.execute("INSERT INTO runs VALUES ('r', 1, 'running', 'v1', 'compact', NULL)")
        connection.execute(
            "INSERT INTO run_revisions VALUES (1, 'r', 1, 'running', NULL, NULL, 'http', NULL, '{}', 'h', 1)"
        )
        connection.execute(
            "INSERT INTO target_instances VALUES "
            "(1, 'r', 'm', 'http', 's', 'm', 0, NULL, NULL, 'sig', 1, NULL)"
        )
        connection.execute("INSERT INTO plugin_definitions VALUES ('p', '1', 'P', 20, 1, NULL)")
        connection.execute("INSERT INTO cells VALUES (1, 'r', 1, 'p', '1', 1)")
        connection.execute("INSERT INTO revision_cells VALUES (1, 1, 1, 'done', NULL, 1)")
        connection.execute("INSERT INTO judge_contracts VALUES ('c', 'p', '1', 'jv', 'iv', 'sh', '{}', 'ch')")
        connection.execute(
            "INSERT INTO judge_attempts "
            "(judge_attempt_id, revision_id, cell_id, judge_model, contract_id, attempt_number) "
            "VALUES (1, 1, 1, 'judge-a', 'c', 1)"
        )
        connection.execute(
            "INSERT INTO judge_vote_attempts VALUES (1, 1, 90, 'high', 'ok', NULL, 1, 1)"
        )
        connection.execute(
            "INSERT INTO current_judge_votes VALUES (1, 1, 'judge-a', 'c', 1, 2, 'latest')"
        )
        with self.assertRaises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO current_judge_votes VALUES (1, 1, 'judge-b', 'c', 1, 2, 'wrong identity')"
            )
        connection.close()


if __name__ == "__main__":
    unittest.main()
