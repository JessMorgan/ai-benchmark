"""Tests for the legacy JSON to SQLite importer."""
import json
import os
import tempfile
import unittest

from benchmark.sqlite_import import LegacySQLiteImporter
from benchmark.sqlite_schema import connect_database


class TestSQLiteImport(unittest.TestCase):
    def test_imports_state_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "benchmark_state.json")
            database = os.path.join(tmp, "run.sqlite3")
            state = {
                "score_schema": "v1",
                "runner": "http",
                "session_seed": 7,
                "active_plugins": ["rate-limiter"],
                "plugin_versions": {"rate-limiter": "1.0.0"},
                "model_info": {"model-a": {}},
                "results": [{
                    "model": "model-a", "source": "Local", "api_model": "model-a",
                    "status": "ok", "rate-limiter_score": 18,
                    "rate-limiter_output_tokens": 20,
                    "rate-limiter_attempts": [
                        {"attempt": 1, "retry_reason": "transport"},
                        {"attempt": 2, "retry_reason": "thinking"},
                    ],
                    "rate-limiter_selected_attempt": 2,
                    "rate-limiter_judge_votes": [{
                        "model": "judge-a", "score": 17, "confidence": "high",
                        "rationale": "good", "usable": True,
                    }],
                }, {
                    "not": "mappable",
                }],
            }
            with open(source, "w", encoding="utf-8") as handle:
                json.dump(state, handle)

            summary = LegacySQLiteImporter.import_path(source, database)
            self.assertEqual(summary.imported_targets, 1)
            self.assertEqual(summary.imported_cells, 1)
            self.assertEqual(summary.imported_attempts, 1)
            self.assertEqual(summary.imported_votes, 1)
            self.assertEqual(summary.ambiguous_records, 1)

            connection = connect_database(database)
            self.addCleanup(connection.close)
            self.assertEqual(connection.execute("SELECT count(*) FROM runs").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT count(*) FROM benchmark_attempts").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT count(*) FROM judge_vote_attempts").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT count(*) FROM legacy_import_records").fetchone()[0], 2)

            again = LegacySQLiteImporter.import_path(source, database)
            self.assertEqual(again.imported_attempts, 0)
            self.assertEqual(connection.execute("SELECT count(*) FROM runs").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT count(*) FROM benchmark_attempts").fetchone()[0], 1)

    def test_import_uses_latest_duplicate_result_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "benchmark_state.json")
            database = os.path.join(tmp, "run.sqlite3")
            state = {
                "score_schema": "v1",
                "runner": "http",
                "active_plugins": ["rate-limiter"],
                "plugin_versions": {"rate-limiter": "1.0.0"},
                "model_info": {"model-a": {}},
                "results": [
                    {"model": "model-a", "source": "Local", "api_model": "model-a",
                     "status": "ok", "rate-limiter_score": 4,
                     "rate-limiter_selected_attempt": 1},
                    {"model": "model-a", "source": "Local", "api_model": "model-a",
                     "status": "ok", "rate-limiter_score": 17,
                     "rate-limiter_selected_attempt": 1},
                ],
            }
            with open(source, "w", encoding="utf-8") as handle:
                json.dump(state, handle)

            summary = LegacySQLiteImporter.import_path(source, database)
            self.assertEqual(summary.imported_attempts, 1)
            connection = connect_database(database)
            self.addCleanup(connection.close)
            score = connection.execute(
                "SELECT score FROM benchmark_attempts"
            ).fetchone()[0]
            self.assertEqual(score, 17)

    def test_import_uses_explicit_selected_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "benchmark_state.json")
            database = os.path.join(tmp, "run.sqlite3")
            state = {
                "score_schema": "v1", "runner": "http",
                "active_plugins": ["rate-limiter"],
                "plugin_versions": {"rate-limiter": "1.0.0"},
                "model_info": {"model-a": {}},
                "results": [{
                    "model": "model-a", "source": "Local", "api_model": "model-a",
                    "status": "ok", "rate-limiter_score": 12,
                    "rate-limiter_selected_attempt": 1,
                    "rate-limiter_attempts": [
                        {"attempt": 1, "score": 12, "content": "selected"},
                        {"attempt": 2, "score": 99, "content": "later retry"},
                    ],
                }],
            }
            with open(source, "w", encoding="utf-8") as handle:
                json.dump(state, handle)
            LegacySQLiteImporter.import_path(source, database)
            connection = connect_database(database)
            self.addCleanup(connection.close)
            row = connection.execute(
                """
                SELECT a.attempt_number, p.data, a.score
                FROM benchmark_attempts a
                JOIN payloads p ON p.payload_id = a.content_payload_id
                """
            ).fetchone()
            import gzip
            self.assertEqual((row[0], gzip.decompress(row[1]), row[2]), (1, b"selected", 12))

    def test_import_rejects_non_object_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "state.json")
            database = os.path.join(tmp, "run.sqlite3")
            with open(source, "w", encoding="utf-8") as handle:
                json.dump([], handle)
            with self.assertRaises(TypeError):
                LegacySQLiteImporter.import_path(source, database)


if __name__ == "__main__":
    unittest.main()
