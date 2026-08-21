"""Tests for normalized SQLite benchmark attempt persistence."""
import os
import sqlite3
import tempfile
import unittest

from benchmark.sqlite_benchmarks import SQLiteBenchmarkStore
from benchmark.sqlite_schema import connect_database


class TestSQLiteBenchmarkStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.connection = connect_database(os.path.join(self.tmpdir.name, "run.sqlite3"))
        self.addCleanup(self.connection.close)
        self.store = SQLiteBenchmarkStore(self.connection)
        self.revision = self.store.create_run(
            "run-a",
            score_schema="score-v1",
            storage_profile="compact",
            runner_mode="http",
            config={"models": {"model-a": "Local"}},
            session_seed=42,
        )
        self.store.register_plugin(
            "plugin", "1.0.0", name="Plugin", max_score=20,
            supports_streaming=True,
        )
        self.store.activate_plugin(self.revision, "plugin", "1.0.0")
        self.target = self.store.register_target(
            self.revision,
            run_id="run-a",
            logical_name="model-a",
            runner="http",
            source="Local",
            api_model="model-a",
            target_signature="sig-a",
        )
        self.cell = self.store.ensure_cell(
            self.revision, self.target, "plugin", "1.0.0",
        )

    def test_setup_creates_revision_membership_and_cell(self):
        self.assertEqual(self.revision, 1)
        self.assertEqual(
            self.connection.execute("SELECT current_revision_id FROM runs WHERE run_id = 'run-a'").fetchone()[0],
            1,
        )
        self.assertEqual(
            self.connection.execute("SELECT active FROM revision_plugins WHERE revision_id = 1").fetchone()[0],
            1,
        )
        self.assertEqual(
            self.connection.execute("SELECT status FROM revision_cells WHERE cell_id = ?", (self.cell,)).fetchone()[0],
            "pending",
        )

    def test_attempts_are_immutable_and_selection_is_revision_local(self):
        first = self.store.record_attempt(
            self.revision,
            self.cell,
            {
                "attempt_number": 1,
                "prompt": "prompt",
                "content": "partial",
                "thinking": "reasoning",
                "max_tokens": 100,
                "output_tokens": 2,
                "thinking_tokens": 3,
                "total_tokens": 5,
                "error": "transport failure",
                "failure_cause": "transport",
            },
        )
        second = self.store.record_attempt(
            self.revision,
            self.cell,
            {
                "attempt_number": 2,
                "prompt": "prompt",
                "content": "answer",
                "thinking": "short reasoning",
                "score": 18,
                "rubric": [{"name": "quality", "points": 18}],
                "diagnostics": {"finish_reason": "stop"},
            },
            selected=True,
        )
        current = self.store.current_attempt(self.revision, self.cell)
        self.assertEqual(current["attempt_id"], second)
        self.assertNotEqual(first, second)
        self.assertEqual(current["score"], 18)
        self.assertIsNotNone(current["prompt_payload_id"])
        self.assertIsNotNone(current["content_payload_id"])
        self.assertEqual(
            self.connection.execute("SELECT count(*) FROM benchmark_attempts").fetchone()[0],
            2,
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.record_attempt(
                self.revision, self.cell, {"attempt_number": 2, "score": 19}
            )

    def test_resume_decision_reruns_failed_but_reuses_success(self):
        self.assertTrue(self.store.should_run_cell(self.revision, self.cell))
        self.store.record_attempt(
            self.revision, self.cell,
            {"attempt_number": 1, "error": "timeout"}, selected=True,
        )
        self.assertTrue(self.store.should_run_cell(self.revision, self.cell))
        self.assertFalse(self.store.should_run_cell(self.revision, self.cell, rerun_failed=False))
        self.store.record_attempt(
            self.revision, self.cell,
            {"attempt_number": 2, "score": 10}, selected=True,
        )
        self.assertFalse(self.store.should_run_cell(self.revision, self.cell))

    def test_payloads_are_deduplicated_and_legacy_files_are_on_demand(self):
        self.store.record_attempt(
            self.revision, self.cell,
            {
                "attempt_number": 1,
                "prompt": "same prompt",
                "content": "same answer",
                "thinking": "same thought",
                "score": 10,
            },
            selected=True,
        )
        paths = self.store.materialize_response(
            self.revision, self.cell, self.tmpdir.name, "model-a", "plugin",
        )
        self.assertEqual(len(paths), 3)
        self.assertTrue(all(os.path.isfile(path) for path in paths))
        self.assertEqual(
            self.connection.execute("SELECT count(*) FROM payloads").fetchone()[0],
            3,
        )

    def test_invalid_cell_selection_is_rejected(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.select_attempt(self.revision, self.cell, 999)


if __name__ == "__main__":
    unittest.main()
