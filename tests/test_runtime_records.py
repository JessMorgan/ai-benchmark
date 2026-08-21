"""Tests for structured records used by the storage façade."""
import os
import tempfile
import unittest

from benchmark.runtime_records import (
    BenchmarkAttemptRecord,
    JudgeVoteRecord,
    PluginRecord,
    RunContext,
    TargetRecord,
)
from benchmark.storage import RunIdentity, SQLiteRunStore


class TestRuntimeRecords(unittest.TestCase):
    def test_records_are_serializable(self):
        attempt = BenchmarkAttemptRecord(1, prompt="p", content="c", score=10)
        self.assertEqual(attempt.as_dict()["score"], 10)
        self.assertEqual(JudgeVoteRecord(score=8, usable=True).as_dict()["score"], 8)
        self.assertEqual(RunContext("r", 1).revision_id, 1)

    def test_sqlite_facade_registers_graph_and_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRunStore(os.path.join(tmp, "run.sqlite3"), flush_interval=0.01)
            store.start_run(RunIdentity("r", 1), score_schema="v1")
            target = TargetRecord("model", "http", "Local", "model", "sig")
            plugin = PluginRecord("p", "1.0.0", "P", 20, True)
            target_id = store.register_target(target)
            store.register_plugin(plugin)
            cell_id = store.ensure_cell(target, plugin)
            self.assertIsNotNone(target_id)
            self.assertIsNotNone(cell_id)
            attempt_id = store.record_benchmark_attempt(
                cell_id, BenchmarkAttemptRecord(1, prompt="p", content="c", score=12),
                selected=True,
            )
            self.assertIsNotNone(attempt_id)
            self.assertTrue(store.close(timeout=2))


if __name__ == "__main__":
    unittest.main()
