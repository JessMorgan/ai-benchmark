"""Tests for the backend-neutral storage façade."""
import os
import tempfile
import unittest

from benchmark.runtime_records import (
    BenchmarkAttemptRecord,
    PluginRecord,
    TargetRecord,
)
from benchmark.sqlite_schema import connect_database
from benchmark.state import BenchmarkState
from benchmark.storage import JsonRunStore, RunIdentity, RunStore, SQLiteRunStore


class TestRunStoreContract(unittest.TestCase):
    def test_json_store_implements_facade_and_preserves_state(self):
        state = BenchmarkState({"m": "Local"}, ["p"])
        store = JsonRunStore(state)
        self.assertIsInstance(store, RunStore)
        store.start_run(RunIdentity("run", 1), source="test")
        store.record_result({"model": "m", "status": "ok", "p_score": 10})
        self.assertEqual(store.latest_results()[0]["p_score"], 10)
        self.assertTrue(store.close())

    def test_sqlite_store_queues_operations_and_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.sqlite3")
            store = SQLiteRunStore(path, flush_interval=0.01)
            self.assertIsInstance(store, RunStore)
            store.start_run(RunIdentity("run", 1), source="test")
            future = store.submit(lambda connection: connection.execute(
                "INSERT INTO schema_meta(key, value) VALUES ('test', 'ok')"
            ))
            future.result(timeout=2)
            store.flush(timeout=2)
            connection = connect_database(path)
            self.addCleanup(connection.close)
            self.assertEqual(
                connection.execute("SELECT value FROM schema_meta WHERE key = 'test'").fetchone()[0],
                "ok",
            )
            self.assertTrue(store.close(timeout=2))

    def test_sqlite_facade_latest_results_reads_normalized_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRunStore(os.path.join(tmp, "run.sqlite3"), flush_interval=0.01)
            store.start_run(RunIdentity("run", 1), source="test")
            target = TargetRecord(
                logical_name="m", runner="http", source="Local",
                api_model="m", target_signature="Local/m",
            )
            plugin = PluginRecord(
                plugin_id="p", plugin_version="1.0.0", name="Plugin",
                max_score=20.0, supports_streaming=True,
            )
            store.prepare_run([target], [plugin])
            cell_id = store.get_cell_id("m", "http", "p")
            self.assertIsNotNone(cell_id)
            store.record_benchmark_attempt(
                cell_id,
                BenchmarkAttemptRecord(
                    attempt_number=1, prompt="q", content="a",
                    output_tokens=1, thinking_tokens=0, total_tokens=1,
                    score=12, status="completed",
                ),
                selected=True,
            )
            store.flush(timeout=5)
            self.assertEqual(store.latest_results()[0]["p_score"], 12)
            self.assertTrue(store.close(timeout=2))

    def test_sqlite_continuation_reuses_completed_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.sqlite3")
            store = SQLiteRunStore(path, flush_interval=0.01)
            store.start_run(RunIdentity("run", 1), source="test")
            target = TargetRecord(
                logical_name="m", runner="http", source="Local",
                api_model="m", target_signature="Local/m",
            )
            plugin = PluginRecord(
                plugin_id="p", plugin_version="1.0.0", name="Plugin",
                max_score=20.0, supports_streaming=True,
            )
            store.prepare_run([target], [plugin])
            cell_id = store.get_cell_id("m", "http", "p")
            store.record_benchmark_attempt(
                cell_id,
                BenchmarkAttemptRecord(
                    attempt_number=1, prompt="q", content="a",
                    output_tokens=1, thinking_tokens=0, total_tokens=1,
                    score=12, status="completed",
                ),
                selected=True,
            )
            store.flush(timeout=5)
            # Second invocation: fresh facade over the same database file.
            resumed = SQLiteRunStore(path, flush_interval=0.01)
            resumed.start_run(RunIdentity("run", 2), source="test")
            self.assertFalse(resumed._is_new_run)
            resumed.prepare_run([target], [plugin])
            resumed.continue_run(
                config={}, runner_mode="http", session_seed=None,
            )
            rows = resumed.latest_results()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["p_score"], 12)
            self.assertTrue(resumed.close(timeout=2))


if __name__ == "__main__":
    unittest.main()
