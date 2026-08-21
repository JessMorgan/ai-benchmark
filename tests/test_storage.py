"""Tests for the backend-neutral storage façade."""
import os
import tempfile
import unittest

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

    def test_sqlite_facade_judge_projection_uses_common_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRunStore(os.path.join(tmp, "run.sqlite3"))
            store.start_run(RunIdentity("run", 1))
            store.record_judge_result(
                "m", "http", "p", score=12, status="complete",
            )
            self.assertEqual(store.latest_results()[0]["p_score"], 12)
            self.assertTrue(store.close(timeout=2))


if __name__ == "__main__":
    unittest.main()
