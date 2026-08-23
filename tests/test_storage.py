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
from benchmark.storage_validation import compare_read_models


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

    def test_json_and_sqlite_current_read_models_are_equivalent(self):
        state = BenchmarkState({"m": "Local"}, ["p"])
        json_store = JsonRunStore(state)
        json_store.start_run(RunIdentity("run", 1))
        json_store.record_result({
            "model": "m", "state_key": "m", "runner": "http",
            "source": "Local", "api_model": "m", "status": "ok",
            "p_score": 12, "p_output_tokens": 3, "p_thinking_tokens": 1,
            "p_total_tokens": 4, "p_response_time": 1.25, "p_tps": 2.4,
            "p_judge_score": 11, "p_judge_confidence": "high",
            "p_judge_rationale": "good", "p_judge_complete": True,
            "p_judge_votes": [{
                "model": "judge-a", "judge_contract_id": "contract-1",
                "score": 11, "confidence": "high",
                "rationale": "good", "usable": True,
            }],
        })

        with tempfile.TemporaryDirectory() as tmp:
            sqlite_store = SQLiteRunStore(os.path.join(tmp, "run.sqlite3"), flush_interval=0.01)
            sqlite_store.start_run(RunIdentity("run", 1))
            target = TargetRecord(
                logical_name="m", runner="http", source="Local",
                api_model="m", target_signature="Local/m",
            )
            plugin = PluginRecord(
                plugin_id="p", plugin_version="1.0.0", name="Plugin",
                max_score=20.0, supports_streaming=True,
            )
            sqlite_store.prepare_run([target], [plugin])
            cell_id = sqlite_store.get_cell_id("m", "http", "p")
            sqlite_store.record_benchmark_attempt(
                cell_id,
                BenchmarkAttemptRecord(
                    attempt_number=1, content="answer", output_tokens=3,
                    thinking_tokens=1, total_tokens=4, response_time=1.25,
                    tps=2.4, score=12, status="completed",
                ),
                selected=True,
            )
            sqlite_store.register_judge("judge-a", "Local")
            sqlite_store.register_contract(
                "contract-1", plugin_id="p", plugin_version="1.0.0",
                prompt_version="judge-v8", instructions_version="1.0.0",
            )
            sqlite_store.record_judge_attempt(
                cell_id,
                __import__("benchmark.runtime_records", fromlist=["JudgeAttemptRecord"]).JudgeAttemptRecord(
                    judge_model="judge-a", contract_id="contract-1",
                    attempt_number=1, raw_response='{"score": 11}', status="completed",
                ),
                vote=__import__("benchmark.runtime_records", fromlist=["JudgeVoteRecord"]).JudgeVoteRecord(
                    score=11, confidence="high", rationale="good", usable=True,
                ),
            )
            sqlite_store.flush(timeout=5)
            sqlite_rows = sqlite_store.latest_results()
            report = compare_read_models(json_store.latest_results(), sqlite_rows)
            self.assertTrue(report.equivalent, report.as_dict())
            self.assertTrue(sqlite_store.close(timeout=2))

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
