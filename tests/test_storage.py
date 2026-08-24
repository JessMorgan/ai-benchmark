"""Tests for the backend-neutral storage façade."""
import os
import random
import sqlite3
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
            store.update_model("m", status="running", elapsed=1.25, last_error="")
            store.flush(timeout=5)
            hydrated = store.latest_results()
            self.assertEqual(hydrated[0]["status"], "running")
            self.assertEqual(hydrated[0]["elapsed"], 1.25)
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

    def test_register_contract_activates_canonical_contract_id(self):
        from benchmark.contracts import JudgeContract

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.sqlite3")
            store = SQLiteRunStore(path, flush_interval=0.01)
            store.start_run(RunIdentity("run", 1))
            target = TargetRecord(
                logical_name="m", runner="http", source="Local",
                api_model="m", target_signature="Local/m",
            )
            plugin = PluginRecord(
                plugin_id="p", plugin_version="1.0.0", name="Plugin",
                max_score=20.0, supports_streaming=True,
            )
            store.prepare_run([target], [plugin])
            contract = JudgeContract.from_definition(
                plugin_id="p", plugin_version="1.0.0", prompt_version="judge-v8",
                instructions_version="1.0.0", response_schema={"type": "object"},
                instructions="Evaluate explicit requirements.",
            )
            store.register_contract(
                "legacy-id", plugin_id="p", plugin_version="1.0.0",
                prompt_version="judge-v8", instructions_version="1.0.0",
                contract=contract,
            )
            store.flush(timeout=5)
            connection = sqlite3.connect(path)
            try:
                active = connection.execute(
                    "SELECT contract_id FROM revision_judge_contracts WHERE revision_id = ?",
                    (store.revision_id,),
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(active[0], contract.contract_id)
            self.assertTrue(store.close(timeout=2))

    def test_seeded_mixed_event_sequence_preserves_backend_parity(self):
        """Exercise a reproducible mix of completed, failed, and retried cells."""
        rng = random.Random(20260823)
        targets = [
            TargetRecord(
                logical_name=f"m-{index}", runner="http", source="Local",
                api_model=f"m-{index}", target_signature=f"Local/m-{index}",
            )
            for index in range(8)
        ]
        plugin = PluginRecord(
            plugin_id="p", plugin_version="1.0.0", name="Plugin",
            max_score=20.0, supports_streaming=True,
        )
        state = BenchmarkState({target.logical_name: "Local" for target in targets}, ["p"])
        json_store = JsonRunStore(state)
        json_store.start_run(RunIdentity("run", 1))

        with tempfile.TemporaryDirectory() as tmp:
            sqlite_store = SQLiteRunStore(os.path.join(tmp, "run.sqlite3"), flush_interval=0.01)
            sqlite_store.start_run(RunIdentity("run", 1))
            sqlite_store.prepare_run(targets, [plugin])
            for index, target in enumerate(targets):
                score = rng.choice([None, 8, 12, 17])
                retry = rng.choice([None, "transport", "thinking-budget"])
                attempts = 2 if retry else 1
                row = {
                    "model": target.logical_name, "state_key": target.logical_name,
                    "runner": "http", "source": "Local", "api_model": target.api_model,
                    "status": "ok" if score is not None else "error",
                    "p_score": score if score is not None else "fail",
                    "p_output_tokens": 10 + index,
                    "p_thinking_tokens": index % 3,
                    "p_total_tokens": 10 + index + index % 3,
                    "p_response_time": round(0.5 + index / 10, 2),
                    "p_tps": round(10.0 + index, 2),
                    "p_attempt_count": attempts,
                }
                if retry:
                    row["p_retry_reasons"] = ["transport", retry] if attempts == 2 else [retry]
                json_store.record_result(row)
                cell_id = sqlite_store.get_cell_id(target.logical_name, "http", "p")
                assert cell_id is not None
                for attempt_number in range(1, attempts + 1):
                    sqlite_store.record_benchmark_attempt(
                        cell_id,
                        BenchmarkAttemptRecord(
                            attempt_number=attempt_number,
                            content="answer" if score is not None else "",
                            output_tokens=10 + index,
                            thinking_tokens=index % 3,
                            total_tokens=10 + index + index % 3,
                            response_time=round(0.5 + index / 10, 2),
                            tps=round(10.0 + index, 2),
                            score=score if attempt_number == attempts else None,
                            status="completed" if score is not None else "failed",
                            retry_reason=retry if attempt_number == attempts else "transport",
                        ),
                        selected=attempt_number == attempts,
                    )
            sqlite_store.flush(timeout=5)
            report = compare_read_models(json_store.latest_results(), sqlite_store.latest_results())
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
