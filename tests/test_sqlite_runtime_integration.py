"""End-to-end integration tests for the SQLite runtime persistence path.

These exercise the same facade calls the live benchmark/judge runtime makes:
register the identity graph, record immutable attempts and votes, then resume
via a continuation and read the durable read model back for reports.
"""
import os
import tempfile
import unittest

from benchmark.runtime_records import (
    BenchmarkAttemptRecord,
    JudgeAttemptRecord,
    JudgeVoteRecord,
    PluginRecord,
    TargetRecord,
)
from benchmark.sqlite_reports import SQLiteReportSource
from benchmark.storage import RunIdentity, SQLiteRunStore


def _target(name="m"):
    return TargetRecord(
        logical_name=name, runner="http", source="Local",
        api_model=name, target_signature=f"Local/{name}",
    )


def _plugin():
    return PluginRecord(
        plugin_id="rate-limiter", plugin_version="1.0.0", name="Rate Limiter",
        max_score=20.0, supports_streaming=True,
    )


def _benchmark_attempt(score=15, attempt_number=1, status="completed"):
    return BenchmarkAttemptRecord(
        attempt_number=attempt_number, prompt="prompt", content="answer",
        thinking="", max_tokens=2048, output_tokens=10, thinking_tokens=0,
        total_tokens=10, tps=1.0, finish_reason="stop",
        response_nature="completed", stream_ok=True, score=score,
        rubric=[{"name": "a", "passed": True}], diagnostics={"ok": True},
        status=status,
    )


class TestSQLiteRuntimeIntegration(unittest.TestCase):
    def test_full_lifecycle_and_report_readback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.sqlite3")
            store = SQLiteRunStore(path, flush_interval=0.01)
            store.start_run(RunIdentity("run", 1), source="test")
            target = _target()
            plugin = _plugin()
            store.prepare_run([target], [plugin])
            cell_id = store.get_cell_id("m", "http", "rate-limiter")
            self.assertIsNotNone(cell_id)

            # Two attempts: the second is selected as the final result.
            store.record_benchmark_attempt(
                cell_id, _benchmark_attempt(score=5, attempt_number=1),
                selected=False,
            )
            store.record_benchmark_attempt(
                cell_id, _benchmark_attempt(score=15, attempt_number=2),
                selected=True,
            )
            # Register the judge and its contract before recording attempts,
            # mirroring the runtime's prepare_run wiring.
            store.register_judge("judge-a", "Local")
            store.register_contract(
                "contract-1", plugin_id="rate-limiter",
                plugin_version="1.0.0", prompt_version="judge-v8",
                instructions_version="1.0.0",
            )
            # Judge transport attempt + parsed vote.
            store.record_judge_attempt(
                cell_id,
                JudgeAttemptRecord(
                    judge_model="judge-a", contract_id="contract-1",
                    attempt_number=1, raw_response='{"score": 15}',
                    max_tokens=2048, status="completed",
                ),
                vote=JudgeVoteRecord(
                    score=15, confidence="high", rationale="ok",
                    criteria=[{"id": "c1", "criterion": "C", "evidence": "E"}],
                    usable=True,
                ),
            )
            store.flush(timeout=10)

            # Read back through the report source (what --generate-reports uses).
            connection = __import__("sqlite3").connect(path)
            connection.row_factory = __import__("sqlite3").Row
            try:
                source = SQLiteReportSource(connection)
                rows, active_plugins, _seed, _revision = source.load_results()
            finally:
                connection.close()
            self.assertEqual(active_plugins, ["rate-limiter"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["rate-limiter_score"], 15)
            self.assertEqual(rows[0]["rate-limiter_judge_score"], 15)
            self.assertTrue(store.close(timeout=10))

    def test_resume_reuses_completed_cells_and_reports_reused_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.sqlite3")
            store = SQLiteRunStore(path, flush_interval=0.01)
            store.start_run(RunIdentity("run", 1), source="test")
            target = _target()
            plugin = _plugin()
            store.prepare_run([target], [plugin])
            cell_id = store.get_cell_id("m", "http", "rate-limiter")
            store.record_benchmark_attempt(
                cell_id, _benchmark_attempt(score=18, attempt_number=1),
                selected=True,
            )
            store.flush(timeout=10)
            store.close(timeout=10)

            # A new facade over the same file simulates a resumed invocation.
            resumed = SQLiteRunStore(path, flush_interval=0.01)
            resumed.start_run(RunIdentity("run", 2), source="test")
            self.assertFalse(resumed._is_new_run)
            resumed.prepare_run([target], [plugin])
            resumed.continue_run(config={}, runner_mode="http", session_seed=None)
            rows = resumed.latest_results()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["rate-limiter_score"], 18)
            self.assertTrue(resumed.close(timeout=10))


if __name__ == "__main__":
    unittest.main()
