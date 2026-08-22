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

    def test_existing_database_with_imported_run_attaches_by_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.sqlite3")
            imported = SQLiteRunStore(path, flush_interval=0.01)
            imported.start_run(RunIdentity("imported-run", 1), source="test")
            target = _target()
            plugin = _plugin()
            imported.prepare_run([target], [plugin])
            cell_id = imported.get_cell_id("m", "http", "rate-limiter")
            imported.record_benchmark_attempt(
                cell_id, _benchmark_attempt(score=18, attempt_number=1), selected=True,
            )
            imported.flush(timeout=10)
            imported.close(timeout=10)

            # The CLI may not yet have a manifest matching the imported run.
            # A sole existing logical run must be continued, not shadowed by
            # an unrelated empty UUID run.
            resumed = SQLiteRunStore(path, flush_interval=0.01)
            resumed.start_run(RunIdentity("cli-generated-uuid", 2), source="test")
            self.assertFalse(resumed._is_new_run)
            self.assertEqual(resumed.identity.run_id, "imported-run")
            resumed.prepare_run([target], [plugin])
            resumed.continue_run(config={}, runner_mode="http", session_seed=None)
            rows = resumed.latest_results()
            self.assertEqual(rows[0]["rate-limiter_score"], 18)
            self.assertEqual(
                resumed._connection_operation(
                    lambda connection: connection.execute(
                        "SELECT count(*) FROM runs"
                    ).fetchone()[0]
                ),
                1,
            )
            resumed.close(timeout=10)

    def test_continuation_reuses_legacy_import_target_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.sqlite3")
            old = SQLiteRunStore(path, flush_interval=0.01)
            old.start_run(RunIdentity("imported-run", 1), source="test")
            legacy_target = TargetRecord(
                logical_name="m", runner="http", source="Local", api_model="m",
                target_signature="a" * 64,
            )
            plugin = _plugin()
            old.prepare_run([legacy_target], [plugin])
            cell_id = old.get_cell_id("m", "http", "rate-limiter")
            old.record_benchmark_attempt(
                cell_id, _benchmark_attempt(score=19, attempt_number=1), selected=True,
            )
            old.flush(timeout=10)
            old.close(timeout=10)

            resumed = SQLiteRunStore(path, flush_interval=0.01)
            resumed.start_run(RunIdentity("imported-run", 2), source="test")
            resumed.prepare_run([_target()], [plugin])
            resumed.continue_run(config={}, runner_mode="http", session_seed=None)
            rows = resumed.latest_results()
            self.assertEqual(rows[0]["rate-limiter_score"], 19)
            self.assertEqual(
                resumed._connection_operation(
                    lambda connection: connection.execute(
                        "SELECT count(*) FROM target_instances"
                    ).fetchone()[0]
                ),
                1,
            )
            resumed.close(timeout=10)

    def test_stale_shadow_run_recovers_to_legacy_imported_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.sqlite3")
            imported = SQLiteRunStore(path, flush_interval=0.01)
            imported.start_run(RunIdentity("imported-run", 1), source="test")
            imported.close(timeout=10)
            connection = __import__("sqlite3").connect(path)
            try:
                connection.execute(
                    "INSERT INTO legacy_import_records "
                    "(run_id, source_file, source_sha256, source_row_number, "
                    "record_kind, raw_json, mapping_status) "
                    "VALUES (?, '', ?, 0, 'result', '{}', 'mapped')",
                    ("imported-run", "import-hash"),
                )
                connection.commit()
            finally:
                connection.close()

            shadow = SQLiteRunStore(path, flush_interval=0.01)
            shadow.start_run(RunIdentity("stale-shadow", 1), source="test")
            shadow.close(timeout=10)

            resumed = SQLiteRunStore(path, flush_interval=0.01)
            resumed.start_run(RunIdentity("stale-shadow", 2), source="test")
            self.assertFalse(resumed._is_new_run)
            self.assertEqual(resumed.identity.run_id, "imported-run")
            resumed.close(timeout=10)

    def test_multiple_existing_runs_reject_unknown_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.sqlite3")
            first = SQLiteRunStore(path, flush_interval=0.01)
            first.start_run(RunIdentity("run-a", 1), source="test")
            first.close(timeout=10)
            second = SQLiteRunStore(path, flush_interval=0.01)
            second.start_run(RunIdentity("run-a", 1), source="test")
            second.close(timeout=10)
            # Add a second logical run through the schema store, as an
            # explicit restart/import operation would.
            import sqlite3
            from benchmark.sqlite_benchmarks import SQLiteBenchmarkStore
            connection = sqlite3.connect(path)
            try:
                SQLiteBenchmarkStore(connection).create_run(
                    "run-b", score_schema="v1", storage_profile="compact",
                    runner_mode="http", config={},
                )
            finally:
                connection.close()

            resumed = SQLiteRunStore(path, flush_interval=0.01)
            with self.assertRaisesRegex(RuntimeError, "multiple logical runs"):
                resumed.start_run(RunIdentity("unknown", 1), source="test")
            resumed.close(timeout=10)

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


    def test_continuation_preserves_registered_judges_and_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.sqlite3")
            store = SQLiteRunStore(path, flush_interval=0.01)
            store.start_run(RunIdentity("run", 1), source="test")
            target = _target()
            plugin = _plugin()
            store.prepare_run([target], [plugin])
            cell_id = store.get_cell_id("m", "http", "rate-limiter")
            store.record_benchmark_attempt(
                cell_id, _benchmark_attempt(score=18, attempt_number=1), selected=True,
            )
            store.register_judge("judge-a", "Local")
            store.register_contract(
                "contract-1", plugin_id="rate-limiter", plugin_version="1.0.0",
                prompt_version="judge-v8", instructions_version="1.0.0",
            )
            store.flush(timeout=10)
            store.close(timeout=10)

            resumed = SQLiteRunStore(path, flush_interval=0.01)
            resumed.start_run(RunIdentity("run", 2), source="test")
            resumed.prepare_run([target], [plugin])
            # Match the CLI ordering: current configuration is registered
            # before the continuation revision is created.
            resumed.register_judge("judge-a", "Local")
            resumed.register_contract(
                "contract-1", plugin_id="rate-limiter", plugin_version="1.0.0",
                prompt_version="judge-v8", instructions_version="1.0.0",
            )
            resumed.continue_run(config={}, runner_mode="http", session_seed=None)
            active_judge = resumed._connection_operation(
                lambda connection: connection.execute(
                    "SELECT active FROM revision_judges "
                    "WHERE revision_id = ? AND judge_model = 'judge-a'",
                    (resumed.revision_id,),
                ).fetchone()
            )
            active_contract = resumed._connection_operation(
                lambda connection: connection.execute(
                    "SELECT active FROM revision_judge_contracts "
                    "WHERE revision_id = ? AND plugin_id = 'rate-limiter'",
                    (resumed.revision_id,),
                ).fetchone()
            )
            self.assertEqual(active_judge[0], 1)
            self.assertEqual(active_contract[0], 1)
            resumed.close(timeout=10)

    def test_continuation_updates_plugin_definition_on_mismatch(self):
        """Plugin metadata can change between continuations without raising."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.sqlite3")
            store = SQLiteRunStore(path, flush_interval=0.01)
            store.start_run(RunIdentity("run", 1), source="test")
            target = _target()
            plugin = _plugin()  # version=1.0.0, max_score=20
            store.prepare_run([target], [plugin])
            cell_id = store.get_cell_id("m", "http", "rate-limiter")
            store.record_benchmark_attempt(
                cell_id, _benchmark_attempt(score=18, attempt_number=1),
                selected=True,
            )
            store.flush(timeout=10)
            store.close(timeout=10)

            # Simulate plugin edit: same version but changed metadata
            plugin_edited = PluginRecord(
                plugin_id="rate-limiter", plugin_version="1.0.0",
                name="Rate Limiter Updated", max_score=25.0,
                supports_streaming=False, metadata={"changed": True},
            )

            resumed = SQLiteRunStore(path, flush_interval=0.01)
            resumed.start_run(RunIdentity("run", 2), source="test")
            resumed.prepare_run([target], [plugin_edited])
            # This should NOT raise — should update the definition
            resumed.continue_run(config={}, runner_mode="http", session_seed=None)
            self.assertTrue(resumed.close(timeout=10))


if __name__ == "__main__":
    unittest.main()
