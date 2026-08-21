"""Tests for revision-aware SQLite continuation and lifecycle behavior."""
import os
import tempfile
import unittest

from benchmark.sqlite_benchmarks import SQLiteBenchmarkStore
from benchmark.sqlite_continuation import (
    ContractSpec,
    JudgeSpec,
    PluginSpec,
    SQLiteContinuationStore,
    TargetSpec,
)
from benchmark.sqlite_judges import SQLiteJudgeStore
from benchmark.sqlite_schema import connect_database


class TestSQLiteContinuation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.connection = connect_database(os.path.join(self.tmpdir.name, "run.sqlite3"))
        self.addCleanup(self.connection.close)
        self.benchmark = SQLiteBenchmarkStore(self.connection)
        self.revision = self.benchmark.create_run(
            "run-a", score_schema="v1", storage_profile="compact",
            runner_mode="http", config={"revision": 1}, session_seed=1,
        )
        self.plugin = PluginSpec("plugin", "1.0.0", "Plugin", 20, True)
        self.benchmark.register_plugin(
            self.plugin.plugin_id, self.plugin.plugin_version,
            name=self.plugin.name, max_score=self.plugin.max_score,
            supports_streaming=self.plugin.supports_streaming,
        )
        self.benchmark.activate_plugin(self.revision, "plugin", "1.0.0")
        self.target = self.benchmark.register_target(
            self.revision, run_id="run-a", logical_name="model-a", runner="http",
            source="Local", api_model="model-a", target_signature="sig-a",
        )
        self.cell = self.benchmark.ensure_cell(self.revision, self.target, "plugin", "1.0.0")
        self.judge = SQLiteJudgeStore(self.connection)
        self.judge.register_judge(self.revision, "judge-a", source="Local", config={"v": 1})
        self.contract = ContractSpec(
            "contract-v1", "plugin", "1.0.0", "judge-v1", "1.0.0",
            "schema-v1", {"version": 1}, "contract-hash-v1",
        )
        self.judge.register_contract(
            self.contract.contract_id, plugin_id="plugin", plugin_version="1.0.0",
            prompt_version=self.contract.prompt_version,
            instructions_version=self.contract.instructions_version,
            response_schema_hash=self.contract.response_schema_hash,
            contract=self.contract.contract, contract_hash=self.contract.contract_hash,
        )
        self.judge.activate_contract(self.revision, "plugin", self.contract.contract_id)
        attempt = self.benchmark.record_attempt(
            self.revision, self.cell,
            {"attempt_number": 1, "prompt": "p", "content": "answer", "score": 18},
            selected=True,
        )
        judge_attempt = self.judge.record_attempt(
            self.revision, self.cell, "judge-a", self.contract.contract_id,
            {"attempt_number": 1, "raw_response": '{"score": 18}'},
        )
        vote = self.judge.record_vote(judge_attempt, {"score": 18, "usable": True})
        self.judge.select_vote(
            self.revision, self.cell, "judge-a", self.contract.contract_id, vote,
        )
        self.assertIsNotNone(attempt)

    def _continuation(self, **kwargs):
        defaults = {
            "config": {"revision": 2},
            "runner_mode": "http",
            "targets": [TargetSpec("model-a", "http", "Local", "model-a", "sig-a")],
            "plugins": [self.plugin],
            "judges": [JudgeSpec("judge-a", "Local", {"v": 1})],
            "contracts": {"plugin": self.contract},
            "session_seed": 2,
        }
        defaults.update(kwargs)
        return SQLiteContinuationStore(self.connection).create_continuation(
            "run-a", **defaults,
        )

    def test_compatible_attempt_and_vote_are_reused_in_new_revision(self):
        summary = self._continuation()
        self.assertEqual(summary.reused_cells, 1)
        self.assertEqual(summary.scheduled_cells, 0)
        self.assertEqual(summary.reused_votes, 1)
        new_revision = summary.revision_id
        selected = self.connection.execute(
            "SELECT attempt_id FROM benchmark_selections WHERE revision_id = ?", (new_revision,)
        ).fetchone()[0]
        old_selected = self.connection.execute(
            "SELECT attempt_id FROM benchmark_selections WHERE revision_id = 1"
        ).fetchone()[0]
        self.assertEqual(selected, old_selected)
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM benchmark_attempts WHERE revision_id = 1"
            ).fetchone()[0],
            1,
        )
        self.assertEqual(
            self.judge.current_votes(new_revision, self.cell, self.contract.contract_id)[0]["score"],
            18,
        )

    def test_added_removed_and_changed_memberships_are_revision_local(self):
        new_plugin = PluginSpec("new-plugin", "1.0.0", "New", 10, False)
        new_target = TargetSpec("model-b", "http", "Local", "model-b", "sig-b")
        summary = self._continuation(
            targets=[new_target],
            plugins=[new_plugin],
            judges=[],
            contracts={},
        )
        self.assertEqual(summary.removed_targets, 1)
        self.assertEqual(summary.added_targets, 1)
        self.assertEqual(summary.removed_plugins, 1)
        self.assertEqual(summary.added_plugins, 1)
        self.assertEqual(
            self.connection.execute(
                "SELECT active FROM revision_targets WHERE revision_id = ? AND target_instance_id = ?",
                (summary.revision_id, self.target),
            ).fetchone()[0],
            0,
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM benchmark_attempts"
            ).fetchone()[0],
            1,
        )

    def test_changed_plugin_version_and_contract_do_not_reuse_old_results(self):
        changed_plugin = PluginSpec("plugin", "2.0.0", "Plugin", 20, True)
        changed_contract = ContractSpec(
            "contract-v2", "plugin", "2.0.0", "judge-v2", "1.0.0",
            "schema-v2", {"version": 2}, "contract-hash-v2",
        )
        summary = self._continuation(
            plugins=[changed_plugin],
            contracts={"plugin": changed_contract},
        )
        self.assertEqual(summary.reused_cells, 0)
        self.assertEqual(summary.scheduled_cells, 1)
        self.assertEqual(summary.reused_votes, 0)
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM cells WHERE plugin_id = 'plugin'"
            ).fetchone()[0],
            2,
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM judge_contracts"
            ).fetchone()[0],
            2,
        )

    def test_failed_result_respects_rerun_policy(self):
        self.benchmark.register_plugin(
            "failed-plugin", "1.0.0", name="Failed", max_score=20,
            supports_streaming=True,
        )
        self.benchmark.activate_plugin(self.revision, "failed-plugin", "1.0.0")
        failed_cell = self.benchmark.ensure_cell(
            self.revision, self.target, "failed-plugin", "1.0.0",
        )
        self.benchmark.record_attempt(
            self.revision, failed_cell,
            {"attempt_number": 1, "error": "timeout"}, selected=True,
        )
        summary = self._continuation(
            plugins=[self.plugin, PluginSpec("failed-plugin", "1.0.0", "Failed", 20, True)],
            rerun_failed=False,
        )
        new_cell = self.connection.execute(
            "SELECT cell_id FROM cells WHERE target_instance_id = ? AND plugin_id = 'failed-plugin'",
            (self.target,),
        ).fetchone()[0]
        self.assertEqual(summary.scheduled_cells, 0)
        self.assertFalse(self.benchmark.should_run_cell(
            summary.revision_id, new_cell, rerun_failed=False,
        ))
        summary2 = SQLiteContinuationStore(self.connection).create_continuation(
            "run-a", config={"revision": 3}, runner_mode="http",
            targets=[TargetSpec("model-a", "http", "Local", "model-a", "sig-a")],
            plugins=[self.plugin, PluginSpec("failed-plugin", "1.0.0", "Failed", 20, True)],
            judges=[JudgeSpec("judge-a", "Local", {"v": 1})],
            contracts={"plugin": self.contract}, rerun_failed=True,
        )
        self.assertEqual(summary2.scheduled_cells, 1)

    def test_stop_marks_only_in_flight_rows_abandoned(self):
        running = self.benchmark.record_attempt(
            self.revision, self.cell,
            {"attempt_number": 2, "status": "running"},
        )
        judge_running = self.judge.record_attempt(
            self.revision, self.cell, "judge-a", self.contract.contract_id,
            {"attempt_number": 2, "status": "in_flight"},
        )
        SQLiteContinuationStore(self.connection).stop_revision(self.revision, reason="SIGINT")
        self.assertEqual(
            self.connection.execute(
                "SELECT status, failure_cause, error FROM benchmark_attempts WHERE attempt_id = ?",
                (running,),
            ).fetchone()[:],
            ("abandoned", "abandoned", "SIGINT"),
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT status, error FROM judge_attempts WHERE judge_attempt_id = ?",
                (judge_running,),
            ).fetchone()[:],
            ("abandoned", "SIGINT"),
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT status FROM run_revisions WHERE revision_id = ?", (self.revision,)
            ).fetchone()[0],
            "interrupted",
        )

    def test_purge_keeps_history_and_resets_current_projection(self):
        summary = SQLiteContinuationStore(self.connection).purge_revision(
            self.revision, cell_ids=[self.cell],
        )
        self.assertEqual(summary.benchmark_selections, 1)
        self.assertEqual(summary.judge_votes, 1)
        self.assertEqual(
            self.connection.execute("SELECT count(*) FROM benchmark_attempts").fetchone()[0], 1
        )
        self.assertEqual(
            self.connection.execute("SELECT count(*) FROM judge_vote_attempts").fetchone()[0], 1
        )
        self.assertTrue(self.benchmark.should_run_cell(self.revision, self.cell))

    def test_restart_creates_a_distinct_logical_run(self):
        new_revision = SQLiteContinuationStore(self.connection).restart_run(
            "run-b", score_schema="v1", storage_profile="compact",
            runner_mode="http", config={"revision": 1}, session_seed=3,
        )
        self.assertNotEqual(new_revision, self.revision)
        self.assertEqual(
            self.connection.execute("SELECT count(*) FROM runs").fetchone()[0], 2
        )
        self.assertEqual(
            self.connection.execute("SELECT run_id FROM run_revisions WHERE revision_id = ?", (new_revision,)).fetchone()[0],
            "run-b",
        )


if __name__ == "__main__":
    unittest.main()
