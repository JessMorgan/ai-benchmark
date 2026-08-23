"""Tests for SQLite report-only read-model generation."""
import os
import tempfile
import unittest

from benchmark.sqlite_benchmarks import SQLiteBenchmarkStore
from benchmark.sqlite_reports import SQLiteReportSource
from benchmark.sqlite_schema import connect_database


class TestSQLiteReports(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.path = os.path.join(self.tmpdir.name, "run.sqlite3")
        self.connection = connect_database(self.path)
        self.addCleanup(self.connection.close)
        self.store = SQLiteBenchmarkStore(self.connection)
        self.revision = self.store.create_run(
            "run-a", score_schema="v1", storage_profile="compact",
            runner_mode="http", config={"revision": 1}, session_seed=42,
        )
        self.store.register_plugin(
            "rate-limiter", "1.0.0", name="Rate Limiter", max_score=20,
            supports_streaming=True,
        )
        self.store.activate_plugin(self.revision, "rate-limiter", "1.0.0")
        self.target = self.store.register_target(
            self.revision, run_id="run-a", logical_name="model-a", runner="http",
            source="Local", api_model="model-a", target_signature="sig-a",
        )
        self.cell = self.store.ensure_cell(
            self.revision, self.target, "rate-limiter", "1.0.0",
        )

    def test_loads_selected_attempt_as_report_row(self):
        self.store.record_attempt(
            self.revision, self.cell,
            {
                "attempt_number": 1, "prompt": "prompt", "content": "answer",
                "thinking": "thought", "score": 15, "output_tokens": 4,
                "thinking_tokens": 2, "total_tokens": 6, "tps": 3.5,
                "response_time": 2.4, "gen_time": 1.8,
                "stream_ok": True, "rubric": [{"name": "quality", "points": 15}],
            },
            selected=True,
        )
        source = SQLiteReportSource.open(self.path)
        self.addCleanup(source.close)
        rows, plugins, seed, revision_id = source.load_results()
        self.assertEqual(plugins, ["rate-limiter"])
        self.assertEqual(seed, 42)
        self.assertEqual(revision_id, self.revision)
        self.assertEqual(rows[0]["model"], "model-a")
        self.assertEqual(rows[0]["rate-limiter_score"], 15)
        self.assertEqual(rows[0]["rate-limiter_total_tokens"], 6)
        self.assertTrue(rows[0]["rate-limiter_stream_ok"])
        self.assertEqual(rows[0]["rate-limiter_rubric"][0]["name"], "quality")
        self.assertEqual(rows[0]["rate-limiter_response_time"], 2.4)
        self.assertEqual(rows[0]["rate-limiter_gen_time"], 1.8)

    def test_default_report_revision_can_be_scoped_to_run_id(self):
        second = self.store.create_run(
            "run-b", score_schema="v1", storage_profile="compact",
            runner_mode="http", config={}, session_seed=99,
        )
        source = SQLiteReportSource.open(self.path)
        self.addCleanup(source.close)
        _rows, _plugins, seed, revision_id = source.load_results(run_id="run-a")
        self.assertEqual(seed, 42)
        self.assertEqual(revision_id, self.revision)
        _rows, _plugins, seed, revision_id = source.load_results(run_id="run-b")
        self.assertEqual(seed, 99)
        self.assertEqual(revision_id, second)

    def test_judge_votes_are_scoped_to_selected_revision_target(self):
        from benchmark.sqlite_judges import SQLiteJudgeStore

        judge = SQLiteJudgeStore(self.connection)
        judge.register_judge(self.revision, "judge-a", source="Local")
        judge.register_contract(
            "contract-1", plugin_id="rate-limiter", plugin_version="1.0.0",
            prompt_version="v1", instructions_version="v1",
            response_schema_hash="schema-1", contract={"v": 1},
            contract_hash="hash-1",
        )
        judge.activate_contract(self.revision, "rate-limiter", "contract-1")
        attempt = judge.record_attempt(
            self.revision, self.cell, "judge-a", "contract-1", {"attempt_number": 1},
        )
        vote = judge.record_vote(attempt, {"score": 11, "usable": True})
        judge.select_vote(self.revision, self.cell, "judge-a", "contract-1", vote)

        second = self.store.create_run(
            "run-b", score_schema="v1", storage_profile="compact",
            runner_mode="http", config={}, session_seed=99,
        )
        target_b = self.store.register_target(
            second, run_id="run-b", logical_name="model-a", runner="http",
            source="Other", api_model="model-a", target_signature="other/model-a",
        )
        self.store.register_plugin(
            "rate-limiter", "1.0.0", name="Rate Limiter", max_score=20,
            supports_streaming=True,
        )
        self.store.activate_plugin(second, "rate-limiter", "1.0.0")
        cell_b = self.store.ensure_cell(second, target_b, "rate-limiter", "1.0.0")
        self.store.record_attempt(second, cell_b, {"attempt_number": 1, "score": 7}, selected=True)

        source = SQLiteReportSource.open(self.path)
        self.addCleanup(source.close)
        rows, _plugins, _seed, _revision = source.load_results(revision=second)
        self.assertNotIn("rate-limiter_judge_score", rows[0])

    def test_judge_votes_attach_with_canonical_keys(self):
        """Votes read back from SQLite must use the canonical vote shape.

        The live judge path, footer progress seeding, ``merge_judge_vote``,
        and disagreement analysis all key votes by ``model`` and
        ``judge_contract_id``. If the SQLite report source emits the raw
        column names (``judge_model``/``contract_id``) instead, a resumed
        run sees zero existing votes and the footer shows every judge at
        0\u2705 despite hundreds of persisted votes.
        """
        from benchmark.core import is_successful_judge_vote
        from benchmark.sqlite_judges import SQLiteJudgeStore

        judge = SQLiteJudgeStore(self.connection)
        judge.register_judge(self.revision, "judge-a", source="Local")
        judge.register_contract(
            "contract-1", plugin_id="rate-limiter", plugin_version="1.0.0",
            prompt_version="v1", instructions_version="v1",
            response_schema_hash="schema-1", contract={"v": 1},
            contract_hash="hash-1",
        )
        judge.activate_contract(self.revision, "rate-limiter", "contract-1")
        attempt = judge.record_attempt(
            self.revision, self.cell, "judge-a", "contract-1", {"attempt_number": 1},
        )
        vote = judge.record_vote(attempt, {
            "score": 15, "confidence": "high", "rationale": "solid", "usable": True,
        })
        judge.select_vote(self.revision, self.cell, "judge-a", "contract-1", vote)

        source = SQLiteReportSource.open(self.path)
        self.addCleanup(source.close)
        rows, _plugins, _seed, _revision = source.load_results()
        votes = rows[0].get("rate-limiter_judge_votes", [])
        self.assertEqual(len(votes), 1)
        vote_dict = votes[0]
        self.assertEqual(vote_dict.get("model"), "judge-a")
        self.assertEqual(vote_dict.get("judge_contract_id"), "contract-1")
        self.assertTrue(is_successful_judge_vote(vote_dict))

    def test_judge_projection_and_criteria_attach_from_stored_votes(self):
        """Judge criteria, consensus, confidence, and completion must be rebuilt.

        The legacy JSON path stored the flat projection (confidence, rationale,
        consensus-by-contract, criteria) alongside votes. SQLite keeps only the
        vote + criteria rows, so the report source must reconstruct those
        fields or reports silently lose the judge detail sections.
        """
        from benchmark.sqlite_judges import SQLiteJudgeStore

        judge = SQLiteJudgeStore(self.connection)
        judge.register_judge(self.revision, "judge-a", source="Local")
        judge.register_judge(self.revision, "judge-b", source="Local")
        judge.register_contract(
            "contract-1", plugin_id="rate-limiter", plugin_version="1.0.0",
            prompt_version="v1", instructions_version="v1",
            response_schema_hash="schema-1", contract={"v": 1},
            contract_hash="hash-1",
        )
        judge.activate_contract(self.revision, "rate-limiter", "contract-1")
        for judge_name in ("judge-a", "judge-b"):
            attempt = judge.record_attempt(
                self.revision, self.cell, judge_name, "contract-1",
                {"attempt_number": 1},
            )
            vote = judge.record_vote(attempt, {
                "score": 15, "confidence": "high", "rationale": "solid",
                "usable": True,
                "criteria": [
                    {"id": "R1", "criterion": "Use headings.",
                     "status": "met", "evidence": "All present."},
                ],
            })
            judge.select_vote(
                self.revision, self.cell, judge_name, "contract-1", vote,
            )

        source = SQLiteReportSource.open(self.path)
        self.addCleanup(source.close)
        rows, _plugins, _seed, _revision = source.load_results()
        row = rows[0]
        self.assertEqual(row.get("rate-limiter_judge_score"), 15.0)
        self.assertEqual(row.get("rate-limiter_judge_confidence"), "high")
        self.assertEqual(row.get("rate-limiter_judge_rationale"), "solid | solid")
        self.assertEqual(row.get("rate-limiter_judge_selected_contract"), "contract-1")
        self.assertTrue(row.get("rate-limiter_judge_complete"))
        consensus = row.get("rate-limiter_judge_consensus_by_contract", {})
        self.assertIn("contract-1", consensus)
        criteria = row.get("rate-limiter_judge_criteria", [])
        self.assertEqual(len(criteria), 2)  # one report per valid judge
        self.assertEqual(criteria[0]["judge"], "judge-a")
        self.assertEqual(criteria[0]["criteria"][0]["id"], "R1")
        # Each stored vote also carries its criteria for the vote-based helper.
        votes = row.get("rate-limiter_judge_votes", [])
        self.assertEqual(votes[0]["criteria"][0]["criterion"], "Use headings.")

    def test_attempt_meta_and_model_level_attach(self):
        """Attempt counts/retry reasons and model-level judge identities attach."""
        from benchmark.sqlite_judges import SQLiteJudgeStore

        judge = SQLiteJudgeStore(self.connection)
        judge.register_judge(self.revision, "judge-a", source="Local")
        self.store.record_attempt(
            self.revision, self.cell,
            {"attempt_number": 1, "score": 10, "retry_reason": "none"},
            selected=True,
        )
        self.store.record_attempt(
            self.revision, self.cell,
            {"attempt_number": 2, "score": 15, "retry_reason": "transport_error"},
        )
        source = SQLiteReportSource.open(self.path)
        self.addCleanup(source.close)
        rows, _plugins, _seed, _revision = source.load_results()
        row = rows[0]
        self.assertEqual(row.get("rate-limiter_attempt_count"), 2)
        self.assertEqual(row.get("rate-limiter_retry_reasons"), ["transport_error"])
        self.assertEqual(row.get("judge_models"), ["judge-a"])

    def test_historical_revision_can_be_selected(self):
        self.store.record_attempt(
            self.revision, self.cell,
            {"attempt_number": 1, "content": "old", "score": 10}, selected=True,
        )
        second = self.store.create_run(
            "run-b", score_schema="v1", storage_profile="compact",
            runner_mode="http", config={}, session_seed=99,
        )
        self.assertNotEqual(second, self.revision)
        source = SQLiteReportSource.open(self.path)
        self.addCleanup(source.close)
        _rows, _plugins, seed, revision_id = source.load_results(revision=1)
        self.assertEqual(seed, 42)
        self.assertEqual(revision_id, self.revision)


if __name__ == "__main__":
    unittest.main()
