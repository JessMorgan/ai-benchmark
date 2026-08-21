"""Tests for normalized SQLite judge persistence."""
import os
import tempfile
import unittest

from benchmark.sqlite_benchmarks import SQLiteBenchmarkStore
from benchmark.sqlite_judges import SQLiteJudgeStore
from benchmark.sqlite_schema import connect_database


class TestSQLiteJudgeStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.connection = connect_database(os.path.join(self.tmpdir.name, "run.sqlite3"))
        self.addCleanup(self.connection.close)
        benchmark = SQLiteBenchmarkStore(self.connection)
        self.revision = benchmark.create_run(
            "run-a", score_schema="v1", storage_profile="compact",
            runner_mode="http", config={}, session_seed=1,
        )
        benchmark.register_plugin(
            "plugin", "1.0.0", name="Plugin", max_score=20,
            supports_streaming=True,
        )
        benchmark.activate_plugin(self.revision, "plugin", "1.0.0")
        target = benchmark.register_target(
            self.revision, run_id="run-a", logical_name="model", runner="http",
            source="Local", api_model="model", target_signature="sig",
        )
        self.cell = benchmark.ensure_cell(self.revision, target, "plugin", "1.0.0")
        self.store = SQLiteJudgeStore(self.connection)
        self.store.register_judge(self.revision, "judge-a", source="Local")
        self.store.register_contract(
            "contract-v1", plugin_id="plugin", plugin_version="1.0.0",
            prompt_version="judge-v1", instructions_version="1.0.0",
            response_schema_hash="schema-v1", contract={"version": 1},
            contract_hash="hash-v1",
        )
        self.store.activate_contract(self.revision, "plugin", "contract-v1")

    def test_failed_and_successful_judge_attempts_are_preserved(self):
        failed = self.store.record_attempt(
            self.revision, self.cell, "judge-a", "contract-v1",
            {"attempt_number": 1, "error": "timeout", "request": "private request"},
        )
        successful = self.store.record_attempt(
            self.revision, self.cell, "judge-a", "contract-v1",
            {
                "attempt_number": 2,
                "request": "private request",
                "raw_response": '{"score": 18}',
                "usage": {"total_tokens": 20},
            },
            retain_request=True,
        )
        failed_vote = self.store.record_vote(failed, {"error": "invalid", "usable": False})
        successful_vote = self.store.record_vote(
            successful,
            {
                "score": 18,
                "confidence": "high",
                "rationale": "Strong answer",
                "criteria": [{
                    "id": "correctness", "criterion": "Correctness",
                    "status": "pass", "evidence": "Evidence",
                }],
            },
        )
        self.store.select_vote(
            self.revision, self.cell, "judge-a", "contract-v1", successful_vote,
        )
        votes = self.store.current_votes(self.revision, self.cell, "contract-v1")
        self.assertEqual(len(votes), 1)
        self.assertEqual(votes[0]["vote_attempt_id"], successful_vote)
        self.assertNotEqual(failed_vote, successful_vote)
        self.assertEqual(self.store.criteria(successful_vote)[0]["criterion_key"], "correctness")
        request_id = self.connection.execute(
            "SELECT request_payload_id FROM judge_attempts WHERE judge_attempt_id = ?",
            (successful,),
        ).fetchone()[0]
        self.assertIsNotNone(request_id)
        self.assertIsNotNone(
            self.connection.execute(
                "SELECT raw_response_payload_id FROM judge_attempts WHERE judge_attempt_id = ?",
                (successful,),
            ).fetchone()[0]
        )
        self.assertIsNone(
            self.connection.execute(
                "SELECT request_payload_id FROM judge_attempts WHERE judge_attempt_id = ?",
                (failed,),
            ).fetchone()[0]
        )

    def test_current_projection_does_not_overwrite_history(self):
        first_attempt = self.store.record_attempt(
            self.revision, self.cell, "judge-a", "contract-v1", {"attempt_number": 1},
        )
        first_vote = self.store.record_vote(first_attempt, {"score": 10})
        self.store.select_vote(
            self.revision, self.cell, "judge-a", "contract-v1", first_vote,
        )
        second_attempt = self.store.record_attempt(
            self.revision, self.cell, "judge-a", "contract-v1", {"attempt_number": 2},
        )
        second_vote = self.store.record_vote(second_attempt, {"score": 19})
        self.store.select_vote(
            self.revision, self.cell, "judge-a", "contract-v1", second_vote,
        )
        self.assertEqual(
            self.connection.execute("SELECT count(*) FROM judge_vote_attempts").fetchone()[0],
            2,
        )
        self.assertEqual(
            self.store.current_votes(self.revision, self.cell)[0]["score"],
            19,
        )

    def test_consensus_cache_is_invalidated_when_current_vote_changes(self):
        attempt = self.store.record_attempt(
            self.revision, self.cell, "judge-a", "contract-v1", {"attempt_number": 1},
        )
        vote = self.store.record_vote(attempt, {"score": 15, "confidence": "medium"})
        self.store.select_vote(self.revision, self.cell, "judge-a", "contract-v1", vote)
        vote_hash = self.store.cache_consensus(
            self.revision, self.cell, "contract-v1", score=15, confidence="medium",
        )
        self.assertEqual(
            self.store.cached_consensus(self.revision, self.cell, "contract-v1")["vote_set_hash"],
            vote_hash,
        )
        attempt2 = self.store.record_attempt(
            self.revision, self.cell, "judge-a", "contract-v1", {"attempt_number": 2},
        )
        vote2 = self.store.record_vote(attempt2, {"score": 17})
        self.store.select_vote(self.revision, self.cell, "judge-a", "contract-v1", vote2)
        self.assertIsNone(self.store.cached_consensus(self.revision, self.cell, "contract-v1"))

    def test_old_and_new_contracts_coexist(self):
        self.store.register_contract(
            "contract-v2", plugin_id="plugin", plugin_version="1.0.0",
            prompt_version="judge-v2", instructions_version="1.0.0",
            response_schema_hash="schema-v1", contract={"version": 2},
            contract_hash="hash-v2",
        )
        attempt = self.store.record_attempt(
            self.revision, self.cell, "judge-a", "contract-v2", {"attempt_number": 1},
        )
        vote = self.store.record_vote(attempt, {"score": 12})
        self.store.select_vote(self.revision, self.cell, "judge-a", "contract-v2", vote)
        self.assertEqual(
            self.connection.execute("SELECT count(*) FROM judge_contracts").fetchone()[0],
            2,
        )
        self.assertEqual(len(self.store.current_votes(self.revision, self.cell, "contract-v2")), 1)
        self.assertEqual(self.store.current_votes(self.revision, self.cell, "contract-v1"), [])


if __name__ == "__main__":
    unittest.main()
