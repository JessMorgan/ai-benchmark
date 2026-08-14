"""Tests for explicit CSV-based benchmark state recovery."""
import csv
import json
import os
import tempfile
import unittest

from scripts.recover_state_from_csv import reconstruct_run_state


class TestStateRecovery(unittest.TestCase):
    def _make_run(self, tmpdir):
        run_dir = os.path.join(tmpdir, "run")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "benchmark-config.yml"), "w", encoding="utf-8") as handle:
            handle.write(
                "sources:\n"
                "  Local:\n"
                "    api_url: http://127.0.0.1:1/chat/completions\n"
                "    headers: {}\n"
                "models:\n"
                "  model-a: Local\n"
                "  model-b: Local\n"
            )
        fields = [
            "Model", "Runner", "Source", "TTFT_s", "Total", "Time_s", "Status", "Error",
            "code-review_Score_15", "code-review_Response_s", "code-review_Thinking_Tokens",
            "code-review_Content_Tokens", "code-review_Total_Tokens", "code-review_TPS",
            "code-review_Empty_Reason", "unknown-plugin_Score_20",
        ]
        with open(os.path.join(run_dir, "results.csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({field: "" for field in fields})
        return run_dir

    def test_recovery_rejects_unknown_plugin_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._make_run(tmpdir)
            with self.assertRaisesRegex(ValueError, "unknown plugin score columns"):
                reconstruct_run_state(run_dir)

    def _make_recovery_fixture(self, tmpdir):
        """Create a deterministic multi-row run without repository artifacts."""
        run_dir = os.path.join(tmpdir, "fixture-run")
        os.makedirs(run_dir)
        models = [f"fixture-model-{index:03d}" for index in range(221)]
        with open(os.path.join(run_dir, "benchmark-config.yml"), "w", encoding="utf-8") as handle:
            handle.write(
                "sources:\n"
                "  Local:\n"
                "    api_url: http://127.0.0.1:1/chat/completions\n"
                "    headers: {}\n"
                "models:\n"
                + "".join(f"  {model}: Local\n" for model in models)
            )
        fields = [
            "Model", "Runner", "Source", "TTFT_s", "Total", "Time_s", "Status", "Error",
            "code-review_Score_15", "code-review_Response_s", "code-review_Thinking_Tokens",
            "code-review_Content_Tokens", "code-review_Total_Tokens", "code-review_TPS",
            "code-review_Empty_Reason",
        ]
        with open(os.path.join(run_dir, "results.csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, model in enumerate(models):
                completed = index < 161
                writer.writerow({
                    "Model": model,
                    "Runner": "http",
                    "Source": "Local",
                    "TTFT_s": "0.1",
                    "Total": "1.0",
                    "Time_s": "1.0",
                    "Status": "OK" if completed else "FAIL",
                    "Error": "" if completed else "fixture failure",
                    "code-review_Score_15": "10" if completed else "fail",
                    "code-review_Response_s": "0.5",
                    "code-review_Thinking_Tokens": "0",
                    "code-review_Content_Tokens": "10",
                    "code-review_Total_Tokens": "10",
                    "code-review_TPS": "20",
                    "code-review_Empty_Reason": "",
                })
        return run_dir

    def test_recovery_apply_preserves_backup_and_reloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._make_recovery_fixture(tmpdir)
            state_path = os.path.join(run_dir, "benchmark_state.json")
            corrupt = b'{"model_info": : invalid}'
            with open(state_path, "wb") as handle:
                handle.write(corrupt)
            with open(state_path, "rb") as handle:
                before = handle.read()
            report, _reconstructed = reconstruct_run_state(run_dir, apply=True)
            self.assertEqual(report["rows"], 221)
            self.assertEqual(report["loaded_completed"], 161)
            self.assertEqual(report["loaded_pending"], 60)
            self.assertTrue(report["identities_match"])
            self.assertEqual(report["score_mismatches"], 0)
            self.assertIsNotNone(report["backup"])
            with open(report["backup"], "rb") as handle:
                self.assertEqual(handle.read(), before)
            with open(state_path, encoding="utf-8") as handle:
                self.assertEqual(len(json.load(handle)["results"]), 221)

    def test_recovery_is_dry_run_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._make_recovery_fixture(tmpdir)
            report, reconstructed = reconstruct_run_state(run_dir)
            self.assertEqual(report["rows"], 221)
            self.assertEqual(report["completed"], 161)
            self.assertEqual(report["failed"], 60)
            self.assertEqual(len(reconstructed["results"]), 221)
            self.assertIsNone(report["backup"])
            self.assertFalse(os.path.exists(os.path.join(run_dir, "benchmark_state.json")))

    def test_failed_judge_votes_are_filtered(self):
        """Only valid votes with no error are retained for resume."""
        from scripts.recover_state_from_csv import _successful_judge_votes

        votes = json.dumps([
            {"model": "good", "score": 80, "confidence": "high", "rationale": "valid"},
            {"model": "bad", "score": None, "confidence": None, "rationale": None, "error": "timeout"},
            {"model": "empty", "score": 40, "confidence": "medium", "rationale": ""},
        ])
        filtered = _successful_judge_votes(votes)
        self.assertEqual([vote["model"] for vote in filtered], ["good"])

    def test_judge_completion_requires_aggregate_score(self):
        """Judge names alone cannot make a missing consensus score complete."""
        from scripts.recover_state_from_csv import _judge_complete

        votes = [
            {"model": "judge-a", "score": 80, "confidence": "high", "rationale": "valid"},
            {"model": "judge-b", "score": 70, "confidence": "medium", "rationale": "valid"},
        ]
        self.assertFalse(_judge_complete(["judge-a", "judge-b"], votes, None))
        self.assertTrue(_judge_complete(["judge-a", "judge-b"], votes, 75))


if __name__ == "__main__":
    unittest.main()
