"""Tests for explicit CSV-based benchmark state recovery."""
import contextlib
import csv
import io
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

    def _make_runner_run(self, tmpdir, runner, model="model-a"):
        """Create a single-row run for the given runner."""
        run_dir = os.path.join(tmpdir, f"{runner}-run")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "benchmark-config.yml"), "w", encoding="utf-8") as handle:
            handle.write(
                "sources:\n"
                "  Local:\n"
                "    api_url: http://127.0.0.1:1/chat/completions\n"
                "    headers: {}\n"
                "models:\n"
                f"  {model}: Local\n"
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
            writer.writerow({
                "Model": model, "Runner": runner, "Source": "Local",
                "TTFT_s": "0.1", "Total": "1.0", "Time_s": "1.0",
                "Status": "OK", "Error": "",
                "code-review_Score_15": "10", "code-review_Response_s": "0.5",
                "code-review_Thinking_Tokens": "0", "code-review_Content_Tokens": "10",
                "code-review_Total_Tokens": "10", "code-review_TPS": "20",
                "code-review_Empty_Reason": "",
            })
        return run_dir

    def test_pi_state_key_recovery(self):
        """Pi-runner rows recover to the ``[pi]`` key family, not ``[opencode]``.

        Regression: the recovery script once hard-coded ``[opencode]`` for every
        non-HTTP runner, so a ``pi`` leg was mislabeled as ``model [opencode]``
        and the recovered state was unusable for pi runs.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._make_runner_run(tmpdir, "pi")
            report, reconstructed = reconstruct_run_state(run_dir)
            self.assertEqual(report["rows"], 1)
            self.assertEqual(report["models"], 1)
            self.assertTrue(report["identities_match"])
            self.assertEqual(report["score_mismatches"], 0)
            results = reconstructed["results"]
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["state_key"], "model-a [pi]")
            self.assertEqual(results[0]["runner"], "pi")
            self.assertIn("model-a [pi]", reconstructed["model_info"])
            self.assertNotIn("model-a [opencode]", reconstructed["model_info"])

    def test_opencode_state_key_recovery(self):
        """OpenCode-runner rows still recover to the ``[opencode]`` key family."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._make_runner_run(tmpdir, "opencode")
            _report, reconstructed = reconstruct_run_state(run_dir)
            results = reconstructed["results"]
            self.assertEqual(results[0]["state_key"], "model-a [opencode]")
            self.assertEqual(results[0]["runner"], "opencode")
            self.assertIn("model-a [opencode]", reconstructed["model_info"])

    def test_config_path_read_from_run_info(self):
        """The config capsule filename is read from run-info.json metadata.

        Regression: the recovery script once hard-coded ``benchmark-config.yml``,
        so a run whose operator used a differently-named config file could not
        be recovered.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._make_runner_run(tmpdir, "http", model="model-a")
            os.remove(os.path.join(run_dir, "benchmark-config.yml"))
            with open(os.path.join(run_dir, "my-config.json"), "w", encoding="utf-8") as handle:
                json.dump({
                    "sources": {"Local": {
                        "api_url": "http://127.0.0.1:1/chat/completions",
                        "headers": {},
                    }},
                    "models": {"model-a": "Local"},
                }, handle)
            with open(os.path.join(run_dir, "run-info.json"), "w", encoding="utf-8") as handle:
                json.dump({"config_file": "/operator/path/my-config.json"}, handle)
            report, _reconstructed = reconstruct_run_state(run_dir)
            self.assertEqual(report["rows"], 1)
            self.assertTrue(report["identities_match"])
            self.assertEqual(report["score_mismatches"], 0)

    def test_config_path_falls_back_to_default_with_warning(self):
        """A missing run-info.json config path falls back to the default name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._make_runner_run(tmpdir, "http", model="model-a")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                report, _reconstructed = reconstruct_run_state(run_dir)
            self.assertEqual(report["rows"], 1)
            self.assertTrue(report["identities_match"])
            self.assertIn("falling back to the default config name", stderr.getvalue())

    def test_invalid_utf8_run_info_falls_back_with_warning(self):
        """A run-info.json truncated mid-multibyte must not crash recovery.

        Regression: a run-info.json cut mid-UTF-8 sequence raised
        UnicodeDecodeError, which was not caught, so the recovery tool
        crashed with a traceback instead of falling back with a warning.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._make_runner_run(tmpdir, "http", model="model-a")
            with open(os.path.join(run_dir, "run-info.json"), "wb") as handle:
                # Valid JSON prefix cut mid-emoji (U+1F600 is a 4-byte
                # sequence; the trailing 3 bytes are an invalid prefix).
                handle.write(b'{"config_file": "my-config.json", "note": "\xf0\x9f\x98')
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                report, _reconstructed = reconstruct_run_state(run_dir)
            self.assertEqual(report["rows"], 1)
            self.assertTrue(report["identities_match"])
            self.assertEqual(report["score_mismatches"], 0)
            self.assertIn("falling back to the default config name", stderr.getvalue())

    def test_config_path_falls_back_to_json_default(self):
        """A .json config with no run-info.json recovers via the .json file.

        Regression: the fallback once hard-coded ``benchmark-config.yml``, so a
        hard-crashed run (run-info.json never written) whose operator used the
        default ``.json`` config failed with FileNotFoundError.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._make_runner_run(tmpdir, "http", model="model-a")
            os.remove(os.path.join(run_dir, "benchmark-config.yml"))
            with open(os.path.join(run_dir, "benchmark-config.json"), "w", encoding="utf-8") as handle:
                json.dump({
                    "sources": {"Local": {
                        "api_url": "http://127.0.0.1:1/chat/completions",
                        "headers": {},
                    }},
                    "models": {"model-a": "Local"},
                }, handle)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                report, _reconstructed = reconstruct_run_state(run_dir)
            self.assertEqual(report["rows"], 1)
            self.assertTrue(report["identities_match"])
            self.assertEqual(report["score_mismatches"], 0)
            self.assertIn("falling back to the default config name", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
