"""Tests for BenchmarkState."""
import json
import os
import tempfile
import unittest

from plugins import discover_plugins
from tests.utils import load_benchmark_module


class TestBenchmarkState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()
        cls.plugins = discover_plugins()
        cls.plugin_ids = [p.id for p in cls.plugins]

    def test_state_tracks_plugin_fields(self):
        models = {"model-a": "Source1", "model-b": "Source2"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        snap = state.snapshot()
        for name in models:
            for pid in self.plugin_ids:
                self.assertIn(f"{pid}_score", snap[name])
                self.assertIn(f"{pid}_tps", snap[name])

    def test_save_and_load_state(self):
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids, session_seed=12345)
        state.update("model-a", status="completed", **{"rate-limiter_score": 10.0})
        state.add_result({"model": "model-a", "status": "ok", "rate-limiter_score": 10.0})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path, plugin_versions={"rate-limiter": "1.0.0"})
            loaded = self.module.BenchmarkState.load_state(path, models, self.plugin_ids)
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["status"], "completed")
            self.assertEqual(snap["model-a"]["rate-limiter_score"], 10.0)
            self.assertEqual(loaded.session_seed, 12345)

    def test_latest_results_deduplicates(self):
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.add_result({"model": "model-a", "status": "ok", "rate-limiter_score": 5.0})
        state.add_result({"model": "model-a", "status": "ok", "rate-limiter_score": 10.0})
        results = state.latest_results()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["rate-limiter_score"], 10.0)

    def test_load_state_preserves_attempt_start(self):
        """Regression test: attempt_start must survive load_state for TUI."""
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.update("model-a", status="completed")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            loaded = self.module.BenchmarkState.load_state(path, models, self.plugin_ids)
            snap = loaded.snapshot()
            self.assertIn("attempt_start", snap["model-a"])

    def test_save_and_load_state_without_session_seed(self):
        """State files without session_seed load with None."""
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.update("model-a", status="completed")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            # Simulate an older state file without session_seed.
            with open(path) as f:
                data = json.load(f)
            data.pop("session_seed", None)
            with open(path, "w") as f:
                json.dump(data, f)
            loaded = self.module.BenchmarkState.load_state(path, models, self.plugin_ids)
            self.assertIsNone(loaded.session_seed)

    def test_load_state_with_missing_tui_keys(self):
        """Older state files missing newer TUI keys still load without KeyError."""
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.update("model-a", status="completed")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            with open(path) as f:
                data = json.load(f)
            # Strip newer keys that older state files may not have.
            for info in data.get("model_info", {}).values():
                info.pop("phase_detail", None)
                info.pop("attempt", None)
                info.pop("max_tok", None)
            with open(path, "w") as f:
                json.dump(data, f)
            loaded = self.module.BenchmarkState.load_state(path, models, self.plugin_ids)
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["status"], "completed")
            self.assertEqual(snap["model-a"]["phase_detail"], "")
            self.assertEqual(snap["model-a"]["attempt"], 0)
            self.assertEqual(snap["model-a"]["max_tok"], 0)

    def test_load_state_with_dict_model_config(self):
        """Regression test: dict-valued model entries resolve to source strings."""
        raw_models = {
            "model-a": "Source1",
            "model-b": {"source": "Source2", "drop_params": ["seed"]},
        }
        models_source_map = {
            name: (val.get("source", "Default") if isinstance(val, dict) else val)
            for name, val in raw_models.items()
        }
        state = self.module.BenchmarkState(models_source_map, self.plugin_ids)
        state.update("model-a", status="completed")
        state.update("model-b", status="completed")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            loaded = self.module.BenchmarkState.load_state(path, models_source_map, self.plugin_ids)
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["source"], "Source1")
            self.assertEqual(snap["model-b"]["source"], "Source2")
            # The TUI builds a set of source strings; ensure sources are hashable.
            self.assertEqual({s["source"] for s in snap.values()}, {"Source1", "Source2"})

    def test_completed_counts_only_successful_models(self):
        """completed should not treat failed models as finished."""
        models = {"model-a": "Source1", "model-b": "Source2", "model-c": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.update("model-a", status="completed")
        state.update("model-b", status="failed")
        state.update("model-c", status="pending")
        self.assertEqual(state.completed, 1)

    def test_load_state_resets_failed_models_to_pending(self):
        """Failed models are queued for rerun when a saved state is resumed."""
        models = {"model-a": "Source1", "model-b": "Source2"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.update("model-a", status="completed")
        state.update("model-b", status="failed", error="boom", last_error="boom")
        state.add_result({"model": "model-b", "status": "error", "error": "boom"})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            loaded = self.module.BenchmarkState.load_state(path, models, self.plugin_ids)
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["status"], "completed")
            self.assertEqual(snap["model-b"]["status"], "pending")
            self.assertEqual(snap["model-b"]["error"], None)
            self.assertEqual(snap["model-b"]["last_error"], "")

    def test_load_state_with_no_rerun_failed_preserves_failed_status(self):
        """With rerun_failed=False, failed models keep their failed status."""
        models = {"model-a": "Source1", "model-b": "Source2"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.update("model-a", status="completed")
        state.update("model-b", status="failed", error="boom", last_error="boom")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            loaded = self.module.BenchmarkState.load_state(
                path, models, self.plugin_ids, rerun_failed=False)
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["status"], "completed")
            self.assertEqual(snap["model-b"]["status"], "failed")
            self.assertEqual(snap["model-b"]["error"], "boom")

    def test_load_state_with_new_plugins_resets_completed_models(self):
        """When new plugins are added, completed models are re-queued to run them."""
        models = {"model-a": "Source1"}
        original_plugins = ["rate-limiter"]
        state = self.module.BenchmarkState(models, original_plugins)
        state.update("model-a", status="completed", **{"rate-limiter_score": 5.0})
        state.add_result({"model": "model-a", "status": "ok", "rate-limiter_score": 5.0})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            # Now load with an expanded plugin set.
            expanded_plugins = ["rate-limiter", "moe-dense"]
            loaded = self.module.BenchmarkState.load_state(path, models, expanded_plugins)
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["status"], "pending")
            self.assertIn("moe-dense_score", snap["model-a"])
            self.assertIsNone(snap["model-a"]["moe-dense_score"])
            # Old plugin results should be preserved in results.
            latest = loaded.latest_results()[0]
            self.assertEqual(latest["rate-limiter_score"], 5.0)

    def test_load_state_without_new_plugins_preserves_completed_status(self):
        """Without plugin changes, completed models stay completed."""
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.update("model-a", status="completed")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            loaded = self.module.BenchmarkState.load_state(path, models, self.plugin_ids)
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
