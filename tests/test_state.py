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


if __name__ == "__main__":
    unittest.main()
