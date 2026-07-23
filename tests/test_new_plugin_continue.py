"""Integration tests for running new plugins on completed models."""
import os
import tempfile
import unittest
from unittest import mock

from plugins import discover_plugins
from tests.utils import load_benchmark_module


class TestNewPluginContinue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()
        cls.plugins = discover_plugins()

    def test_completed_model_runs_new_plugin_on_continue(self):
        """A model completed for plugin A should run plugin B when B is added."""
        plugins = [p for p in self.plugins if p.id in ("rate-limiter", "moe-dense")]
        models = {"dummy-model": "Local"}
        state = self.module.BenchmarkState(models, ["rate-limiter"])
        state.update("dummy-model", status="completed", **{"rate-limiter_score": 5.0})
        state.add_result({
            "model": "dummy-model",
            "status": "ok",
            "rate-limiter_score": 5.0,
            "rate-limiter_response_time": 1.0,
            "rate-limiter_output_tokens": 100,
            "rate-limiter_tps": 50.0,
            "rate-limiter_stream_ok": True,
            "total_time": 1.0,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)

            loaded = self.module.BenchmarkState.load_state(path, models, ["rate-limiter", "moe-dense"])
            snap = loaded.snapshot()
            self.assertEqual(snap["dummy-model"]["status"], "pending")

            source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}, "plugin_thread_limit": 1}}

            calls = []

            def fake_run_plugin_task(target_name, api_model, source, plugin, *args, **kwargs):
                calls.append(plugin.id)
                if plugin.id == "moe-dense":
                    return {
                        "moe-dense_score": 7.0,
                        "moe-dense_response_time": 2.0,
                        "moe-dense_output_tokens": 200,
                        "moe-dense_tps": 100.0,
                        "moe-dense_stream_ok": False,
                    }, None
                return None, "should not be called"

            with mock.patch.object(self.module, "_run_plugin_task", side_effect=fake_run_plugin_task):
                self.module.run_model(
                    "dummy-model", "Local", loaded, plugins, source_config,
                    timeout=1, token_levels=[100], output_dir=tmpdir,
                    session_seed=0, global_cfg={},
                )

            self.assertEqual(calls, ["moe-dense"])
            result = loaded.latest_results()[0]
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["rate-limiter_score"], 5.0)
            self.assertEqual(result["moe-dense_score"], 7.0)

    def test_all_completed_models_run_new_plugin_on_continue(self):
        """When multiple models are completed and a new plugin is added, ALL models run it."""
        plugins = [p for p in self.plugins if p.id in ("rate-limiter", "moe-dense")]
        models = {"model-a": "Src1", "model-b": "Src1", "model-c": "Src1"}
        state = self.module.BenchmarkState(models, ["rate-limiter"])
        for name in models:
            state.update(name, status="completed", **{"rate-limiter_score": 5.0})
            state.add_result({
                "model": name, "status": "ok",
                "rate-limiter_score": 5.0, "rate-limiter_response_time": 1.0,
                "rate-limiter_output_tokens": 100, "rate-limiter_tps": 50.0,
                "rate-limiter_stream_ok": True, "total_time": 1.0,
            })

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)

            loaded = self.module.BenchmarkState.load_state(path, models, ["rate-limiter", "moe-dense"])
            snap = loaded.snapshot()
            for name in models:
                self.assertEqual(snap[name]["status"], "pending",
                                 f"{name} should be pending after load_state with new plugin")

            source_config = {"Src1": {"api_url": "http://localhost:11434/chat/completions", "headers": {}, "plugin_thread_limit": 1}}

            plugin_calls = {}

            def fake_run_plugin_task(target_name, api_model, source, plugin, *args, **kwargs):
                plugin_calls.setdefault(target_name, []).append(plugin.id)
                if plugin.id == "moe-dense":
                    return {
                        "moe-dense_score": 7.0, "moe-dense_response_time": 2.0,
                        "moe-dense_output_tokens": 200, "moe-dense_tps": 100.0,
                        "moe-dense_stream_ok": False,
                    }, None
                return None, "should not be called"

            with mock.patch.object(self.module, "_run_plugin_task", side_effect=fake_run_plugin_task):
                for name in models:
                    self.module.run_model(
                        name, "Src1", loaded, plugins, source_config,
                        timeout=1, token_levels=[100], output_dir=tmpdir,
                        session_seed=0, global_cfg={},
                    )

            for name in models:
                self.assertEqual(plugin_calls.get(name), ["moe-dense"],
                                 f"{name} should only run moe-dense (new plugin)")
                results = [r for r in loaded.latest_results() if r["model"] == name]
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["rate-limiter_score"], 5.0)
                self.assertEqual(results[0]["moe-dense_score"], 7.0)

    def test_source_queues_include_models_with_new_plugins(self):
        """Simulates CLI source_queues construction: models reset to pending are queued."""
        models = {"model-a": "Src1", "model-b": "Src2", "model-c": "Src1"}
        state = self.module.BenchmarkState(models, ["rate-limiter"])
        state.update("model-a", status="completed", **{"rate-limiter_score": 5.0})
        state.update("model-b", status="completed", **{"rate-limiter_score": 8.0})
        state.update("model-c", status="failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)

            loaded = self.module.BenchmarkState.load_state(path, models, ["rate-limiter", "moe-dense"])

            source_queues = {"Src1": [], "Src2": []}
            for name, src in models.items():
                info = loaded.snapshot().get(name, {})
                if info.get("status") in ("completed",):
                    continue
                source_queues[src].append(name)

            self.assertIn("model-a", source_queues["Src1"])
            self.assertIn("model-c", source_queues["Src1"])
            self.assertIn("model-b", source_queues["Src2"])

    def test_no_plugin_change_completed_models_stay_completed(self):
        """Without new plugins, completed models remain completed and are not queued."""
        models = {"model-a": "Src1", "model-b": "Src1"}
        state = self.module.BenchmarkState(models, ["rate-limiter"])
        state.update("model-a", status="completed", **{"rate-limiter_score": 5.0})
        state.add_result({
            "model": "model-a", "status": "ok",
            "rate-limiter_score": 5.0, "rate-limiter_response_time": 1.0,
            "rate-limiter_output_tokens": 100, "rate-limiter_tps": 50.0,
            "rate-limiter_stream_ok": True, "total_time": 1.0,
        })
        state.update("model-b", status="failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)

            loaded = self.module.BenchmarkState.load_state(path, models, ["rate-limiter"])

            source_queues = {"Src1": []}
            for name, src in models.items():
                info = loaded.snapshot().get(name, {})
                if info.get("status") in ("completed",):
                    continue
                source_queues["Src1"].append(name)

            self.assertNotIn("model-a", source_queues["Src1"])
            self.assertIn("model-b", source_queues["Src1"])

    def test_new_plugin_only_called_for_new_not_existing(self):
        """When model has old plugin results AND new plugin, only new plugin is run."""
        plugins = [p for p in self.plugins if p.id in ("rate-limiter", "moe-dense")]
        models = {"dummy-model": "Local"}
        state = self.module.BenchmarkState(models, ["rate-limiter"])
        state.update("dummy-model", status="completed", **{"rate-limiter_score": 12.0})
        state.add_result({
            "model": "dummy-model", "status": "ok",
            "rate-limiter_score": 12.0, "rate-limiter_response_time": 3.0,
            "rate-limiter_output_tokens": 300, "rate-limiter_tps": 40.0,
            "rate-limiter_stream_ok": True, "total_time": 3.0,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)

            loaded = self.module.BenchmarkState.load_state(path, models, ["rate-limiter", "moe-dense"])
            source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}, "plugin_thread_limit": 1}}

            run_calls = []

            def fake_run_plugin_task(target_name, api_model, source, plugin, *args, **kwargs):
                run_calls.append(plugin.id)
                return {
                    f"{plugin.id}_score": 9.0,
                    f"{plugin.id}_response_time": 1.0,
                    f"{plugin.id}_output_tokens": 50,
                    f"{plugin.id}_tps": 25.0,
                    f"{plugin.id}_stream_ok": True,
                }, None

            with mock.patch.object(self.module, "_run_plugin_task", side_effect=fake_run_plugin_task):
                self.module.run_model(
                    "dummy-model", "Local", loaded, plugins, source_config,
                    timeout=1, token_levels=[100], output_dir=tmpdir,
                    session_seed=0, global_cfg={},
                )

            self.assertEqual(run_calls, ["moe-dense"])

            result = loaded.latest_results()[0]
            self.assertEqual(result["rate-limiter_score"], 12.0)
            self.assertEqual(result["moe-dense_score"], 9.0)

    def test_load_state_resets_completed_models_across_all_sources(self):
        """Completed models on different sources are all reset when new plugins added."""
        models = {
            "m1": "SrcA", "m2": "SrcB", "m3": "SrcA", "m4": "SrcB",
        }
        state = self.module.BenchmarkState(models, ["rate-limiter"])
        for name in models:
            state.update(name, status="completed", **{"rate-limiter_score": 10.0})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)

            loaded = self.module.BenchmarkState.load_state(path, models, ["rate-limiter", "moe-dense"])
            snap = loaded.snapshot()
            for name in models:
                self.assertEqual(snap[name]["status"], "pending",
                                 f"{name} on {models[name]} should be pending")

    def test_load_state_resets_model_when_plugin_listed_but_not_scored(self):
        """Plugin in active_plugins without a score for a model resets that model."""
        models = {"model-a": "Src1", "model-b": "Src1"}
        state = self.module.BenchmarkState(models, ["rate-limiter", "moe-dense"])
        state.update("model-a", status="completed",
                     **{"rate-limiter_score": 5.0, "moe-dense_score": 8.0})
        state.add_result({
            "model": "model-a", "status": "ok",
            "rate-limiter_score": 5.0, "rate-limiter_response_time": 1.0,
            "rate-limiter_output_tokens": 100, "rate-limiter_tps": 50.0,
            "rate-limiter_stream_ok": True,
            "moe-dense_score": 8.0, "moe-dense_response_time": 2.0,
            "moe-dense_output_tokens": 200, "moe-dense_tps": 100.0,
            "moe-dense_stream_ok": True,
            "total_time": 3.0,
            "plugin_versions": {"rate-limiter": "0.1.0", "moe-dense": "0.1.0"},
        })
        # model-b completed but moe-dense never ran (partial previous run)
        state.update("model-b", status="completed", **{"rate-limiter_score": 6.0})
        state.add_result({
            "model": "model-b", "status": "ok",
            "rate-limiter_score": 6.0, "rate-limiter_response_time": 1.5,
            "rate-limiter_output_tokens": 150, "rate-limiter_tps": 60.0,
            "rate-limiter_stream_ok": True,
            "total_time": 1.5,
            "plugin_versions": {"rate-limiter": "0.1.0", "moe-dense": "0.1.0"},
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)

            loaded = self.module.BenchmarkState.load_state(
                path, models, ["rate-limiter", "moe-dense"])
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["status"], "completed",
                             "model-a has all plugin scores, should stay completed")
            self.assertEqual(snap["model-b"]["status"], "pending",
                             "model-b missing moe-dense_score, should be reset to pending")

    def test_load_state_resets_model_when_result_plugin_versions_is_subset(self):
        """Result from an older run with fewer plugins resets for missing plugins."""
        models = {"model-a": "Src1", "model-b": "Src1"}
        state = self.module.BenchmarkState(
            models, ["code-review", "moe-dense", "multi-step"])
        state.update("model-a", status="completed",
                     **{"code-review_score": 10.0, "moe-dense_score": 8.0,
                        "multi-step_score": 5.0})
        state.add_result({
            "model": "model-a", "status": "ok",
            "code-review_score": 10.0, "code-review_response_time": 1.0,
            "code-review_output_tokens": 100, "code-review_tps": 50.0,
            "code-review_stream_ok": True,
            "moe-dense_score": 8.0, "moe-dense_response_time": 2.0,
            "moe-dense_output_tokens": 200, "moe-dense_tps": 100.0,
            "moe-dense_stream_ok": True,
            "multi-step_score": 5.0, "multi-step_response_time": 3.0,
            "multi-step_output_tokens": 300, "multi-step_tps": 80.0,
            "multi-step_stream_ok": True,
            "total_time": 6.0,
            "plugin_versions": {"code-review": "0.2.0", "moe-dense": "0.1.0",
                                "multi-step": "0.1.0"},
        })
        # model-b ran before multi-step existed — its result has fewer plugins
        state.update("model-b", status="completed",
                     **{"code-review_score": 7.0, "moe-dense_score": 9.0})
        state.add_result({
            "model": "model-b", "status": "ok",
            "code-review_score": 7.0, "code-review_response_time": 1.5,
            "code-review_output_tokens": 150, "code-review_tps": 60.0,
            "code-review_stream_ok": True,
            "moe-dense_score": 9.0, "moe-dense_response_time": 2.5,
            "moe-dense_output_tokens": 250, "moe-dense_tps": 90.0,
            "moe-dense_stream_ok": True,
            "total_time": 4.0,
            "plugin_versions": {"code-review": "0.2.0", "moe-dense": "0.1.0"},
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)

            loaded = self.module.BenchmarkState.load_state(
                path, models, ["code-review", "moe-dense", "multi-step"])
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["status"], "completed",
                             "model-a has all plugin scores, should stay completed")
            self.assertEqual(snap["model-b"]["status"], "pending",
                             "model-b result has fewer plugins, missing multi-step_score")


if __name__ == "__main__":
    unittest.main()
