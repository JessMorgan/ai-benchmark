"""Tests for CLI argument handling and plugin execution modes."""
import json
import subprocess
import sys
import unittest
from unittest import mock

from plugins import discover_plugins
from tests.utils import load_benchmark_module


class TestCLIArgs(unittest.TestCase):
    def test_list_plugins_shows_id_name_version(self):
        result = subprocess.run(
            [sys.executable, "ai-benchmark.py", "--list-plugins"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        output = result.stdout
        self.assertIn("ID", output)
        self.assertIn("Name", output)
        self.assertIn("Version", output)
        self.assertIn("rate-limiter", output)
        self.assertIn("Rate Limiter", output)
        self.assertIn("moe-dense", output)
        self.assertIn("MoE vs Dense", output)
        self.assertIn("tool-calling", output)
        self.assertIn("Tool Calling Agent", output)
        self.assertIn("structured-output", output)
        self.assertIn("Structured Output", output)
        # Check a specific ID/name/version line
        self.assertRegex(output, r"structured-output\s+Structured Output\s+0\.2\.0")
        # Footer hint helps users use the IDs
        self.assertIn("--plugins-whitelist", output)
        self.assertIn("--plugins-blacklist", output)

    def test_format_plugin_list_empty(self):
        from plugins import format_plugin_list
        self.assertEqual(format_plugin_list([]), "No plugins discovered.")

    def test_dump_default_config_has_plugin_thread_limit(self):
        result = subprocess.run(
            [sys.executable, "ai-benchmark.py", "--dump-default-config"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        cfg = json.loads(result.stdout)
        self.assertIn("plugin_thread_limit", cfg)
        self.assertEqual(cfg["plugin_thread_limit"], 1)

    def test_dump_default_config_has_per_plugin_temperatures(self):
        result = subprocess.run(
            [sys.executable, "ai-benchmark.py", "--dump-default-config"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        cfg = json.loads(result.stdout)
        self.assertIn("rate-limiter_temperature", cfg)
        self.assertIn("moe-dense_temperature", cfg)
        self.assertNotIn("code_temperature", cfg)
        self.assertNotIn("general_temperature", cfg)


class TestPluginExecutionMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()
        cls.plugins = discover_plugins()

    def test_run_model_thread_limit_one_completes(self):
        plugins = [p for p in self.plugins if p.id in ("rate-limiter", "moe-dense")]
        models = {"dummy-model": "Local"}
        state = self.module.BenchmarkState(models, [p.id for p in plugins])
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}}}
        global_cfg = {"plugin_thread_limit": 1}

        with mock.patch.object(self.module, "stream_request", return_value=("", None, 0, "connection refused", None, {})):
            with mock.patch.object(self.module, "nonstream_request", return_value=("", {}, 0.1, "connection refused", None)):
                self.module.run_model(
                    "dummy-model", "Local", state, plugins, source_config,
                    timeout=1, token_levels=[100], output_dir="/tmp/benchmark-test",
                    session_seed=0, global_cfg=global_cfg,
                )

        snap = state.snapshot()["dummy-model"]
        self.assertIn(snap["status"], ("completed", "failed"))

    def test_run_model_thread_limit_zero_completes(self):
        plugins = [p for p in self.plugins if p.id in ("rate-limiter", "moe-dense")]
        models = {"dummy-model": "Local"}
        state = self.module.BenchmarkState(models, [p.id for p in plugins])
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}}}
        global_cfg = {"plugin_thread_limit": 0}

        with mock.patch.object(self.module, "stream_request", return_value=("", None, 0, "connection refused", None, {})):
            with mock.patch.object(self.module, "nonstream_request", return_value=("", {}, 0.1, "connection refused", None)):
                self.module.run_model(
                    "dummy-model", "Local", state, plugins, source_config,
                    timeout=1, token_levels=[100], output_dir="/tmp/benchmark-test",
                    session_seed=0, global_cfg=global_cfg,
                )

        snap = state.snapshot()["dummy-model"]
        self.assertIn(snap["status"], ("completed", "failed"))

    def test_run_model_thread_limit_two_completes(self):
        plugins = [p for p in self.plugins if p.id in ("rate-limiter", "moe-dense")]
        models = {"dummy-model": "Local"}
        state = self.module.BenchmarkState(models, [p.id for p in plugins])
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}}}
        global_cfg = {"plugin_thread_limit": 2}

        with mock.patch.object(self.module, "stream_request", return_value=("", None, 0, "connection refused", None, {})):
            with mock.patch.object(self.module, "nonstream_request", return_value=("", {}, 0.1, "connection refused", None)):
                self.module.run_model(
                    "dummy-model", "Local", state, plugins, source_config,
                    timeout=1, token_levels=[100], output_dir="/tmp/benchmark-test",
                    session_seed=0, global_cfg=global_cfg,
                )

        snap = state.snapshot()["dummy-model"]
        self.assertIn(snap["status"], ("completed", "failed"))


class TestPartialPluginFailure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()
        cls.plugins = discover_plugins()

    def test_partial_failure_records_success_and_fail_values(self):
        """When one plugin fails and another succeeds, both results are recorded."""
        plugins = [p for p in self.plugins if p.id in ("rate-limiter", "moe-dense")]
        models = {"dummy-model": "Local"}
        state = self.module.BenchmarkState(models, [p.id for p in plugins])
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}}}

        def fake_run_plugin_task(model_name, source, plugin, *args, **kwargs):
            if plugin.id == "rate-limiter":
                return {
                    "rate-limiter_score": 5,
                    "rate-limiter_response_time": 1.2,
                    "rate-limiter_output_tokens": 100,
                    "rate-limiter_tps": 50.0,
                    "rate-limiter_stream_ok": True,
                }, None
            return None, "connection refused"

        with mock.patch.object(self.module, "_run_plugin_task", side_effect=fake_run_plugin_task):
            self.module.run_model(
                "dummy-model", "Local", state, plugins, source_config,
                timeout=1, token_levels=[100], output_dir="/tmp/benchmark-test",
                session_seed=0, global_cfg={"plugin_thread_limit": 1},
            )

        snap = state.snapshot()["dummy-model"]
        self.assertEqual(snap["status"], "failed")

        result = state.latest_results()[0]
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["rate-limiter_score"], 5)
        self.assertEqual(result["rate-limiter_response_time"], 1.2)
        self.assertEqual(result["rate-limiter_output_tokens"], 100)
        self.assertEqual(result["rate-limiter_tps"], 50.0)
        self.assertEqual(result["moe-dense_score"], "fail")
        self.assertEqual(result["moe-dense_response_time"], "fail")
        self.assertEqual(result["moe-dense_output_tokens"], "fail")
        self.assertEqual(result["moe-dense_tps"], "fail")


class TestPerPluginTemperature(unittest.TestCase):
    def test_plugin_temperature_from_config(self):
        cfg = {
            "rate-limiter_temperature": 0.1,
            "moe-dense_temperature": 0.9,
        }
        plugin_temperatures = {}
        for key, value in cfg.items():
            if key.endswith("_temperature"):
                plugin_id = key[:-len("_temperature")].replace("_", "-")
                plugin_temperatures[plugin_id] = value
        self.assertEqual(plugin_temperatures["rate-limiter"], 0.1)
        self.assertEqual(plugin_temperatures["moe-dense"], 0.9)

    def test_legacy_temperature_keys_map_correctly(self):
        cfg = {"code_temperature": 0.2, "general_temperature": 0.7}
        plugin_temperatures = {}
        for key, value in cfg.items():
            if key.endswith("_temperature"):
                plugin_id = key[:-len("_temperature")].replace("_", "-")
                plugin_temperatures[plugin_id] = value
        if "code_temperature" in cfg and "rate-limiter" not in plugin_temperatures:
            plugin_temperatures["rate-limiter"] = cfg["code_temperature"]
        if "general_temperature" in cfg and "moe-dense" not in plugin_temperatures:
            plugin_temperatures["moe-dense"] = cfg["general_temperature"]
        self.assertEqual(plugin_temperatures["rate-limiter"], 0.2)
        self.assertEqual(plugin_temperatures["moe-dense"], 0.7)



if __name__ == "__main__":
    unittest.main()
