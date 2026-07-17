"""Tests for CLI argument handling and plugin execution modes."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from plugins import discover_plugins
from tests.utils import load_benchmark_module, MockResponse


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

    def test_dump_default_config_has_per_source_plugin_thread_limit(self):
        result = subprocess.run(
            [sys.executable, "ai-benchmark.py", "--dump-default-config"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        cfg = json.loads(result.stdout)
        for src_cfg in cfg["sources"].values():
            self.assertIn("plugin_thread_limit", src_cfg)
            self.assertEqual(src_cfg["plugin_thread_limit"], 1)

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

    def test_dump_default_config_shows_per_model_object_syntax(self):
        result = subprocess.run(
            [sys.executable, "ai-benchmark.py", "--dump-default-config"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        cfg = json.loads(result.stdout)
        self.assertIn("models", cfg)
        self.assertIn("example-model-3", cfg["models"])
        self.assertEqual(cfg["models"]["example-model-3"]["source"], "Local Server 2")
        self.assertEqual(cfg["models"]["example-model-3"]["drop_params"], ["seed"])


class TestPluginExecutionMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()
        cls.plugins = discover_plugins()

    def test_run_model_thread_limit_one_completes(self):
        plugins = [p for p in self.plugins if p.id in ("rate-limiter", "moe-dense")]
        models = {"dummy-model": "Local"}
        state = self.module.BenchmarkState(models, [p.id for p in plugins])
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}, "plugin_thread_limit": 1}}

        with mock.patch.object(self.module, "stream_request", return_value=("", None, 0, "connection refused", None, {})):
            with mock.patch.object(self.module, "nonstream_request", return_value=("", {}, 0.1, "connection refused", None)):
                self.module.run_model(
                    "dummy-model", "Local", state, plugins, source_config,
                    timeout=1, token_levels=[100], output_dir="/tmp/benchmark-test",
                    session_seed=0, global_cfg={},
                )

        snap = state.snapshot()["dummy-model"]
        self.assertIn(snap["status"], ("completed", "failed"))

    def test_run_model_thread_limit_zero_completes(self):
        plugins = [p for p in self.plugins if p.id in ("rate-limiter", "moe-dense")]
        models = {"dummy-model": "Local"}
        state = self.module.BenchmarkState(models, [p.id for p in plugins])
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}, "plugin_thread_limit": 0}}

        with mock.patch.object(self.module, "stream_request", return_value=("", None, 0, "connection refused", None, {})):
            with mock.patch.object(self.module, "nonstream_request", return_value=("", {}, 0.1, "connection refused", None)):
                self.module.run_model(
                    "dummy-model", "Local", state, plugins, source_config,
                    timeout=1, token_levels=[100], output_dir="/tmp/benchmark-test",
                    session_seed=0, global_cfg={},
                )

        snap = state.snapshot()["dummy-model"]
        self.assertIn(snap["status"], ("completed", "failed"))

    def test_run_model_thread_limit_two_completes(self):
        plugins = [p for p in self.plugins if p.id in ("rate-limiter", "moe-dense")]
        models = {"dummy-model": "Local"}
        state = self.module.BenchmarkState(models, [p.id for p in plugins])
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}, "plugin_thread_limit": 2}}

        with mock.patch.object(self.module, "stream_request", return_value=("", None, 0, "connection refused", None, {})):
            with mock.patch.object(self.module, "nonstream_request", return_value=("", {}, 0.1, "connection refused", None)):
                self.module.run_model(
                    "dummy-model", "Local", state, plugins, source_config,
                    timeout=1, token_levels=[100], output_dir="/tmp/benchmark-test",
                    session_seed=0, global_cfg={},
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
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}, "plugin_thread_limit": 1}}

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
                session_seed=0, global_cfg={},
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

    def test_partial_failure_rerun_only_runs_failed_plugins(self):
        """On restart, only plugins that previously failed are re-run."""
        plugins = [p for p in self.plugins if p.id in ("rate-limiter", "moe-dense")]
        models = {"dummy-model": "Local"}
        state = self.module.BenchmarkState(models, [p.id for p in plugins])
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}, "plugin_thread_limit": 1}}

        # Seed a previous partial result: rate-limiter succeeded, moe-dense failed.
        state.add_result({
            "model": "dummy-model",
            "status": "error",
            "rate-limiter_score": 5,
            "rate-limiter_response_time": 1.2,
            "rate-limiter_output_tokens": 100,
            "rate-limiter_tps": 50.0,
            "rate-limiter_stream_ok": True,
            "moe-dense_score": "fail",
            "moe-dense_response_time": "fail",
            "moe-dense_output_tokens": "fail",
            "moe-dense_tps": "fail",
            "moe-dense_stream_ok": False,
        })

        calls = []

        def fake_run_plugin_task(model_name, source, plugin, *args, **kwargs):
            calls.append(plugin.id)
            if plugin.id == "moe-dense":
                return {
                    "moe-dense_score": 7,
                    "moe-dense_response_time": 2.0,
                    "moe-dense_output_tokens": 200,
                    "moe-dense_tps": 100.0,
                    "moe-dense_stream_ok": True,
                }, None
            return None, "should not be called"

        with mock.patch.object(self.module, "_run_plugin_task", side_effect=fake_run_plugin_task):
            self.module.run_model(
                "dummy-model", "Local", state, plugins, source_config,
                timeout=1, token_levels=[100], output_dir="/tmp/benchmark-test",
                session_seed=0, global_cfg={},
            )

        self.assertEqual(calls, ["moe-dense"])
        result = state.latest_results()[0]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["rate-limiter_score"], 5)
        self.assertEqual(result["moe-dense_score"], 7)


class TestSaveResponses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()
        cls.plugins = discover_plugins()

    def test_save_responses_writes_prompt_and_response_files(self):
        plugins = [p for p in self.plugins if p.id == "rate-limiter"]
        models = {"dummy-model": "Local"}
        state = self.module.BenchmarkState(models, [p.id for p in plugins])
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}, "plugin_thread_limit": 1}}

        expected_response = "This is the model response for rate limiter."

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                self.module, "stream_request", return_value=(expected_response, 1.0, 1.5, None, "stop", {})
            ):
                with mock.patch.object(
                    self.module, "nonstream_request", return_value=(expected_response, {}, 0.1, None, "stop")
                ):
                    self.module.run_model(
                        "dummy-model", "Local", state, plugins, source_config,
                        timeout=1, token_levels=[100], output_dir=tmpdir,
                        session_seed=12345, global_cfg={},
                        save_responses=True,
                    )

            responses_dir = os.path.join(tmpdir, "responses", "dummy-model")
            prompt_path = os.path.join(responses_dir, "rate-limiter.prompt.txt")
            response_path = os.path.join(responses_dir, "rate-limiter.txt")

            self.assertTrue(os.path.isfile(prompt_path))
            self.assertTrue(os.path.isfile(response_path))

            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_content = f.read()
            with open(response_path, "r", encoding="utf-8") as f:
                response_content = f.read()

            self.assertEqual(prompt_content, plugins[0].get_prompt())
            self.assertEqual(response_content, expected_response)

            meta_path = os.path.join(responses_dir, "rate-limiter.meta.json")
            self.assertTrue(os.path.isfile(meta_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["plugin"], "rate-limiter")
            self.assertEqual(meta["plugin_version"], plugins[0].version)
            self.assertEqual(meta["model"], "dummy-model")
            self.assertIn("score", meta)
            self.assertIn("response_time", meta)
            self.assertIn("output_tokens", meta)
            self.assertIn("tps", meta)
            self.assertIn("seed", meta)
            self.assertEqual(meta["seed"], 12345)
            self.assertIn("timestamp", meta)
            self.assertIn("rubric", meta)
            self.assertIsInstance(meta["rubric"], list)
            self.assertTrue(all("name" in item and "max" in item and "earned" in item and "missed" in item for item in meta["rubric"]))

    def test_save_responses_disabled_does_not_write_files(self):
        plugins = [p for p in self.plugins if p.id == "rate-limiter"]
        models = {"dummy-model": "Local"}
        state = self.module.BenchmarkState(models, [p.id for p in plugins])
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}, "plugin_thread_limit": 1}}

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                self.module, "stream_request", return_value=("response", 1.0, 1.5, None, "stop", {})
            ):
                with mock.patch.object(
                    self.module, "nonstream_request", return_value=("response", {}, 0.1, None, "stop")
                ):
                    self.module.run_model(
                        "dummy-model", "Local", state, plugins, source_config,
                        timeout=1, token_levels=[100], output_dir=tmpdir,
                        session_seed=0, global_cfg={},
                        save_responses=False,
                    )

            responses_dir = os.path.join(tmpdir, "responses")
            self.assertFalse(os.path.exists(responses_dir))


class TestDropParams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()
        cls.plugins = discover_plugins()

    def test_stream_request_drop_params_omits_seed(self):
        """stream_request omits seed when drop_params contains 'seed'."""
        captured = {}
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}}}

        def fake_post(url, **kwargs):
            captured["body"] = kwargs.get("json")
            return MockResponse()

        with mock.patch("requests.post", side_effect=fake_post):
            self.module.stream_request(
                source_config, timeout=1, model="m", source="Local",
                prompt="hello", max_tokens=10, session_seed=12345,
                temperature=0.5, drop_params=["seed"],
            )

        self.assertIn("body", captured)
        self.assertNotIn("seed", captured["body"])
        self.assertIn("temperature", captured["body"])

    def test_nonstream_request_drop_params_omits_temperature(self):
        """nonstream_request omits temperature when drop_params contains 'temperature'."""
        captured = {}
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}}}

        def fake_post(url, **kwargs):
            captured["body"] = kwargs.get("json")
            return MockResponse()

        with mock.patch("requests.post", side_effect=fake_post):
            self.module.nonstream_request(
                source_config, timeout=1, model="m", source="Local",
                prompt="hello", max_tokens=10, session_seed=12345,
                temperature=0.5, drop_params=["temperature"],
            )

        self.assertIn("body", captured)
        self.assertNotIn("temperature", captured["body"])
        self.assertIn("seed", captured["body"])

    def test_run_plugin_task_threads_drop_params_to_request(self):
        """_run_plugin_task reads drop_params from global_cfg and omits them from requests."""
        plugins = [p for p in self.plugins if p.id == "rate-limiter"]
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}}}

        captured = {}

        def fake_post(url, **kwargs):
            captured["body"] = kwargs.get("json")
            return MockResponse()

        global_cfg = {
            "models": {
                "dummy-model": {
                    "source": "Local",
                    "drop_params": ["seed"],
                }
            }
        }

        with mock.patch("requests.post", side_effect=fake_post):
            self.module._run_plugin_task(
                "dummy-model", "Local", plugins[0], source_config,
                timeout=1, token_levels=[100], session_seed=12345,
                log_file=None, global_cfg=global_cfg,
            )

        self.assertIn("body", captured)
        self.assertNotIn("seed", captured["body"])


class TestSeedCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()
        cls.plugins = discover_plugins()

    def test_fixed_seed_passed_to_request_body(self):
        """A fixed session_seed appears in the API request body."""
        plugins = [p for p in self.plugins if p.id == "rate-limiter"]
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}}}

        captured = {}

        def fake_post(url, **kwargs):
            captured["body"] = kwargs.get("json")
            return MockResponse()

        with mock.patch("requests.post", side_effect=fake_post):
            self.module._run_plugin_task(
                "dummy-model", "Local", plugins[0], source_config,
                timeout=1, token_levels=[100], session_seed=42,
                log_file=None, global_cfg={},
            )

        self.assertIn("body", captured)
        self.assertEqual(captured["body"]["seed"], 42)


class TestStopEventInterruption(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()

    def test_stream_request_respects_stop_event(self):
        """stream_request returns 'Cancelled' when stop_event is set mid-stream."""
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}}}
        stop_event = threading.Event()

        class SlowMockResponse:
            status_code = 200

            def iter_lines(self, decode_unicode=False):
                # Yield many lines; the outer loop will see stop_event and break.
                for _ in range(100):
                    yield "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}"
                    time.sleep(0.01)

            def close(self):
                pass

        def fake_post(url, **kwargs):
            return SlowMockResponse()

        def set_stop_after_delay():
            time.sleep(0.05)
            stop_event.set()

        with mock.patch("requests.post", side_effect=fake_post):
            thread = threading.Thread(target=set_stop_after_delay)
            thread.start()
            text, first_tok, stream_end, err, finish_reason, usage = self.module.stream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hello", max_tokens=10, stop_event=stop_event,
            )
            thread.join()

        self.assertEqual(err, "Cancelled")

    def test_nonstream_request_respects_stop_event(self):
        """nonstream_request returns 'Cancelled' when stop_event is set mid-read."""
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}}}
        stop_event = threading.Event()

        class SlowMockResponse:
            status_code = 200

            def iter_content(self, chunk_size=8192):
                # Yield many chunks; the outer loop will see stop_event and break.
                for _ in range(100):
                    yield b'{"choices":[{"message":{"content":"x"}}]}'
                    time.sleep(0.01)

            def close(self):
                pass

        def fake_post(url, **kwargs):
            return SlowMockResponse()

        def set_stop_after_delay():
            time.sleep(0.05)
            stop_event.set()

        with mock.patch("requests.post", side_effect=fake_post):
            thread = threading.Thread(target=set_stop_after_delay)
            thread.start()
            text, usage, gen_time, err, finish_reason = self.module.nonstream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hello", max_tokens=10, stop_event=stop_event,
            )
            thread.join()

        self.assertEqual(err, "Cancelled")


class TestPerPluginTemperature(unittest.TestCase):
    def test_plugin_temperature_from_config(self):
        from benchmark_core import parse_plugin_temperatures
        cfg = {
            "rate-limiter_temperature": 0.1,
            "moe-dense_temperature": 0.9,
        }
        plugin_temperatures = parse_plugin_temperatures(cfg)
        self.assertEqual(plugin_temperatures["rate-limiter"], 0.1)
        self.assertEqual(plugin_temperatures["moe-dense"], 0.9)

    def test_legacy_temperature_keys_are_ignored(self):
        from benchmark_core import parse_plugin_temperatures
        cfg = {"code_temperature": 0.2, "general_temperature": 0.7}
        plugin_temperatures = parse_plugin_temperatures(cfg)
        self.assertNotIn("rate-limiter", plugin_temperatures)
        self.assertNotIn("moe-dense", plugin_temperatures)

    def test_default_temperature_overrides_config_for_all_plugins(self):
        """--temperature applies to every active plugin, overriding config."""
        from benchmark_core import parse_plugin_temperatures
        cfg = {
            "rate-limiter_temperature": 0.1,
            "moe-dense_temperature": 0.9,
        }
        plugin_temperatures = parse_plugin_temperatures(cfg)
        active_plugins = [
            type("P", (), {"id": "rate-limiter"}),
            type("P", (), {"id": "moe-dense"}),
        ]
        default_temp = 0.5
        for plugin in active_plugins:
            plugin_temperatures[plugin.id] = default_temp
        self.assertEqual(plugin_temperatures["rate-limiter"], 0.5)
        self.assertEqual(plugin_temperatures["moe-dense"], 0.5)

    def test_per_plugin_temperature_overrides_default_temperature(self):
        """--plugin-temperature takes priority over --temperature."""
        from benchmark_core import parse_plugin_temperatures
        cfg = {"rate-limiter_temperature": 0.1}
        plugin_temperatures = parse_plugin_temperatures(cfg)
        active_plugins = [type("P", (), {"id": "rate-limiter"})]
        for plugin in active_plugins:
            plugin_temperatures[plugin.id] = 0.5
        plugin_temperatures["rate-limiter"] = 0.3
        self.assertEqual(plugin_temperatures["rate-limiter"], 0.3)



if __name__ == "__main__":
    unittest.main()
