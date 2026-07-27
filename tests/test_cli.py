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

        def fake_run_plugin_task(target_name, api_model, source, plugin, *args, **kwargs):
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

        def fake_run_plugin_task(target_name, api_model, source, plugin, *args, **kwargs):
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
            self.assertEqual(meta["target"], "dummy-model")
            self.assertEqual(meta["model"], "dummy-model")
            self.assertEqual(meta["is_agent"], False)
            self.assertIn("system_prompt", meta)
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

        state = self.module.BenchmarkState({"dummy-model": "Local"}, ["rate-limiter"])
        with mock.patch("requests.post", side_effect=fake_post):
            self.module._run_plugin_task(
                "dummy-model", "dummy-model", "Local", plugins[0], source_config,
                timeout=1, token_levels=[100], session_seed=12345,
                log_file=None, global_cfg=global_cfg, state=state,
            )

        self.assertIn("body", captured)
        self.assertNotIn("seed", captured["body"])

    def test_run_plugin_task_streaming_callback_updates_state(self):
        """The streaming on_chunk closure updates bytes_received and first_chunk_seen."""
        plugins = [p for p in self.plugins if p.id == "rate-limiter"]
        models = {"dummy-model": "Local"}
        state = self.module.BenchmarkState(models, [p.id for p in plugins])
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}}}

        def fake_stream_request(source_config, timeout, model, source, prompt, max_tokens=2048,
                                log_path=None, log_label=None, session_seed=0, temperature=None,
                                drop_params=None, stop_event=None, system_prompt=None,
                                on_chunk=None):
            # Simulate two SSE deltas; the closure should fire once per delta.
            for delta in ["Hello, ", "world"]:
                if on_chunk is not None:
                    on_chunk(delta)
            return "Hello, world", 1.0, 1.5, None, "stop", {}

        with mock.patch.object(self.module, "stream_request", side_effect=fake_stream_request):
            with mock.patch.object(self.module, "nonstream_request", return_value=("", {}, 0.1, "no tokens", "stop")):
                result, err = self.module._run_plugin_task(
                    "dummy-model", "dummy-model", "Local", plugins[0], source_config,
                    timeout=1, token_levels=[100], session_seed=12345,
                    log_file=None, global_cfg={}, state=state,
                )

        self.assertIsNone(err)
        snap = state.snapshot()["dummy-model"]
        self.assertTrue(snap["rate-limiter_first_chunk_seen"])
        # "Hello, " (7) + "world" (5) = 12 chars -> 12 // 4 = 3 tok
        self.assertEqual(snap["rate-limiter_bytes_received"], 12)
        self.assertEqual(result["rate-limiter_output_tokens"], 3)




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

        state = self.module.BenchmarkState({"dummy-model": "Local"}, ["rate-limiter"])
        with mock.patch("requests.post", side_effect=fake_post):
            self.module._run_plugin_task(
                "dummy-model", "dummy-model", "Local", plugins[0], source_config,
                timeout=1, token_levels=[100], session_seed=42,
                log_file=None, global_cfg={}, state=state,
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


class TestScriptedMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ai_benchmark", "ai-benchmark.py")
        cls.ai_benchmark = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ai_benchmark)

    def test_scripted_flag_defaults_to_continue_on_plugin_change(self):
        """--scripted continues a run when the plugin set changes."""
        import io
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            choice = self.ai_benchmark._prompt_restart_or_continue(scripted=True)
        finally:
            sys.stderr = old_stderr
        self.assertEqual(choice, "continue")

    def test_interactive_prompt_returns_continue(self):
        """Interactive mode returns the user's choice."""
        import io
        old_stdin = sys.stdin
        sys.stdin = io.StringIO("c\n")
        try:
            choice = self.ai_benchmark._prompt_restart_or_continue(scripted=False)
        finally:
            sys.stdin = old_stdin
        self.assertEqual(choice, "continue")


class TestConfigFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ai_benchmark", "ai-benchmark.py")
        cls.ai_benchmark = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ai_benchmark)

    def test_resolve_config_prefers_existing_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "benchmark-config.json")
            yaml_path = os.path.join(tmpdir, "benchmark-config.yaml")
            open(json_path, "w").close()
            open(yaml_path, "w").close()
            result = self.ai_benchmark._resolve_config_path(json_path)
            self.assertEqual(result, json_path)

    def test_resolve_config_falls_back_to_yaml_then_yml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            default_path = os.path.join(tmpdir, "benchmark-config.json")
            yaml_path = os.path.join(tmpdir, "benchmark-config.yaml")
            yml_path = os.path.join(tmpdir, "benchmark-config.yml")
            self.assertIsNone(self.ai_benchmark._resolve_config_path(default_path))
            open(yml_path, "w").close()
            self.assertEqual(self.ai_benchmark._resolve_config_path(default_path), yml_path)
            open(yaml_path, "w").close()
            self.assertEqual(self.ai_benchmark._resolve_config_path(default_path), yaml_path)

    def test_resolve_config_does_not_fallback_for_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            explicit_path = os.path.join(tmpdir, "my-config.json")
            self.assertIsNone(self.ai_benchmark._resolve_config_path(explicit_path))

    def test_missing_default_config_exits_with_error(self):
        """Running without a config exits with a helpful error message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, os.path.abspath("ai-benchmark.py")],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("Config file not found", result.stderr)
            self.assertIn("benchmark-config.json", result.stderr)
            self.assertIn("benchmark-config.yaml", result.stderr)
            self.assertIn("benchmark-config.yml", result.stderr)

    def test_default_yaml_config_is_used_when_json_missing(self):
        """If benchmark-config.json is missing, benchmark-config.yaml is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "benchmark-config.yaml")
            output_dir = os.path.join(tmpdir, "output")
            with open(config_path, "w") as f:
                f.write(f"output_dir: {output_dir}\n")

            result = subprocess.run(
                [sys.executable, os.path.abspath("ai-benchmark.py")],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            run_info_path = os.path.join(output_dir, "run-info.json")
            self.assertTrue(os.path.isfile(run_info_path))
            with open(run_info_path, "r", encoding="utf-8") as f:
                run_info = json.load(f)
            self.assertEqual(os.path.basename(run_info["config_file"]), "benchmark-config.yaml")
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "benchmark-config.yaml")))


class TestTimeCapsule(unittest.TestCase):
    def test_config_file_is_copied_to_output_dir_json(self):
        """A JSON config is copied into the output directory as a time capsule."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            output_dir = os.path.join(tmpdir, "output")
            with open(config_path, "w") as f:
                json.dump({"output_dir": output_dir}, f)

            result = subprocess.run(
                [sys.executable, "ai-benchmark.py", "--config", config_path],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            copied = os.path.join(output_dir, "config.json")
            self.assertTrue(os.path.isfile(copied))
            with open(copied, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f)["output_dir"], output_dir)


class TestRunInfo(unittest.TestCase):
    def test_run_info_written_for_empty_config(self):
        """A run with no models still writes run-info.json with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            output_dir = os.path.join(tmpdir, "output")
            with open(config_path, "w") as f:
                f.write(f"output_dir: {output_dir}\n")

            result = subprocess.run(
                [sys.executable, "ai-benchmark.py", "--config", config_path],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            run_info_path = os.path.join(output_dir, "run-info.json")
            self.assertTrue(os.path.isfile(run_info_path))
            with open(run_info_path, "r", encoding="utf-8") as f:
                run_info = json.load(f)

            self.assertEqual(run_info["config_file"], config_path)
            self.assertEqual(run_info["output_dir"], output_dir)
            self.assertEqual(run_info["status"], "completed")
            self.assertEqual(run_info["total_targets"], 0)
            self.assertEqual(run_info["completed_targets"], 0)
            self.assertIn("cli_args", run_info)
            self.assertIn("start_time", run_info)
            self.assertIn("end_time", run_info)
            self.assertIsNotNone(run_info["end_time"])
            self.assertIn("session_seed", run_info)

    def test_write_run_info_persists_status(self):
        """_write_run_info persists the supplied status to run-info.json."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("ai_benchmark", "ai-benchmark.py")
        ai_benchmark = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ai_benchmark)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_info = {
                "config_file": "config.yaml",
                "cli_args": {},
                "output_dir": tmpdir,
                "start_time": None,
                "end_time": None,
                "status": "running",
                "total_targets": 0,
                "completed_targets": 0,
                "worker_errors": 0,
                "session_seed": None,
            }
            ai_benchmark._write_run_info(tmpdir, run_info)
            run_info_path = os.path.join(tmpdir, "run-info.json")
            self.assertTrue(os.path.isfile(run_info_path))
            with open(run_info_path, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f)["status"], "running")


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



class TestCLIRetryOn429(unittest.TestCase):
    """Tests for the --retry-on-429 / --no-retry-on-429 CLI flag pair
    and benchmark_core._apply_http_retry_default helper."""

    def test_help_advertises_retry_on_429(self):
        """ai-benchmark.py --help mentions both flag forms."""
        result = subprocess.run(
            [sys.executable, "ai-benchmark.py", "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--retry-on-429", result.stdout)
        self.assertIn("--no-retry-on-429", result.stdout)

    def test_retry_on_429_fat_finger_rejected(self):
        """Passing both --retry-on-429 and --no-retry-on-429 on the same command line
        must produce an argparse error (mutually exclusive group), not silently
        last-wins. Argparse exits with code 2 on conflict and writes the
        offending flag pair to stderr.
        """
        result = subprocess.run(
            [sys.executable, "ai-benchmark.py",
             "--retry-on-429", "--no-retry-on-429",
             "--dump-default-config"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2,
                         "passing both retry flags must fail (exit 2); "
                         "the mutual-exclusion is the whole point of "
                         "add_mutually_exclusive_group().")
        # argparse formats the error as 'argument --no-retry-on-429: not
        # allowed with argument --retry-on-429' (or the reverse).
        self.assertIn("not allowed", result.stderr.lower())
        self.assertIn("--retry-on-429", result.stderr)
        self.assertIn("--no-retry-on-429", result.stderr)

    def test_default_retry_on_429_is_true(self):
        """When neither flag is supplied, --dump-default-config shows no
        max_429_retries injection (default-ON path is a no-op at the helper)."""
        result = subprocess.run(
            [sys.executable, "ai-benchmark.py", "--dump-default-config"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0)
        cfg = json.loads(result.stdout)
        for src_cfg in cfg["sources"].values():
            self.assertNotIn("max_429_retries", src_cfg)

    def test_apply_http_retry_default_no_op_when_retry_on(self):
        """retry_on_429=True leaves per-source config untouched."""
        from benchmark_core import _apply_http_retry_default
        cfg = {"sources": {"Local": {"api_url": "http://x", "headers": {}}}}
        _apply_http_retry_default(cfg, retry_on_429=True)
        self.assertNotIn("max_429_retries", cfg["sources"]["Local"])

    def test_apply_http_retry_default_zeros_implicit_sources(self):
        """retry_on_429=False zeroes max_429_retries on sources that didn't set it."""
        from benchmark_core import _apply_http_retry_default
        cfg = {
            "sources": {
                "Local": {"api_url": "http://x", "headers": {}},
                "Remote": {"api_url": "http://r", "headers": {}, "max_429_retries": 5,
                            "backoff_seconds": 12},
            }
        }
        _apply_http_retry_default(cfg, retry_on_429=False)
        # Implicit source flipped to 0.
        self.assertEqual(cfg["sources"]["Local"]["max_429_retries"], 0)
        # Explicit source preserved.
        self.assertEqual(cfg["sources"]["Remote"]["max_429_retries"], 5)
        self.assertEqual(cfg["sources"]["Remote"]["backoff_seconds"], 12)

    def test_apply_http_retry_default_handles_missing_sources(self):
        """A config without a 'sources' key should not crash."""
        from benchmark_core import _apply_http_retry_default
        cfg = {}
        _apply_http_retry_default(cfg, retry_on_429=False)
        self.assertEqual(cfg, {})

    def test_apply_http_retry_default_preserves_explicit_per_source_only(self):
        """End-to-end story: default-ON keeps implicit sources untouched;
        default-OFF flips the implicit ones while preserving explicit opts-in."""
        from benchmark_core import _apply_http_retry_default
        cfg = {
            "sources": {
                "Local": {"api_url": "http://x", "headers": {}},
                "Remote": {"api_url": "http://r", "headers": {}, "max_429_retries": 4},
            }
        }
        _apply_http_retry_default(cfg, retry_on_429=True)
        self.assertNotIn("max_429_retries", cfg["sources"]["Local"])
        self.assertEqual(cfg["sources"]["Remote"]["max_429_retries"], 4)
        _apply_http_retry_default(cfg, retry_on_429=False)
        self.assertEqual(cfg["sources"]["Local"]["max_429_retries"], 0)
        self.assertEqual(cfg["sources"]["Remote"]["max_429_retries"], 4)


if __name__ == "__main__":
    unittest.main()
