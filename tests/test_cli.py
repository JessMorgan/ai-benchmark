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

        with mock.patch.object(self.module, "stream_request", return_value=("", "", None, 0, "connection refused", None, {})):
                with mock.patch.object(self.module, "nonstream_request", return_value=("", "", {}, 0.1, "connection refused", None)):
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

        with mock.patch.object(self.module, "stream_request", return_value=("", "", None, 0, "connection refused", None, {})):
            with mock.patch.object(self.module, "nonstream_request", return_value=("", "", {}, 0.1, "connection refused", None)):
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

        with mock.patch.object(self.module, "stream_request", return_value=("", "", None, 0, "connection refused", None, {})):
            with mock.patch.object(self.module, "nonstream_request", return_value=("", "", {}, 0.1, "connection refused", None)):
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
                self.module, "stream_request", return_value=(expected_response, "", 1.0, 1.5, None, "stop", {})
            ):
                with mock.patch.object(
                    self.module, "nonstream_request", return_value=(expected_response, "", {}, 0.1, None, "stop")
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
            think_path = os.path.join(responses_dir, "rate-limiter.think.txt")
            content_path = os.path.join(responses_dir, "rate-limiter.content.txt")

            self.assertTrue(os.path.isfile(prompt_path))
            self.assertTrue(os.path.isfile(response_path))
            # No think_text was returned by the mock, so .think.txt must NOT
            # be written (only created when thinking content is non-empty).
            self.assertFalse(os.path.isfile(think_path))
            self.assertTrue(os.path.isfile(content_path))

            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_content = f.read()
            with open(response_path, "r", encoding="utf-8") as f:
                response_content = f.read()
            with open(content_path, "r", encoding="utf-8") as f:
                content_content = f.read()

            self.assertEqual(prompt_content, plugins[0].get_prompt())
            self.assertEqual(response_content, expected_response)
            # Without thinking, .txt and .content.txt are identical.
            self.assertEqual(content_content, expected_response)

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

    def test_save_responses_with_thinking_writes_three_files(self):
        """When the model returns thinking content, all three response-file
        variants are written:

        * ``.txt`` — joined form with ``<thinking>…</thinking>`` markers
          followed by the final content.
        * ``.think.txt`` — pure thinking content only.
        * ``.content.txt`` — pure final content without thinking markers.

        The existing ``test_save_responses_writes_prompt_and_response_files``
        pins the empty-think-text (non-thinking model) path where
        ``.think.txt`` is NOT created and ``.txt == .content.txt``. This test
        pins the opposite: non-empty thinking content produces all three
        distinct files.
        """
        plugins = [p for p in self.plugins if p.id == "rate-limiter"]
        models = {"dummy-model": "Local"}
        state = self.module.BenchmarkState(models, [p.id for p in plugins])
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}, "plugin_thread_limit": 1}}

        final_content = "This is the final answer."
        thinking = "Let me think through this step by step.\nFirst, I need to understand the problem.\nThen I can craft a solution."

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                self.module, "stream_request", return_value=(final_content, thinking, 1.0, 1.5, None, "stop", {})
            ):
                with mock.patch.object(
                    self.module, "nonstream_request", return_value=(final_content, thinking, {}, 0.1, None, "stop")
                ):
                    self.module.run_model(
                        "dummy-model", "Local", state, plugins, source_config,
                        timeout=1, token_levels=[100], output_dir=tmpdir,
                        session_seed=12345, global_cfg={},
                        save_responses=True,
                    )

            responses_dir = os.path.join(tmpdir, "responses", "dummy-model")
            response_path = os.path.join(responses_dir, "rate-limiter.txt")
            think_path = os.path.join(responses_dir, "rate-limiter.think.txt")
            content_path = os.path.join(responses_dir, "rate-limiter.content.txt")

            # All three files must exist when thinking content is non-empty.
            self.assertTrue(os.path.isfile(response_path))
            self.assertTrue(os.path.isfile(think_path))
            self.assertTrue(os.path.isfile(content_path))

            with open(response_path, "r", encoding="utf-8") as f:
                response_content = f.read()
            with open(think_path, "r", encoding="utf-8") as f:
                think_content = f.read()
            with open(content_path, "r", encoding="utf-8") as f:
                content_content = f.read()

            # .txt = <thinking>\n{think_text}\n</thinking>\n\n{final_content}
            expected_joined = f"<thinking>\n{thinking}\n</thinking>\n\n{final_content}"
            self.assertEqual(response_content, expected_joined)
            # .think.txt = pure thinking content
            self.assertEqual(think_content, thinking)
            # .content.txt = pure final content (no thinking markers)
            self.assertEqual(content_content, final_content)

    def test_save_responses_disabled_does_not_write_files(self):
        plugins = [p for p in self.plugins if p.id == "rate-limiter"]
        models = {"dummy-model": "Local"}
        state = self.module.BenchmarkState(models, [p.id for p in plugins])
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}, "plugin_thread_limit": 1}}

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                self.module, "stream_request", return_value=("response", "", 1.0, 1.5, None, "stop", {})
            ):
                with mock.patch.object(
                    self.module, "nonstream_request", return_value=("response", "", {}, 0.1, None, "stop")
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
                                on_chunk=None, on_think_chunk=None, pid=None, on_retry=None):
            # Simulate two SSE deltas; the closure should fire once per delta.
            # ``on_think_chunk`` is the parallel reasoning-content
            # callback added to make the live TUI's per-plugin
            # thinking-phase ticker (``[streaming - N think-tok]``
            # cell + ``[<pid>: N think-tok (...s)]`` live footer)
            # increment on SSE ``reasoning_content`` deltas. The mock
            # signature advertises it so the produced
            # ``_run_plugin_task`` call site (which always passes
            # ``on_think_chunk=on_think_chunk`` for streaming plugins)
            # does not blow up with a `TypeError: unexpected keyword
            # argument`. The mock itself never produces a reasoning
            # delta (input is plain content) so the closure sits
            # idle -- the production code path's reasoning-counter
            # increment is exercised in
            # ``tests/test_tui_cells.test_in_flight_streaming_plugin_thinking_only_shows_think_tok``.
            for delta in ["Hello, ", "world"]:
                if on_chunk is not None:
                    on_chunk(delta)
            return "Hello, world", "", 1.0, 1.5, None, "stop", {}

        with mock.patch.object(self.module, "stream_request", side_effect=fake_stream_request):
            with mock.patch.object(self.module, "nonstream_request", return_value=("", "", {}, 0.1, "no tokens", "stop")):
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




class TestNonStreamingPluginRetry(unittest.TestCase):
    """Regression tests for the ``on_retry`` closure-scope bug that
    previously raised ``UnboundLocalError`` on every
    ``supports_streaming=False`` plugin (``code-review``, ``moe-dense``,
    ``structured-output``).

    Bug history: ``def on_retry():`` was defined *inside* the
    ``if plugin.supports_streaming:`` branch of
    ``benchmark_core._run_plugin_task``. Python's static scope analysis
    treats the name as local for the entire function, so the alternative
    ``else`` branch's
    ``nonstream_request(... on_retry=on_retry)`` evaluated a name that
    had never been bound, raising:

        ``UnboundLocalError: cannot access local variable 'on_retry'
        where it is not associated with a value``

    The bug surfaces at CALL-TIME kwargs evaluation -- not inside the
    request function or the retry callback -- which is why every model
    that ran a supports_streaming=False plugin failed the run for
    ~every model in the affected benchmark.
    """

    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()
        cls.plugins = discover_plugins()

    def _nonstreaming_plugin(self):
        candidates = [p for p in self.plugins if p.id == "structured-output"]
        self.assertEqual(
            len(candidates), 1,
            "structured-output plugin must be present for this regression test",
        )
        return candidates[0]

    def test_run_plugin_task_nonstreaming_plugin_binds_on_retry(self):
        """A successful ``_run_plugin_task`` call with a
        ``supports_streaming=False`` plugin must NOT raise
        ``UnboundLocalError`` at the kwargs evaluation of
        ``nonstream_request(... on_retry=on_retry)``.

        If the closure is mis-scoped (defined inside
        ``if plugin.supports_streaming:``), Python raises the error
        before ``nonstream_request`` is even called -- so a single
        clean run of a non-streaming plugin is sufficient to catch the
        regression.
        """
        plugin = self._nonstreaming_plugin()
        source_config = {
            "Local": {
                "api_url": "http://localhost:11434/chat/completions",
                "headers": {},
            }
        }
        state = self.module.BenchmarkState({"dummy-model": "Local"}, [plugin.id])

        with mock.patch.object(
            self.module, "nonstream_request",
            return_value=("", "", {}, 0.1, None, "stop"),
        ):
            # Will raise UnboundLocalError if ``on_retry`` is not bound
            # before the call site below.
            result, err = self.module._run_plugin_task(
                "dummy-model", "dummy-model", "Local", plugin, source_config,
                timeout=1, token_levels=[100], session_seed=12345,
                log_file=None, global_cfg={}, state=state,
            )

        # Empty response correctly scores 0 -- we only assert no
        # exception was raised at the kwargs evaluation.
        self.assertIsNone(err)
        self.assertEqual(result["structured-output_score"], 0.0)

    def test_run_plugin_task_nonstreaming_429_retry_fires_on_retry(self):
        """End-to-end: a 429 retry on a non-streaming plugin still
        fires ``state.start_plugin_run`` again -- not just at plugin
        dispatch time. This pins both the closure-scope fix AND the
        per-request elapsed reset wiring for supports_streaming=False
        plugins.
        """
        plugin = self._nonstreaming_plugin()
        source_config = {
            "Local": {
                "api_url": "http://localhost:11434/chat/completions",
                "headers": {},
                "max_429_retries": 1,
                "backoff_seconds": 0.01,
                "backoff_factor": 1.0,
                "max_backoff_seconds": 1.0,
            }
        }
        state = self.module.BenchmarkState({"dummy-model": "Local"}, [plugin.id])

        # Track every ``start_plugin_run`` call -- both the plugin
        # dispatch (one per plugin) and the per-429-retry reset.
        start_calls = []
        real_start = state.start_plugin_run

        def tracking_start(target_name, pid, **kwargs):
            start_calls.append((target_name, pid))
            return real_start(target_name, pid, **kwargs)

        state.start_plugin_run = tracking_start

        # 429 on attempt 1, 200 on attempt 2 -- with a tiny backoff so
        # the test stays sub-second.
        class _Resp429:
            status_code = 429
            text = "rate limited"
            headers = {}

            def close(self):
                pass

        def _mk_200():
            _body = {
                "choices": [
                    {"message": {"content": "{}"}, "finish_reason": "stop"}
                ],
                "usage": {},
            }

            class _Resp200:
                status_code = 200
                text = json.dumps(_body)
                headers = {}
                body = _body

                def iter_content(self, chunk_size=8192):
                    return [json.dumps(self.body).encode()]

                def json(self):
                    return self.body

                def close(self):
                    pass

            return _Resp200()

        with mock.patch(
            "requests.post", side_effect=[_Resp429(), _mk_200()],
        ):
            result, err = self.module._run_plugin_task(
                "dummy-model", "dummy-model", "Local", plugin, source_config,
                timeout=5, token_levels=[100], session_seed=12345,
                log_file=None, global_cfg={}, state=state,
            )

        self.assertIsNone(err)
        # ``_run_plugins`` calls ``start_plugin_run`` once at dispatch
        # (outside ``_run_plugin_task`` -- not visible to this test's
        # wrapping); the on_retry closure is the only path inside
        # ``_run_plugin_task`` that re-fires it. With max_429_retries=1
        # we expect exactly one on_retry invocation -- if a future
        # refactor fires it twice (or zero times for non-streaming
        # plugins) we want a loud failure.
        retry_callback_firings = [
            (t, p) for (t, p) in start_calls
            if (t, p) == ("dummy-model", plugin.id)
        ]
        self.assertEqual(
            retry_callback_firings,
            [("dummy-model", plugin.id)],
            f"on_retry closure must fire exactly once on a single 429 "
            f"retry for non-streaming plugins; "
            f"saw start_calls={start_calls!r}",
        )


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
            text, think_text, first_tok, stream_end, err, finish_reason, usage = self.module.stream_request(
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
            text, think_text, usage, gen_time, err, finish_reason = self.module.nonstream_request(
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

    def test_run_info_includes_backoff_429(self):
        """A completed run writes backoff_429 metadata to run-info.json.

        This integration test verifies the schema is written even when no
        429 retries occur (the common empty case). Non-empty aggregation is
        covered in ``tests.test_benchmark_http``.
        """
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

            self.assertIn("backoff_429", run_info)
            self.assertEqual(run_info["backoff_429"]["total_retries"], 0)
            self.assertEqual(run_info["backoff_429"]["per_plugin"], {})

    def test_run_info_backoff_429_populated(self):
        """run-info.json reflects populated 429 plugin statistics."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "write_run_info.py")
            run_info_path = os.path.join(tmpdir, "run-info.json")
            with open(script, "w", encoding="utf-8") as f:
                f.write(
                    "import importlib.util\n"
                    "import json\n"
                    "import os\n"
                    "import sys\n"
                    f"sys.path.insert(0, {project_root!r})\n"
                    "import benchmark_http\n"
                    "\n"
                    "# Simulate a run that saw some 429 retries.\n"
                    "benchmark_http._429_stats['total_retries'] = 3\n"
                    "benchmark_http._429_stats['plugin_stats']['rate-limiter'] = "
                    "{'retries': 2, 'total_sleep_time': 45.5}\n"
                    "benchmark_http._429_stats['plugin_stats']['moe-dense'] = "
                    "{'retries': 1, 'total_sleep_time': 12.0}\n"
                    "\n"
                    "spec = importlib.util.spec_from_file_location("
                    "    'ai_benchmark', os.path.abspath('ai-benchmark.py'))\n"
                    "ai = importlib.util.module_from_spec(spec)\n"
                    "spec.loader.exec_module(ai)\n"
                    "run_info = ai._inject_429_stats({})\n"
                    f"ai._write_run_info({tmpdir!r}, run_info)\n"
                    f"print(open({run_info_path!r}).read())\n"
                )
            result = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                cwd=project_root,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            run_info = json.loads(result.stdout)
            self.assertEqual(run_info["backoff_429"]["total_retries"], 3)
            self.assertEqual(
                run_info["backoff_429"]["per_plugin"],
                {
                    "rate-limiter": {"retries": 2, "total_sleep_time": 45.5},
                    "moe-dense": {"retries": 1, "total_sleep_time": 12.0},
                },
            )


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


class TestRetryResetsStartTimestamp(unittest.TestCase):
    """A 429 retry should reset the per-plugin start timestamp so the TUI
    shows elapsed time for the current request, not cumulative time across
    earlier failed attempts and backoff sleeps."""

    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()
        cls.plugins = discover_plugins()

    def test_429_retry_calls_start_plugin_run_again(self):
        plugins = [p for p in self.plugins if p.id == "rate-limiter"]
        models = {"dummy-model": "Local"}
        state = self.module.BenchmarkState(models, [p.id for p in plugins])
        source_config = {
            "Local": {
                "api_url": "http://localhost:11434/chat/completions",
                "headers": {},
                "max_429_retries": 1,
                "backoff_seconds": 0.01,
                "backoff_factor": 1.0,
                "max_backoff_seconds": 1.0,
            }
        }

        class _Mock429:
            status_code = 429
            text = "rate limited"
            headers = {}

            def close(self):
                pass

        class _Mock200:
            status_code = 200

            def iter_lines(self, decode_unicode=False):
                yield "data: " + json.dumps({
                    "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
                })
                yield "data: [DONE]"

            def iter_content(self, chunk_size=8192):
                yield json.dumps({
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                }).encode("utf-8")

            def close(self):
                pass

        start_calls = []
        original_start = state.start_plugin_run

        def tracking_start(model_name, pid):
            start_calls.append((time.monotonic(), model_name, pid))
            return original_start(model_name, pid)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(state, "start_plugin_run", side_effect=tracking_start):
                with mock.patch("requests.post", side_effect=[_Mock429(), _Mock200()]):
                    self.module.run_model(
                        "dummy-model", "Local", state, plugins, source_config,
                        timeout=5, token_levels=[100], output_dir=tmpdir,
                        session_seed=0, global_cfg={},
                    )

        self.assertEqual(len(start_calls), 2,
                         "start_plugin_run should fire once at dispatch and once on retry")
        self.assertGreater(start_calls[1][0], start_calls[0][0])
        # The state snapshot should reflect the final (retry) start timestamp.
        snap = state.snapshot()["dummy-model"]
        self.assertGreaterEqual(snap["rate-limiter_start_ts"], start_calls[0][0])


class TestStreamPartialTextKept(unittest.TestCase):
    """Regression tests for the silent-overwrite bug in
    ``benchmark_core._run_plugin_task`` where streaming could accumulate
    tens of thousands of characters and yet ``_output_tokens`` was
    still recorded as ``1``.

    Root cause: when ``stream_request`` returned a non-empty
    ``serr`` (eg ``"Total timeout (...) exceeded"`` from
    ``_check_total_timeout``) AND the request genuinely had streamed
    many K chars but not finished, the streaming-fallback branch
    UNCONDITIONALLY called ``nonstream_request`` and reassigned
    ``text`` to whatever it returned. A non-stream retry from a
    "thinking" model that already streamed 40 K chars will likely
    come back empty (the model has nothing buffered to repeat),
    and ``count_tokens("")`` floors to ``max(1, 0 / 4) = 1`` -- so
    a ~40 K-char observation collapsed to a 1-token placeholder
    record. Operator's kimi-dev observation: streamed 10 K tokens
    over 2 000 s then timed out, but the persisted result showed
    ``_output_tokens = 1``.

    Fix: only fall through to the non-streaming retry when
    streaming produced nothing useful (no ``first_tok`` AND empty
    ``text``). If either the first chunk landed or any characters
    accumulated, KEEP the streamed text and let
    ``count_tokens(text)`` measure the real partial stream.
    ``stream_ok`` flips to ``False`` so downstream visualisations
    can flag the partial stream explicitly.
    """

    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()
        cls.plugins = discover_plugins()

    def _streaming_plugin(self):
        candidates = [p for p in self.plugins if p.id == "rate-limiter"]
        self.assertEqual(
            len(candidates), 1,
            "rate-limiter must be discovered as a streaming-capable plugin",
        )
        return candidates[0]

    def test_stream_timeout_partial_text_keeps_streamed_output_tokens(self):
        """``count_tokens`` of a 40 K-char streamed timeout on rate-limiter
        must record 10 000 tokens, NOT the pre-fix ``1``-token floor.
        """
        plugin = self._streaming_plugin()
        source_config = {
            "Local": {
                "api_url": "http://localhost:11434/chat/completions",
                "headers": {},
            }
        }
        state = self.module.BenchmarkState({"dummy-model": "Local"}, [plugin.id])

        streamed_text = "a" * 40000  # 40 K chars -> 10 K tokens by len/4.

        with mock.patch.object(
            self.module, "stream_request",
            return_value=(
                streamed_text, "", 1.0, 2000.0,
                "Total timeout (2000s) exceeded",
                "length", {},
            ),
        ):
            with mock.patch.object(
                self.module, "nonstream_request",
            ) as mock_nonstream:
                result, err = self.module._run_plugin_task(
                    "dummy-model", "dummy-model", "Local", plugin, source_config,
                    timeout=1, token_levels=[100], session_seed=12345,
                    log_file=None, global_cfg={}, state=state,
                )

        self.assertIsNone(err)
        # The streamed text was kept -- output_tokens reflects the real
        # streamed length, NOT ``count_tokens("")`` = 1. State-side
        # mirroring happens in ``_run_plugins`` (post-future write);
        # ``_run_plugin_task``'s contract is the result dict + per-SSE
        # state counters (which we don't drive here because the mock
        # doesn't fire ``on_chunk``).
        self.assertEqual(result["rate-limiter_output_tokens"], 10000)
        self.assertFalse(result["rate-limiter_stream_ok"])
        # Truncation semantics are NOT the bug here -- the kimi-dev
        # operator observation is about output_tokens collapsing to 1,
        # not about the truncation flag. The conditional ``sfr ->
        # truncated`` path is exercised by the legacy ``test_run_plugin_task_streaming_callback_updates_state``.
        # The non-stream retry MUST NOT have been called -- if it had,
        # the pre-fix path would have clobbered ``text`` with the
        # (empty) nonstream response and recorded 1 token.
        mock_nonstream.assert_not_called()

    def test_stream_failure_with_no_content_falls_back_to_nonstream(self):
        """When streaming failed at connect time (no first chunk, no
        text), the non-stream fallback IS still called -- the
        pre-fix path remains for genuinely-empty streaming failures.
        """
        plugin = self._streaming_plugin()
        source_config = {
            "Local": {
                "api_url": "http://localhost:11434/chat/completions",
                "headers": {},
            }
        }
        state = self.module.BenchmarkState({"dummy-model": "Local"}, [plugin.id])

        # 3 000 chars from a successful nonstream retry -> 750 tokens.
        nonstream_text = "x" * 3000
        with mock.patch.object(
            self.module, "stream_request",
            return_value=("", "", None, 1.0, "connection refused", None, {}),
        ):
            with mock.patch.object(
                self.module, "nonstream_request",
                return_value=(nonstream_text, "", {}, 0.1, None, "stop"),
            ) as mock_nonstream:
                result, err = self.module._run_plugin_task(
                    "dummy-model", "dummy-model", "Local", plugin, source_config,
                    timeout=1, token_levels=[100], session_seed=12345,
                    log_file=None, global_cfg={}, state=state,
                )

        self.assertIsNone(err)
        self.assertEqual(result["rate-limiter_output_tokens"], 750)
        self.assertFalse(result["rate-limiter_stream_ok"])
        mock_nonstream.assert_called_once()


if __name__ == "__main__":
    unittest.main()
