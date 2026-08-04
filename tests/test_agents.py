"""Tests for agent support in the benchmark config and runtime."""
import json
import os
import tempfile
import unittest
from unittest import mock

from benchmark_core import resolve_targets
from plugins import discover_plugins
from tests.utils import load_benchmark_module
from benchmark_http import NonStreamResult, StreamResult
from benchmark_plugin import PluginTaskResult


class TestResolveTargets(unittest.TestCase):
    def test_resolve_targets_for_models(self):
        cfg = {
            "models": {
                "model-a": "Source1",
                "model-b": {"source": "Source2", "drop_params": ["seed"]},
            }
        }
        targets = resolve_targets(cfg)
        self.assertEqual(targets["model-a"]["source"], "Source1")
        self.assertEqual(targets["model-a"]["api_model"], "model-a")
        self.assertEqual(targets["model-a"]["is_agent"], False)
        self.assertEqual(targets["model-a"]["system_prompt"], None)
        self.assertEqual(targets["model-b"]["source"], "Source2")
        self.assertEqual(targets["model-b"]["drop_params"], ["seed"])

    def test_resolve_targets_for_agents(self):
        cfg = {
            "agents": {
                "agent-a": {
                    "model": "gpt-4",
                    "source": "OpenAI",
                    "system_prompt": "You are a coder.",
                }
            }
        }
        targets = resolve_targets(cfg)
        self.assertEqual(targets["agent-a"]["source"], "OpenAI")
        self.assertEqual(targets["agent-a"]["api_model"], "gpt-4")
        self.assertEqual(targets["agent-a"]["is_agent"], True)
        self.assertEqual(targets["agent-a"]["system_prompt"], "You are a coder.")

    def test_resolve_targets_merges_models_and_agents(self):
        cfg = {
            "models": {"model-a": "Source1"},
            "agents": {
                "agent-a": {
                    "model": "gpt-4",
                    "source": "Source1",
                    "system_prompt": "be helpful",
                }
            },
        }
        targets = resolve_targets(cfg)
        self.assertIn("model-a", targets)
        self.assertIn("agent-a", targets)
        self.assertEqual(targets["agent-a"]["is_agent"], True)
        self.assertEqual(targets["model-a"]["is_agent"], False)

    def test_resolve_targets_agent_requires_model(self):
        cfg = {
            "agents": {
                "agent-a": {
                    "source": "OpenAI",
                    "system_prompt": "You are a coder.",
                }
            }
        }
        with self.assertRaises(ValueError):
            resolve_targets(cfg)

    def test_resolve_targets_agent_requires_system_prompt(self):
        cfg = {
            "agents": {
                "agent-a": {
                    "model": "gpt-4",
                    "source": "OpenAI",
                }
            }
        }
        with self.assertRaises(ValueError):
            resolve_targets(cfg)

    def test_resolve_targets_per_model_token_levels(self):
        """Per-target token_levels resolve from (1) the model dict entry,
        (2) the top-level model_token_levels map keyed by target name, and
        (3) the map keyed by "{source}/{api_model}" — in that precedence."""
        cfg = {
            "models": {
                "inline-model": {"source": "S1", "token_levels": [32768]},
                "by-name": "S2",
                "by-source-model": "S1",
                "plain": "S2",
            },
            "model_token_levels": {
                "by-name": [65536],
                "S1/by-source-model": [4096],
            },
        }
        targets = resolve_targets(cfg)
        self.assertEqual(targets["inline-model"]["token_levels"], [32768])
        self.assertEqual(targets["by-name"]["token_levels"], [65536])
        self.assertEqual(targets["by-source-model"]["token_levels"], [4096])
        self.assertIsNone(targets["plain"]["token_levels"])

    def test_resolve_targets_agent_token_levels(self):
        """Agents accept the same per-target token_levels override."""
        cfg = {
            "agents": {
                "agent-a": {
                    "model": "gpt-4",
                    "source": "OpenAI",
                    "system_prompt": "You are a coder.",
                    "token_levels": [16384],
                }
            }
        }
        targets = resolve_targets(cfg)
        self.assertEqual(targets["agent-a"]["token_levels"], [16384])

    def test_resolve_targets_token_levels_normalizes_and_rejects_garbage(self):
        """A scalar int is coerced to a one-element list; a string or empty
        list is treated as unset rather than crashing or splintering into
        per-character levels."""
        cfg = {
            "models": {
                "scalar-model": {"source": "S1", "token_levels": 32768},
                "bad-model": {"source": "S1", "token_levels": "32768"},
                "empty-model": {"source": "S1", "token_levels": []},
                "bool-model": {"source": "S1", "token_levels": True},
            }
        }
        targets = resolve_targets(cfg)
        self.assertEqual(targets["scalar-model"]["token_levels"], [32768])
        self.assertIsNone(targets["bad-model"]["token_levels"])
        self.assertIsNone(targets["empty-model"]["token_levels"])
        self.assertIsNone(targets["bool-model"]["token_levels"])


class TestModelPreload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()

    def test_resolve_preload_timeout_uses_configured_positive_value(self):
        self.assertEqual(
            self.module.resolve_preload_timeout(
                {"Local": {"preload_timeout": 17}}, "Local"
            ),
            17,
        )

    def test_resolve_preload_timeout_defaults_for_invalid_values(self):
        for value in (None, 0, -1, "not-a-number"):
            with self.subTest(value=value):
                self.assertEqual(
                    self.module.resolve_preload_timeout(
                        {"Local": {"preload_timeout": value}}, "Local"
                    ),
                    300,
                )

    def test_preload_model_uses_direct_nonstream_probe_without_429_retries(self):
        response = NonStreamResult("OK", "", {}, 1.25, None, "stop")
        with mock.patch.object(self.module, "nonstream_request", return_value=response) as request:
            result = self.module.preload_model(
                {"Local": {"api_url": "http://localhost/chat", "max_429_retries": 4}},
                "Local",
                "model-a",
                timeout=23,
                session_seed=9,
                drop_params=["seed"],
            )

        self.assertTrue(result.success)
        self.assertEqual(result.text, "OK")
        self.assertIsNone(result.error)
        request.assert_called_once()
        args, kwargs = request.call_args
        self.assertEqual(
            args[1:6],
            (23, "model-a", "Local", "Reply with the single word OK.",
             self.module.PRELOAD_MAX_TOKENS),
        )
        self.assertEqual(kwargs["session_seed"], 9)
        self.assertEqual(kwargs["drop_params"], ["seed"])
        # The helper copies the source config so the caller's retry policy is
        # not mutated while the one-shot probe disables 429 retries.
        self.assertEqual(kwargs["stop_event"], None)
        self.assertEqual(request.call_args.args[0]["Local"]["max_429_retries"], 0)

    def test_preload_model_accepts_reasoning_only_response(self):
        """A thinking model that burns the probe budget on reasoning_content
        (empty content, non-empty think_text, finish_reason="length") still
        proves it is warm and must count as preloaded, not as an
        ``empty preload response`` failure. Regression for the 2026-08-02
        run where 83/122 probes were thinking-truncation and the affected
        models were skipped for the entire benchmark."""
        response = NonStreamResult(
            "", "\nOkay, so I need to figure out how to respond",
            {}, 1.25, None, "length",
        )
        with mock.patch.object(self.module, "nonstream_request", return_value=response):
            result = self.module.preload_model(
                {"Local": {"api_url": "http://localhost/chat"}},
                "Local", "model-a", timeout=23,
            )
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_preload_model_rejects_empty_response(self):
        response = NonStreamResult("", "", {}, 0.2, None, "stop")
        with mock.patch.object(self.module, "nonstream_request", return_value=response):
            result = self.module.preload_model(
                {"Local": {"api_url": "http://localhost/chat"}},
                "Local", "model-a", timeout=3,
            )
        self.assertFalse(result.success)
        self.assertEqual(result.error, "empty preload response")

    def test_preload_model_error_still_fails_with_reasoning(self):
        """A transport error must fail the probe even when the response
        carries reasoning content -- the ``empty preload response`` gate
        only fires when there is no error, so this pins that precedence
        so a future refactor cannot let errored reasoning-only responses
        slip through as "warm"."""
        response = NonStreamResult(
            "", "\nSome reasoning despite the error",
            {}, 0.3, "connection reset", "length",
        )
        with mock.patch.object(self.module, "nonstream_request", return_value=response):
            result = self.module.preload_model(
                {"Local": {"api_url": "http://localhost/chat"}},
                "Local", "model-a", timeout=3,
            )
        self.assertFalse(result.success)
        self.assertEqual(result.error, "connection reset")

    def test_preload_model_accepts_thinking_with_escalated_budget(self):
        """The probe must send a token budget large enough for a thinking
        model to emit at least one content token after its reasoning
        preamble -- the old hardcoded 16 was fully consumed by
        ``reasoning_content``. Assert the constant is well above 16 so a
        regression to a tiny budget fails loudly."""
        self.assertGreaterEqual(self.module.PRELOAD_MAX_TOKENS, 64)
        response = NonStreamResult("OK", "some reasoning", {}, 1.0, None, "stop")
        with mock.patch.object(self.module, "nonstream_request", return_value=response) as request:
            result = self.module.preload_model(
                {"Local": {"api_url": "http://localhost/chat"}},
                "Local", "model-a", timeout=10,
            )
        self.assertTrue(result.success)
        args, _ = request.call_args
        self.assertGreaterEqual(args[5], 64)

    def test_save_state_omits_session_only_preload_fields(self):
        state = self.module.BenchmarkState({"model-a": "Local"}, ["rate-limiter"])
        state.update(
            "model-a", preloading=True, preload_start_ts=12.0,
            preload_status="running", preload_time=3.0,
            preload_error=None,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
        self.assertNotIn("preloading", saved["model_info"]["model-a"])
        self.assertNotIn("preload_start_ts", saved["model_info"]["model-a"])
        self.assertNotIn("preload_status", saved["model_info"]["model-a"])


class TestAgentMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()
        cls.plugins = discover_plugins()

    def test_run_model_saves_agent_metadata(self):
        plugins = [p for p in self.plugins if p.id in ("rate-limiter", "moe-dense")]
        targets = {
            "my-agent": {
                "source": "Local",
                "api_model": "underlying-model",
                "system_prompt": "You are a coding agent.",
                "is_agent": True,
                "drop_params": [],
                "plugins_blacklist": [],
            }
        }
        state = self.module.BenchmarkState(targets, [p.id for p in plugins])
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}, "plugin_thread_limit": 1}}

        def fake_run_plugin_task(target_name, api_model, source, plugin, *args, **kwargs):
            return PluginTaskResult({
                f"{plugin.id}_score": 5.0,
                f"{plugin.id}_response_time": 1.0,
                f"{plugin.id}_output_tokens": 100,
                f"{plugin.id}_tps": 50.0,
                f"{plugin.id}_stream_ok": True,
            }, None)

        with mock.patch.object(self.module, "_run_plugin_task", side_effect=fake_run_plugin_task):
            self.module.run_model(
                "my-agent", "Local", state, plugins, source_config,
                timeout=1, token_levels=[100], output_dir="/tmp/benchmark-test",
                session_seed=12345, global_cfg={},
                api_model="underlying-model",
                system_prompt="You are a coding agent.",
                is_agent=True,
            )

        snap = state.snapshot()["my-agent"]
        self.assertEqual(snap["status"], "completed")

        result = state.latest_results()[0]
        self.assertEqual(result["model"], "my-agent")
        self.assertEqual(result["api_model"], "underlying-model")
        self.assertEqual(result["is_agent"], True)
        self.assertEqual(result["system_prompt"], "You are a coding agent.")

    def test_run_model_saves_model_metadata_for_plain_models(self):
        plugins = [p for p in self.plugins if p.id in ("rate-limiter",)]
        models = {"dummy-model": "Local"}
        state = self.module.BenchmarkState(models, [p.id for p in plugins])
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}, "plugin_thread_limit": 1}}

        def fake_run_plugin_task(target_name, api_model, source, plugin, *args, **kwargs):
            return PluginTaskResult({
                f"{plugin.id}_score": 5.0,
                f"{plugin.id}_response_time": 1.0,
                f"{plugin.id}_output_tokens": 100,
                f"{plugin.id}_tps": 50.0,
                f"{plugin.id}_stream_ok": True,
            }, None)

        with mock.patch.object(self.module, "_run_plugin_task", side_effect=fake_run_plugin_task):
            self.module.run_model(
                "dummy-model", "Local", state, plugins, source_config,
                timeout=1, token_levels=[100], output_dir="/tmp/benchmark-test",
                session_seed=0, global_cfg={},
            )

        result = state.latest_results()[0]
        self.assertEqual(result["model"], "dummy-model")
        self.assertEqual(result["api_model"], "dummy-model")
        self.assertEqual(result["is_agent"], False)
        self.assertIsNone(result["system_prompt"])


class TestAgentHTTPRequest(unittest.TestCase):
    def test_run_plugin_task_sends_system_prompt(self):
        module = load_benchmark_module()
        plugins = [p for p in discover_plugins() if p.id == "rate-limiter"]
        source_config = {"Local": {"api_url": "http://localhost:11434/chat/completions", "headers": {}}}
        captured = {}

        def fake_nonstream(source_config, timeout, model, source, prompt, max_tokens=2048,
                           log_path=None, log_label=None, session_seed=0, temperature=None,
                           drop_params=None, stop_event=None, system_prompt=None,
                           pid=None, on_retry=None):
            captured["body"] = {
                "model": model,
                "messages": [],
            }
            if system_prompt:
                captured["body"]["messages"].append({"role": "system", "content": system_prompt})
            captured["body"]["messages"].append({"role": "user", "content": prompt})
            return NonStreamResult("ok", "", {}, 0.1, None, "stop")

        state = module.BenchmarkState({"my-agent": "Local"}, ["rate-limiter"])
        with mock.patch.object(module, "nonstream_request", side_effect=fake_nonstream):
            with mock.patch.object(module, "stream_request", return_value=StreamResult("", "", None, 0, "no tokens", None, {})):
                module._run_plugin_task(
                    "my-agent", "underlying-model", "Local", plugins[0], source_config,
                    timeout=1, token_levels=[100], session_seed=12345,
                    log_file=None, global_cfg={}, state=state,
                    system_prompt="You are a coding agent.",
                )

        self.assertIn("body", captured)
        messages = captured["body"]["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "You are a coding agent."})
        self.assertEqual(messages[1], {"role": "user", "content": plugins[0].get_prompt()})


class TestAgentStatePersistence(unittest.TestCase):
    def test_agent_metadata_round_trips_through_state(self):
        module = load_benchmark_module()
        plugins = [p for p in discover_plugins() if p.id == "rate-limiter"]
        targets = {
            "my-agent": {
                "source": "Local",
                "api_model": "underlying-model",
                "system_prompt": "You are a coding agent.",
                "is_agent": True,
                "drop_params": [],
                "plugins_blacklist": [],
            }
        }
        state = module.BenchmarkState(targets, [p.id for p in plugins], session_seed=999)
        state.update("my-agent", status="completed")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path, plugin_versions={"rate-limiter": "1.0.0"})
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["model_info"]["my-agent"]["api_model"], "underlying-model")
            self.assertEqual(data["model_info"]["my-agent"]["system_prompt"], "You are a coding agent.")
            self.assertEqual(data["model_info"]["my-agent"]["is_agent"], True)


if __name__ == "__main__":
    unittest.main()
