"""Tests for agent support in the benchmark config and runtime."""
import json
import os
import tempfile
import unittest
from unittest import mock

from benchmark_core import resolve_targets
from plugins import discover_plugins
from tests.utils import load_benchmark_module


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
            return {
                f"{plugin.id}_score": 5.0,
                f"{plugin.id}_response_time": 1.0,
                f"{plugin.id}_output_tokens": 100,
                f"{plugin.id}_tps": 50.0,
                f"{plugin.id}_stream_ok": True,
            }, None

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
            return {
                f"{plugin.id}_score": 5.0,
                f"{plugin.id}_response_time": 1.0,
                f"{plugin.id}_output_tokens": 100,
                f"{plugin.id}_tps": 50.0,
                f"{plugin.id}_stream_ok": True,
            }, None

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
                           drop_params=None, stop_event=None, system_prompt=None):
            captured["body"] = {
                "model": model,
                "messages": [],
            }
            if system_prompt:
                captured["body"]["messages"].append({"role": "system", "content": system_prompt})
            captured["body"]["messages"].append({"role": "user", "content": prompt})
            return "ok", {}, 0.1, None, "stop"

        with mock.patch.object(module, "nonstream_request", side_effect=fake_nonstream):
            with mock.patch.object(module, "stream_request", return_value=("", None, 0, "no tokens", None, {})):
                module._run_plugin_task(
                    "my-agent", "underlying-model", "Local", plugins[0], source_config,
                    timeout=1, token_levels=[100], session_seed=12345,
                    log_file=None, global_cfg={},
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
