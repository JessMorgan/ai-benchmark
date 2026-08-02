"""Focused tests for the optional OpenCode execution runner."""
import importlib.util
import json
import os
import stat
import tempfile
import unittest
from unittest import mock

from benchmark_core import run_model
from benchmark_plugin import PluginTaskResult
from benchmark_state import BenchmarkState
from opencode_runner import (
    OpenCodeProcessResult,
    generate_config,
    opencode_model_name,
    run_process,
    slugify_source,
)
from plugins import discover_plugins


class TestOpenCodeMapping(unittest.TestCase):
    def test_cli_imports_model_mapping_helper(self):
        spec = importlib.util.spec_from_file_location("ai_benchmark_cli_test", "ai-benchmark.py")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertIs(module.opencode_model_name, opencode_model_name)

    def test_slugify_and_literal_model_mapping(self):
        self.assertEqual(slugify_source("  Remote/OpenAI 2  "), "remote-openai-2")
        self.assertEqual(opencode_model_name("Local Server 1", "vendor/model-x"),
                         "local-server-1/vendor/model-x")

    def test_empty_mapping_parts_are_rejected(self):
        with self.assertRaises(ValueError):
            opencode_model_name("!!!", "model")
        with self.assertRaises(ValueError):
            opencode_model_name("Source", "")


class TestOpenCodeConfig(unittest.TestCase):
    def setUp(self):
        self.sources = {
            "Local Server": {
                "api_url": "https://example.test/v1/chat/completions",
                "headers": {
                    "Authorization": "Bearer secret-value",
                    "X-Tenant": "benchmark",
                    "Content-Type": "application/json",
                },
            }
        }
        self.targets = {
            "model-a": {
                "source": "Local Server",
                "api_model": "model-a",
                "is_agent": False,
            },
            "agent-a": {
                "source": "Local Server",
                "api_model": "model-b",
                "is_agent": True,
                "system_prompt": "You are a coding agent.",
            },
        }

    def test_config_merges_models_and_retains_exact_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "opencode.generated.json")
            generated = generate_config(self.sources, self.targets, path,
                                        timeout=30, token_levels=[100, 200])
            with open(path, encoding="utf-8") as handle:
                on_disk = json.load(handle)

            provider = on_disk["provider"]["local-server"]
            self.assertEqual(provider["options"]["baseURL"], "https://example.test/v1")
            self.assertEqual(provider["options"]["apiKey"], "secret-value")
            self.assertEqual(provider["options"]["headers"]["X-Tenant"], "benchmark")
            self.assertEqual(set(provider["models"]), {"model-a", "model-b"})
            self.assertEqual(on_disk, generated["config"])
            self.assertEqual(generated["mappings"]["local-server/model-b"], ["agent-a"])
            self.assertEqual(generated["agent_ids"]["agent-a"], "benchmark-agent-a")
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)

    def test_mapping_collision_is_rejected_before_write(self):
        targets = {
            "one": {"source": "Same Source", "api_model": "model"},
            "two": {"source": "Same Source", "api_model": "model"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "collision"):
                generate_config(
                    {"Same Source": {"api_url": "http://localhost/v1"}},
                    targets,
                    os.path.join(tmpdir, "generated.json"),
                )


class _FakeProcess:
    def __init__(self, returncode=0, stdout=b"final answer\n", stderr=b"diagnostic\n"):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.pid = 1234

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        return self._stdout, self._stderr

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class TestOpenCodeProcess(unittest.TestCase):
    def test_invocation_uses_argument_list_config_env_and_separate_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            open(config_path, "w", encoding="utf-8").close()
            fake = _FakeProcess()
            with mock.patch("opencode_runner.subprocess.Popen", return_value=fake) as popen:
                result = run_process(
                    "line one\nline two",
                    config_path=config_path,
                    model="local-server/model-a",
                    timeout=5,
                    binary="opencode-test",
                    agent="benchmark-agent-a",
                    output_dir=tmpdir,
                    target_key="agent-a",
                    plugin_id="rate-limiter",
                )

            command = popen.call_args.args[0]
            self.assertEqual(command[:7], [
                "opencode-test", "run", "--model", "local-server/model-a",
                "--format", "plain", "--agent",
            ])
            self.assertEqual(command[-2:], ["benchmark-agent-a", "line one\nline two"])
            env = popen.call_args.kwargs["env"]
            self.assertEqual(env["OPENCODE_CONFIG"], os.path.abspath(config_path))
            self.assertEqual(result.text, "final answer")
            self.assertEqual(result.stderr, "diagnostic\n")
            self.assertIsNone(result.error)
            self.assertTrue(os.path.exists(os.path.join(
                tmpdir, "logs", "agent-a", "rate-limiter.stdout.txt")))
            self.assertTrue(os.path.exists(os.path.join(
                tmpdir, "logs", "agent-a", "rate-limiter.stderr.txt")))

    def test_nonzero_exit_is_reported(self):
        fake = _FakeProcess(returncode=7, stdout=b"", stderr=b"bad config")
        with mock.patch("opencode_runner.subprocess.Popen", return_value=fake):
            result = run_process("prompt", config_path="/tmp/config.json",
                                 model="provider/model", timeout=5,
                                 binary="opencode-test")
        self.assertEqual(result.error, "OpenCode exited with status 7")


class TestRunnerAwareExecution(unittest.TestCase):
    def test_opencode_result_is_distinct_from_http_result(self):
        plugin = next(p for p in discover_plugins() if p.id == "rate-limiter")
        target_key = "model-a [opencode]"
        state = BenchmarkState({
            target_key: {
                "source": "Local",
                "api_model": "model-a",
                "runner": "opencode",
            }
        }, [plugin.id])
        source_config = {"Local": {"plugin_thread_limit": 1}}
        process_result = OpenCodeProcessResult(
            "This is a valid benchmark response.", "", 0.2, None, 0)

        with mock.patch("benchmark_core.run_process", return_value=process_result):
            run_model(
                target_key, "Local", state, [plugin], source_config,
                timeout=5, token_levels=[100], output_dir="/tmp/opencode-test",
                global_cfg={}, runner="opencode", api_model="model-a",
                opencode_config_path="/tmp/config.json",
                opencode_model="local/model-a", display_name="model-a",
                config_target_name="model-a",
            )

        result = state.latest_results()[0]
        self.assertEqual(result["model"], "model-a")
        self.assertEqual(result["state_key"], target_key)
        self.assertEqual(result["runner"], "opencode")
        self.assertEqual(result["opencode_model"], "local/model-a")

    def test_latest_results_keeps_http_and_opencode_variants(self):
        state = BenchmarkState({"model-a": "Local", "model-a [opencode]": "Local"}, ["p"])
        state.add_result({"model": "model-a", "state_key": "model-a", "runner": "http"})
        state.add_result({"model": "model-a", "state_key": "model-a [opencode]", "runner": "opencode"})
        self.assertEqual({r["runner"] for r in state.latest_results()}, {"http", "opencode"})


if __name__ == "__main__":
    unittest.main()
