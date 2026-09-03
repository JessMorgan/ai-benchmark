"""Focused tests for the optional OpenCode execution runner."""
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

from benchmark.core import run_model
from benchmark.opencode import (
    OPENCODE_BINARY,
    OPENCODE_INSTALL_SUBDIR,
    OPENCODE_NEUTRAL_AGENT_PERMISSION,
    OPENCODE_NEUTRAL_AGENT_PROMPT,
    OPENCODE_NO_OUTPUT_GRACE,
    OPENCODE_PURE_FLAG,
    OPENCODE_RUN_FORMAT,
    OPENCODE_THINKING_FLAG,
    OpenCodeExtract,
    OpenCodeProcessResult,
    _extract_final_text,
    _local_binary_path,
    _model_context_limit,
    _platform_asset_name,
    _StreamGuard,
    generate_config,
    install_opencode,
    opencode_model_name,
    resolve_opencode_binary,
    resolve_opencode_timeout,
    run_process,
    slugify_source,
    validate_cli,
)
from benchmark.state import BenchmarkState
from plugins import discover_plugins


class TestOpenCodeMapping(unittest.TestCase):
    def test_cli_imports_model_mapping_helper(self):
        spec = importlib.util.spec_from_file_location("ai_benchmark_cli_test", "benchmark/cli.py")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertIs(module.opencode_model_name, opencode_model_name)
        self.assertIs(module.resolve_opencode_binary, resolve_opencode_binary)

    def test_cli_accepts_no_install_opencode_flag(self):
        """--no-install-opencode must parse (and be ignorable with
        --dump-default-config, which exits before any runner preflight)."""
        result = subprocess.run(
            [sys.executable, "ai-benchmark.py",
             "--no-install-opencode", "--dump-default-config"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("output_dir", result.stdout)

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

    def test_per_source_opencode_timeout_defaults_and_overrides(self):
        sources = {
            "Default Source": {"api_url": "https://example.test/chat/completions"},
            "Slow Source": {
                "api_url": "https://slow.example.test/chat/completions",
                "opencode_timeout": 900,
            },
            "Disabled Source": {
                "api_url": "https://disabled.example.test/chat/completions",
                "opencode_timeout": 0,
            },
            "Invalid Source": {
                "api_url": "https://invalid.example.test/chat/completions",
                "opencode_timeout": -1,
            },
        }
        self.assertEqual(OPENCODE_NO_OUTPUT_GRACE, 300.0)
        self.assertEqual(resolve_opencode_timeout(sources, "Default Source"),
                         OPENCODE_NO_OUTPUT_GRACE)
        self.assertEqual(resolve_opencode_timeout(sources, "Slow Source"), 900)
        self.assertEqual(resolve_opencode_timeout(sources, "Disabled Source"), 0)
        self.assertEqual(resolve_opencode_timeout(sources, "Invalid Source"),
                         OPENCODE_NO_OUTPUT_GRACE)
        self.assertEqual(resolve_opencode_timeout(
            {"Boolean Source": {"opencode_timeout": True}}, "Boolean Source"),
            OPENCODE_NO_OUTPUT_GRACE)
        self.assertEqual(resolve_opencode_timeout(sources, "Missing Source"),
                         OPENCODE_NO_OUTPUT_GRACE)

    def test_config_merges_models_and_retains_exact_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "opencode.generated.json")
            generated = generate_config(self.sources, self.targets, path,
                                        timeout=30, max_tokens=100)
            with open(path, encoding="utf-8") as handle:
                on_disk = json.load(handle)

            provider = on_disk["provider"]["local-server"]
            self.assertEqual(provider["options"]["baseURL"], "https://example.test/v1")
            self.assertEqual(provider["options"]["apiKey"], "secret-value")
            self.assertEqual(provider["options"]["headers"]["X-Tenant"], "benchmark")
            # The source's OpenCode inactivity timeout is consumed by the
            # subprocess runner rather than projected into the provider request
            # timeout, which remains the benchmark-wide request setting.
            self.assertNotIn("opencode_timeout", provider["options"])
            self.assertEqual(set(provider["models"]), {"model-a", "model-b"})
            self.assertEqual(on_disk, generated["config"])
            self.assertEqual(generated["mappings"]["local-server/model-b"], ["agent-a"])
            self.assertEqual(generated["agent_ids"]["agent-a"], "benchmark-agent-a")
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)

    def test_model_limit_contains_context_and_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "opencode.generated.json")
            generate_config(self.sources, self.targets, path, max_tokens=100)
            with open(path, encoding="utf-8") as handle:
                on_disk = json.load(handle)
            limits = {
                name: model["limit"]
                for provider in on_disk["provider"].values()
                for name, model in provider["models"].items()
            }
            self.assertEqual(limits["model-a"], {"context": 131072, "output": 100})

    def test_model_limit_uses_per_target_max_tokens(self):
        """A target's resolved ``max_tokens`` beats the global budget for its
        OpenCode output budget (thinking-heavy models get a bigger cap)."""
        targets = {
            "thinker": {
                "source": "Local Server",
                "api_model": "thinker-model",
                "is_agent": False,
                "max_tokens": 32768,
            },
            "plain": {
                "source": "Local Server",
                "api_model": "plain-model",
                "is_agent": False,
                "max_tokens": None,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "opencode.generated.json")
            generate_config(self.sources, targets, path, max_tokens=100)
            with open(path, encoding="utf-8") as handle:
                on_disk = json.load(handle)
            limits = {
                name: model["limit"]["output"]
                for provider in on_disk["provider"].values()
                for name, model in provider["models"].items()
            }
            self.assertEqual(limits["thinker-model"], 32768)
            self.assertEqual(limits["plain-model"], 100)

    def test_model_context_limit_infers_suffix(self):
        self.assertEqual(_model_context_limit("qwen3.6:27b-128k"), 131072)
        self.assertEqual(_model_context_limit("nemotron-3-nano:30b-1m"), 1048576)
        self.assertEqual(_model_context_limit("big-pickle"), 131072)

    def test_mapping_collision_is_rejected_before_write(self):
        targets = {
            "one": {"source": "Same Source", "api_model": "model"},
            "two": {"source": "Same Source", "api_model": "model"},
        }
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            self.assertRaisesRegex(ValueError, "collision"),
        ):
            generate_config(
                {"Same Source": {"api_url": "http://localhost/v1"}},
                targets,
                os.path.join(tmpdir, "generated.json"),
            )

    def test_plain_model_target_registers_neutral_agent(self):
        """Non-agent targets must get a registered agent whose prompt has no
        conciseness instruction and whose permission denies every tool, so
        OpenCode never falls back to its default Build agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "opencode.generated.json")
            generated = generate_config(self.sources, self.targets, path)
            with open(path, encoding="utf-8") as handle:
                on_disk = json.load(handle)

        agents = on_disk["agent"]
        neutral = agents["benchmark-model-a"]
        self.assertEqual(neutral["mode"], "primary")
        self.assertEqual(neutral["prompt"], OPENCODE_NEUTRAL_AGENT_PROMPT)
        self.assertEqual(neutral["permission"], OPENCODE_NEUTRAL_AGENT_PERMISSION)
        self.assertEqual(neutral["model"], "local-server/model-a")
        # The neutral prompt must not contain the default agent's conciseness
        # instruction nor mention any tool.
        self.assertNotIn("concisely", OPENCODE_NEUTRAL_AGENT_PROMPT)
        self.assertNotIn("fewer than 4 lines", OPENCODE_NEUTRAL_AGENT_PROMPT)
        # No tool *definitions* or tool-family names in the neutral prompt
        # (webfetch/todowrite/bash are distinctive; plain words like "task"
        # appear naturally in "written benchmark task").
        for tool in ("webfetch", "todowrite", "bash", "edit", "websearch",
                     "tool_use", "<tool", "sequentialthinking"):
            self.assertNotIn(tool, OPENCODE_NEUTRAL_AGENT_PROMPT)
        # Every tool family is denied -> no tool definitions reach the model.
        for key, action in OPENCODE_NEUTRAL_AGENT_PERMISSION.items():
            self.assertEqual(action, "deny", key)
        # Both targets are registered, so every invocation passes --agent.
        self.assertEqual(generated["agent_ids"]["model-a"], "benchmark-model-a")
        self.assertEqual(generated["agent_ids"]["agent-a"], "benchmark-agent-a")
        self.assertEqual(agents["benchmark-agent-a"]["prompt"], "You are a coding agent.")

    def test_agent_persona_keeps_custom_prompt_and_edit_bash_deny(self):
        """Agent persona targets keep their explicit system prompt; only
        edit/bash are denied so personas can still use read/search tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "opencode.generated.json")
            generate_config(self.sources, self.targets, path)
            with open(path, encoding="utf-8") as handle:
                on_disk = json.load(handle)
        persona = on_disk["agent"]["benchmark-agent-a"]
        self.assertEqual(persona["prompt"], "You are a coding agent.")
        self.assertEqual(persona["permission"], {"edit": "deny", "bash": "deny"})

    def test_agent_id_collision_is_rejected(self):
        """Targets whose names collide after _agent_id slugging (e.g. ``foo:3b``
        and ``foo-3b``) must be rejected instead of silently overwriting."""
        targets = {
            "model:one": {"source": "Local Server", "api_model": "model-a"},
            "model-one": {"source": "Local Server", "api_model": "model-b"},
        }
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            self.assertRaisesRegex(ValueError, "agent id collision"),
        ):
            generate_config(
                self.sources, targets,
                os.path.join(tmpdir, "generated.json"),
            )


def _ndjson_event(event_type, **extra):
    payload = {"type": event_type, "timestamp": 1, "sessionID": "s1"}
    payload.update(extra)
    return json.dumps(payload).encode()


def _text_event(text):
    return _ndjson_event("text", part={
        "type": "text", "text": text, "time": {"end": 1},
    })


class _FakeProcess:
    """A process that has already exited (poll() != None) with full output."""
    def __init__(self, returncode=0, stdout=b"", stderr=b"diagnostic\n"):
        self.returncode = returncode
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.pid = 1234

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        return self.stdout.getvalue(), self.stderr.getvalue()

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class _StallingFakeProcess(_FakeProcess):
    """A process that never exits on its own (poll() -> None until killed)."""
    def __init__(self, stdout=b"", stderr=b""):
        super().__init__(returncode=None, stdout=stdout, stderr=stderr)


class TestOpenCodeExtraction(unittest.TestCase):
    def test_extract_final_text_joins_text_events_in_order(self):
        stream = b"\n".join([
            _ndjson_event("step_start", part={"type": "step-start"}),
            _text_event("part one"),
            _ndjson_event("tool_use", part={"type": "tool"}),
            _text_event("part two"),
        ])
        extracted = _extract_final_text(stream)
        self.assertEqual(extracted, OpenCodeExtract("part one\npart two", "", None))
        self.assertEqual(extracted.text, "part one\npart two")
        self.assertEqual(extracted.think_text, "")
        self.assertIsNone(extracted.error)

    def test_extract_final_text_joins_reasoning_events_into_think_text(self):
        stream = b"\n".join([
            _ndjson_event("reasoning", part={
                "type": "reasoning", "text": "step one reasoning",
                "time": {"end": 1},
            }),
            _text_event("final answer"),
            _ndjson_event("reasoning", part={
                "type": "reasoning", "text": "step two reasoning",
                "time": {"end": 1},
            }),
        ])
        extracted = _extract_final_text(stream)
        self.assertEqual(extracted.text, "final answer")
        self.assertEqual(extracted.think_text, "step one reasoning\nstep two reasoning")
        self.assertIsNone(extracted.error)

    def test_extract_final_text_ignores_empty_reasoning_parts(self):
        stream = _ndjson_event("reasoning", part={"type": "reasoning", "text": "", "time": {}})
        extracted = _extract_final_text(stream)
        self.assertEqual(extracted.text, "")
        self.assertEqual(extracted.think_text, "")
        self.assertIsNone(extracted.error)

    def test_extract_final_text_surfaces_session_error(self):
        stream = b"\n".join([
            _text_event("partial"),
            _ndjson_event("error", error={"message": "rate limited"}),
        ])
        extracted = _extract_final_text(stream)
        self.assertEqual(extracted.text, "partial")
        self.assertEqual(extracted.error, "rate limited")

    def test_extract_final_text_falls_back_to_raw_stdout(self):
        extracted = _extract_final_text(b"plain text\nnot json")
        self.assertEqual(extracted.text, "plain text\nnot json")
        self.assertEqual(extracted.think_text, "")
        self.assertIsNone(extracted.error)

    def test_extract_final_text_ignores_incomplete_parts(self):
        stream = _ndjson_event("text", part={"type": "text", "text": "", "time": {}})
        extracted = _extract_final_text(stream)
        self.assertEqual(extracted.text, "")
        self.assertEqual(extracted.think_text, "")


class TestOpenCodeProcess(unittest.TestCase):
    def test_invocation_uses_argument_list_config_env_and_separate_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            open(config_path, "w", encoding="utf-8").close()
            stdout = b"\n".join([_text_event("final answer")])
            fake = _FakeProcess(stdout=stdout)
            with mock.patch("benchmark.opencode.subprocess.Popen", return_value=fake) as popen:
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
                    debug_logs=True,
                )

            command = popen.call_args.args[0]
            self.assertEqual(command[:9], [
                "opencode-test", "run", OPENCODE_PURE_FLAG,
                "--model", "local-server/model-a", "--format",
                OPENCODE_RUN_FORMAT, OPENCODE_THINKING_FLAG, "--agent",
            ])
            self.assertEqual(command[-2:], ["benchmark-agent-a", "line one\nline two"])
            env = popen.call_args.kwargs["env"]
            self.assertEqual(env["OPENCODE_CONFIG"], os.path.abspath(config_path))
            self.assertEqual(result.text, "final answer")
            self.assertEqual(result.stderr, "diagnostic\n")
            self.assertIsNone(result.error)
            self.assertTrue(os.path.exists(os.path.join(
                tmpdir, "logs", "agent-a", "rate-limiter.stdout.txt.gz")))
            self.assertTrue(os.path.exists(os.path.join(
                tmpdir, "logs", "agent-a", "rate-limiter.stderr.txt.gz")))

    def test_nonzero_exit_is_reported(self):
        fake = _FakeProcess(returncode=7, stderr=b"bad config")
        with mock.patch("benchmark.opencode.subprocess.Popen", return_value=fake):
            result = run_process("prompt", config_path="/tmp/config.json",
                                 model="provider/model", timeout=5,
                                 binary="opencode-test")
        self.assertEqual(result.error, "OpenCode exited with status 7")

    def test_session_error_is_reported(self):
        stream = _ndjson_event("error", error={"message": "provider down"})
        fake = _FakeProcess(stdout=stream)
        with mock.patch("benchmark.opencode.subprocess.Popen", return_value=fake):
            result = run_process("prompt", config_path="/tmp/config.json",
                                 model="provider/model", timeout=5,
                                 binary="opencode-test")
        self.assertEqual(result.error, "OpenCode session error: provider down")

    def test_help_advertising_only_default_format_is_rejected(self):
        help_text = (
            "Options:\n"
            "  --model  model to use  [string]\n"
            "  --format format  [string] [choices: \"default\"] [default: \"default\"]\n"
            "  --agent  agent to use  [string]\n"
            "  --pure  disable external plugins  [boolean]\n"
            "  --thinking  show thinking blocks  [boolean]\n"
        )
        probe = mock.Mock()
        probe.returncode = 0
        probe.stdout = help_text
        probe.stderr = ""
        with (
            mock.patch("benchmark.opencode.subprocess.run", return_value=probe),
            self.assertRaisesRegex(RuntimeError, "'json' run format"),
        ):
            validate_cli("opencode-test")

    def test_help_without_pure_is_rejected(self):
        help_text = (
            "Options:\n"
            "  --model  model to use  [string]\n"
            "  --format format  [string] [choices: \"default\", \"json\"] [default: \"default\"]\n"
            "  --agent  agent to use  [string]\n"
        )
        probe = mock.Mock()
        probe.returncode = 0
        probe.stdout = help_text
        probe.stderr = ""
        with (
            mock.patch("benchmark.opencode.subprocess.run", return_value=probe),
            self.assertRaisesRegex(RuntimeError, "--pure"),
        ):
            validate_cli("opencode-test")

    def test_help_without_thinking_is_rejected(self):
        help_text = (
            "Options:\n"
            "  --model  model to use  [string]\n"
            "  --format format  [string] [choices: \"default\", \"json\"] [default: \"default\"]\n"
            "  --agent  agent to use  [string]\n"
            "  --pure  disable external plugins  [boolean]\n"
        )
        probe = mock.Mock()
        probe.returncode = 0
        probe.stdout = help_text
        probe.stderr = ""
        with (
            mock.patch("benchmark.opencode.subprocess.run", return_value=probe),
            self.assertRaisesRegex(RuntimeError, "--thinking"),
        ):
            validate_cli("opencode-test")

    def test_help_advertising_json_format_passes(self):
        help_text = (
            "Options:\n"
            "  --model  model to use  [string]\n"
            "  --format format  [string] [choices: \"default\", \"json\"] [default: \"default\"]\n"
            "  --agent  agent to use  [string]\n"
            "  --pure  disable external plugins  [boolean]\n"
            "  --thinking  show thinking blocks  [boolean]\n"
        )
        probe = mock.Mock()
        probe.returncode = 0
        probe.stdout = help_text
        probe.stderr = ""
        with mock.patch("benchmark.opencode.subprocess.run", return_value=probe):
            validate_cli("opencode-test")


def _make_archive(tmpdir, name="opencode-linux-x64.tar.gz", binary_name="opencode",
                  contents=b"#!/bin/sh\necho fake-opencode\n"):
    """Build a real tar.gz/zip archive containing a fake opencode binary."""
    archive = os.path.join(tmpdir, name)
    if name.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(binary_name, contents)
    else:
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w:gz") as tar:
            info = tarfile.TarInfo(binary_name)
            info.size = len(contents)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(contents))
        with open(archive, "wb") as handle:
            handle.write(payload.getvalue())
    return archive


class TestOpenCodeAutoInstall(unittest.TestCase):
    def setUp(self):
        patchers = [
            mock.patch("benchmark.opencode.platform.system", return_value="Linux"),
            mock.patch("benchmark.opencode.platform.machine", return_value="x86_64"),
            mock.patch("benchmark.opencode._cpu_has_avx2", return_value=True),
            mock.patch("benchmark.opencode._is_musl_libc", return_value=False),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_platform_asset_linux_x64_avx2(self):
        self.assertEqual(_platform_asset_name(), "opencode-linux-x64.tar.gz")

    def test_platform_asset_linux_baseline_without_avx2(self):
        with mock.patch("benchmark.opencode._cpu_has_avx2", return_value=False):
            self.assertEqual(_platform_asset_name(), "opencode-linux-x64-baseline.tar.gz")

    def test_platform_asset_linux_musl(self):
        with mock.patch("benchmark.opencode._is_musl_libc", return_value=True):
            self.assertEqual(_platform_asset_name(), "opencode-linux-x64-musl.tar.gz")

    def test_platform_asset_arm64(self):
        with mock.patch("benchmark.opencode.platform.machine", return_value="aarch64"):
            self.assertEqual(_platform_asset_name(), "opencode-linux-arm64.tar.gz")

    def test_platform_asset_macos_zip(self):
        with mock.patch("benchmark.opencode.platform.system", return_value="Darwin"), \
             mock.patch("benchmark.opencode._darwin_translated", return_value=False), \
             mock.patch("benchmark.opencode._darwin_avx2", return_value=True):
            self.assertEqual(_platform_asset_name(), "opencode-darwin-x64.zip")

    def test_platform_asset_macos_rosetta_uses_arm64(self):
        with mock.patch("benchmark.opencode.platform.system", return_value="Darwin"), \
             mock.patch("benchmark.opencode._darwin_translated", return_value=True):
            self.assertEqual(_platform_asset_name(), "opencode-darwin-arm64.zip")

    def test_platform_asset_unsupported_os_raises(self):
        with (
            mock.patch("benchmark.opencode.platform.system", return_value="Plan9"),
            self.assertRaisesRegex(RuntimeError, "Unsupported OS"),
        ):
            _platform_asset_name()

    def test_install_opencode_downloads_extracts_and_chmods(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            install_dir = os.path.join(tmpdir, "install", "opencode")
            archive = _make_archive(tmpdir)

            def fake_download(url, dest, *, timeout):
                self.assertIn("releases/latest/download/opencode-linux-x64.tar.gz", url)
                with (
                    open(dest, "wb") as handle,
                    open(archive, "rb") as src,
                ):
                    handle.write(src.read())

            with mock.patch("benchmark.opencode._download_to", side_effect=fake_download) as dl, \
                 mock.patch("benchmark.opencode._latest_opencode_version", return_value="9.9.9"):
                binary = install_opencode(install_dir)

            self.assertTrue(os.path.isfile(binary))
            self.assertTrue(os.access(binary, os.X_OK))
            self.assertEqual(binary, os.path.join(install_dir, "opencode"))
            with open(os.path.join(install_dir, "version.txt"), encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "9.9.9")
            dl.assert_called_once()

    def test_install_opencode_windows_zip_extracts_exe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            install_dir = os.path.join(tmpdir, "install")
            archive = _make_archive(tmpdir, name="opencode-windows-x64.zip",
                                    binary_name="opencode.exe")

            def fake_download(url, dest, *, timeout):
                with (
                    open(dest, "wb") as handle,
                    open(archive, "rb") as src,
                ):
                    handle.write(src.read())

            with mock.patch("benchmark.opencode._platform_asset_name",
                            return_value="opencode-windows-x64.zip"), \
                 mock.patch("benchmark.opencode._download_to", side_effect=fake_download), \
                 mock.patch("benchmark.opencode._latest_opencode_version", return_value=None):
                binary = install_opencode(install_dir)

            self.assertTrue(binary.endswith("opencode.exe"))
            self.assertTrue(os.path.isfile(binary))

    def test_install_opencode_download_failure_raises_actionable_error(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir, mock.patch("benchmark.opencode._download_to", side_effect=OSError("network down")), mock.patch("benchmark.opencode._latest_opencode_version", return_value="1.0.0"),
            self.assertRaisesRegex(RuntimeError, "Could not auto-install OpenCode"),
        ):
            install_opencode(os.path.join(tmpdir, "install"))


class TestOpenCodeBinaryResolution(unittest.TestCase):
    def test_path_binary_that_validates_is_used(self):
        with mock.patch("benchmark.opencode.shutil.which", return_value="/usr/bin/opencode") as which, \
             mock.patch("benchmark.opencode.validate_cli") as validate:
            resolved = resolve_opencode_binary()
        self.assertEqual(resolved, "/usr/bin/opencode")
        which.assert_called_once_with("opencode")
        validate.assert_called_once_with("/usr/bin/opencode", timeout=10)

    def test_stale_path_binary_falls_back_to_local_install(self):
        """A PATH binary that fails preflight is replaced by a fresh local install."""
        def selective_validate(binary, **kwargs):
            if binary == "/usr/bin/opencode":
                raise RuntimeError("missing --pure")

        with mock.patch("benchmark.opencode.shutil.which", return_value="/usr/bin/opencode"), \
             mock.patch("benchmark.opencode._local_binary_path",
                        return_value=mock.Mock(is_file=lambda: False)), \
             mock.patch("benchmark.opencode.validate_cli", side_effect=selective_validate), \
             mock.patch("benchmark.opencode.install_opencode", return_value="/proj/.tools/opencode/opencode") as install:
            resolved = resolve_opencode_binary()
        self.assertEqual(resolved, "/proj/.tools/opencode/opencode")
        install.assert_called_once()

    def test_reuses_valid_local_copy_when_path_missing(self):
        local = os.path.join(tempfile.gettempdir(), "tools-opencode", "opencode")
        with mock.patch("benchmark.opencode.shutil.which", return_value=None), \
             mock.patch("benchmark.opencode._local_binary_path", return_value=mock.Mock(
                 is_file=lambda: True, __str__=lambda self: local)), \
             mock.patch("benchmark.opencode.validate_cli") as validate:
            resolved = resolve_opencode_binary()
        self.assertEqual(resolved, local)
        validate.assert_called_once_with(local, timeout=10)

    def test_install_disabled_and_missing_raises_actionable_error(self):
        with (
            mock.patch("benchmark.opencode.shutil.which", return_value=None), mock.patch("benchmark.opencode._local_binary_path", return_value=mock.Mock(is_file=lambda: False)),
            self.assertRaisesRegex(RuntimeError, "not found on PATH"),
        ):
            resolve_opencode_binary(allow_install=False)

    def test_install_disabled_and_stale_path_raises(self):
        def bad_validate(binary, **kwargs):
            raise RuntimeError("missing --thinking")

        with (
            mock.patch("benchmark.opencode.shutil.which", return_value="/usr/bin/opencode"), mock.patch("benchmark.opencode._local_binary_path", return_value=mock.Mock(is_file=lambda: False)), mock.patch("benchmark.opencode.validate_cli", side_effect=bad_validate),
            self.assertRaisesRegex(RuntimeError, "incompatible"),
        ):
            resolve_opencode_binary(allow_install=False)

    def test_fresh_install_failing_preflight_raises(self):
        def bad_validate(binary, **kwargs):
            raise RuntimeError("missing --pure")

        with (
            mock.patch("benchmark.opencode.shutil.which", return_value=None), mock.patch("benchmark.opencode._local_binary_path", return_value=mock.Mock(is_file=lambda: False)), mock.patch("benchmark.opencode.install_opencode", return_value="/proj/.tools/opencode/opencode"), mock.patch("benchmark.opencode.validate_cli", side_effect=bad_validate),
            self.assertRaisesRegex(RuntimeError, "failed preflight"),
        ):
            resolve_opencode_binary()

    def test_local_binary_path_defaults_to_project_tools_dir(self):
        project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        expected = os.path.join(project, OPENCODE_INSTALL_SUBDIR, OPENCODE_BINARY)
        self.assertEqual(str(_local_binary_path()), expected)


class TestOpenCodeLoopGuards(unittest.TestCase):
    """Fast-fail and loop guards on the OpenCode subprocess."""

    def _stall(self, stdout=b"", **run_kwargs):
        fake = _StallingFakeProcess(stdout=stdout)
        popen = mock.patch("benchmark.opencode.subprocess.Popen", return_value=fake)
        term = mock.patch("benchmark.opencode._terminate_process")
        popen.start()
        term_mock = term.start()
        self.addCleanup(popen.stop)
        self.addCleanup(term.stop)
        # Callers may supply their own short timeout; default to 30s otherwise.
        run_kwargs.setdefault("timeout", 30)
        result = run_process("prompt", config_path="/tmp/config.json",
                             model="provider/model", **run_kwargs)
        return result, fake, term_mock

    def test_run_model_passes_source_opencode_timeout_to_staleness_guard(self):
        plugin = next(p for p in discover_plugins() if p.id == "rate-limiter")
        target_key = "model-a [opencode]"
        state = BenchmarkState({
            target_key: {
                "source": "Local",
                "api_model": "model-a",
                "runner": "opencode",
            }
        }, [plugin.id])
        source_config = {
            "Local": {"plugin_thread_limit": 1, "opencode_timeout": 777}
        }
        process_result = OpenCodeProcessResult(
            "valid benchmark response", "", 0.2, None, 0)
        with mock.patch("benchmark.core.run_process", return_value=process_result) as run:
            run_model(
                target_key, "Local", state, [plugin], source_config,
                timeout=5, max_tokens=100, output_dir="/tmp/opencode-test",
                global_cfg={}, runner="opencode", api_model="model-a",
                opencode_config_path="/tmp/config.json",
                opencode_model="local/model-a", display_name="model-a",
                config_target_name="model-a",
            )
        self.assertEqual(run.call_args.kwargs["no_output_grace"], 777)

    def test_run_model_uses_300_second_default_for_omitted_source_timeout(self):
        plugin = next(p for p in discover_plugins() if p.id == "rate-limiter")
        target_key = "model-a [opencode]"
        state = BenchmarkState({
            target_key: {
                "source": "Local",
                "api_model": "model-a",
                "runner": "opencode",
            }
        }, [plugin.id])
        process_result = OpenCodeProcessResult(
            "valid benchmark response", "", 0.2, None, 0)
        with mock.patch("benchmark.core.run_process", return_value=process_result) as run:
            run_model(
                target_key, "Local", state, [plugin],
                {"Local": {"plugin_thread_limit": 1}},
                timeout=5, max_tokens=100, output_dir="/tmp/opencode-test",
                global_cfg={}, runner="opencode", api_model="model-a",
                opencode_config_path="/tmp/config.json",
                opencode_model="local/model-a", display_name="model-a",
                config_target_name="model-a",
            )
        self.assertEqual(run.call_args.kwargs["no_output_grace"], 300.0)

    def test_fast_fail_when_no_output_for_grace_period(self):
        """A subprocess that emits nothing for the grace period is killed early."""
        result, _, term = self._stall(stdout=b"", no_output_grace=0.15)
        self.assertIn("produced no output within", result.error)
        self.assertTrue(term.called)

    def test_steps_only_stream_is_killed_when_grace_elapses(self):
        """step_start then silence (mid-stream stall) must also fast-fail."""
        stream = _ndjson_event("step_start", part={"type": "step-start"})
        result, _, _ = self._stall(stdout=stream, no_output_grace=0.15)
        self.assertIn("produced no output within", result.error)

    def test_step_budget_kills_planning_loop(self):
        """A reasoning/tool planning loop (many steps, no final text) dies at the cap."""
        stream = b"\n".join([
            _ndjson_event("step_finish", part={"type": "step-finish"})
            for _ in range(60)
        ])
        result, _, term = self._stall(stdout=stream, step_limit=10)
        self.assertIn("reached 10 agent steps", result.error)
        self.assertTrue(term.called)

    def test_text_repetition_kills_loop(self):
        """The same non-trivial text event repeated past the threshold trips the guard."""
        event = _text_event("Continue if you have next steps, or stop and ask for clarification.")
        result, _, term = self._stall(stdout=b"\n".join([event] * 7), repeat_threshold=5)
        self.assertIn("repeated the same text 5 times", result.error)
        self.assertTrue(term.called)

    def test_repeat_guard_ignores_short_acknowledgements(self):
        """Trivial short repeats below repeat_min_len must not false-positive."""
        result, _, _ = self._stall(
            stdout=b"\n".join([_text_event("Yes")] * 10),
            timeout=0.3, repeat_threshold=3,
        )
        self.assertIn("timed out", result.error)
        self.assertNotIn("loop", result.error)

    def test_guards_can_be_disabled(self):
        """All guards off: a silent process burns the outer timeout instead."""
        result, _, term = self._stall(
            stdout=b"", timeout=0.3, no_output_grace=0, step_limit=0, repeat_threshold=0)
        self.assertIn("timed out after 0.3s", result.error)
        self.assertTrue(term.called)

    def test_healthy_stream_outlives_grace_and_keeps_partial_output(self):
        """Flowing content resets the staleness timer; on outer timeout the
        streamed text is retained in the result."""
        stream = b"\n".join([
            _ndjson_event("step_start", part={"type": "step-start"}),
            _text_event("A real answer with enough text to exceed the repeat-min length."),
            _ndjson_event("step_finish", part={"type": "step-finish"}),
        ])
        result, _, _ = self._stall(stdout=stream, timeout=0.3, no_output_grace=5)
        self.assertIn("timed out", result.error)
        self.assertNotIn("loop", result.error)
        self.assertNotIn("no output", result.error)
        self.assertIn("real answer", result.text)

    def test_guard_parses_events_split_across_chunk_boundaries(self):
        """The incremental parser must handle a line that spans two feeds."""
        guard = _StreamGuard(step_limit=2)
        line = _ndjson_event("step_finish", part={"type": "step-finish"})
        mid = len(line) // 2
        guard.feed(line[:mid])
        guard.feed(line[mid:] + b"\n")
        guard.feed(line + b"\n")
        self.assertEqual(guard.step_count, 2)
        self.assertTrue(guard.steps_exceeded)

    def test_guard_stops_counting_after_trip(self):
        """Once a guard trips, feed() must not keep accumulating counts."""
        guard = _StreamGuard(step_limit=2, repeat_threshold=3, repeat_min_len=5)
        guard.feed(_text_event("aaaaaaaaaa") + b"\n")  # 1 repeat
        guard.feed(_text_event("aaaaaaaaaa") + b"\n")  # 2 repeats
        guard.feed(_text_event("aaaaaaaaaa") + b"\n")  # 3 repeats -> tripped
        self.assertTrue(guard.repeated)
        guard.feed(_text_event("aaaaaaaaaa") + b"\n")  # ignored
        self.assertEqual(guard._text_counts["aaaaaaaaaa"], 3)

    def test_natural_exit_wins_over_deadline_and_guards(self):
        """A process that exits on its own must not be reported as a timeout
        even when the deadline has already passed."""
        stream = b"\n".join([
            _text_event("A complete final answer that finishes the task."),
        ])
        fake = _FakeProcess(returncode=0, stdout=stream)
        with mock.patch("benchmark.opencode.subprocess.Popen", return_value=fake):
            result = run_process(
                "prompt", config_path="/tmp/config.json",
                model="provider/model", timeout=0,  # deadline already passed
                no_output_grace=0, step_limit=0, repeat_threshold=0,
                binary="opencode-test",
            )
        self.assertIsNone(result.error)
        self.assertIn("complete final answer", result.text)


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

        with mock.patch("benchmark.core.run_process", return_value=process_result):
            run_model(
                target_key, "Local", state, [plugin], source_config,
                timeout=5, max_tokens=100, output_dir="/tmp/opencode-test",
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

    def test_opencode_thinking_writes_think_sidecar(self):
        """Reasoning events must flow into ``think.txt`` inside the plugin dir."""
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
            "final answer", "", 0.2, None, 0, think_text="chain of thought")

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch("benchmark.core.run_process", return_value=process_result),
        ):
            run_model(
                target_key, "Local", state, [plugin], source_config,
                timeout=5, max_tokens=100, output_dir=tmpdir,
                global_cfg={}, runner="opencode", api_model="model-a",
                opencode_config_path="/tmp/config.json",
                opencode_model="local/model-a", display_name="model-a",
                config_target_name="model-a", save_responses=True,
            )

            plugin_dir = os.path.join(tmpdir, "responses", "model-a", plugin.id)
            think_path = os.path.join(plugin_dir, "think.txt")
            self.assertTrue(os.path.isfile(think_path), "expected a think.txt sidecar")
            with open(think_path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "chain of thought")
            joined_path = os.path.join(plugin_dir, "response.txt")
            with open(joined_path, encoding="utf-8") as handle:
                self.assertIn("<thinking>\nchain of thought\n</thinking>", handle.read())

    def test_opencode_failure_preserves_captured_thinking(self):
        """A failed OpenCode run must keep reasoning captured pre-failure."""
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
            "partial answer", "", 1800.1, "OpenCode timed out after 1800s", 0,
            think_text="reasoning before timeout",
        )

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch("benchmark.core.run_process", return_value=process_result),
        ):
            run_model(
                target_key, "Local", state, [plugin], source_config,
                timeout=5, max_tokens=100, output_dir=tmpdir,
                global_cfg={}, runner="opencode", api_model="model-a",
                opencode_config_path="/tmp/config.json",
                opencode_model="local/model-a", display_name="model-a",
                config_target_name="model-a", save_responses=True,
            )

            plugin_dir = os.path.join(tmpdir, "responses", "model-a", plugin.id)
            think_path = os.path.join(plugin_dir, "think.txt")
            self.assertTrue(os.path.isfile(think_path), "failed run must keep think.txt")
            with open(think_path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "reasoning before timeout")
            with open(os.path.join(plugin_dir, "meta.json"), encoding="utf-8") as handle:
                meta = json.load(handle)
            self.assertEqual(meta["error"], "OpenCode timed out after 1800s")
            self.assertEqual(meta["think_text"], "reasoning before timeout")

    def test_latest_results_keeps_http_and_opencode_variants(self):
        state = BenchmarkState({"model-a": "Local", "model-a [opencode]": "Local"}, ["p"])
        state.add_result({"model": "model-a", "state_key": "model-a", "runner": "http"})
        state.add_result({"model": "model-a", "state_key": "model-a [opencode]", "runner": "opencode"})
        self.assertEqual({r["runner"] for r in state.latest_results()}, {"http", "opencode"})

    def test_resolved_binary_is_passed_to_run_process(self):
        """The resolved binary path must reach ``run_process`` as ``binary``."""
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

        with mock.patch("benchmark.core.run_process", return_value=process_result) as run:
            run_model(
                target_key, "Local", state, [plugin], source_config,
                timeout=5, max_tokens=100, output_dir="/tmp/opencode-test",
                global_cfg={}, runner="opencode", api_model="model-a",
                opencode_config_path="/tmp/config.json",
                opencode_model="local/model-a",
                opencode_binary="/proj/.tools/opencode/opencode",
                display_name="model-a", config_target_name="model-a",
            )

        self.assertEqual(run.call_args.kwargs["binary"],
                         "/proj/.tools/opencode/opencode")

    def test_run_process_defaults_to_opencode_name_without_binary(self):
        """Without a resolved binary, ``run_process`` receives ``opencode``."""
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

        with mock.patch("benchmark.core.run_process", return_value=process_result) as run:
            run_model(
                target_key, "Local", state, [plugin], source_config,
                timeout=5, max_tokens=100, output_dir="/tmp/opencode-test",
                global_cfg={}, runner="opencode", api_model="model-a",
                opencode_config_path="/tmp/config.json",
                opencode_model="local/model-a",
                display_name="model-a", config_target_name="model-a",
            )

        self.assertEqual(run.call_args.kwargs["binary"], OPENCODE_BINARY)


if __name__ == "__main__":
    unittest.main()
