"""Coverage-focused tests for benchmark.opencode.

Targets branches not exercised by tests/test_opencode_runner.py: timeout
normalization edge cases, validate_cli failure modes, platform detection
helpers, download/extract helpers, provider options translation, and the
remaining guard/cleanup paths in run_process.
"""
import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from benchmark.opencode import (
    OPENCODE_NEUTRAL_AGENT_PERMISSION,
    OpenCodeExtract,
    OpenCodeProcessResult,
    _StreamGuard,
    _cpu_has_avx2,
    _darwin_avx2,
    _darwin_translated,
    _download_to,
    _extract_binary,
    _extract_final_text,
    _is_musl_libc,
    _latest_opencode_version,
    _local_binary_path,
    _platform_asset_name,
    _provider_options,
    _pump_stream,
    _terminate_process,
    generate_config,
    opencode_version,
    resolve_opencode_binary,
    resolve_opencode_timeout,
    run_process,
    validate_cli,
)


class TestResolveOpencodeTimeout(unittest.TestCase):
    def test_bool_value_falls_back_to_default(self):
        cfg = {"S": {"opencode_timeout": True}}
        self.assertEqual(resolve_opencode_timeout(cfg, "S"), 300.0)

    def test_invalid_value_falls_back_to_default(self):
        cfg = {"S": {"opencode_timeout": "not-a-number"}}
        self.assertEqual(resolve_opencode_timeout(cfg, "S"), 300.0)

    def test_negative_value_falls_back_to_default(self):
        cfg = {"S": {"opencode_timeout": -5}}
        self.assertEqual(resolve_opencode_timeout(cfg, "S"), 300.0)

    def test_zero_disables_guard(self):
        cfg = {"S": {"opencode_timeout": 0}}
        self.assertEqual(resolve_opencode_timeout(cfg, "S"), 0.0)

    def test_non_mapping_source_uses_default(self):
        self.assertEqual(resolve_opencode_timeout({}, "S"), 300.0)


class TestValidateCli(unittest.TestCase):
    def test_subprocess_failure_raises(self):
        with mock.patch("benchmark.opencode.subprocess.run",
                        side_effect=OSError("no binary")):
            with self.assertRaisesRegex(RuntimeError, "Could not validate"):
                validate_cli("opencode")

    def test_nonzero_help_status_raises(self):
        probe = mock.Mock()
        probe.returncode = 3
        probe.stdout = ""
        probe.stderr = ""
        with mock.patch("benchmark.opencode.subprocess.run", return_value=probe):
            with self.assertRaisesRegex(RuntimeError, "status 3"):
                validate_cli("opencode")

    def test_missing_required_flags_raises(self):
        probe = mock.Mock()
        probe.returncode = 0
        probe.stdout = "run --help text"
        probe.stderr = ""
        with mock.patch("benchmark.opencode.subprocess.run", return_value=probe):
            with self.assertRaisesRegex(RuntimeError, "missing required run options"):
                validate_cli("opencode")

    def test_help_without_json_choice_raises(self):
        probe = mock.Mock()
        probe.returncode = 0
        probe.stdout = "--model --format --agent --pure --thinking --format choices: default"
        probe.stderr = ""
        with mock.patch("benchmark.opencode.subprocess.run", return_value=probe):
            with self.assertRaisesRegex(RuntimeError, "json"):
                validate_cli("opencode")


class TestLocalBinaryPath(unittest.TestCase):
    def test_custom_install_dir_uses_it(self):
        self.assertEqual(
            _local_binary_path("/some/dir"),
            Path("/some/dir") / ("opencode.exe" if os.name == "nt" else "opencode"),
        )


class TestPlatformDetection(unittest.TestCase):
    def test_cpu_has_avx2_true_when_flag_present(self):
        with mock.patch("builtins.open", mock.mock_open(
                read_data="flags\t\t: fpu vme avx2 sse\n")):
            self.assertTrue(_cpu_has_avx2())

    def test_cpu_has_avx2_false_when_absent(self):
        with mock.patch("builtins.open", mock.mock_open(
                read_data="flags\t\t: fpu vme sse\n")):
            self.assertFalse(_cpu_has_avx2())

    def test_cpu_has_avx2_oserror_defaults_true(self):
        with mock.patch("builtins.open", side_effect=OSError):
            self.assertTrue(_cpu_has_avx2())

    def test_is_musl_alpine_release(self):
        with mock.patch("os.path.exists", return_value=True):
            self.assertTrue(_is_musl_libc())

    def test_is_musl_via_ldd(self):
        probe = mock.Mock()
        probe.stdout = "musl libc"
        probe.stderr = ""
        with mock.patch("os.path.exists", return_value=False), \
                mock.patch("benchmark.opencode.subprocess.run", return_value=probe):
            self.assertTrue(_is_musl_libc())

    def test_is_musl_ldd_error_returns_false(self):
        with mock.patch("os.path.exists", return_value=False), \
                mock.patch("benchmark.opencode.subprocess.run",
                           side_effect=OSError):
            self.assertFalse(_is_musl_libc())

    def test_darwin_translated_when_rosetta(self):
        probe = mock.Mock()
        probe.returncode = 0
        probe.stdout = "1"
        with mock.patch("benchmark.opencode.subprocess.run", return_value=probe):
            self.assertTrue(_darwin_translated())

    def test_darwin_translated_false_otherwise(self):
        probe = mock.Mock()
        probe.returncode = 0
        probe.stdout = "0"
        with mock.patch("benchmark.opencode.subprocess.run", return_value=probe):
            self.assertFalse(_darwin_translated())

    def test_darwin_avx2_true_and_false(self):
        probe = mock.Mock()
        probe.returncode = 0
        probe.stdout = "1"
        with mock.patch("benchmark.opencode.subprocess.run", return_value=probe):
            self.assertTrue(_darwin_avx2())
        probe.stdout = "0"
        with mock.patch("benchmark.opencode.subprocess.run", return_value=probe):
            self.assertFalse(_darwin_avx2())

    def test_darwin_avx2_oserror_defaults_true(self):
        with mock.patch("benchmark.opencode.subprocess.run",
                        side_effect=OSError):
            self.assertTrue(_darwin_avx2())


class TestPlatformAssetName(unittest.TestCase):
    def test_linux_musl_appends_musl(self):
        with mock.patch("platform.system", return_value="Linux"), \
                mock.patch("platform.machine", return_value="x86_64"), \
                mock.patch("benchmark.opencode._cpu_has_avx2", return_value=True), \
                mock.patch("benchmark.opencode._is_musl_libc", return_value=True):
            self.assertIn("-musl", _platform_asset_name())

    def test_windows_uses_zip(self):
        with mock.patch("platform.system", return_value="Windows"), \
                mock.patch("platform.machine", return_value="AMD64"):
            self.assertTrue(_platform_asset_name().endswith(".zip"))

    def test_unsupported_os_raises(self):
        with mock.patch("platform.system", return_value="BeOS"):
            with self.assertRaises(RuntimeError):
                _platform_asset_name()

    def test_unsupported_arch_raises(self):
        with mock.patch("platform.system", return_value="Linux"), \
                mock.patch("platform.machine", return_value="mips"):
            with self.assertRaises(RuntimeError):
                _platform_asset_name()


class TestDownloadAndExtract(unittest.TestCase):
    def test_download_to_writes_body(self):
        class _FakeResponse:
            def __init__(self):
                self._remaining = b"body"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, n):
                chunk, self._remaining = self._remaining[:n], self._remaining[n:]
                return chunk

        with mock.patch("benchmark.opencode.urllib.request.urlopen",
                        return_value=_FakeResponse()), \
                tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "asset"
            _download_to("http://x/asset", dest, timeout=1)
            self.assertEqual(dest.read_bytes(), b"body")

    def test_extract_binary_tar_gz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "opencode.tar.gz"
            dest = Path(tmpdir) / "out"
            dest.mkdir()
            with tarfile.open(archive, "w:gz") as tar:
                member = tarfile.TarInfo("opencode")
                member.mode = 0o755
                member.size = 4
                member.type = tarfile.REGTYPE
                tar.addfile(member, io.BytesIO(b"data"))
            binary = _extract_binary(archive, dest)
            self.assertEqual(binary.name, "opencode")
            self.assertEqual(binary.read_bytes(), b"data")

    def test_extract_binary_zip_exe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "opencode.zip"
            dest = Path(tmpdir) / "out"
            dest.mkdir()
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("opencode.exe", b"data")
            binary = _extract_binary(archive, dest)
            self.assertEqual(binary.name, "opencode.exe")

    def test_extract_binary_missing_binary_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "opencode.zip"
            dest = Path(tmpdir) / "out"
            dest.mkdir()
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("readme.txt", "hello")
            with self.assertRaisesRegex(RuntimeError, "did not contain"):
                _extract_binary(archive, dest)

    def test_latest_version_invalid_payload_returns_none(self):
        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b""

            def __iter__(self):
                return iter(())

        with mock.patch("benchmark.opencode.urllib.request.urlopen",
                        return_value=_FakeResponse()):
            self.assertIsNone(_latest_opencode_version())


class TestOpencodeVersion(unittest.TestCase):
    def test_oserror_returns_none(self):
        with mock.patch("benchmark.opencode.subprocess.run",
                        side_effect=OSError):
            self.assertIsNone(opencode_version("opencode"))

    def test_empty_output_returns_none(self):
        probe = mock.Mock()
        probe.stdout = ""
        probe.stderr = ""
        with mock.patch("benchmark.opencode.subprocess.run", return_value=probe):
            self.assertIsNone(opencode_version("opencode"))

    def test_first_line_returned(self):
        probe = mock.Mock()
        probe.stdout = "v1.2.3\nmore\n"
        probe.stderr = ""
        with mock.patch("benchmark.opencode.subprocess.run", return_value=probe):
            self.assertEqual(opencode_version("opencode"), "v1.2.3")


class TestResolveOpencodeBinary(unittest.TestCase):
    def test_path_binary_incompatible_without_install_raises(self):
        with mock.patch("benchmark.opencode.shutil.which", return_value="/usr/bin/opencode"), \
                mock.patch("benchmark.opencode.validate_cli",
                           side_effect=RuntimeError("old")), \
                mock.patch("benchmark.opencode._local_binary_path",
                           return_value=Path("/nonexistent/opencode")):
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                resolve_opencode_binary(allow_install=False)

    def test_stale_local_without_install_raises(self):
        local = Path(tempfile.mkdtemp()) / "opencode"
        local.write_text("old binary")
        with mock.patch("benchmark.opencode.shutil.which", return_value=None), \
                mock.patch("benchmark.opencode.validate_cli",
                           side_effect=RuntimeError("stale")), \
                mock.patch("benchmark.opencode._local_binary_path", return_value=local):
            with self.assertRaisesRegex(RuntimeError, "does not pass"):
                resolve_opencode_binary(allow_install=False)

    def test_nothing_anywhere_without_install_raises(self):
        with mock.patch("benchmark.opencode.shutil.which", return_value=None), \
                mock.patch("benchmark.opencode._local_binary_path",
                           return_value=Path("/nonexistent/opencode")):
            with self.assertRaisesRegex(RuntimeError, "not found on PATH"):
                resolve_opencode_binary(allow_install=False)

    def test_fresh_install_failing_preflight_raises(self):
        with mock.patch("benchmark.opencode.shutil.which", return_value=None), \
                mock.patch("benchmark.opencode._local_binary_path",
                           return_value=Path("/nonexistent/opencode")), \
                mock.patch("benchmark.opencode.install_opencode",
                           return_value="/installed/opencode"), \
                mock.patch("benchmark.opencode.validate_cli",
                           side_effect=RuntimeError("nope")):
            with self.assertRaisesRegex(RuntimeError, "failed preflight"):
                resolve_opencode_binary(allow_install=True)


class TestProviderOptions(unittest.TestCase):
    def test_missing_api_url_raises(self):
        with self.assertRaises(ValueError):
            _provider_options({})

    def test_non_mapping_headers_raises(self):
        with self.assertRaises(ValueError):
            _provider_options({"api_url": "http://x/v1/chat/completions",
                               "headers": "nope"})

    def test_bearer_authorization_becomes_api_key(self):
        options = _provider_options({
            "api_url": "http://x/v1/chat/completions",
            "headers": {"Authorization": "Bearer sk-123",
                        "Content-Type": "application/json",
                        "X-Custom": "v"},
        })
        self.assertEqual(options["apiKey"], "sk-123")
        self.assertNotIn("content-type", {k.lower() for k in options.get("headers", {})})
        self.assertEqual(options["headers"]["X-Custom"], "v")

    def test_authorization_without_bearer_becomes_custom_header(self):
        options = _provider_options({
            "api_url": "http://x/v1/chat/completions",
            "headers": {"Authorization": "Basic abc"},
        })
        self.assertNotIn("apiKey", options)
        self.assertEqual(options["headers"]["Authorization"], "Basic abc")

    def test_timeout_option_passthrough(self):
        options = _provider_options({
            "api_url": "http://x/v1/completions",
            "headers": {},
            "timeout": 30,
        })
        self.assertEqual(options["timeout"], 30)
        self.assertIn("baseURL", options)


class TestGenerateConfigUnsupported(unittest.TestCase):
    def _base(self):
        return {
            "sources": {"S": {"api_url": "http://x/v1/chat/completions",
                              "headers": {"Authorization": "Bearer k"}}},
            "targets": {"m1": {"source": "S", "api_model": "model-a"}},
        }

    def test_unsupported_fields_are_projected(self):
        targets = {"m1": {"source": "S", "api_model": "model-a",
                          "drop_params": ["seed"]}}
        with tempfile.TemporaryDirectory() as tmpdir:
            out = generate_config(
                {"S": {"api_url": "http://x/v1/chat/completions",
                       "headers": {}}},
                targets, Path(tmpdir) / "oc.json",
                timeout=60, token_levels=[4096],
                benchmark_config={"seed": 42, "retry_on_429": True},
                plugin_temperatures={"rate-limiter": 0.2},
            )
            self.assertIn("per-target drop_params", out["projection"]["unsupported"])
            self.assertIn("seed", out["projection"]["unsupported"])
            self.assertIn("HTTP retry/backoff controls", out["projection"]["unsupported"])
            self.assertIn("per-plugin temperature overrides", out["projection"]["unsupported"])
            self.assertIn("HTTP streaming/TTFT telemetry", out["projection"]["unsupported"])
            self.assertEqual(out["config"]["model"], "s/model-a")
            # Written file is chmod 600 where supported.
            self.assertTrue(Path(out["path"]).exists())

    def test_mapping_collision_detected(self):
        targets = {
            "m1": {"source": "S", "api_model": "model-a"},
            "m2": {"source": "S", "api_model": "model-a"},
        }
        with self.assertRaisesRegex(ValueError, "model mapping collision"):
            generate_config({"S": {"api_url": "http://x", "headers": {}}},
                            targets, "/tmp/oc.json")


class TestExtractFinalTextEdgeCases(unittest.TestCase):
    def test_error_event_with_string_payload(self):
        stream = b'{"type": "error", "error": "boom"}\n'
        extract = _extract_final_text(stream)
        self.assertEqual(extract.error, "boom")

    def test_error_event_with_dict_payload(self):
        stream = b'{"type": "error", "error": {"message": "msg"}}\n'
        self.assertEqual(_extract_final_text(stream).error, "msg")

    def test_non_ndjson_falls_back_to_raw_stdout(self):
        extract = _extract_final_text(b"plain text output")
        self.assertEqual(extract.text, "plain text output")
        self.assertIsNone(extract.error)

    def test_incomplete_parts_are_ignored(self):
        stream = (b'{"type": "text", "part": {"type": "text", "text": "   "}}\n'
                  b'{"type": "reasoning", "part": {"type": "reasoning"}}\n')
        extract = _extract_final_text(stream)
        self.assertEqual(extract.text, "")
        self.assertEqual(extract.think_text, "")


class TestTerminateProcess(unittest.TestCase):
    def test_already_exited_returns(self):
        process = mock.Mock()
        process.poll.return_value = 0
        _terminate_process(process)
        process.terminate.assert_not_called()

    def test_posix_killpg_then_wait(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.pid = 1234
        with mock.patch("os.name", "posix"), \
                mock.patch("os.killpg") as killpg, \
                mock.patch("benchmark.opencode.signal.SIGTERM", 15):
            _terminate_process(process)
        killpg.assert_called_once()
        process.wait.assert_called_once()

    def test_killpg_error_falls_back_to_terminate(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.pid = 1234
        with mock.patch("os.name", "posix"), \
                mock.patch("os.killpg", side_effect=OSError), \
                mock.patch("benchmark.opencode.signal.SIGTERM", 15):
            _terminate_process(process)
        process.terminate.assert_called_once()

    def test_wait_timeout_then_sigkill(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.pid = 1234
        process.wait.side_effect = subprocess.TimeoutExpired("x", 1)
        with mock.patch("os.name", "posix"), \
                mock.patch("os.killpg") as killpg, \
                mock.patch("benchmark.opencode.signal.SIGTERM", 15), \
                mock.patch("benchmark.opencode.signal.SIGKILL", 9):
            _terminate_process(process)
        self.assertEqual(killpg.call_count, 2)


class TestStreamGuard(unittest.TestCase):
    def test_feed_counts_step_finish(self):
        guard = _StreamGuard(step_limit=3)
        guard.feed(b'{"type": "step_finish"}\n')
        self.assertEqual(guard.step_count, 1)
        self.assertFalse(guard.steps_exceeded)

    def test_steps_exceeded_trips(self):
        guard = _StreamGuard(step_limit=2)
        guard.feed(b'{"type": "step_finish"}\n' * 2)
        self.assertTrue(guard.steps_exceeded)

    def test_ignores_non_dict_and_bad_json(self):
        guard = _StreamGuard(repeat_threshold=2)
        guard.feed(b"not json\n[1, 2]\n")
        self.assertFalse(guard.repeated)

    def test_repetition_detected_after_threshold(self):
        guard = _StreamGuard(repeat_threshold=2, repeat_min_len=5)
        event = json.dumps({"type": "text",
                            "part": {"type": "text", "text": "aaaaa"}})
        guard.feed((event + "\n").encode() * 2)
        self.assertTrue(guard.repeated)

    def test_short_text_below_min_len_ignored(self):
        guard = _StreamGuard(repeat_threshold=2, repeat_min_len=20)
        event = json.dumps({"type": "text",
                            "part": {"type": "text", "text": "x"}})
        guard.feed((event + "\n").encode() * 5)
        self.assertFalse(guard.repeated)

    def test_stops_counting_after_trip(self):
        guard = _StreamGuard(step_limit=1)
        guard.feed(b'{"type": "step_finish"}\n')
        guard.feed(b'{"type": "step_finish"}\n')
        self.assertEqual(guard.step_count, 1)


class TestPumpStream(unittest.TestCase):
    def test_reads_to_eof_and_feeds_guard(self):
        class _Stream:
            def __init__(self, data):
                self._data = data

            def read(self, n):
                chunk, self._data = self._data[:n], self._data[n:]
                return chunk

        sink = []
        guard = _StreamGuard(step_limit=1)
        _pump_stream(_Stream(b'{"type": "step_finish"}\n'), sink, guard)
        self.assertEqual(b"".join(sink), b'{"type": "step_finish"}\n')
        self.assertEqual(guard.step_count, 1)

    def test_read_error_is_swallowed(self):
        class _Broken:
            def read(self, n):
                raise OSError("closed")

        sink = []
        _pump_stream(_Broken(), sink, None)
        self.assertEqual(sink, [])


class TestRunProcessEdgeCases(unittest.TestCase):
    def _popen_fake(self, stdout=b"", stderr=b"", returncode=0):
        fake = mock.Mock()
        fake.stdout = io.BytesIO(stdout)
        fake.stderr = io.BytesIO(stderr)
        fake.returncode = returncode
        # First poll: still running. Every later poll (loop re-check + the
        # finally-block cleanup) reports the final returncode.
        state = {"calls": 0}

        def _poll():
            state["calls"] += 1
            return None if state["calls"] == 1 else returncode

        fake.poll.side_effect = _poll
        return fake

    def test_popen_oserror_reported(self):
        with mock.patch("benchmark.opencode.subprocess.Popen",
                        side_effect=OSError("nope")):
            result = run_process("p", config_path="/tmp/c.json", model="m",
                                 timeout=1, output_dir=None, target_key="t",
                                 plugin_id="p")
        self.assertIn("Could not start OpenCode", result.error)

    def test_session_error_surfaces(self):
        stream = b'{"type": "error", "error": "session down"}\n'
        fake = self._popen_fake(stdout=stream)
        with mock.patch("benchmark.opencode.subprocess.Popen", return_value=fake):
            result = run_process("p", config_path="/tmp/c.json", model="m",
                                 timeout=5, output_dir=None, target_key="t",
                                 plugin_id="p")
        self.assertIn("session error: session down", result.error)

    def test_nonzero_exit_surfaces(self):
        fake = self._popen_fake(stderr=b"oops", returncode=7)
        with mock.patch("benchmark.opencode.subprocess.Popen", return_value=fake):
            result = run_process("p", config_path="/tmp/c.json", model="m",
                                 timeout=5, output_dir=None, target_key="t",
                                 plugin_id="p")
        self.assertIn("exited with status 7", result.error)

    def test_empty_response_surfaces(self):
        fake = self._popen_fake(stdout=b"")
        with mock.patch("benchmark.opencode.subprocess.Popen", return_value=fake):
            result = run_process("p", config_path="/tmp/c.json", model="m",
                                 timeout=5, output_dir=None, target_key="t",
                                 plugin_id="p")
        self.assertEqual(result.error, "OpenCode returned an empty response")

    def test_writes_logs_when_output_dir_given(self):
        stream = b'{"type": "text", "part": {"type": "text", "text": "hello there"}}\n'
        fake = self._popen_fake(stdout=stream)
        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch("benchmark.opencode.subprocess.Popen", return_value=fake):
            result = run_process("p", config_path="/tmp/c.json", model="m",
                                 timeout=5, output_dir=tmpdir, target_key="t",
                                 plugin_id="p")
            self.assertEqual(result.text, "hello there")
            log_dir = Path(tmpdir) / "logs" / "t"
            self.assertTrue((log_dir / "p.stdout.txt").exists())
            self.assertTrue((log_dir / "p.stderr.txt").exists())


if __name__ == "__main__":
    unittest.main()
