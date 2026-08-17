"""Tests for the ChatPlayground.ai subprocess-isolated source adapter.

The parent module (:mod:`benchmark.chatplayground`) never imports Playwright;
it proxies a JSON-lines protocol to a worker subprocess. These tests exercise
that proxy with a fake worker script (real subprocess + pipes, no browser) and
mock the request layer for the config/request helpers. Browser-level behaviour
lives in ``benchmark.chatplayground_worker`` and is tested separately in
``tests/test_chatplayground_worker.py``.
"""

import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import benchmark.chatplayground as cp
from benchmark.http import nonstream_request, stream_request

DEFAULT_BASE_URL = cp.DEFAULT_BASE_URL

# A fake worker that answers every op deterministically over the same
# JSON-lines protocol the real worker uses.
_HEALTHY_WORKER = r"""
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except ValueError:
        continue
    op = msg.get("op")
    if op == "send":
        resp = {"id": msg.get("id"), "ok": True,
                "text": "fake answer for " + str(msg.get("prompt", ""))[:20]}
    elif op == "list_models":
        resp = {"id": msg.get("id"), "ok": True,
                "models": ["gpt-5.6-terra", "deepseek-v4-pro"]}
    elif op == "probe":
        resp = {"id": msg.get("id"), "ok": True,
                "probe": {"url": "https://x", "title": "t",
                          "textarea_count": 1, "input_count": 0,
                          "button_count": 2, "models": ["gpt-5.6-terra"]}}
    else:
        resp = {"id": msg.get("id"), "ok": False, "error": "unknown op"}
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
"""

# Reads one request then exits non-zero, so the parent sees EOF + a bad code.
_CRASH_WORKER = "import sys; sys.stdin.readline(); sys.exit(3)\n"

# Reads one request then sleeps forever, so the parent's stop_event/deadline
# path has to terminate it.
_HANG_WORKER = "import sys, time; sys.stdin.readline(); time.sleep(3600)\n"


def _cfg(**overrides):
    base = {
        "api_protocol": "chatplayground",
        "base_url": DEFAULT_BASE_URL,
        "email": "a@b.com",
        "password": "hunter2",
        "headless": True,
    }
    base.update(overrides)
    return base


def _write_fake_worker(body):
    fd, path = tempfile.mkstemp(suffix=".py", text=True)
    with os.fdopen(fd, "w") as handle:
        handle.write(body)
    return path


def _patch_worker(script):
    return mock.patch.object(cp, "_worker_command", return_value=[sys.executable, script])


class TestConfigHelpers(unittest.TestCase):
    def test_is_chatplayground(self):
        self.assertTrue(cp.is_chatplayground(_cfg()))
        self.assertFalse(cp.is_chatplayground({}))
        self.assertFalse(cp.is_chatplayground(None))
        self.assertFalse(cp.is_chatplayground({"api_protocol": "1min"}))

    def test_credentials_email_and_username_fallback(self):
        self.assertEqual(cp.credentials(_cfg()), ("a@b.com", "hunter2"))
        self.assertEqual(
            cp.credentials({"email": "x@y.z", "password": "pw"}), ("x@y.z", "pw")
        )
        self.assertEqual(
            cp.credentials({"username": "user", "password": "pw"}), ("user", "pw")
        )
        self.assertEqual(cp.credentials(None), ("", ""))

    def test_selectors_merge_overrides(self):
        merged = cp.selectors(_cfg(selectors={"settle_ms": 0, "prompt_input": "textarea#x"}))
        self.assertEqual(merged["settle_ms"], 0)
        self.assertEqual(merged["prompt_input"], "textarea#x")
        # Untouched keys keep their defaults.
        self.assertEqual(merged["email_input"], cp.DEFAULT_SELECTORS["email_input"])

    def test_config_from_env_reads_environment(self):
        env = {
            "CHATPLAYGROUND_EMAIL": "a@b.com",
            "CHATPLAYGROUND_PASSWORD": "pw",
            "CHATPLAYGROUND_BASE_URL": "https://cp.example",
            "CHATPLAYGROUND_HEADLESS": "0",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            cfg = cp.config_from_env()
        self.assertEqual(cfg["api_protocol"], "chatplayground")
        self.assertEqual(cfg["email"], "a@b.com")
        self.assertEqual(cfg["password"], "pw")
        self.assertEqual(cfg["base_url"], "https://cp.example")
        self.assertFalse(cfg["headless"])

    def test_config_from_env_headless_defaults_true(self):
        with mock.patch.dict("os.environ", clear=True):
            cfg = cp.config_from_env()
        self.assertTrue(cfg["headless"])
        self.assertEqual(cfg["base_url"], DEFAULT_BASE_URL)

    def test_generate_config_builds_ready_to_run_config(self):
        with mock.patch.object(
            cp, "list_models", return_value=["gpt-5.6-terra", "deepseek-v4-pro"]
        ):
            cfg = cp.generate_config(_cfg())

        self.assertEqual(cfg["output_dir"], "benchmark-results")
        self.assertEqual(cfg["token_levels"], [16384])
        source = cfg["sources"]["ChatPlayground"]
        self.assertEqual(source["api_protocol"], "chatplayground")
        # Browser-safe scheduler defaults are applied automatically.
        self.assertEqual(source["model_thread_limit"], 1)
        self.assertEqual(source["plugin_thread_limit"], 1)
        self.assertFalse(source["preload"])
        self.assertEqual(
            cfg["models"],
            {"gpt-5.6-terra": "ChatPlayground", "deepseek-v4-pro": "ChatPlayground"},
        )

    def test_generate_config_requires_credentials(self):
        with self.assertRaises(ValueError):
            cp.generate_config({"api_protocol": "chatplayground", "email": "", "password": ""})

    def test_generate_config_fails_without_models(self):
        with (
            mock.patch.object(cp, "list_models", return_value=[]),
            self.assertRaises(RuntimeError),
        ):
            cp.generate_config(_cfg())


class TestRequestProxy(unittest.TestCase):
    def test_request_returns_buffered_text_and_passes_payload(self):
        with mock.patch.object(
            cp, "_send_request", return_value={"ok": True, "text": "the answer"}
        ) as send:
            text, error, elapsed = cp.request(_cfg(), "gpt-5.6-terra", "hi", timeout=10)

        self.assertEqual(text, "the answer")
        self.assertIsNone(error)
        self.assertGreaterEqual(elapsed, 0.0)
        send.assert_called_once()
        call = send.call_args
        self.assertEqual(call[0][0], "send")
        self.assertEqual(call.kwargs["model"], "gpt-5.6-terra")
        self.assertEqual(call.kwargs["prompt"], "hi")
        self.assertEqual(call.kwargs["timeout"], 10)
        self.assertIsNone(call.kwargs["system_prompt"])
        self.assertIsNone(call.kwargs["stop_event"])

    def test_request_folds_system_prompt_into_payload(self):
        with mock.patch.object(cp, "_send_request", return_value={"ok": True, "text": "x"}) as send:
            cp.request(_cfg(), "gpt-5.6-terra", "hi", timeout=10, system_prompt="You are a coder.")
        self.assertEqual(send.call_args.kwargs["system_prompt"], "You are a coder.")

    def test_request_surfaces_worker_error(self):
        with mock.patch.object(
            cp, "_send_request", return_value={"ok": False, "error": "RuntimeError: login failed"}
        ):
            text, error, _elapsed = cp.request(_cfg(), "gpt-5.6-terra", "hi", timeout=10)
        self.assertEqual(text, "")
        self.assertIn("RuntimeError: login failed", error)

    def test_request_cancel_before_send_short_circuits(self):
        stop = threading.Event()
        stop.set()
        with mock.patch.object(cp, "_send_request") as send:
            text, error, _elapsed = cp.request(
                _cfg(), "gpt-5.6-terra", "hi", timeout=10, stop_event=stop
            )
        send.assert_not_called()
        self.assertEqual(text, "")
        self.assertIn("cancelled", error)

    def test_list_models_proxies(self):
        with mock.patch.object(
            cp, "_send_request", return_value={"ok": True, "models": ["a", "b"]}
        ):
            self.assertEqual(cp.list_models(_cfg()), ["a", "b"])

    def test_list_models_raises_on_worker_error(self):
        with (
            mock.patch.object(cp, "_send_request", return_value={"ok": False, "error": "boom"}),
            self.assertRaises(RuntimeError),
        ):
            cp.list_models(_cfg())

    def test_probe_proxies(self):
        with mock.patch.object(
            cp, "_send_request",
            return_value={"ok": True, "probe": {"url": "u", "models": ["a"]}},
        ):
            info = cp.probe(_cfg())
        self.assertEqual(info["url"], "u")
        self.assertEqual(info["models"], ["a"])

    def test_probe_raises_on_worker_error(self):
        with (
            mock.patch.object(cp, "_send_request", return_value={"ok": False, "error": "boom"}),
            self.assertRaises(RuntimeError),
        ):
            cp.probe(_cfg())

    def test_cli_probe_reads_env(self):
        env = {"CHATPLAYGROUND_EMAIL": "e", "CHATPLAYGROUND_PASSWORD": "p",
               "CHATPLAYGROUND_HEADLESS": "0"}
        with mock.patch.dict("os.environ", env, clear=True), \
             mock.patch.object(cp, "probe", return_value={"ok": True}) as pr:
            out = cp._cli_probe()
        self.assertEqual(out, {"ok": True})
        pr.assert_called_once()


class TestWorkerSubprocess(unittest.TestCase):
    """Real subprocess + pipe round-trips against a fake worker script."""

    def tearDown(self):
        cp._close_session()

    def test_request_roundtrip_through_worker(self):
        script = _write_fake_worker(_HEALTHY_WORKER)
        with _patch_worker(script):
            text, error, elapsed = cp.request(_cfg(), "gpt-5.6-terra", "hi", timeout=10)
        self.assertEqual(text, "fake answer for hi")
        self.assertIsNone(error)
        self.assertGreaterEqual(elapsed, 0.0)

    def test_worker_crash_is_surfaced_and_next_request_recovers(self):
        crash_script = _write_fake_worker(_CRASH_WORKER)
        healthy_script = _write_fake_worker(_HEALTHY_WORKER)

        with _patch_worker(crash_script):
            text, error, _elapsed = cp.request(_cfg(), "gpt-5.6-terra", "hi", timeout=10)
        self.assertEqual(text, "")
        self.assertIn("crashed", error)
        self.assertIn("exit code 3", error)
        self.assertIsNone(cp._proc)

        # A fresh worker is spawned for the next request and succeeds.
        with _patch_worker(healthy_script):
            text, error, _elapsed = cp.request(_cfg(), "gpt-5.6-terra", "hi", timeout=10)
        self.assertEqual(text, "fake answer for hi")
        self.assertIsNone(error)

    def test_cancel_terminates_hung_worker(self):
        script = _write_fake_worker(_HANG_WORKER)
        stop = threading.Event()

        def _set_stop():
            time.sleep(0.4)
            stop.set()

        threading.Thread(target=_set_stop, daemon=True).start()
        with _patch_worker(script):
            text, error, _elapsed = cp.request(_cfg(), "gpt-5.6-terra", "hi", timeout=10,
                                               stop_event=stop)
        self.assertEqual(text, "")
        self.assertIn("cancelled", error)
        self.assertIsNone(cp._proc)

    def test_hung_worker_times_out(self):
        script = _write_fake_worker(_HANG_WORKER)
        with _patch_worker(script), mock.patch.object(cp, "_WORKER_GRACE", 0.5):
            text, error, _elapsed = cp.request(_cfg(), "gpt-5.6-terra", "hi", timeout=0.01)
        self.assertEqual(text, "")
        self.assertIn("timed out", error)
        self.assertIsNone(cp._proc)

    def test_close_session_terminates_worker(self):
        script = _write_fake_worker(_HEALTHY_WORKER)
        with _patch_worker(script):
            cp.request(_cfg(), "gpt-5.6-terra", "hi", timeout=10)
            self.assertIsNotNone(cp._proc)
            cp._close_session()
        self.assertIsNone(cp._proc)

    def test_send_request_surfaces_broken_pipe(self):
        proc = mock.MagicMock()
        proc.stdin.write.side_effect = BrokenPipeError()
        proc.stdin.flush.side_effect = BrokenPipeError()
        # ``_ensure_worker`` is mocked, so seed the queue the wait loop needs.
        cp._queue = queue.Queue()
        with mock.patch.object(cp, "_ensure_worker", return_value=proc):
            resp = cp._send_request("send", _cfg(), model="m", prompt="p", timeout=1)
        self.assertFalse(resp.get("ok"))
        self.assertIn("died before handling", resp.get("error", ""))

    def test_send_request_tolerates_non_numeric_timeout(self):
        script = _write_fake_worker(_HEALTHY_WORKER)
        with _patch_worker(script):
            resp = cp._send_request("send", _cfg(), timeout="oops", model="m", prompt="p")
        self.assertTrue(resp.get("ok"))

    def test_send_request_reports_unavailable_worker(self):
        proc = mock.MagicMock()
        proc.stdin = None
        with mock.patch.object(cp, "_ensure_worker", return_value=proc):
            resp = cp._send_request("send", _cfg(), model="m", prompt="p", timeout=1)
        self.assertFalse(resp.get("ok"))
        self.assertIn("unavailable", resp.get("error", ""))


class TestWorkerInternals(unittest.TestCase):
    """Low-level subprocess plumbing: termination and stream pumps."""

    def setUp(self):
        cp._teardown_worker()

    def tearDown(self):
        cp._teardown_worker()

    def test_terminate_noop_when_already_exited(self):
        proc = mock.MagicMock()
        proc.poll.return_value = 0
        cp._terminate(proc)
        proc.terminate.assert_not_called()
        proc.kill.assert_not_called()

    def test_terminate_escalates_to_kill_on_timeout(self):
        proc = mock.MagicMock()
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired("p", 1)
        # First killpg (SIGTERM) raises ProcessLookupError -> terminate() fallback;
        # second (SIGKILL) succeeds -> the timeout escalation is exercised.
        with mock.patch(
            "os.killpg",
            side_effect=[ProcessLookupError, None],
        ) as killpg:
            cp._terminate(proc)
        proc.terminate.assert_called()
        killpg.assert_has_calls([
            mock.call(proc.pid, signal.SIGTERM),
            mock.call(proc.pid, signal.SIGKILL),
        ])

    def test_ensure_worker_raises_on_spawn_failure(self):
        with (
            mock.patch("subprocess.Popen", side_effect=OSError("boom")),
            self.assertRaises(RuntimeError),
        ):
            cp._ensure_worker()
        self.assertIsNone(cp._proc)

    def test_atexit_cleanup_tears_down_worker(self):
        script = _write_fake_worker(_HEALTHY_WORKER)
        with _patch_worker(script):
            cp.request(_cfg(), "gpt-5.6-terra", "hi", timeout=10)
            self.assertIsNotNone(cp._proc)
            cp._atexit_cleanup()
        self.assertIsNone(cp._proc)

    def test_reader_loop_skips_empty_and_invalid_lines(self):
        out_queue = queue.Queue()
        proc = mock.MagicMock()
        proc.stdout.__iter__.return_value = iter([
            "", "not-json\n", '{"id": 1, "ok": true}\n',
        ])
        cp._reader_loop(proc, out_queue)
        items = []
        while not out_queue.empty():
            items.append(out_queue.get())
        self.assertEqual(items, [("msg", {"id": 1, "ok": True}), ("eof", None)])

    def test_reader_loop_surfaces_stream_failure(self):
        out_queue = queue.Queue()
        proc = mock.MagicMock()

        def _boom():
            raise OSError("pipe closed")

        proc.stdout.__iter__.side_effect = _boom
        cp._reader_loop(proc, out_queue)
        items = []
        while not out_queue.empty():
            items.append(out_queue.get())
        self.assertEqual(items, [("eof", None)])


class TestHttpDelegation(unittest.TestCase):
    def test_nonstream_request_routes_chatplayground(self):
        cfg = {"cp": {"api_protocol": "chatplayground", "email": "a@b.com", "password": "pw"}}
        with mock.patch.object(cp, "request", return_value=("hello", None, 1.5)) as req:
            result = nonstream_request(cfg, timeout=5, model="gpt-5.6-terra", source="cp",
                                       prompt="hi", max_tokens=10)
        self.assertEqual(result.text, "hello")
        self.assertIsNone(result.error)
        self.assertEqual(result.gen_time, 1.5)
        req.assert_called_once()

    def test_stream_request_routes_chatplayground(self):
        cfg = {"cp": {"api_protocol": "chatplayground", "email": "a@b.com", "password": "pw"}}
        with mock.patch.object(cp, "request", return_value=("hello", None, 1.5)) as req:
            result = stream_request(cfg, timeout=5, model="gpt-5.6-terra", source="cp",
                                    prompt="hi", max_tokens=10)
        self.assertEqual(result.text, "hello")
        self.assertIsNone(result.error)
        # Buffered: no first-token time for browser sources.
        self.assertIsNone(result.first_tok)
        req.assert_called_once()


if __name__ == "__main__":
    unittest.main()
