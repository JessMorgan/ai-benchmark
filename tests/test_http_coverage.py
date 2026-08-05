"""Coverage-focused tests for benchmark.http.

Targets small helpers and error branches not covered by test_benchmark_http:
the SSE iterator sentinel, active-request cleanup, curl-cmd with a system
prompt, total-timeout check, labelled request logging, tool-call merging
fragments, the invalid-response fallback in the non-streaming path, retry
config fallbacks, and the stream/non-stream error branches.
"""
import contextlib
import json
import tempfile
import threading
import time
import unittest
from unittest import mock

from benchmark.http import (
    _check_total_timeout,
    _merge_tool_calls,
    _parse_sse_line,
    _render_tool_calls,
    _safe_iter_lines,
    _StreamLineError,
    build_curl_cmd,
    close_active_requests,
    get_active_request_count,
    log_request_entry,
    nonstream_request,
    PostRequestResult,
    ResponseBodyResult,
    stream_request,
)
from tests.utils import MockResponse


class TestSafeIterLines(unittest.TestCase):
    def test_yields_lines(self):
        resp = mock.Mock()
        resp.iter_lines.return_value = ["data: one", "data: two"]
        self.assertEqual(
            list(_safe_iter_lines(resp)), ["data: one", "data: two"])

    def test_iteration_failure_yields_sentinel(self):
        resp = mock.Mock()

        def boom(decode_unicode=False):
            raise OSError("connection reset")
            yield  # pragma: no cover

        resp.iter_lines.side_effect = boom
        results = list(_safe_iter_lines(resp))
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], _StreamLineError)
        self.assertIn("OSError", results[0].error)


class TestCloseActiveRequests(unittest.TestCase):
    def test_closes_all_active_responses(self):
        resp1 = mock.Mock()
        resp2 = mock.Mock()
        with mock.patch("benchmark.http._active_requests", {resp1, resp2}):
            close_active_requests()
        resp1.close.assert_called_once()
        resp2.close.assert_called_once()

    def test_close_error_is_swallowed(self):
        resp = mock.Mock()
        resp.close.side_effect = OSError("closed")
        with mock.patch("benchmark.http._active_requests", {resp}):
            close_active_requests()  # must not raise


class TestBuildCurlCmdSystemPrompt(unittest.TestCase):
    def test_system_prompt_added_as_first_message(self):
        command = build_curl_cmd(
            "model-a", "user prompt", 100, True,
            "http://x/v1/chat/completions",
            {"Authorization": "Bearer k", "Content-Type": "application/json"},
            system_prompt="be concise",
        )
        self.assertIn("system", command)
        self.assertIn("be concise", command)
        self.assertIn("Bearer k", command)


class TestLogRequestEntry(unittest.TestCase):
    def test_writes_with_label_and_creates_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/nested/request.log"
            log_request_entry(log_path, "curl ...", "body", request_label="probe")
            with open(log_path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("# === probe ===", content)
            self.assertIn("curl ...", content)
            self.assertIn("body", content)

    def test_writes_without_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/request.log"
            log_request_entry(log_path, "curl ...", "body")
            with open(log_path, encoding="utf-8") as f:
                self.assertNotIn("# ===", f.read())


class TestCheckTotalTimeout(unittest.TestCase):
    def test_returns_timeout_error_when_exceeded(self):
        error = _check_total_timeout(time.time() - 10, 5, None)
        self.assertIn("Total timeout", error)

    def test_keeps_existing_error(self):
        error = _check_total_timeout(time.time() - 10, 5, "previous")
        self.assertEqual(error, "previous")

    def test_finish_reason_suppresses_timeout(self):
        error = _check_total_timeout(time.time() - 10, 5, None, finish_reason="stop")
        self.assertIsNone(error)

    def test_within_timeout_returns_none(self):
        self.assertIsNone(_check_total_timeout(time.time(), 60, None))


class TestMergeToolCalls(unittest.TestCase):
    def test_merges_fragments_by_index(self):
        acc = [
            {"id": "call_1", "type": "function",
             "function": {"name": "read", "arguments": "{\"path\":"}},
        ]
        merged = _merge_tool_calls(acc, [
            {"index": 0, "function": {"arguments": " \"/x\"}"}},
            {"index": 1, "id": "call_2", "type": "function",
             "function": {"name": "write", "arguments": "{}"}},
        ])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["function"]["arguments"], '{"path": "/x"}')
        self.assertEqual(merged[1]["id"], "call_2")

    def test_ignores_non_dict_fragments(self):
        self.assertEqual(_merge_tool_calls([], [None, "junk"]), [])


class TestNonStreamInvalidResponse(unittest.TestCase):
    def _config(self):
        return {"S": {"api_url": "http://x/v1/chat/completions", "headers": {}}}

    def test_invalid_json_reports_error(self):
        resp = MockResponse(text="not json", status_code=200)
        with mock.patch("benchmark.http.requests.post", return_value=resp):
            result = nonstream_request(self._config(), 5, "m", "S", "p", 100)
        self.assertIn("Invalid completion response", result.error)

    def test_missing_choices_key_reports_error(self):
        resp = MockResponse(text=json.dumps({"unexpected": True}), status_code=200)
        with mock.patch("benchmark.http.requests.post", return_value=resp):
            result = nonstream_request(self._config(), 5, "m", "S", "p", 100)
        self.assertIn("Invalid completion response", result.error)

    def test_empty_body_reports_error(self):
        resp = mock.Mock()
        resp.status_code = 200
        resp.headers = {}
        resp.close = mock.Mock()
        resp.iter_lines.return_value = []
        resp.text = ""
        with mock.patch("benchmark.http.requests.post", return_value=resp):
            result = nonstream_request(self._config(), 5, "m", "S", "p", 100)
        self.assertIsNotNone(result.error)


class TestRetryConfigFallbacks(unittest.TestCase):
    """Bad per-source retry values fall back to the documented defaults."""

    def _streaming_ok(self):
        m = mock.MagicMock()
        m.status_code = 200
        m.text = "ok"
        m.headers = {}
        m.iter_lines = mock.MagicMock(return_value=iter([
            "data: " + json.dumps({"choices": [{"delta": {"content": "ok"},
                                                    "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]))
        m.close = mock.MagicMock()
        return m

    def test_invalid_retry_values_use_defaults(self):
        cfg = {"S": {"api_url": "http://x/v1/chat/completions", "headers": {},
                     "max_429_retries": "abc", "backoff_seconds": "x",
                     "backoff_factor": "y", "max_backoff_seconds": "z"}}
        with mock.patch("benchmark.http.requests.post", return_value=self._streaming_ok()):
            result = stream_request(cfg, 5, "m", "S", "p", 100)
        self.assertIsNone(result.error)
        self.assertEqual(result.text, "ok")


class TestOnRetryObserver(unittest.TestCase):
    """A raising on_retry observer is reported but never aborts the retry."""

    def _cfg(self):
        return {"S": {"api_url": "http://x/v1/chat/completions", "headers": {},
                       "max_429_retries": 1, "backoff_seconds": 0.01,
                       "backoff_factor": 1.0, "max_backoff_seconds": 1.0}}

    def _mock_429(self):
        m = mock.MagicMock()
        m.status_code = 429
        m.text = "slow down"
        m.headers = {}
        m.close = mock.MagicMock()
        return m

    def _mock_200(self):
        m = mock.MagicMock()
        m.status_code = 200
        m.text = "ok"
        m.headers = {}
        m.iter_lines = mock.MagicMock(return_value=iter([
            "data: " + json.dumps({"choices": [{"delta": {"content": "ok"},
                                                    "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]))
        m.close = mock.MagicMock()
        return m

    def test_raising_on_retry_is_swallowed(self):
        import io
        def boom():
            raise RuntimeError("observer bug")
        old_stderr = __import__("sys").stderr
        __import__("sys").stderr = io.StringIO()
        try:
            with mock.patch("benchmark.http.requests.post",
                            side_effect=[self._mock_429(), self._mock_200()]):
                result = stream_request(self._cfg(), 5, "m", "S", "p", 100,
                                        on_retry=boom)
        finally:
            __import__("sys").stderr = old_stderr
        self.assertIsNone(result.error)
        self.assertEqual(result.text, "ok")

    def test_requests_post_raising_yields_error(self):
        with mock.patch("benchmark.http.requests.post",
                        side_effect=RuntimeError("conn refused")):
            result = stream_request(self._cfg(), 5, "m", "S", "p", 100)
        self.assertIn("RuntimeError", result.error)


class TestHttpErrorStatus(unittest.TestCase):
    def test_non_200_yields_error(self):
        resp = MockResponse(text="boom", status_code=500)
        with mock.patch("benchmark.http.requests.post", return_value=resp):
            result = stream_request({"S": {"api_url": "http://x", "headers": {}}},
                                    5, "m", "S", "p", 100)
        self.assertIn("HTTP 500", result.error)

    def test_429_exhausted_writes_log(self):
        cfg = {"S": {"api_url": "http://x/v1/chat/completions", "headers": {},
                     "max_429_retries": 0}}
        m = mock.MagicMock()
        m.status_code = 429
        m.text = "slow down"
        m.headers = {}
        m.close = mock.MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/r.log"
            with mock.patch("benchmark.http.requests.post", return_value=m):
                result = stream_request(cfg, 5, "m", "S", "p", 100,
                                        log_path=log_path, log_label="probe")
            with open(log_path, encoding="utf-8") as f:
                content = f.read()
        self.assertIn("HTTP 429", result.error)
        self.assertIn("probe", content)


class TestRetryAfterHttpDate(unittest.TestCase):
    """Retry-After in RFC 7231 HTTP-date form is honoured."""

    def _cfg(self):
        return {"S": {"api_url": "http://x/v1/chat/completions", "headers": {},
                       "max_429_retries": 1, "backoff_seconds": 0.01,
                       "backoff_factor": 1.0, "max_backoff_seconds": 1.0}}

    def _mock_429(self, retry_after):
        m = mock.MagicMock()
        m.status_code = 429
        m.text = "slow down"
        m.headers = {"Retry-After": retry_after}
        m.close = mock.MagicMock()
        return m

    def _mock_200(self):
        m = mock.MagicMock()
        m.status_code = 200
        m.text = "ok"
        m.headers = {}
        m.iter_lines = mock.MagicMock(return_value=iter([
            "data: " + json.dumps({"choices": [{"delta": {"content": "ok"},
                                                    "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]))
        m.close = mock.MagicMock()
        return m

    def test_http_date_in_past_is_ignored(self):
        # "Sun, 06 Nov 1994 08:49:37 GMT" parses as a naive timestamp with a
        # GMT offset; the delta against now is negative so retry_after clamps
        # to 0 and the exponential backoff governs the delay.
        with mock.patch("benchmark.http.requests.post",
                        side_effect=[self._mock_429("Sun, 06 Nov 1994 08:49:37 GMT"),
                                     self._mock_200()]):
            result = stream_request(self._cfg(), 5, "m", "S", "p", 100)
        self.assertIsNone(result.error)
        self.assertEqual(result.text, "ok")

    def test_invalid_date_falls_back_to_backoff(self):
        with mock.patch("benchmark.http.requests.post",
                        side_effect=[self._mock_429("not-a-date"), self._mock_200()]):
            result = stream_request(self._cfg(), 5, "m", "S", "p", 100)
        self.assertIsNone(result.error)
        self.assertEqual(result.text, "ok")


class TestBackoffTeardownAndCancellation(unittest.TestCase):
    """429 teardown-before-sleep and stop_event cancellation paths."""

    def _cfg(self):
        return {"S": {"api_url": "http://x/v1/chat/completions", "headers": {},
                       "max_429_retries": 1, "backoff_seconds": 0.01,
                       "backoff_factor": 1.0, "max_backoff_seconds": 1.0}}

    def _mock_429(self):
        m = mock.MagicMock()
        m.status_code = 429
        m.text = "slow down"
        m.headers = {}
        m.close = mock.MagicMock()
        return m

    def _mock_200(self):
        m = mock.MagicMock()
        m.status_code = 200
        m.text = "ok"
        m.headers = {}
        m.iter_lines = mock.MagicMock(return_value=iter([
            "data: " + json.dumps({"choices": [{"delta": {"content": "ok"},
                                                    "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]))
        m.close = mock.MagicMock()
        return m

    def test_watchdog_cancel_and_close_errors_are_swallowed(self):
        fake_timer = mock.MagicMock()
        fake_timer.cancel.side_effect = RuntimeError("cancel boom")
        with mock.patch("benchmark.http.threading.Timer", return_value=fake_timer), \
                mock.patch("benchmark.http.requests.post",
                           side_effect=[self._mock_429(), self._mock_200()]):
            # max_429_retries=1 with a 429 then a 200 -> teardown then
            # backoff sleep; the watchdogs cancel() raising must not
            # propagate.
            result = stream_request(self._cfg(), 5, "m", "S", "p", 100,
                                    stop_event=threading.Event())
        self.assertIsNone(result.error)
        self.assertEqual(result.text, "ok")

    def test_stop_event_during_backoff_yields_cancelled(self):
        class _FakeStop:
            def __init__(self):
                self.calls = 0

            def is_set(self):
                self.calls += 1
                # Pre-request check (first call) passes; the wait below
                # then reports the event as tripped mid-backoff.
                return self.calls > 1

            def wait(self, delay):
                return True

        with mock.patch("benchmark.http.requests.post", return_value=self._mock_429()):
            result = stream_request(self._cfg(), 5, "m", "S", "p", 100,
                                    stop_event=_FakeStop())
        self.assertEqual(result.error, "Cancelled")

    def test_resp_close_error_swallowed_in_finally(self):
        resp = mock.MagicMock()
        resp.status_code = 200
        resp.text = "ok"
        resp.headers = {}
        resp.iter_lines.return_value = []
        resp.close.side_effect = RuntimeError("close boom")
        with mock.patch("benchmark.http.requests.post", return_value=resp):
            result = stream_request(self._cfg(), 5, "m", "S", "p", 100)
        resp.close.assert_called()  # cleanup still attempted despite the raise


class TestRenderToolCallsBadArgs(unittest.TestCase):
    def test_non_json_arguments_fall_back_to_raw(self):
        rendered = _render_tool_calls([
            {"function": {"name": "f", "arguments": "not json"}},
        ])
        self.assertIn("not json", rendered)


class TestParseSseLineEdges(unittest.TestCase):
    def test_bad_json_payload_is_ignored(self):
        result = _parse_sse_line("data: {bad", None, "", "", None, {})
        self.assertFalse(result.done)
        self.assertIsNone(result.error)

    def test_error_dict_message_surface(self):
        result = _parse_sse_line(
            "data: " + json.dumps({"error": {"message": "mid-stream fail"}}),
            None, "", "", None, {})
        self.assertEqual(result.error, "mid-stream fail")

    def test_error_string_surface(self):
        result = _parse_sse_line(
            "data: " + json.dumps({"error": "boom"}),
            None, "", "", None, {})
        self.assertEqual(result.error, "boom")


class TestStreamErrorPaths(unittest.TestCase):
    def _cfg(self):
        return {"S": {"api_url": "http://x/v1/chat/completions", "headers": {}}}

    def _fake_ctx(self, result):
        @contextlib.contextmanager
        def fake_ctx(*a, **kw):
            yield result
        return fake_ctx

    def test_request_error_short_circuits(self):
        with mock.patch("benchmark.http._post_request_context",
                        self._fake_ctx(PostRequestResult(None, "boom", None))):
            result = stream_request(self._cfg(), 5, "m", "S", "p", 100)
        self.assertEqual(result.error, "boom")

    def test_no_response_reports_error(self):
        with mock.patch("benchmark.http._post_request_context",
                        self._fake_ctx(PostRequestResult(None, None, None))):
            result = stream_request(self._cfg(), 5, "m", "S", "p", 100)
        self.assertIn("no response", result.error)

    def test_stream_line_error_propagates(self):
        resp = mock.MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.close = mock.MagicMock()
        resp.iter_lines.return_value = iter([_StreamLineError("reset")])
        with mock.patch("benchmark.http.requests.post", return_value=resp):
            result = stream_request(self._cfg(), 5, "m", "S", "p", 100)
        self.assertIn("reset", result.error)

    def test_empty_line_skipped(self):
        resp = mock.MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.close = mock.MagicMock()
        resp.iter_lines.return_value = iter(["", "data: [DONE]"])
        with mock.patch("benchmark.http.requests.post", return_value=resp):
            result = stream_request(self._cfg(), 5, "m", "S", "p", 100)
        self.assertIsNone(result.error)

    def test_parse_exception_yields_sse_error(self):
        resp = mock.MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.close = mock.MagicMock()
        resp.iter_lines.return_value = iter(["data: x"])
        with mock.patch("benchmark.http.requests.post", return_value=resp), \
                mock.patch("benchmark.http._parse_sse_line",
                           side_effect=ValueError("bad")):
            result = stream_request(self._cfg(), 5, "m", "S", "p", 100)
        self.assertIn("SSE parse error", result.error)

    def test_on_chunk_raising_swallowed(self):
        resp = mock.MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.close = mock.MagicMock()
        resp.iter_lines.return_value = iter([
            "data: " + json.dumps({"choices": [{"delta": {"content": "hi"}}]}),
            "data: [DONE]",
        ])
        calls = []
        with mock.patch("benchmark.http.requests.post", return_value=resp):
            result = stream_request(
                self._cfg(), 5, "m", "S", "p", 100,
                on_chunk=lambda delta: (_ for _ in ()).throw(RuntimeError("observer")))
        self.assertEqual(result.text, "hi")
        self.assertEqual(calls, [])

    def test_on_think_chunk_raising_swallowed(self):
        resp = mock.MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.close = mock.MagicMock()
        resp.iter_lines.return_value = iter([
            "data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "think"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": "done"}}]}),
            "data: [DONE]",
        ])
        with mock.patch("benchmark.http.requests.post", return_value=resp):
            result = stream_request(
                self._cfg(), 5, "m", "S", "p", 100,
                on_think_chunk=lambda delta: (_ for _ in ()).throw(RuntimeError("observer")))
        self.assertEqual(result.text, "done")
        self.assertEqual(result.think_text, "think")

    def test_tool_calls_rendered_into_text(self):
        resp = mock.MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.close = mock.MagicMock()
        resp.iter_lines.return_value = iter([
            "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": "f", "arguments": "{}"}}]}}]}),
            "data: [DONE]",
        ])
        with mock.patch("benchmark.http.requests.post", return_value=resp):
            result = stream_request(self._cfg(), 5, "m", "S", "p", 100)
        self.assertIn("<tool_call>", result.text)


class TestNonStreamErrorPaths(unittest.TestCase):
    def _cfg(self):
        return {"S": {"api_url": "http://x/v1/chat/completions", "headers": {}}}

    def _fake_ctx(self, result):
        @contextlib.contextmanager
        def fake_ctx(*a, **kw):
            yield result
        return fake_ctx

    def test_request_error_short_circuits(self):
        with mock.patch("benchmark.http._post_request_context",
                        self._fake_ctx(PostRequestResult(None, "boom", None))):
            result = nonstream_request(self._cfg(), 5, "m", "S", "p", 100)
        self.assertEqual(result.error, "boom")

    def test_no_response_reports_error(self):
        with mock.patch("benchmark.http._post_request_context",
                        self._fake_ctx(PostRequestResult(None, None, None))):
            result = nonstream_request(self._cfg(), 5, "m", "S", "p", 100)
        self.assertIn("no response", result.error)

    def test_empty_body_reports_error(self):
        with mock.patch("benchmark.http._post_request_context",
                        self._fake_ctx(PostRequestResult(mock.MagicMock(), None, None))), \
                mock.patch("benchmark.http._read_response_body",
                           return_value=ResponseBodyResult(None, None)):
            result = nonstream_request(self._cfg(), 5, "m", "S", "p", 100)
        self.assertIn("Empty response body", result.error)

    def test_tool_calls_rendered_into_text(self):
        resp = MockResponse(text=json.dumps({
            "choices": [{"message": {
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
            }}],
        }), status_code=200)
        with mock.patch("benchmark.http.requests.post", return_value=resp):
            result = nonstream_request(self._cfg(), 5, "m", "S", "p", 100)
        self.assertIn("<tool_call>", result.text)


class TestGetActiveRequestCount(unittest.TestCase):
    def test_counts_and_clears(self):
        resp = mock.MagicMock()
        with mock.patch("benchmark.http._active_requests", {resp}):
            self.assertEqual(get_active_request_count(), 1)
        with mock.patch("benchmark.http._active_requests", set()):
            self.assertEqual(get_active_request_count(), 0)


if __name__ == "__main__":
    unittest.main()
