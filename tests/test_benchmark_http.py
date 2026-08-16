"""Tests for benchmark.http request helpers."""
import contextlib
import json
import shlex
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from typing import ClassVar
from unittest import mock

from benchmark.http import (
    PostRequestResult,
    build_curl_cmd,
    fetch_models_v1,
    get_429_stats,
    nonstream_request,
    stream_request,
)


class TestBuildCurlCmd(unittest.TestCase):
    """Tests for shell-safe curl command generation."""

    def test_payload_round_trips_through_shell_with_special_characters(self):
        prompt = "It's a test\\nwith \\\"quotes\\\" and 🦄."
        api_url = "http://localhost/v1/weird'path"
        headers = {
            "Authorization": "Bearer it's-secret",
            "Content-Type": "application/json; profile='custom'",
        }
        command = build_curl_cmd(
            model="test-model",
            prompt=prompt,
            max_tokens=10,
            stream=False,
            api_url=api_url,
            headers=headers,
        )

        parts = shlex.split(command)
        payload = parts[parts.index("-d") + 1]

        self.assertEqual(parts[parts.index("-X") + 2], api_url)
        authorization_index = parts.index("Authorization: Bearer it's-secret")
        self.assertEqual(parts[authorization_index], "Authorization: Bearer it's-secret")
        content_type_index = parts.index("Content-Type: application/json; profile='custom'")
        self.assertEqual(parts[content_type_index], "Content-Type: application/json; profile='custom'")
        self.assertEqual(json.loads(payload)["messages"][0]["content"], prompt)

    def test_request_body_includes_provider_parameters_in_curl(self):
        request_body = {
            "model": "judge",
            "messages": [{"role": "user", "content": "judge this"}],
            "max_tokens": 4096,
            "stream": False,
            "chat_template_kwargs": {"thinking_token_budget": 2048},
            "response_format": {"type": "json_object"},
        }
        command = build_curl_cmd(
            "judge", "judge this", 4096, False,
            "http://localhost/v1/chat/completions", {},
            request_body=request_body,
        )
        payload = shlex.split(command)[shlex.split(command).index("-d") + 1]
        self.assertEqual(json.loads(payload), request_body)


class TestStreamRequest(unittest.TestCase):
    """Tests for the streaming request helper."""

    def test_stream_request_calls_on_chunk_on_non_empty_deltas_only(self):
        """Pins the SSE-parse-layer wiring contract: ``on_chunk`` must
        fire ONLY when a parsed delta carries non-empty content. This
        is the operator's live counter trigger -- gauge against this
        expectation is that:
          * role-only deltas (`{"choices": [{"delta": {"role": "assistant"}}]}`)
            must NOT fire -- they don't carry content
          * heartbeat lines / `: heartbeat` / blank lines / `[DONE]`
            must NOT fire -- they're filtered inside ``_parse_sse_line``
          * malformed-JSON lines must NOT fire
          * content deltas fire ONCE each with the accumulated delta
            (joined across all ``choices`` in the same data event)

        The first-chunk marker called before ``add_bytes_received`` in
        the runtime (``benchmark.core._run_plugin_task``) relies on
        this gating: if it ever fires on a role-only delta, the cell
        would render `[streaming - 0 tok]` (or worse, the runtime would
        fire `mark_first_chunk_seen` then call `add_bytes_received``
        with ``len(delta) == 0`` -- and the current contract says 0
        is a no-op so the cells would stay bare).
        """
        calls = []
        fake_response = mock.MagicMock()
        fake_response.status_code = 200
        fake_response.__enter__ = lambda self_: self_
        fake_response.__exit__ = lambda self_, *a: None
        fake_response.iter_lines = mock.MagicMock(return_value=iter([
            "data: " + json.dumps({"choices": [{"delta": {"role": "assistant"}}]}),
            json.dumps({"choices": [{"delta": {"content": "Hello, "}}]}),  # no 'data: ' prefix -> ignored by parser
            "data: " + json.dumps({"choices": [{"delta": {"content": "world"}}]}),
            "data: [DONE]",
        ]))

        @contextlib.contextmanager
        def fake_ctx(*a, **kw):
            yield PostRequestResult(fake_response, None, None)

        with mock.patch("benchmark.http._post_request_context", fake_ctx):
            result = stream_request(
                {"src": {"api_url": "http://x", "headers": {}}},
                10, "m", "src", "p", 100,
                on_chunk=lambda delta: calls.append(delta),
            )
        self.assertEqual(result.error or "", "", "no error expected for a clean stream")
        # Only the two valid content lines should have fired the callback.
        self.assertEqual(calls, ["world"],
                         "on_chunk must fire only on parsed non-empty content deltas; "
                         "role-only deltas (1st line), non-`data:`-prefixed JSON (2nd line), "
                         "and `[DONE]` (4th line) are all skipped inside `_parse_sse_line`")

    def test_stream_request_returns_text_and_usage(self):
        """stream_request parses SSE chunks and returns the assembled text."""
        source_config = {"Local": {"api_url": "http://localhost/chat/completions", "headers": {}}}

        class MockResponse:
            status_code = 200

            def iter_lines(self, decode_unicode=False):
                yield "data: " + json.dumps({
                    "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                })
                yield "data: " + json.dumps({
                    "choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}],
                })
                yield "data: [DONE]"

            def close(self):
                pass

        with mock.patch("requests.post", return_value=MockResponse()):
            result = stream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )


        self.assertEqual(result.error, None)
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage, {"prompt_tokens": 1, "completion_tokens": 1})

    def test_stream_request_captures_native_tool_calls(self):
        """Native ``tool_calls`` SSE deltas are merged and rendered into text.

        Agent-style responses emit tool calls in ``delta.tool_calls`` (not
        ``delta.content``). Previously those deltas were ignored and the
        leg scored 0 with an empty response; now they must accumulate into
        the ``tool_calls`` field AND render as ``<tool_call>{...}</tool_call>``
        blocks in ``text`` so the tool-calling plugin can score them.
        """
        source_config = {"Local": {"api_url": "http://localhost/chat/completions", "headers": {}}}

        class MockResponse:
            status_code = 200

            def iter_lines(self, decode_unicode=False):
                # First fragment: id + type + function name.
                yield "data: " + json.dumps({
                    "choices": [{"delta": {
                        "tool_calls": [{
                            "index": 0, "id": "call_abc", "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }]
                    }, "finish_reason": None}]
                })
                # Second fragment: arguments chunk 1.
                yield "data: " + json.dumps({
                    "choices": [{"delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": "{\"location\": "}}]
                    }}]
                })
                # Third fragment: arguments chunk 2.
                yield "data: " + json.dumps({
                    "choices": [{"delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": "\"Tokyo\", \"unit\": \"celsius\"}"}}]
                    }}]
                })
                yield "data: [DONE]"

            def close(self):
                pass

        with mock.patch("requests.post", return_value=MockResponse()):
            result = stream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )

        self.assertEqual(result.error, None)
        # Merged arguments across fragments, rendered as the plugin format.
        self.assertIn("<tool_call>", result.text)
        self.assertIn('"name": "get_weather"', result.text)
        self.assertIn('"location": "Tokyo"', result.text)
        # The raw merged tool call is retained for diagnostics.
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0]["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(result.tool_calls[0]["function"]["arguments"]),
            {"location": "Tokyo", "unit": "celsius"},
        )

    def test_stream_request_tool_calls_appended_after_content(self):
        """Content and native tool calls in the same stream are both kept."""
        source_config = {"Local": {"api_url": "http://localhost/chat/completions", "headers": {}}}

        class MockResponse:
            status_code = 200

            def iter_lines(self, decode_unicode=False):
                yield "data: " + json.dumps({
                    "choices": [{"delta": {"content": "Let me check the weather."}, "finish_reason": None}]
                })
                yield "data: " + json.dumps({
                    "choices": [{"delta": {
                        "tool_calls": [{
                            "index": 0, "id": "call_xyz", "type": "function",
                            "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"},
                        }]
                    }, "finish_reason": "tool_calls"}]
                })
                yield "data: [DONE]"

            def close(self):
                pass

        with mock.patch("requests.post", return_value=MockResponse()):
            result = stream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )

        self.assertEqual(result.error, None)
        self.assertIn("Let me check the weather.", result.text)
        self.assertIn("<tool_call>", result.text)
        # Tool calls are appended AFTER content, never clobbering it.
        self.assertLess(result.text.index("Let me check"), result.text.index("<tool_call>"))

    def test_stream_request_tool_calls_are_scorable_by_plugin(self):
        """End-to-end: rendered tool calls score with the real plugin.

        The whole point of capturing native ``tool_calls`` deltas is that
        the tool-calling plugin (which regex-scans ``<tool_call>`` blocks)
        can score a response that emitted tools instead of text. This test
        drives a stream with two parallel tool calls through
        ``stream_request`` and asserts the assembled text earns a non-zero
        score from the actual plugin.
        """
        from plugins.challenges.tool_calling import ToolCallingPlugin

        source_config = {"Local": {"api_url": "http://localhost/chat/completions", "headers": {}}}

        class MockResponse:
            status_code = 200

            def iter_lines(self, decode_unicode=False):
                # Parallel tool calls: two indices interleaved by the server.
                yield "data: " + json.dumps({
                    "choices": [{"delta": {
                        "tool_calls": [{
                            "index": 0, "id": "call_a", "type": "function",
                            "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\", \"unit\": \"celsius\"}"},
                        }, {
                            "index": 1, "id": "call_b", "type": "function",
                            "function": {"name": "search_flights", "arguments": "{\"origin\": \"JFK\", \"destination\": \"Tokyo\", \"date\": \"2024-08-15\"}"},
                        }]
                    }, "finish_reason": "tool_calls"}]
                })
                yield "data: [DONE]"

            def close(self):
                pass

        with mock.patch("requests.post", return_value=MockResponse()):
            result = stream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )

        self.assertEqual(result.error, None)
        self.assertEqual(len(result.tool_calls), 2, "both parallel tool calls must be captured")
        # The rendered text must be scorable by the real plugin.
        score = ToolCallingPlugin().score(result.text)
        self.assertGreater(score, 0.0,
                           "rendered native tool calls should earn plugin points, not 0")

    def test_stream_request_surfaces_sse_error_line(self):
        """SSE ``{"error": …}`` payloads abort the stream with the error.

        litellm/Ollama signal a mid-reasoning connection drop (EOF) by
        emitting a final data line ``{"error": {...}}`` and closing the
        stream without ``[DONE]``. Previously ``_parse_sse_line`` ignored
        the payload (it only reads ``choices``), so the aborted stream
        looked like a clean empty completion: ``stream_ok=True``, score 0.
        Now the error must surface on the result so ``benchmark.core``
        records ``stream_ok=False`` + ``stream_error`` instead.
        """
        source_config = {"Local": {"api_url": "http://localhost/chat/completions", "headers": {}}}

        class MockResponse:
            status_code = 200

            def iter_lines(self, decode_unicode=False):
                # Model starts reasoning, then the backend dies with EOF.
                yield "data: " + json.dumps({
                    "choices": [{"delta": {"reasoning_content": "Let me think about the tools..."}, "finish_reason": None}]
                })
                yield "data: " + json.dumps({
                    "error": {"message": "litellm.APIConnectionError: Ollama_chatException - EOF", "type": None},
                })
                # No [DONE] sentinel -- stream just ends after the error.

            def close(self):
                pass

        with mock.patch("requests.post", return_value=MockResponse()):
            result = stream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )

        self.assertIsNotNone(result.error)
        self.assertIn("EOF", result.error)
        self.assertIn("Ollama_chatException", result.error)
        # Reasoning captured before the abort is preserved for think.txt.
        self.assertIn("Let me think", result.think_text)
        # No content was produced, and no finish_reason was ever seen.
        self.assertEqual(result.text, "")
        self.assertIsNone(result.finish_reason)

    def test_stream_request_surfaces_plain_error_string(self):
        """An SSE ``{"error": "string"}`` form is surfaced too."""
        source_config = {"Local": {"api_url": "http://localhost/chat/completions", "headers": {}}}

        class MockResponse:
            status_code = 200

            def iter_lines(self, decode_unicode=False):
                yield "data: " + json.dumps({"error": "upstream failure"})

            def close(self):
                pass

        with mock.patch("requests.post", return_value=MockResponse()):
            result = stream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )
        self.assertEqual(result.error, "upstream failure")

    def test_stream_request_without_tool_calls_unchanged(self):
        """Ordinary content streams keep an empty ``tool_calls`` field."""
        source_config = {"Local": {"api_url": "http://localhost/chat/completions", "headers": {}}}

        class MockResponse:
            status_code = 200

            def iter_lines(self, decode_unicode=False):
                yield "data: " + json.dumps({
                    "choices": [{"delta": {"content": "Hello"}, "finish_reason": "stop"}]
                })
                yield "data: [DONE]"

            def close(self):
                pass

        with mock.patch("requests.post", return_value=MockResponse()):
            result = stream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )
        self.assertEqual(result.text, "Hello")
        self.assertEqual(result.tool_calls, [])

    def test_stream_request_respects_stop_event(self):
        """stream_request returns 'Cancelled' when stop_event is set mid-stream."""
        source_config = {"Local": {"api_url": "http://localhost/chat/completions", "headers": {}}}
        stop_event = threading.Event()

        class SlowMockResponse:
            status_code = 200

            def iter_lines(self, decode_unicode=False):
                for _ in range(100):
                    yield "data: " + json.dumps({
                        "choices": [{"delta": {"content": "x"}, "finish_reason": None}],
                    })
                    time.sleep(0.01)

            def close(self):
                pass

        def set_stop_after_delay():
            time.sleep(0.05)
            stop_event.set()

        with mock.patch("requests.post", return_value=SlowMockResponse()):
            thread = threading.Thread(target=set_stop_after_delay)
            thread.start()
            result = stream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10, stop_event=stop_event,
            )
            thread.join()

        self.assertEqual(result.error, "Cancelled")


class TestNonstreamRequest(unittest.TestCase):
    """Tests for the non-streaming request helper."""

    def test_nonstream_request_returns_text_and_usage(self):
        """nonstream_request parses a JSON response body."""
        source_config = {"Local": {"api_url": "http://localhost/chat/completions", "headers": {}}}

        body = json.dumps({
            "choices": [{"message": {"content": "Hello world"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        })

        class MockResponse:
            status_code = 200

            def iter_content(self, chunk_size=8192):
                yield body.encode("utf-8")

            def close(self):
                pass

        with mock.patch("requests.post", return_value=MockResponse()):
            result = nonstream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )

        self.assertEqual(result.error, None)
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage, {"prompt_tokens": 1, "completion_tokens": 2})

    def test_nonstream_request_captures_native_tool_calls(self):
        """Non-streaming ``message.tool_calls`` render into text + field."""
        source_config = {"Local": {"api_url": "http://localhost/chat/completions", "headers": {}}}

        body = json.dumps({
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {
                            "name": "search_flights",
                            "arguments": "{\"origin\": \"JFK\", \"destination\": \"Tokyo\", \"date\": \"2024-08-15\"}",
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        })

        class MockResponse:
            status_code = 200

            def iter_content(self, chunk_size=8192):
                yield body.encode("utf-8")

            def close(self):
                pass

        with mock.patch("requests.post", return_value=MockResponse()):
            result = nonstream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )

        self.assertEqual(result.error, None)
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertIn("<tool_call>", result.text)
        self.assertIn('"name": "search_flights"', result.text)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0]["function"]["name"], "search_flights")

    def test_nonstream_request_respects_stop_event(self):
        """nonstream_request returns 'Cancelled' when stop_event is set mid-read."""
        source_config = {"Local": {"api_url": "http://localhost/chat/completions", "headers": {}}}
        stop_event = threading.Event()

        class SlowMockResponse:
            status_code = 200

            def iter_content(self, chunk_size=8192):
                for _ in range(100):
                    yield b'{"choices":[{"message":{"content":"x"}}]}'
                    time.sleep(0.01)

            def close(self):
                pass

        def set_stop_after_delay():
            time.sleep(0.05)
            stop_event.set()

        with mock.patch("requests.post", return_value=SlowMockResponse()):
            thread = threading.Thread(target=set_stop_after_delay)
            thread.start()
            result = nonstream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10, stop_event=stop_event,
            )
            thread.join()

        self.assertEqual(result.error, "Cancelled")


class TestRateLimitRetries(unittest.TestCase):
    """Tests for HTTP 429 retry/backoff in `_post_request_context`.

    These tests use tiny backoff values (`backoff_seconds=0.01`) so a test
    run completes in well under a second even when 2 retries fire.
    """

    def _cfg(self, **overrides):
        base = {
            "api_url": "http://localhost/chat/completions",
            "headers": {},
            "backoff_seconds": 0.01,
            "backoff_factor": 1.0,
            "max_backoff_seconds": 1.0,
            "max_429_retries": 2,
        }
        base.update(overrides)
        return {"Local": base}

    def _mock_429(self, retry_after=None, body="rate limited"):
        m = mock.MagicMock()
        m.status_code = 429
        m.text = body
        m.headers = {}
        if retry_after is not None:
            m.headers["Retry-After"] = str(retry_after)
        m.close = mock.MagicMock()
        return m

    def _mock_200(self, content="ok"):
        m = mock.MagicMock()
        m.status_code = 200
        m.text = content
        m.headers = {}
        m.iter_lines = mock.MagicMock(return_value=iter([
            "data: " + json.dumps({
                "choices": [{"delta": {"content": content}, "finish_reason": "stop"}],
            }),
            "data: [DONE]",
        ]))
        m.iter_content = mock.MagicMock(return_value=iter([
            json.dumps({"choices": [{"message": {"content": content},
                                    "finish_reason": "stop"}]}).encode("utf-8"),
        ]))
        m.close = mock.MagicMock()
        return m

    def test_429_eventually_succeeds(self):
        """Two consecutive 429s then a 200 must yield the eventual OK."""
        cfg = self._cfg(max_429_retries=2)
        sequence = [self._mock_429(), self._mock_429(), self._mock_200("hello")]
        with mock.patch("requests.post", side_effect=sequence) as mp:
            result = stream_request(
                cfg, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )
        self.assertEqual(result.error, None)
        self.assertEqual(result.text, "hello")
        self.assertEqual(mp.call_count, 3)

    def test_429_exhausted_returns_error_string(self):
        """All attempts 429 should yield `(None, 'HTTP 429: ...')`."""
        cfg = self._cfg(max_429_retries=2)
        sequence = [self._mock_429()] * 3  # 2 retries + 1 final attempt
        with mock.patch("requests.post", side_effect=sequence):
            result = stream_request(
                cfg, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )
        self.assertEqual(result.text, "")
        self.assertIn("HTTP 429: ", result.error)

    def test_max_429_retries_zero_disables_retry(self):
        """Explicit ``max_429_retries: 0`` is opt-out — fail fast on first 429."""
        cfg = self._cfg(max_429_retries=0)
        with mock.patch("requests.post", side_effect=[self._mock_429()]) as mp:
            result = stream_request(
                cfg, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )
        self.assertEqual(mp.call_count, 1, "max_429_retries=0 must not retry")
        self.assertIn("HTTP 429: ", result.error)

    def test_default_max_429_retries_is_two(self):
        """When no per-source config is supplied, the default is 2 retries."""
        cfg_no_opt = {"Local": {
            "api_url": "http://localhost/chat/completions", "headers": {}}}
        sequence = [self._mock_429(), self._mock_429(), self._mock_200()]
        # The factory defaults (backoff_seconds=30, factor 2.0) would make the
        # two real retries sleep ~90s wall-clock. The point of this test is the
        # DEFAULT RETRY COUNT, not the backoff timing, so mock the sleep: the
        # delay math still runs (30s, then 60s) but returns instantly.
        with mock.patch("benchmark.http.time.sleep"), \
             mock.patch("requests.post", side_effect=sequence) as mp:
            result = stream_request(
                cfg_no_opt, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )
        self.assertEqual(mp.call_count, 3, "default config: 3 attempts -> success on 3rd mock")
        self.assertEqual(result.error, None)
        self.assertEqual(result.text, "ok")

    def test_retry_after_header_beats_computed_delay(self):
        """Retry-After: 0.5s must beat the computed 0.01s floor.

        We pin both bounds so a regression that silently switches in the
        factory-default 30s floor would FAIL loudly.
        """
        cfg = self._cfg(max_429_retries=1, backoff_seconds=0.01,
                        backoff_factor=1.0, max_backoff_seconds=5.0)
        start = time.monotonic()
        with mock.patch("requests.post",
                        side_effect=[self._mock_429(retry_after=0.5), self._mock_200()]):
            result = stream_request(
                cfg, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )
        elapsed = time.monotonic() - start
        self.assertEqual(result.error, None)
        self.assertGreaterEqual(elapsed, 0.45,
                                f"Retry-After:0.5s should beat the 0.01s floor; got {elapsed:.2f}s")
        self.assertLess(elapsed, 1.0,
                        f"sanity upper bound — if 30s factory default leaked in we'd be here; got {elapsed:.2f}s")

    def test_retry_after_http_date_form_is_parsed(self):
        """Retry-After can also be RFC 7231 §7.1.3 form-2: an HTTP-date ~5s from now."""
        future = (datetime.now(timezone.utc) + timedelta(seconds=5)).strftime(
            "%a, %d %b %Y %H:%M:%S GMT")
        cfg = self._cfg(max_429_retries=1, backoff_seconds=0.01,
                        backoff_factor=1.0, max_backoff_seconds=30.0)
        start = time.monotonic()
        with mock.patch("requests.post",
                        side_effect=[self._mock_429(retry_after=future), self._mock_200()]):
            result = stream_request(
                cfg, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )
        elapsed = time.monotonic() - start
        self.assertEqual(result.error, None)
        # HTTP-date ~5s ahead, no jitter because Retry-After present.
        self.assertGreaterEqual(elapsed, 4.0,
                                f"HTTP-date Retry-After:~5s should be ~5s; got {elapsed:.2f}s")
        self.assertLess(elapsed, 6.5,
                        f"upper bound; got {elapsed:.2f}s")

    def test_429_exponential_growth_increases_delay(self):
        """backoff_factor=10 means second sleep ~10x the first.

        Both bounds asserted so a regression that swaps ``time.sleep`` for
        ``stop_event.wait`` (or back) will fail loudly either way.
        """
        cfg = self._cfg(max_429_retries=2, backoff_seconds=0.01,
                        backoff_factor=10.0, max_backoff_seconds=10.0)
        start = time.monotonic()
        with mock.patch("requests.post",
                        side_effect=[self._mock_429(), self._mock_429(), self._mock_200()]):
            result = stream_request(
                cfg, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )
        elapsed = time.monotonic() - start
        self.assertEqual(result.error, None)
        # delay #1 ~= 0.012s (jitter 0.8-1.2x), delay #2 ~= 0.10s.
        # Without growth both come to ~0.024s; with growth total ~0.10-0.13s.
        self.assertGreaterEqual(elapsed, 0.05,
                                f"second sleep should be ~10x the first; total={elapsed:.2f}s")
        self.assertLess(elapsed, 0.3,
                        f"sanity upper bound — not silently 30s; got {elapsed:.2f}s")

    def test_stop_event_aborts_retry(self):
        """If stop_event is set before retry, we should bail out immediately."""
        stop = threading.Event()
        stop.set()  # pre-cancel
        cfg = self._cfg(max_429_retries=2)
        start = time.monotonic()
        with mock.patch("requests.post", side_effect=[self._mock_429()]) as mp:
            result = stream_request(
                cfg, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10, stop_event=stop,
            )
        elapsed = time.monotonic() - start
        self.assertEqual(result.error, "Cancelled")
        self.assertLess(elapsed, 0.1, "must not sleep when stop_event is set")
        # Pre-loop cancellation short-circuits before any HTTP request fires.
        self.assertEqual(mp.call_count, 0)

    def test_429_tracks_per_plugin_pid(self):
        """When a plugin request hits 429, the sleeping key includes the pid."""
        cfg = self._cfg(max_429_retries=1)
        sequence = [self._mock_429(), self._mock_200("ok")]
        captured = {}

        def record_set(source, model, pid, wake_ts, attempts, max_attempts, delay):
            captured["key"] = (source, model, pid)

        with (
            mock.patch("benchmark.http._set_429_sleep", side_effect=record_set),
            mock.patch("requests.post", side_effect=sequence),
        ):
            result = stream_request(
                cfg, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10, pid="rate-limiter",
            )
        self.assertEqual(result.error, None)
        self.assertEqual(result.text, "ok")
        self.assertEqual(captured["key"], ("Local", "m", "rate-limiter"))

    def test_429_tracks_per_plugin_stats(self):
        """get_429_stats returns aggregate retry attempts and sleep time per plugin."""
        from benchmark.http import _set_429_sleep, reset_429_stats

        reset_429_stats()
        self.addCleanup(reset_429_stats)
        # Simulate two 429 sleeps for the same plugin, one for another plugin.
        _set_429_sleep("Local", "m1", "rate-limiter", time.time() + 0.1, 1, 3, 0.5)
        _set_429_sleep("Local", "m1", "rate-limiter", time.time() + 0.2, 2, 3, 1.5)
        _set_429_sleep("Local", "m2", "json-formatter", time.time() + 0.3, 1, 3, 2.0)

        stats = get_429_stats()
        self.assertEqual(stats["total_retries"], 3)
        per_plugin = stats["plugin_stats"]
        self.assertEqual(per_plugin["rate-limiter"]["retries"], 2)
        self.assertEqual(per_plugin["rate-limiter"]["total_sleep_time"], 2.0)
        self.assertEqual(per_plugin["json-formatter"]["retries"], 1)
        self.assertEqual(per_plugin["json-formatter"]["total_sleep_time"], 2.0)

    def test_429_retry_resets_per_request_elapsed(self):
        """Each retry attempt resets the per-request start timestamp so the
        TUI shows elapsed time for the current request, not cumulative
        time across all attempts and sleeps."""
        cfg = self._cfg(max_429_retries=1, backoff_seconds=0.01,
                         backoff_factor=1.0, max_backoff_seconds=1.0)
        sequence = [self._mock_429(), self._mock_200("ok")]
        retry_calls = []

        def on_retry():
            retry_calls.append(time.monotonic())

        start = time.monotonic()
        with mock.patch("requests.post", side_effect=sequence):
            result = stream_request(
                cfg, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10, pid="rate-limiter",
                on_retry=on_retry,
            )
        elapsed = time.monotonic() - start
        self.assertEqual(result.error, None)
        self.assertEqual(result.text, "ok")
        # The retry callback should fire exactly once, between the first
        # request and the retry, after the 429 sleep.
        self.assertEqual(len(retry_calls), 1, "on_retry must fire once per retry")
        self.assertGreater(retry_calls[0], start)
        self.assertLess(retry_calls[0], start + elapsed)

    def test_429_retry_fires_for_nonstream(self):
        """Non-streaming requests also invoke on_retry when they retry."""
        retry_calls = []
        source_config = {
            "Local": {
                "api_url": "http://x",
                "headers": {},
                "max_429_retries": 1,
                "backoff_seconds": 0.01,
                "backoff_factor": 1.0,
                "max_backoff_seconds": 1.0,
            }
        }

        class _Resp429:
            status_code = 429
            text = "rate limited"
            headers: ClassVar[dict] = {}

            def close(self):
                pass

        def _mk_200():
            _body = {
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {},
            }

            class _Resp200:
                status_code = 200
                text = json.dumps(_body)
                headers: ClassVar[dict] = {}
                body = _body

                def iter_content(self, chunk_size=8192):
                    return [json.dumps(self.body).encode()]

                def json(self):
                    return self.body

                def close(self):
                    pass

            return _Resp200()

        from benchmark.http import nonstream_request

        with mock.patch("requests.post", side_effect=[_Resp429(), _mk_200()]):
            nonstream_request(
                source_config,
                timeout=5,
                model="m",
                source="Local",
                prompt="hi",
                max_tokens=10,
                on_retry=lambda: retry_calls.append(True),
            )

        self.assertEqual(len(retry_calls), 1, "on_retry must fire once per retry")


class TestSystemPrompt(unittest.TestCase):
    """Tests for system prompt handling in request bodies."""

    def test_nonstream_request_includes_system_prompt(self):
        """nonstream_request prepends a system message when system_prompt is provided."""
        source_config = {"Local": {"api_url": "http://localhost/chat/completions", "headers": {}}}
        captured = {}

        class MockResponse:
            status_code = 200

            def iter_content(self, chunk_size=8192):
                body = json.dumps({
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                })
                yield body.encode("utf-8")

            def close(self):
                pass

        def fake_post(url, **kwargs):
            captured["body"] = kwargs.get("json")
            return MockResponse()

        with mock.patch("requests.post", side_effect=fake_post):
            nonstream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10, system_prompt="You are a coder.",
            )

        self.assertIn("body", captured)
        messages = captured["body"]["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "You are a coder."})
        self.assertEqual(messages[1], {"role": "user", "content": "hi"})

    def test_nonstream_request_includes_provider_request_params(self):
        """Provider-specific nested request parameters are sent unchanged."""
        source_config = {"Local": {"api_url": "http://localhost/chat/completions", "headers": {}}}
        captured = {}
        request_params = {
            "chat_template_kwargs": {"thinking_token_budget": 2048},
            "response_format": {"type": "json_object"},
        }

        class MockResponse:
            status_code = 200

            def iter_content(self, chunk_size=8192):
                body = json.dumps({
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                })
                yield body.encode("utf-8")

            def close(self):
                pass

        def fake_post(url, **kwargs):
            captured["body"] = kwargs.get("json")
            return MockResponse()

        with tempfile.TemporaryDirectory() as tmp:
            log_path = f"{tmp}/judge.log"
            with mock.patch("requests.post", side_effect=fake_post):
                nonstream_request(
                    source_config, timeout=5, model="m", source="Local",
                    prompt="hi", max_tokens=4096, request_params=request_params,
                    log_path=log_path,
                )

            with open(log_path, encoding="utf-8") as handle:
                logged_command = handle.read()
            logged_payload = shlex.split(logged_command)[
                shlex.split(logged_command).index("-d") + 1
            ]

        self.assertEqual(captured["body"]["chat_template_kwargs"], {
            "thinking_token_budget": 2048,
        })
        self.assertEqual(captured["body"]["response_format"], {
            "type": "json_object",
        })
        self.assertEqual(json.loads(logged_payload), captured["body"])

    def test_nonstream_request_no_system_prompt_when_none(self):
        """nonstream_request only includes a user message when no system_prompt is provided."""
        source_config = {"Local": {"api_url": "http://localhost/chat/completions", "headers": {}}}
        captured = {}

        class MockResponse:
            status_code = 200

            def iter_content(self, chunk_size=8192):
                body = json.dumps({
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                })
                yield body.encode("utf-8")

            def close(self):
                pass

        def fake_post(url, **kwargs):
            captured["body"] = kwargs.get("json")
            return MockResponse()

        with mock.patch("requests.post", side_effect=fake_post):
            nonstream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )

        self.assertIn("body", captured)
        messages = captured["body"]["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0], {"role": "user", "content": "hi"})


class TestFetchModelsV1(unittest.TestCase):
    """Tests for the model discovery helper."""

    def test_fetch_models_v1_returns_ids(self):
        """fetch_models_v1 returns model IDs from the /v1/models endpoint."""
        response_data = {
            "data": [
                {"id": "model-a"},
                {"id": "model-b"},
            ]
        }

        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return response_data

        with mock.patch("requests.get", return_value=MockResponse()):
            result = fetch_models_v1("http://localhost")

        self.assertEqual(result, ["model-a", "model-b"])

    def test_fetch_models_v1_skips_entries_without_id(self):
        """fetch_models_v1 ignores entries missing an 'id' field."""
        response_data = {
            "data": [
                {"id": "model-a"},
                {"object": "model"},
            ]
        }

        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return response_data

        with mock.patch("requests.get", return_value=MockResponse()):
            result = fetch_models_v1("http://localhost")

        self.assertEqual(result, ["model-a"])

    def test_fetch_models_v1_adds_api_key_header(self):
        """fetch_models_v1 adds the Authorization header when an API key is provided."""
        captured = {}

        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": []}

        def fake_get(url, **kwargs):
            captured["headers"] = kwargs.get("headers")
            return MockResponse()

        with mock.patch("requests.get", side_effect=fake_get):
            fetch_models_v1("http://localhost", api_key="secret")

        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")


class TestOneMinProtocol(unittest.TestCase):
    """Tests for the 1min.ai native ``api_protocol: 1min`` handling."""

    _ONEMIN_CFG: ClassVar[dict] = {
        "1min": {
            "api_protocol": "1min",
            "api_url": "https://api.1min.ai/api/chat-with-ai",
            "headers": {"API-KEY": "secret", "Content-Type": "application/json"},
        }
    }

    def test_1min_nonstream_builds_native_body_and_parses_result(self):
        """1min sources send the native body and read ``aiRecord.resultObject``."""
        captured = {}

        class MockResponse:
            status_code = 200

            def iter_content(self, chunk_size=8192):
                yield json.dumps({
                    "aiRecord": {
                        "status": "SUCCESS",
                        "aiRecordDetail": {"resultObject": ["Hello", " world"]},
                    }
                }).encode("utf-8")

            def close(self):
                pass

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["body"] = kwargs.get("json")
            return MockResponse()

        with mock.patch("requests.post", side_effect=fake_post):
            result = nonstream_request(
                self._ONEMIN_CFG, timeout=5, model="gpt-4o-mini", source="1min",
                prompt="hi", max_tokens=10,
            )

        self.assertIsNone(result.error)
        self.assertEqual(result.text, "Hello\n world")
        self.assertEqual(captured["url"], "https://api.1min.ai/api/chat-with-ai")
        self.assertEqual(captured["body"], {
            "type": "UNIFY_CHAT_WITH_AI",
            "model": "gpt-4o-mini",
            "promptObject": {"prompt": "hi"},
        })

    def test_1min_stream_appends_isStreaming_and_parses_named_events(self):
        """1min streaming uses ``?isStreaming=true`` and named SSE events."""
        captured = {}

        class MockResponse:
            status_code = 200

            def iter_lines(self, decode_unicode=False):
                yield "event: content"
                yield "data: " + json.dumps({"content": "Hel"})
                yield "event: content"
                yield "data: " + json.dumps({"content": "lo"})
                yield "event: done"
                yield "data: " + json.dumps({"message": "Stream completed"})

            def close(self):
                pass

        def fake_post(url, **kwargs):
            captured["url"] = url
            return MockResponse()

        with mock.patch("requests.post", side_effect=fake_post):
            result = stream_request(
                self._ONEMIN_CFG, timeout=5, model="gpt-4o-mini", source="1min",
                prompt="hi", max_tokens=10,
            )

        self.assertIsNone(result.error)
        self.assertEqual(result.text, "Hello")
        self.assertIn("isStreaming=true", captured["url"])

    def test_1min_nonstream_error_body_surfaces_message(self):
        """A 1min ``{"success": false}`` body surfaces its error message."""
        class MockResponse:
            status_code = 200

            def iter_content(self, chunk_size=8192):
                yield json.dumps({
                    "success": False,
                    "error": {"code": "RATE_LIMITED", "message": "Too many requests"},
                }).encode("utf-8")

            def close(self):
                pass

        with mock.patch("requests.post", return_value=MockResponse()):
            result = nonstream_request(
                self._ONEMIN_CFG, timeout=5, model="gpt-4o-mini", source="1min",
                prompt="hi", max_tokens=10,
            )

        self.assertIsNotNone(result.error)
        self.assertIn("Too many requests", result.error)

    def test_1min_stream_error_event_surfaces(self):
        """A 1min ``event: error`` aborts the stream with the message."""
        class MockResponse:
            status_code = 200

            def iter_lines(self, decode_unicode=False):
                yield "event: error"
                yield "data: " + json.dumps({"message": "boom"})

            def close(self):
                pass

        with mock.patch("requests.post", return_value=MockResponse()):
            result = stream_request(
                self._ONEMIN_CFG, timeout=5, model="m", source="1min",
                prompt="hi", max_tokens=10,
            )

        self.assertIsNotNone(result.error)
        self.assertIn("boom", result.error)

    def test_1min_folds_system_prompt_into_prompt(self):
        """1min has no system-message field, so the persona is folded in."""
        captured = {}

        class MockResponse:
            status_code = 200

            def iter_content(self, chunk_size=8192):
                yield json.dumps({
                    "aiRecord": {
                        "status": "SUCCESS",
                        "aiRecordDetail": {"resultObject": ["ok"]},
                    }
                }).encode("utf-8")

            def close(self):
                pass

        def fake_post(url, **kwargs):
            captured["body"] = kwargs.get("json")
            return MockResponse()

        with mock.patch("requests.post", side_effect=fake_post):
            nonstream_request(
                self._ONEMIN_CFG, timeout=5, model="gpt-4o-mini", source="1min",
                prompt="hi", max_tokens=10, system_prompt="You are a coder.",
            )

        self.assertEqual(
            captured["body"]["promptObject"]["prompt"],
            "You are a coder.\n\nhi",
        )

    def test_api_protocol_defaults(self):
        """Only ``api_protocol: 1min`` opts out of the OpenAI format."""
        from benchmark.http import _api_protocol
        self.assertEqual(_api_protocol(None), "openai")
        self.assertEqual(_api_protocol({}), "openai")
        self.assertEqual(_api_protocol({"api_protocol": "openai"}), "openai")
        self.assertEqual(_api_protocol({"api_protocol": "1min"}), "1min")

    def test_parse_1min_result_branches(self):
        """Every resultObject/error shape is mapped deterministically."""
        from benchmark.http import _parse_1min_result
        text, err = _parse_1min_result({
            "aiRecord": {"status": "SUCCESS", "aiRecordDetail": {"resultObject": "plain"}},
        })
        self.assertEqual((text, err), ("plain", None))
        text, err = _parse_1min_result({
            "aiRecord": {"status": "SUCCESS", "aiRecordDetail": {"resultObject": {"a": 1}}},
        })
        self.assertEqual(text, '{"a": 1}')
        text, err = _parse_1min_result({"aiRecord": {"status": "SUCCESS"}})
        self.assertEqual((text, err), ("", None))
        text, err = _parse_1min_result({})
        self.assertEqual(text, "")
        self.assertIn("missing aiRecord", err)
        text, err = _parse_1min_result({"aiRecord": {"status": "FAILED"}})
        self.assertIn("FAILED", err)
        text, err = _parse_1min_result({"success": False, "error": "nope"})
        self.assertIn("nope", err)

    def test_1min_sse_events_malformed_and_unnamed(self):
        """Malformed JSON data lines are skipped; unnamed data uses 'message'."""
        from benchmark.http import _iter_1min_sse_events

        class MockResponse:
            def iter_lines(self, decode_unicode=False):
                yield "data: not-json"
                yield "data: " + json.dumps({"content": "unnamed"})

        self.assertEqual(
            list(_iter_1min_sse_events(MockResponse())),
            [("message", {"content": "unnamed"})],
        )

    def test_1min_sse_events_transport_error(self):
        """Iterator failures surface as a stream-line error sentinel."""
        from benchmark.http import _iter_1min_sse_events, _StreamLineError

        class MockResponse:
            def iter_lines(self, decode_unicode=False):
                def boom():
                    raise RuntimeError("dropped")
                    yield
                yield from boom()

        events = list(_iter_1min_sse_events(MockResponse()))
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], _StreamLineError)

    def test_1min_nonstream_invalid_json(self):
        """A non-JSON 1min body produces an Invalid 1min response error."""
        class MockResponse:
            status_code = 200

            def iter_content(self, chunk_size=8192):
                yield b"not json"

            def close(self):
                pass

        with mock.patch("requests.post", return_value=MockResponse()):
            result = nonstream_request(
                self._ONEMIN_CFG, timeout=5, model="m", source="1min",
                prompt="hi", max_tokens=10,
            )
        self.assertIsNotNone(result.error)
        self.assertIn("Invalid 1min response", result.error)

    def test_1min_stream_respects_stop_event(self):
        """A mid-stream stop_event aborts a 1min stream as Cancelled."""
        stop_event = threading.Event()

        class SlowMockResponse:
            status_code = 200

            def iter_lines(self, decode_unicode=False):
                for _ in range(100):
                    yield "event: content"
                    yield "data: " + json.dumps({"content": "x"})
                    time.sleep(0.01)

            def close(self):
                pass

        def set_stop_after_delay():
            time.sleep(0.05)
            stop_event.set()

        with mock.patch("requests.post", return_value=SlowMockResponse()):
            thread = threading.Thread(target=set_stop_after_delay)
            thread.start()
            result = stream_request(
                self._ONEMIN_CFG, timeout=5, model="m", source="1min",
                prompt="hi", max_tokens=10, stop_event=stop_event,
            )
            thread.join()

        self.assertEqual(result.error, "Cancelled")

    def test_curl_command_includes_api_key_header(self):
        """curl commands render every header, including 1min's API-KEY."""
        command = build_curl_cmd(
            model="gpt-4o-mini", prompt="hi", max_tokens=10, stream=False,
            api_url="https://api.1min.ai/api/chat-with-ai",
            headers={"API-KEY": "secret", "Content-Type": "application/json"},
        )
        self.assertIn("API-KEY: secret", command)
        self.assertIn("Content-Type: application/json", command)


if __name__ == "__main__":
    unittest.main()
