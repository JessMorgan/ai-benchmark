"""Tests for benchmark_http request helpers."""
import contextlib
import json
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from benchmark_http import fetch_models_v1, nonstream_request, stream_request


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
        the runtime (``benchmark_core._run_plugin_task``) relies on
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
            yield fake_response, None, None

        with mock.patch("benchmark_http._post_request_context", fake_ctx):
            text, _, _, err, _, _ = stream_request(
                {"src": {"api_url": "http://x", "headers": {}}},
                10, "m", "src", "p", 100,
                on_chunk=lambda delta: calls.append(delta),
            )
        self.assertEqual(err or "", "", "no error expected for a clean stream")
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
            text, first_tok, stream_end, err, finish_reason, usage = stream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )


        self.assertEqual(err, None)
        self.assertEqual(text, "Hello world")
        self.assertEqual(finish_reason, "stop")
        self.assertEqual(usage, {"prompt_tokens": 1, "completion_tokens": 1})

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
            text, first_tok, stream_end, err, finish_reason, usage = stream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10, stop_event=stop_event,
            )
            thread.join()

        self.assertEqual(err, "Cancelled")


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
            text, usage, gen_time, err, finish_reason = nonstream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )

        self.assertEqual(err, None)
        self.assertEqual(text, "Hello world")
        self.assertEqual(finish_reason, "stop")
        self.assertEqual(usage, {"prompt_tokens": 1, "completion_tokens": 2})

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
            text, usage, gen_time, err, finish_reason = nonstream_request(
                source_config, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10, stop_event=stop_event,
            )
            thread.join()

        self.assertEqual(err, "Cancelled")


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
            text, _, _, err, _, _ = stream_request(
                cfg, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )
        self.assertEqual(err, None)
        self.assertEqual(text, "hello")
        self.assertEqual(mp.call_count, 3)

    def test_429_exhausted_returns_error_string(self):
        """All attempts 429 should yield `(None, 'HTTP 429: ...')`."""
        cfg = self._cfg(max_429_retries=2)
        sequence = [self._mock_429()] * 3  # 2 retries + 1 final attempt
        with mock.patch("requests.post", side_effect=sequence):
            text, _, _, err, _, _ = stream_request(
                cfg, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )
        self.assertEqual(text, "")
        self.assertIn("HTTP 429: ", err)

    def test_max_429_retries_zero_disables_retry(self):
        """Explicit ``max_429_retries: 0`` is opt-out — fail fast on first 429."""
        cfg = self._cfg(max_429_retries=0)
        with mock.patch("requests.post", side_effect=[self._mock_429()]) as mp:
            text, _, _, err, _, _ = stream_request(
                cfg, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )
        self.assertEqual(mp.call_count, 1, "max_429_retries=0 must not retry")
        self.assertIn("HTTP 429: ", err)

    def test_default_max_429_retries_is_two(self):
        """When no per-source config is supplied, the default is 2 retries."""
        cfg_no_opt = {"Local": {
            "api_url": "http://localhost/chat/completions", "headers": {}}}
        sequence = [self._mock_429(), self._mock_429(), self._mock_200()]
        with mock.patch("requests.post", side_effect=sequence) as mp:
            text, _, _, err, _, _ = stream_request(
                cfg_no_opt, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )
        self.assertEqual(mp.call_count, 3, "default config: 3 attempts -> success on 3rd mock")
        self.assertEqual(err, None)
        self.assertEqual(text, "ok")

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
            text, _, _, err, _, _ = stream_request(
                cfg, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )
        elapsed = time.monotonic() - start
        self.assertEqual(err, None)
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
            text, _, _, err, _, _ = stream_request(
                cfg, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )
        elapsed = time.monotonic() - start
        self.assertEqual(err, None)
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
            text, _, _, err, _, _ = stream_request(
                cfg, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10,
            )
        elapsed = time.monotonic() - start
        self.assertEqual(err, None)
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
            text, _, _, err, _, _ = stream_request(
                cfg, timeout=5, model="m", source="Local",
                prompt="hi", max_tokens=10, stop_event=stop,
            )
        elapsed = time.monotonic() - start
        self.assertEqual(err, "Cancelled")
        self.assertLess(elapsed, 0.1, "must not sleep when stop_event is set")
        # Pre-loop cancellation short-circuits before any HTTP request fires.
        self.assertEqual(mp.call_count, 0)


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


if __name__ == "__main__":
    unittest.main()
