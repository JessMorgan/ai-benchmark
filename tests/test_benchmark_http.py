"""Tests for benchmark_http request helpers."""
import json
import threading
import time
import unittest
from unittest import mock

from benchmark_http import fetch_models_v1, nonstream_request, stream_request


class TestStreamRequest(unittest.TestCase):
    """Tests for the streaming request helper."""

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
