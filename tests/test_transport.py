import unittest
from unittest import mock

from benchmark.http import NonStreamResult, StreamResult
from benchmark.observer import TaskObserver
from benchmark.opencode import OpenCodeProcessResult
from benchmark.transport import (
    RequestIdentity,
    RetryPolicy,
    TransportRequest,
    execute_task,
    execute_transport,
)
from benchmark.transport_options import (
    HTTPTransportOptions,
    OpenCodeTransportOptions,
    PiTransportOptions,
    TransportOptions,
)


class TestTransport(unittest.TestCase):
    def _request(self, **overrides):
        values = {
            "prompt": "Answer the task.",
            "max_tokens": 128,
            "source_config": {"Local": {"api_url": "http://localhost/v1/chat/completions"}},
            "api_model": "model",
            "source": "Local",
            "timeout": 5,
            "observer": TaskObserver.noop(),
        }
        values.update(overrides)
        return TransportRequest(**values)

    def test_request_identity_is_stable_and_surfaces_on_result(self):
        identity = RequestIdentity(
            run_id="run-1", revision_id=3, target="model-a",
            plugin="rate-limiter", runner="http", attempt=2,
        )
        self.assertEqual(identity.request_id, "run-1:3:model-a:rate-limiter:http:2")
        response = StreamResult("answer", "", 1.0, 1.5, None, "stop", {})
        with mock.patch("benchmark.transport.stream_request", return_value=response):
            result = execute_transport(self._request(identity=identity, attempt=2))
        self.assertEqual(result.request_id, identity.request_id)
        self.assertEqual(result.timeout_seconds, 5)

    def test_logical_retry_updates_request_identity_attempt(self):
        first = StreamResult("", "", None, 1.0, "connection refused", None, {})
        second = StreamResult("answer", "", 1.0, 1.5, None, "stop", {})
        identity = RequestIdentity(
            run_id="run-1", revision_id=3, target="model-a",
            plugin="rate-limiter", runner="http", attempt=1,
        )
        with mock.patch("benchmark.transport.stream_request", side_effect=[first, second]):
            result = execute_task(
                self._request(identity=identity),
                retry_policy=RetryPolicy(max_attempts=2),
                base_prompt="Answer the task.",
            )
        self.assertEqual(result.attempts[0].result.request_id, "run-1:3:model-a:rate-limiter:http:1")
        self.assertEqual(result.attempts[1].result.request_id, "run-1:3:model-a:rate-limiter:http:2")

    def test_grouped_transport_options_select_runner_specific_fields(self):
        request = self._request(
            options=TransportOptions(
                http=HTTPTransportOptions(supports_streaming=False, request_params={"x": 1}),
                opencode=OpenCodeTransportOptions(config_path="oc.json", model="provider/model"),
                pi=PiTransportOptions(worker="worker.mjs", target_key="target"),
            ),
            supports_streaming=True,
            request_params=None,
            opencode_config_path=None,
            opencode_model=None,
            pi_worker=None,
        )
        self.assertFalse(request.http_options().supports_streaming)
        self.assertEqual(request.http_options().request_params, {"x": 1})
        self.assertEqual(request.opencode_options().config_path, "oc.json")
        self.assertEqual(request.pi_options().worker, "worker.mjs")

    def test_typed_request_variants_are_transport_specific(self):
        request = self._request(options=HTTPTransportOptions(supports_streaming=False))
        self.assertEqual(request.kind, "http")
        self.assertFalse(request.options.supports_streaming)


    def test_streaming_http_is_normalized(self):
        response = StreamResult(
            "answer", "thinking", 1.0, 1.5, None, "stop", {"usage": 1},
        )
        with mock.patch("benchmark.transport.stream_request", return_value=response):
            result = execute_transport(self._request())

        self.assertEqual(result.text, "answer")
        self.assertEqual(result.think_text, "thinking")
        self.assertTrue(result.stream_ok)
        self.assertEqual(result.response_nature, "completed")
        self.assertEqual(result.thinking_tokens, 2)
        self.assertEqual(len(result.prompt_sha256), 64)

    def test_nonstream_schema_grammar_failure_falls_back_to_json_object(self):
        first = NonStreamResult(
            "", "", {}, 0.1,
            "HTTP 500: failed to initialize samplers: grammar sampler", None,
        )
        second = NonStreamResult('{"ok": true}', "", {}, 0.2, None, "stop")
        params = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": {}},
            }
        }
        with mock.patch(
            "benchmark.transport.nonstream_request", side_effect=[first, second],
        ) as request:
            result = execute_transport(self._request(
                supports_streaming=False, request_params=params,
            ))

        self.assertTrue(result.schema_fallback_used)
        self.assertIn("failed to initialize", result.schema_fallback_error)
        self.assertEqual(params["response_format"], {"type": "json_object"})
        self.assertEqual(result.text, '{"ok": true}')
        self.assertEqual(request.call_count, 2)

    def test_streaming_provider_rejection_falls_back_without_partial_output(self):
        streamed = StreamResult(
            "", "", None, 1.0, "HTTP 400: streaming is not supported", None, {},
        )
        buffered = NonStreamResult("answer", "", {}, 0.2, None, "stop")
        with (
            mock.patch("benchmark.transport.stream_request", return_value=streamed),
            mock.patch("benchmark.transport.nonstream_request", return_value=buffered) as request,
        ):
            result = execute_transport(self._request())

        self.assertTrue(result.stream_fallback_used)
        self.assertIn("streaming is not supported", result.stream_fallback_error)
        self.assertEqual(result.text, "answer")
        request.assert_called_once()

    def test_streaming_generic_error_does_not_fall_back(self):
        streamed = StreamResult("", "", None, 1.0, "connection refused", None, {})
        with (
            mock.patch("benchmark.transport.stream_request", return_value=streamed),
            mock.patch("benchmark.transport.nonstream_request") as request,
        ):
            result = execute_transport(self._request())

        self.assertFalse(result.stream_fallback_used)
        self.assertEqual(result.response_nature, "transport_error")
        request.assert_not_called()

    def test_opencode_is_normalized(self):
        response = OpenCodeProcessResult(
            "answer", "diagnostic", 2.0, None, 0, think_text="thinking",
        )
        with mock.patch("benchmark.transport.run_process", return_value=response):
            result = execute_transport(self._request(
                transport="opencode",
                opencode_config_path="config.json",
                opencode_model="provider/model",
                opencode_target_key="target",
                opencode_plugin_id="plugin",
            ))

        self.assertEqual(result.text, "answer")
        self.assertEqual(result.think_text, "thinking")
        self.assertFalse(result.stream_ok)
        self.assertEqual(result.response_nature, "completed")
