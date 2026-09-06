import unittest
from unittest import mock

from benchmark.http import StreamResult
from benchmark.observer import TaskObserver
from benchmark.request_models import GenerationFields, HTTPRequest
from benchmark.task_execution import resolve_retry_policy
from benchmark.transport import (
    BENCHMARK_RETRY_POLICY,
    JUDGE_RETRY_POLICY,
    RetryPolicy,
    execute_task,
    execute_task_streaming,
)


class TestTransportRetry(unittest.TestCase):
    def _request(self):
        return HTTPRequest(GenerationFields(
            prompt="base prompt",
            max_tokens=100,
            source_config={"S": {}},
            api_model="model",
            source="S",
            timeout=5,
            observer=TaskObserver.noop(),
        ))

    def test_benchmark_policy_retries_token_limit_with_altered_prompt(self):
        responses = [StreamResult("partial", "r" * 400, 1.0, 2.0, None, "length", {}), StreamResult("answer", "", 1.0, 2.0, None, "stop", {})]
        attempts = []
        with mock.patch("benchmark.transport.stream_request", side_effect=responses) as request:
            execution = execute_task(self._request(), retry_policy=BENCHMARK_RETRY_POLICY, base_prompt="base prompt", attempt_callback=attempts.append)
        self.assertEqual(attempts, [1, 2])
        self.assertEqual(execution.attempt_count, 2)
        self.assertEqual(execution.retry_reasons, ["token_limit"])
        self.assertEqual(execution.attempts[1].prompt_altered, "thinking_50_percent")
        self.assertIn("RETRY GUIDANCE", execution.attempts[1].request_prompt)
        self.assertEqual(request.call_count, 2)

    def test_transport_error_retry_keeps_prompt_unchanged(self):
        responses = [StreamResult("", "", None, 1.0, "connection refused", None, {}), StreamResult("answer", "", 1.0, 2.0, None, "stop", {})]
        with mock.patch("benchmark.transport.stream_request", side_effect=responses):
            execution = execute_task(self._request(), retry_policy=BENCHMARK_RETRY_POLICY, base_prompt="base prompt")
        self.assertEqual(execution.retry_reasons, ["transport_error"])
        self.assertEqual(execution.attempts[0].request_prompt, "base prompt")
        self.assertEqual(execution.attempts[1].request_prompt, "base prompt")
        self.assertEqual(execution.attempts[1].prompt_altered, "none")

    def test_judge_policy_retries_only_json_errors(self):
        responses = [StreamResult("not json", "", 1.0, 2.0, None, "stop", {}), StreamResult('{"ok": true}', "", 1.0, 2.0, None, "stop", {})]
        with mock.patch("benchmark.transport.stream_request", side_effect=responses):
            execution = execute_task(self._request(), retry_policy=JUDGE_RETRY_POLICY, base_prompt="base prompt", json_error_prompt_alterer=lambda result: "\nretry as JSON")
        self.assertEqual(execution.retry_reasons, ["json_error"])
        self.assertIn("retry as JSON", execution.attempts[1].request_prompt)

    def test_judge_policy_does_not_retry_transport_errors(self):
        response = StreamResult("", "", None, 1.0, "connection refused", None, {})
        with mock.patch("benchmark.transport.stream_request", return_value=response) as request:
            execution = execute_task(self._request(), retry_policy=JUDGE_RETRY_POLICY, base_prompt="base prompt", json_error_prompt_alterer=lambda result: "\nretry as JSON")
        self.assertEqual(execution.attempt_count, 1)
        request.assert_called_once()

    def test_streaming_execution_yields_content_and_resolves_metadata(self):
        def fake_stream(*args, observer=None, **kwargs):
            observer.chunk("hello ")
            observer.chunk("world")
            return StreamResult("hello world", "", 1.0, 2.0, None, "stop", {})
        with mock.patch("benchmark.transport.stream_request", side_effect=fake_stream):
            execution = execute_task_streaming(self._request(), retry_policy=RetryPolicy(max_attempts=1), base_prompt="base prompt")
            self.assertEqual(list(execution.stream), ["hello ", "world"])
            attempt = execution.metadata_future.result(timeout=2)
        self.assertEqual(attempt.attempt_number, 1)
        self.assertEqual(attempt.result.text, "hello world")
        self.assertIsNone(execution.next_attempt)

    def test_streaming_execution_links_policy_retry(self):
        responses = [StreamResult("partial", "r" * 400, 1.0, 2.0, None, "length", {}), StreamResult("answer", "", 1.0, 2.0, None, "stop", {})]
        def fake_stream(*args, observer=None, **kwargs):
            response = responses.pop(0)
            if response.text:
                observer.chunk(response.text)
            return response
        with mock.patch("benchmark.transport.stream_request", side_effect=fake_stream):
            execution = execute_task_streaming(self._request(), retry_policy=BENCHMARK_RETRY_POLICY, base_prompt="base prompt")
            self.assertEqual(list(execution.stream), ["partial"])
            first = execution.metadata_future.result(timeout=2)
            retry = execution.next_attempt
            self.assertIsNotNone(retry)
            self.assertEqual(first.attempt_number, 1)
            self.assertEqual(list(retry.stream), ["answer"])
            second = retry.metadata_future.result(timeout=2)
        self.assertEqual(second.attempt_number, 2)
        self.assertEqual(second.retry_reason, "token_limit")
        self.assertIn("RETRY GUIDANCE", second.request_prompt)

    def test_streaming_execution_cancel_sets_stop_event(self):
        execution = execute_task_streaming(self._request(), retry_policy=RetryPolicy(max_attempts=1), base_prompt="base prompt", stream_request_fn=lambda *args, **kwargs: StreamResult("", "", None, 0, "cancelled", None, {}))
        execution.cancel()
        self.assertTrue(execution._stop_event.is_set())

    def test_selection_can_be_replaced_after_scoring(self):
        responses = [StreamResult("first", "", 1.0, 2.0, None, "stop", {}), StreamResult("second", "", 1.0, 2.0, None, "stop", {})]
        with mock.patch("benchmark.transport.stream_request", side_effect=responses):
            execution = execute_task(self._request(), retry_policy=RetryPolicy(max_attempts=2), base_prompt="base prompt")
        execution.select(execution.attempts[0])
        self.assertIs(execution.selected, execution.attempts[0])

    def test_timeout_is_terminal_by_default(self):
        responses = [StreamResult("", "", None, 1.0, "Stream watchdog timeout (1200s) exceeded", None, {})]
        with mock.patch("benchmark.transport.stream_request", side_effect=responses) as request:
            execution = execute_task(self._request(), retry_policy=resolve_retry_policy({"S": {}}, "S"), base_prompt="base prompt")
        self.assertEqual(execution.attempt_count, 1)
        request.assert_called_once()
        self.assertEqual(execution.attempts[0].result.response_nature, "timeout")

    def test_timeout_retries_when_source_opts_in(self):
        responses = [
            StreamResult("", "", None, 1.0, "Stream watchdog timeout (1200s) exceeded", None, {}),
            StreamResult("answer", "", 1.0, 2.0, None, "stop", {}),
        ]
        policy = resolve_retry_policy({"S": {"retry_on_timeout": True}}, "S")
        self.assertIsNot(policy, BENCHMARK_RETRY_POLICY)
        self.assertTrue(policy.retry_on_timeout)
        with mock.patch("benchmark.transport.stream_request", side_effect=responses) as request:
            execution = execute_task(self._request(), retry_policy=policy, base_prompt="base prompt")
        self.assertEqual(execution.attempt_count, 2)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(execution.retry_reasons, ["timeout"])

    def test_resolve_retry_policy_defaults_and_mutation_safety(self):
        # Unknown source: benchmark defaults (timeout terminal).
        self.assertIs(resolve_retry_policy({}, "missing"), BENCHMARK_RETRY_POLICY)
        # Non-dict source config: same.
        self.assertIs(resolve_retry_policy({"S": "bogus"}, "S"), BENCHMARK_RETRY_POLICY)
        # Explicit false matches the default singleton.
        self.assertIs(resolve_retry_policy({"S": {"retry_on_timeout": False}}, "S"), BENCHMARK_RETRY_POLICY)
        # Opt-in returns a new policy and must NOT mutate the shared default.
        policy = resolve_retry_policy({"S": {"retry_on_timeout": True}}, "S")
        self.assertIsNot(policy, BENCHMARK_RETRY_POLICY)
        self.assertTrue(policy.retry_on_timeout)
        self.assertFalse(BENCHMARK_RETRY_POLICY.retry_on_timeout)
        # Judge policy is untouched by the benchmark helper.
        self.assertFalse(JUDGE_RETRY_POLICY.retry_on_timeout)


if __name__ == "__main__":
    unittest.main()
