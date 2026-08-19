import unittest
from unittest import mock

from benchmark.http import StreamResult
from benchmark.observer import TaskObserver
from benchmark.transport import (
    BENCHMARK_RETRY_POLICY,
    JUDGE_RETRY_POLICY,
    RetryPolicy,
    TransportRequest,
    execute_task,
)


class TestTransportRetry(unittest.TestCase):
    def _request(self):
        return TransportRequest(
            prompt="base prompt",
            max_tokens=100,
            source_config={"S": {}},
            api_model="model",
            source="S",
            timeout=5,
            observer=TaskObserver.noop(),
        )

    def test_benchmark_policy_retries_token_limit_with_altered_prompt(self):
        responses = [
            StreamResult("partial", "r" * 400, 1.0, 2.0, None, "length", {}),
            StreamResult("answer", "", 1.0, 2.0, None, "stop", {}),
        ]
        attempts = []
        with mock.patch("benchmark.transport.stream_request", side_effect=responses) as request:
            execution = execute_task(
                self._request(),
                retry_policy=BENCHMARK_RETRY_POLICY,
                base_prompt="base prompt",
                attempt_callback=attempts.append,
            )

        self.assertEqual(attempts, [1, 2])
        self.assertEqual(execution.attempt_count, 2)
        self.assertEqual(execution.retry_reasons, ["token_limit"])
        self.assertEqual(execution.attempts[1].prompt_altered, "thinking_50_percent")
        self.assertIn("RETRY GUIDANCE", execution.attempts[1].request_prompt)
        self.assertEqual(request.call_count, 2)

    def test_transport_error_retry_keeps_prompt_unchanged(self):
        responses = [
            StreamResult("", "", None, 1.0, "connection refused", None, {}),
            StreamResult("answer", "", 1.0, 2.0, None, "stop", {}),
        ]
        with mock.patch("benchmark.transport.stream_request", side_effect=responses):
            execution = execute_task(
                self._request(),
                retry_policy=BENCHMARK_RETRY_POLICY,
                base_prompt="base prompt",
            )

        self.assertEqual(execution.retry_reasons, ["transport_error"])
        self.assertEqual(execution.attempts[0].request_prompt, "base prompt")
        self.assertEqual(execution.attempts[1].request_prompt, "base prompt")
        self.assertEqual(execution.attempts[1].prompt_altered, "none")

    def test_judge_policy_retries_only_json_errors(self):
        responses = [
            StreamResult("not json", "", 1.0, 2.0, None, "stop", {}),
            StreamResult('{"ok": true}', "", 1.0, 2.0, None, "stop", {}),
        ]
        with mock.patch("benchmark.transport.stream_request", side_effect=responses):
            execution = execute_task(
                self._request(),
                retry_policy=JUDGE_RETRY_POLICY,
                base_prompt="base prompt",
                json_error_prompt_alterer=lambda result: "\nretry as JSON",
            )

        self.assertEqual(execution.retry_reasons, ["json_error"])
        self.assertIn("retry as JSON", execution.attempts[1].request_prompt)

    def test_judge_policy_does_not_retry_transport_errors(self):
        response = StreamResult("", "", None, 1.0, "connection refused", None, {})
        with mock.patch("benchmark.transport.stream_request", return_value=response) as request:
            execution = execute_task(
                self._request(),
                retry_policy=JUDGE_RETRY_POLICY,
                base_prompt="base prompt",
                json_error_prompt_alterer=lambda result: "\nretry as JSON",
            )

        self.assertEqual(execution.attempt_count, 1)
        request.assert_called_once()

    def test_selection_can_be_replaced_after_scoring(self):
        responses = [
            StreamResult("first", "", 1.0, 2.0, None, "stop", {}),
            StreamResult("second", "", 1.0, 2.0, None, "stop", {}),
        ]
        with mock.patch("benchmark.transport.stream_request", side_effect=responses):
            execution = execute_task(
                self._request(),
                retry_policy=RetryPolicy(max_attempts=2),
                base_prompt="base prompt",
            )

        execution.select(execution.attempts[0])
        self.assertIs(execution.selected, execution.attempts[0])


if __name__ == "__main__":
    unittest.main()
