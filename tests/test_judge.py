import json
import tempfile
import unittest
from unittest import mock

from benchmark.core import (
    JudgeResult,
    build_judge_prompt,
    judge_response,
    parse_judge_response,
    prepare_judge_sidecar,
)
from benchmark.state import BenchmarkState
from plugins import discover_plugins
from plugins.outputs.output_html import HTMLOutputPlugin
from plugins.outputs.output_markdown import MarkdownOutputPlugin


class FakePlugin:
    id = "fake"
    name = "Fake task"
    version = "1.0"
    max_score = 20

    def get_prompt(self):
        return "Produce a useful answer."


class TestJudgeCore(unittest.TestCase):
    def test_parse_judge_json_and_rejects_invalid(self):
        self.assertEqual(
            parse_judge_response('{"score": 82.4, "confidence": "high", "rationale": "complete"}'),
            JudgeResult(score=82, confidence="high", rationale="complete"),
        )
        self.assertEqual(parse_judge_response("not json").error, "invalid judge JSON: Expecting value")

    def test_build_prompt_blinds_deterministic_score(self):
        prompt = build_judge_prompt(FakePlugin(), "Do this", "Done well")
        self.assertIn("Do this", prompt)
        self.assertIn("Done well", prompt)
        self.assertIn("semantic score", prompt.lower())

    def test_sidecar_is_atomic_and_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/http/model/fake.json"
            prepare_judge_sidecar(path, FakePlugin(), "Prompt", "Response", target="model", runner="http")
            with open(path, encoding="utf-8") as handle:
                item = json.load(handle)
            self.assertEqual(item["target"], "model")
            self.assertEqual(item["response"], "Response")
            self.assertEqual(len(item["response_sha256"]), 64)

    def test_judge_response_retries_invalid_json(self):
        response = mock.Mock(error=None, text='{"score": 75, "confidence": "medium", "rationale": "usable"}')
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = f"{tmp}/input.json"
            prepare_judge_sidecar(sidecar, FakePlugin(), "Prompt", "Response", target="model", runner="http")
            with mock.patch("benchmark.core.nonstream_request", return_value=response) as request:
                result = judge_response({}, "Local", "judge", sidecar, timeout=3)
        self.assertEqual(result.score, 75)
        request.assert_called_once()


class TestJudgeStateAndReports(unittest.TestCase):
    def setUp(self):
        self.plugins = discover_plugins()
        self.plugin = self.plugins[0]

    def test_state_update_judge_result_does_not_append_row(self):
        state = BenchmarkState({"model": "Local"}, [self.plugin.id])
        state.add_result({
            "model": "model", "state_key": "model", "runner": "http", "status": "ok",
            f"{self.plugin.id}_score": 80,
        })
        state.update_judge_result("model", "http", self.plugin.id, score=91, confidence="high", rationale="good")
        self.assertEqual(len(state.results), 1)
        self.assertEqual(state.latest_results()[0][f"{self.plugin.id}_judge_score"], 91)

    def test_html_and_markdown_render_judge_columns(self):
        result = {
            "model": "model", "runner": "http", "status": "ok", "stream_ok": True,
            "ttft": 1, "total_time": 2, "judge_model": "judge", "judge_status": "complete",
            f"{self.plugin.id}_score": 80,
            f"{self.plugin.id}_judge_score": 91,
            f"{self.plugin.id}_judge_confidence": "high",
            f"{self.plugin.id}_judge_error": "",
        }
        self.assertIn("Judge Confidence", MarkdownOutputPlugin().generate([result], [self.plugin]))
        self.assertIn("91", HTMLOutputPlugin().generate([result], [self.plugin]))


if __name__ == "__main__":
    unittest.main()
