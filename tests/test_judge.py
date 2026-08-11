import json
import tempfile
import unittest
from unittest import mock

from benchmark import cli
from benchmark.core import (
    JudgeResult,
    build_judge_prompt,
    confidence_weighted_consensus,
    judge_response,
    parse_judge_response,
    prepare_judge_sidecar,
    save_judge_response,
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
    def test_confidence_weighted_consensus(self):
        result = confidence_weighted_consensus([
            {"score": 90, "confidence": "high", "rationale": "strong"},
            {"score": 50, "confidence": "low", "rationale": "weak"},
        ])
        self.assertEqual(result["score"], 81)
        self.assertEqual(result["confidence"], "high")

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

    def test_judge_response_artifact_uses_existing_response_naming_convention(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_judge_response(
                tmp, "model", "http", "rate-limiter", "judge/model", '{"score": 90}'
            )
            self.assertTrue(path.endswith("rate-limiter.judge.judge_model.txt"))
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), '{"score": 90}')

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


class TestJudgeResumeDiscovery(unittest.TestCase):
    def test_retained_completed_sidecar_is_eligible_on_resume(self):
        """Resume discovery finds completed work absent from benchmark queues."""
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = f"{tmp}/http/model/fake.json"
            prepare_judge_sidecar(
                sidecar, FakePlugin(), "Prompt", "Response",
                target="model", runner="http", state_key="model",
            )
            state = BenchmarkState({"model": "Local"}, ["fake"])
            state.add_result({
                "model": "model", "state_key": "model", "runner": "http",
                "status": "ok", "fake_score": 80,
            })
            state.update("model", status="completed")
            eligible = cli._eligible_judge_sidecars(
                tmp,
                {"model": {"source": "Local"}},
                state,
                {"fake"},
                ["judge"],
            )
            self.assertEqual(len(eligible), 1)
            self.assertEqual(eligible[0][1]["target"], "model")

    def test_retained_opencode_sidecar_matches_runner_specific_state(self):
        """OpenCode sidecars use the suffixed state identity on resume."""
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = f"{tmp}/opencode/model/fake.json"
            prepare_judge_sidecar(
                sidecar, FakePlugin(), "Prompt", "Response",
                target="model", runner="opencode", state_key="model [opencode]",
            )
            state = BenchmarkState({"model [opencode]": "Local"}, ["fake"])
            state.add_result({
                "model": "model", "state_key": "model [opencode]", "runner": "opencode",
                "status": "ok", "fake_score": 80,
            })
            state.update("model [opencode]", status="completed")
            eligible = cli._eligible_judge_sidecars(
                tmp,
                {"model": {"source": "Local"}},
                state,
                {"fake"},
                ["judge"],
            )
            self.assertEqual(len(eligible), 1)

    def test_numeric_plugin_is_eligible_when_sibling_failed(self):
        """A numeric plugin remains judgeable even when its model failed."""
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = f"{tmp}/http/model/fake.json"
            prepare_judge_sidecar(
                sidecar, FakePlugin(), "Prompt", "Response",
                target="model", runner="http", state_key="model",
            )
            state = BenchmarkState({"model": "Local"}, ["fake", "other"])
            state.add_result({
                "model": "model", "state_key": "model", "runner": "http",
                "status": "error", "fake_score": 80, "other_score": "fail",
            })
            state.update("model", status="failed", fake_score=80)
            eligible = cli._eligible_judge_sidecars(
                tmp, {"model": {"source": "Local"}}, state,
                {"fake"}, ["judge"],
            )
            self.assertEqual(len(eligible), 1)

    def test_partial_model_info_score_is_eligible_before_result_row_exists(self):
        """A completed plugin can be judged while its model is still partial."""
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = f"{tmp}/http/model/fake.json"
            prepare_judge_sidecar(
                sidecar, FakePlugin(), "Prompt", "Response",
                target="model", runner="http", state_key="model",
            )
            state = BenchmarkState({"model": "Local"}, ["fake"])
            state.update("model", status="running", fake_score=80)
            eligible = cli._eligible_judge_sidecars(
                tmp, {"model": {"source": "Local"}}, state,
                {"fake"}, ["judge"],
            )
            self.assertEqual(len(eligible), 1)

    def test_missing_judge_remains_eligible_after_another_judge_vote(self):
        """Resume queues only the judge that has not voted yet."""
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = f"{tmp}/http/model/fake.json"
            prepare_judge_sidecar(
                sidecar, FakePlugin(), "Prompt", "Response",
                target="model", runner="http", state_key="model",
            )
            state = BenchmarkState({"model": "Local"}, ["fake"])
            state.add_result({
                "model": "model", "state_key": "model", "runner": "http",
                "status": "error", "fake_score": 80,
                "fake_judge_votes": [{
                    "model": "judge-a", "score": 90, "confidence": "high",
                }],
            })
            state.update("model", status="failed")
            only_b = cli._eligible_judge_sidecars(
                tmp, {"model": {"source": "Local"}}, state,
                {"fake"}, ["judge-a", "judge-b"],
            )
            self.assertEqual(len(only_b), 1)

    def test_retained_sidecar_is_not_eligible_after_all_judges_complete(self):
        """Startup discovery does not requeue a fully judged retained result."""
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = f"{tmp}/http/model/fake.json"
            prepare_judge_sidecar(
                sidecar, FakePlugin(), "Prompt", "Response",
                target="model", runner="http", state_key="model",
            )
            state = BenchmarkState({"model": "Local"}, ["fake"])
            state.add_result({
                "model": "model", "state_key": "model", "runner": "http",
                "status": "ok", "fake_score": 80,
                "fake_judge_votes": [{
                    "model": "judge", "score": 90, "confidence": "high",
                }],
                "fake_judge_complete": True,
            })
            state.update("model", status="completed")
            eligible = cli._eligible_judge_sidecars(
                tmp,
                {"model": {"source": "Local"}},
                state,
                {"fake"},
                ["judge"],
            )
            self.assertEqual(eligible, [])

    def test_add_result_preserves_judge_update_before_row_append(self):
        """A judge update made before result append survives in the row."""
        state = BenchmarkState({"model": "Local"}, ["fake"])
        state.update("model", fake_judge_score=91, fake_judge_votes=[{"model": "judge", "score": 91}], fake_judge_complete=True)
        state.add_result({
            "model": "model", "state_key": "model", "runner": "http",
            "fake_score": 80,
        })
        result = state.latest_results()[0]
        self.assertEqual(result["fake_judge_score"], 91)
        self.assertTrue(result["fake_judge_complete"])


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
            f"{self.plugin.id}_judge_votes": [{"model": "judge", "score": 91, "confidence": "high"}],
        }
        self.assertIn("Judge Confidence", MarkdownOutputPlugin().generate([result], [self.plugin]))
        self.assertIn("91", HTMLOutputPlugin().generate([result], [self.plugin]))


if __name__ == "__main__":
    unittest.main()
