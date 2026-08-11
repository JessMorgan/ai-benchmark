"""Tests for persisted model-as-a-judge analysis helpers."""
import json
import tempfile
import unittest
from pathlib import Path

from benchmark.judge_analysis import (
    build_disagreement_queue,
    judge_statistics,
    write_disagreement_queue,
)


class TestJudgeAnalysis(unittest.TestCase):
    def _state(self):
        return {
            "results": [
                {
                    "model": "model-a",
                    "state_key": "model-a",
                    "runner": "http",
                    "source": "Local",
                    "code-review_score": 10,
                    "code-review_judge_votes": [
                        {"model": "judge-a", "score": 90, "confidence": "high", "rationale": "strong"},
                        {"model": "judge-b", "score": 20, "confidence": "medium", "rationale": "weak"},
                    ],
                },
                {
                    "model": "model-a",
                    "state_key": "model-a",
                    "runner": "http",
                    "source": "Local",
                    "code-review_score": 10,
                    "code-review_judge_votes": [
                        {"model": "judge-a", "score": 95, "confidence": "high", "rationale": "updated"},
                    ],
                },
                {
                    "model": "model-b",
                    "state_key": "model-b",
                    "runner": "http",
                    "source": "Cloud",
                    "code-review_score": 80,
                    "code-review_judge_votes": [
                        {"model": "judge-a", "score": None, "confidence": None, "rationale": None, "error": "HTTP 429"},
                    ],
                },
            ]
        }

    def test_latest_row_and_valid_votes_are_used(self):
        stats = judge_statistics(self._state())
        judge_a = next(item for item in stats["per_judge"] if item["model"] == "judge-a")
        self.assertEqual(judge_a["valid_votes"], 1)
        self.assertEqual(judge_a["failed_attempts"], 1)
        self.assertEqual(judge_a["mean_score"], 95.0)

    def test_queue_deduplicates_latest_judge_votes(self):
        state = self._state()
        state["results"][1]["code-review_judge_votes"].extend([
            {"model": "judge-b", "score": 100, "confidence": "high", "rationale": "duplicate retry"},
            {"model": "judge-b", "score": 20, "confidence": "high", "rationale": "latest weak"},
        ])
        queue = build_disagreement_queue(state, spread_threshold=30, deviation_threshold=40)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["valid_judges"], 2)
        self.assertIn("judge-spread>=30", queue[0]["triggers"])
        self.assertIn("consensus-deviation>=40", queue[0]["triggers"])

    def test_each_queue_criterion_can_be_disabled(self):
        state = self._state()
        state["results"][1]["code-review_judge_votes"].append(
            {"model": "judge-b", "score": 20, "confidence": "high", "rationale": "weak"}
        )
        spread_only = build_disagreement_queue(
            state, spread_threshold=30, deviation_threshold=None
        )
        deviation_only = build_disagreement_queue(
            state, spread_threshold=None, deviation_threshold=40
        )
        neither = build_disagreement_queue(
            state, spread_threshold=None, deviation_threshold=None
        )
        self.assertEqual(len(spread_only), 1)
        self.assertEqual(spread_only[0]["triggers"], ["judge-spread>=30"])
        self.assertEqual(len(deviation_only), 1)
        self.assertEqual(deviation_only[0]["triggers"], ["consensus-deviation>=40"])
        self.assertEqual(neither, [])

    def test_write_queue_is_json_and_paths_are_run_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "benchmark_state.json"
            output_path = Path(tmp) / "queue.json"
            state = self._state()
            state["results"][1]["code-review_judge_votes"].append(
                {"model": "judge-b", "score": 20, "confidence": "high", "rationale": "weak"}
            )
            state_path.write_text(json.dumps(state), encoding="utf-8")
            result = write_disagreement_queue(state_path, output_path)
            self.assertEqual(result, output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "judge-disagreement-queue-v1")
            self.assertEqual(len(payload["entries"]), 1)
            self.assertTrue(
                all(path.startswith(str(Path(tmp))) for path in payload["entries"][0]["judge_response_paths"])
            )


if __name__ == "__main__":
    unittest.main()
