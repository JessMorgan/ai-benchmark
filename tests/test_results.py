import json
import tempfile
import unittest
from types import SimpleNamespace

from benchmark.results import save_judge_result, save_task_result


class _Plugin:
    id = "fake"
    version = "1.2.3"


class TestResults(unittest.TestCase):
    def test_save_task_result_builds_flat_result_and_sidecars(self):
        sidecars = []

        def sidecar_path(root, target, runner, pid):
            return f"{root}/{target}-{runner}-{pid}.json"

        def prepare(path, plugin, prompt, response, **kwargs):
            sidecars.append((path, prompt, response, kwargs))

        attempts = [{
            "attempt": 1,
            "retry_reason": None,
            "score_traceback": "trace",
            "response_nature": "completed",
        }]
        with tempfile.TemporaryDirectory() as tmp:
            result = save_task_result(
                None,
                state=None,
                model_name="model",
                pid="fake",
                plugin=_Plugin(),
                output_dir=tmp,
                save_responses=True,
                judge_input_dir=tmp,
                artifact_target="model",
                score=80,
                rubric=[{"id": "R1"}],
                diagnostics={"ok": True},
                selected_prompt="Prompt",
                selected_text="Answer",
                selected_think="Think",
                response_time=1.2,
                output_tokens=2,
                thinking_tokens=3,
                total_tokens=5,
                attempts=attempts,
                schema_metadata={"schema_requested": True},
                prepare_judge_sidecar_fn=prepare,
                judge_sidecar_path_fn=sidecar_path,
            )
            meta_path = f"{tmp}/responses/model/fake.meta.json"
            response_path = f"{tmp}/responses/model/fake.txt"
            self.assertEqual(result["fake_score"], 80)
            self.assertEqual(result["fake_total_tokens"], 5)
            self.assertEqual(result["fake_attempts"][0]["response_nature"], "completed")
            self.assertTrue(sidecars)
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            self.assertEqual(meta["score"], 80)
            with open(response_path, encoding="utf-8") as handle:
                self.assertIn("<thinking>", handle.read())

    def test_save_judge_result_uses_parsed_values_and_contract(self):
        result = SimpleNamespace(
            score=91,
            confidence="high",
            rationale="good",
            criteria=[{"id": "R1"}],
            error=None,
        )
        vote = save_judge_result(
            result,
            model_name="judge-a",
            judge_prompt_version="judge-v8",
            judge_contract_id="contract-current",
        )
        self.assertEqual(vote["model"], "judge-a")
        self.assertEqual(vote["score"], 91)
        self.assertEqual(vote["judge_contract_id"], "contract-current")

    def test_save_judge_result_can_store_scheduler_failure(self):
        vote = save_judge_result(
            None,
            model_name="judge-a",
            judge_prompt_version="judge-v8",
            judge_contract_id="contract-current",
            parsed_judge={"error": "input failed"},
        )
        self.assertIsNone(vote["score"])
        self.assertEqual(vote["error"], "input failed")
        self.assertEqual(vote["criteria"], [])


if __name__ == "__main__":
    unittest.main()
