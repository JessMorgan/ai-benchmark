import json
import os
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
            meta_path = f"{tmp}/responses/model/fake/meta.json"
            response_path = f"{tmp}/responses/model/fake/response.txt"
            self.assertEqual(result["fake_score"], 80)
            self.assertEqual(result["fake_total_tokens"], 5)
            self.assertEqual(result["fake_attempts"][0]["response_nature"], "completed")
            self.assertTrue(sidecars)
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            self.assertEqual(meta["score"], 80)
            with open(response_path, encoding="utf-8") as handle:
                self.assertIn("<thinking>", handle.read())

    def test_plugin_dir_matches_judge_reader_for_nonslug_plugin_ids(self):
        """Writer and judge reader must derive the SAME plugin directory.

        Plugin ids are only validated as non-empty strings, so a non-slug
        id (spaces, slashes) must sanitize identically on the save-responses
        writer (``save_task_result``) and the judge-artifact writer/reader
        (``judge_response_path``). A mismatch silently strands judge raw
        responses outside the directory the reports link to.
        """
        from benchmark.judging import judge_response_path
        from benchmark.outputs import sanitize_filename

        for plugin_id in ("my plugin", "a/b", "plugin..dots"):
            with self.subTest(plugin_id=plugin_id):
                with tempfile.TemporaryDirectory() as tmp:
                    save_task_result(
                        None,
                        state=None,
                        model_name="model",
                        pid=plugin_id,
                        plugin=_Plugin(),
                        output_dir=tmp,
                        save_responses=True,
                        selected_prompt="Prompt",
                        selected_text="Answer",
                    )
                    expected_dir = os.path.join(
                        tmp, "responses", "model", sanitize_filename(plugin_id),
                    )
                    self.assertTrue(
                        os.path.isfile(os.path.join(expected_dir, "response.txt")),
                        "writer must place response.txt in the sanitized plugin dir",
                    )
                    # The judge path helper must sanitize the plugin id
                    # identically. The judge helper lives under the runner
                    # namespace (``<runner>/responses/...``) while the live
                    # writer does not, so compare the plugin-dir component.
                    judge_dir = os.path.dirname(
                        judge_response_path(tmp, "model", "http", plugin_id, "judge-a")
                    )
                    self.assertEqual(os.path.basename(judge_dir), os.path.basename(expected_dir))

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
