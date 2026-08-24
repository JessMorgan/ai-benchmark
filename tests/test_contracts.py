import unittest

from benchmark.contracts import JudgeContract


class TestJudgeContract(unittest.TestCase):
    def test_identity_contains_complete_definition(self):
        contract = JudgeContract.from_definition(
            plugin_id="p",
            plugin_version="1.0.0",
            prompt_version="judge-v8",
            instructions_version="1.0.0",
            response_schema={"type": "object", "properties": {"score": {"type": "integer"}}},
            instructions="Evaluate explicit requirements.",
        )
        self.assertTrue(contract.contract_id.startswith("judge-contract-v1:"))
        self.assertEqual(contract.contract_hash, contract.contract_id.split(":", 1)[1])
        self.assertNotIn("legacy_contract_id", contract.contract_json)

    def test_changed_schema_changes_identity(self):
        common = {
            "plugin_id": "p",
            "plugin_version": "1.0.0",
            "prompt_version": "judge-v8",
            "instructions_version": "1.0.0",
            "instructions": "Evaluate explicit requirements.",
        }
        first = JudgeContract.from_definition(response_schema={"type": "object"}, **common)
        second = JudgeContract.from_definition(
            response_schema={"type": "object", "additionalProperties": False}, **common,
        )
        self.assertNotEqual(first.contract_id, second.contract_id)

    def test_from_json_rejects_hash_drift(self):
        with self.assertRaises(ValueError, msg="hash mismatch"):
            JudgeContract.from_json(
                contract_id="c",
                plugin_id="p",
                plugin_version="1",
                prompt_version="v1",
                instructions_version="v1",
                response_schema_hash="schema",
                contract_json='{"x":1}',
                contract_hash="wrong",
            )


if __name__ == "__main__":
    unittest.main()
