"""Tests for the multi-step instructions challenge plugin."""
import unittest

from plugins import discover_plugins


class TestMultiStepScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = next(p for p in discover_plugins() if p.id == "multi-step")

    def test_empty_response_scores_zero(self):
        self.assertEqual(self.plugin.score(""), 0.0)

    def test_whitespace_response_scores_zero(self):
        self.assertEqual(self.plugin.score("   \n\t  "), 0.0)

    def test_get_temperature_from_config(self):
        self.assertEqual(self.plugin.get_temperature({"multi_step_temperature": 0.5}), 0.5)

    def test_get_temperature_default(self):
        self.assertIsNone(self.plugin.get_temperature({}))

    def _full_response(self):
        return (
            "```python\n"
            "def greet_user(name: str) -> str:\n"
            "    return \"Hello, {name}! Welcome.\"\n"
            "```\n\n"
            "```python\n"
            "def validate_name(name: str) -> bool:\n"
            "    if len(name) > 50:\n"
            "        return False\n"
            "    if not name.isalpha():\n"
            "        return False\n"
            "    return True\n"
            "```\n\n"
            "```python\n"
            "def format_greeting(greeting: str, times: int) -> str:\n"
            "    if times < 1:\n"
            "        return \"\"\n"
            "    return \"\\n\".join([greeting] * times)\n"
            "```\n\n"
            "[SUMMARY: 3 lines, 3 functions, completed all steps]."
        )

    def test_full_response_scores_high(self):
        score = self.plugin.score(self._full_response())
        self.assertGreater(score, 15.0)
        self.assertLessEqual(score, self.plugin.max_score)

    def test_greet_user_untyped_signature(self):
        text = (
            "```python\n"
            "def greet_user(name):\n"
            "    return \"Hello, \" + name + \"! Welcome.\"\n"
            "```\n\n"
            "```python\n"
            "def validate_name(name):\n"
            "    return True\n"
            "```\n\n"
            "```python\n"
            "def format_greeting(greeting, times):\n"
            "    return greeting\n"
            "```\n\n"
            "[SUMMARY: 1 lines, 3 functions, completed all steps]."
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)
        self.assertLess(score, self.plugin.max_score)

    def test_greet_user_missing_return(self):
        text = (
            "```python\n"
            "def greet_user(name: str):\n"
            "    print(\"Hello, name! Welcome.\")\n"
            "```\n\n"
            "```python\n"
            "def validate_name(name: str):\n"
            "    return True\n"
            "```\n\n"
            "```python\n"
            "def format_greeting(greeting: str, times: int):\n"
            "    return greeting\n"
            "```\n\n"
            "[SUMMARY: 1 lines, 3 functions, completed all steps]."
        )
        score = self.plugin.score(text)
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, self.plugin.max_score)

    def test_validate_name_missing_checks(self):
        text = (
            "```python\n"
            "def greet_user(name: str) -> str:\n"
            "    return \"Hello, name! Welcome.\"\n"
            "```\n\n"
            "```python\n"
            "def validate_name(name: str) -> bool:\n"
            "    return True\n"
            "```\n\n"
            "```python\n"
            "def format_greeting(greeting: str, times: int) -> str:\n"
            "    return greeting\n"
            "```\n\n"
            "[SUMMARY: 1 lines, 3 functions, completed all steps]."
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)
        self.assertLess(score, self.plugin.max_score)

    def test_format_greeting_untyped_and_missing_checks(self):
        text = (
            "```python\n"
            "def greet_user(name: str) -> str:\n"
            "    return \"Hello, name! Welcome.\"\n"
            "```\n\n"
            "```python\n"
            "def validate_name(name: str) -> bool:\n"
            "    if len(name) <= 50 and name.isalpha():\n"
            "        return True\n"
            "    return False\n"
            "```\n\n"
            "```python\n"
            "def format_greeting(greeting, times):\n"
            "    return greeting * times\n"
            "```\n\n"
            "[SUMMARY: 1 lines, 3 functions, completed all steps]."
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)
        self.assertLess(score, self.plugin.max_score)

    def test_summary_partial_match(self):
        text = (
            "```python\n"
            "def greet_user(name: str) -> str:\n"
            "    return \"Hello, name! Welcome.\"\n"
            "```\n\n"
            "```python\n"
            "def validate_name(name: str) -> bool:\n"
            "    return True\n"
            "```\n\n"
            "```python\n"
            "def format_greeting(greeting: str, times: int) -> str:\n"
            "    return greeting\n"
            "```\n\n"
            "[SUMMARY: some text, completed all steps]."
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)
        self.assertLess(score, self.plugin.max_score)

    def test_summary_missing(self):
        text = (
            "```python\n"
            "def greet_user(name: str) -> str:\n"
            "    return \"Hello, name! Welcome.\"\n"
            "```\n\n"
            "```python\n"
            "def validate_name(name: str) -> bool:\n"
            "    return True\n"
            "```\n\n"
            "```python\n"
            "def format_greeting(greeting: str, times: int) -> str:\n"
            "    return greeting\n"
            "```\n\n"
            "All steps completed."
        )
        score = self.plugin.score(text)
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, self.plugin.max_score)

    def test_main_block_penalty(self):
        text = (
            "```python\n"
            "def greet_user(name: str) -> str:\n"
            "    return \"Hello, name! Welcome.\"\n"
            "```\n\n"
            "```python\n"
            "def validate_name(name: str) -> bool:\n"
            "    return True\n"
            "```\n\n"
            "```python\n"
            "def format_greeting(greeting: str, times: int) -> str:\n"
            "    return greeting\n"
            "```\n\n"
            "if __name__ == \"__main__\":\n"
            "    pass\n\n"
            "[SUMMARY: 3 lines, 3 functions, completed all steps]."
        )
        score = self.plugin.score(text)
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, self.plugin.max_score)

    def test_too_few_code_blocks_penalty(self):
        text = (
            "```python\n"
            "def greet_user(name: str) -> str:\n"
            "    return \"Hello, name! Welcome.\"\n"
            "```\n\n"
            "[SUMMARY: 1 lines, 1 functions, completed all steps]."
        )
        score = self.plugin.score(text)
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, self.plugin.max_score)

    def test_zero_code_blocks_penalty(self):
        text = "No code blocks here. [SUMMARY: 0 lines, 0 functions, completed all steps]."
        score = self.plugin.score(text)
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, self.plugin.max_score)


if __name__ == "__main__":
    unittest.main()
