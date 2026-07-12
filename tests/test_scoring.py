"""Tests for plugin scoring functions."""
import unittest

from plugins import discover_plugins


class TestRateLimiterScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = next(p for p in discover_plugins() if p.id == "rate-limiter")

    def test_empty_response_scores_zero(self):
        self.assertEqual(self.plugin.score(""), 0.0)

    def test_token_bucket_class_scores(self):
        text = "class TokenBucket:\n    pass"
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)

    def test_full_response_scores_high(self):
        text = (
            "class TokenBucket:\n"
            "    def __init__(self):\n"
            "        self.tokens = 0\n"
            "    def refill(self):\n"
            "        pass\n"
            "    def allow_request(self, client_id: str) -> bool:\n"
            "        return True\n"
            "    def get_usage_stats(self, client_id: str) -> dict:\n"
            "        return {}\n"
            "import threading\n"
            "lock = threading.Lock()\n"
            "with lock:\n"
            "    pass\n"
            "\"\"\"docstring\"\"\"\n"
            "raise ValueError('bad')\n"
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 5.0)


class TestMoEDenseScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = next(p for p in discover_plugins() if p.id == "moe-dense")

    def test_empty_response_scores_zero(self):
        self.assertEqual(self.plugin.score(""), 0.0)

    def test_moe_mention_scores(self):
        text = "Mixture-of-Experts (MoE) architecture uses sparse routing."
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)

    def test_full_response_scores_high(self):
        text = (
            "MoE uses top-k gating with softmax routing. "
            "The load-balancing loss ensures experts are evenly used. "
            "Training challenges include token dropping and expert collapse. "
            "Inference is memory bandwidth bound. "
            "MoE outperforms dense on MMLU and GSM8K. "
            "See the arxiv paper by Shazeer et al."
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 5.0)


class TestToolCallingScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = next(p for p in discover_plugins() if p.id == "tool-calling")

    def test_empty_response_scores_zero(self):
        self.assertEqual(self.plugin.score(""), 0.0)

    def test_tool_call_format_scores(self):
        text = "<tool_call>{\"name\": \"get_weather\", \"args\": {\"location\": \"Tokyo\"}}</tool_call>"
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)

    def test_full_response_scores_high(self):
        text = (
            "<tool_call>{\"name\": \"get_weather\", \"args\": {\"location\": \"Tokyo\"}}</tool_call>\n"
            "<tool_call>{\"name\": \"get_stock_price\", \"args\": {\"ticker\": \"SONY\"}}</tool_call>\n"
            "<tool_call>{\"name\": \"convert_currency\", \"args\": {\"amount\": 1000, \"from_curr\": \"USD\", \"to_curr\": \"JPY\"}}</tool_call>\n"
            "The weather in Tokyo is sunny, SONY stock price is $100, and 1000 USD is 150000 JPY."
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 5.0)


class TestOrchestrationScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = next(p for p in discover_plugins() if p.id == "orchestration")

    def test_empty_response_scores_zero(self):
        self.assertEqual(self.plugin.score(""), 0.0)

    def test_steps_mentioned_scores(self):
        text = "Step 1: Process logs. Step 2: GeoIP lookup. Step 3: Anomaly detection. Step 4: PDF report."
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)

    def test_full_response_scores_high(self):
        text = (
            "Step 1: Ingest logs [PARALLEL]\n"
            "Step 2: GeoIP lookup [DEPENDS_ON: step1]\n"
            "Step 3: Anomaly detection [DEPENDS_ON: step2]\n"
            "Step 4: Generate PDF report [DEPENDS_ON: step3] [SEQUENTIAL]\n"
            "Execution trace: step1 init -> step1 running -> step1 complete -> "
            "step2 init -> step2 running -> step2 complete -> step3 complete -> step4 complete"
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 5.0)


class TestCodeReviewScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = next(p for p in discover_plugins() if p.id == "code-review")

    def test_empty_response_scores_zero(self):
        self.assertEqual(self.plugin.score(""), 0.0)

    def test_invalid_json_scores_zero(self):
        self.assertEqual(self.plugin.score("Here are some issues"), 0.0)

    def test_identifies_file_handle_issue(self):
        text = (
            '{"issues": [{"description": "The file handle is never closed; use a context manager."}]}'
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)

    def test_full_review_scores_high(self):
        text = (
            '{"issues": ['
            '{"description": "File handle is never closed; use a context manager."},'
            '{"description": "Uses == None instead of is None."},'
            '{"description": "Hardcoded /tmp/data.txt path."},'
            '{"description": "fetch_data may raise an exception; wrap it in try/except."},'
            '{"description": "Unused imports os and time should be removed."},'
            '{"description": "Use enumerate or iterate directly instead of range(len())."}'
            ']}'
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 5.0)

    def test_json_with_markdown_wrapper_scores(self):
        text = (
            "Here is the review:\n"
            "```json\n"
            '{"issues": [{"description": "You need to close the file handle or use a context manager."}]}\n'
            "```"
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)


class TestStructuredOutputScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = next(p for p in discover_plugins() if p.id == "structured-output")

    def test_empty_response_scores_zero(self):
        self.assertEqual(self.plugin.score(""), 0.0)

    def test_valid_json_scores(self):
        text = '{"name": "Alice", "age": 30, "email": "alice@example.com", "roles": ["admin"], "settings": {"theme": "dark", "notifications": true}}'
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)

    def test_full_response_scores_high(self):
        text = (
            '```json\n'
            '{"name": "Alice", "age": 30, "email": "alice@example.com", '
            '"roles": ["admin", "editor"], "settings": {"theme": \"dark\", "notifications": true}}\n'
            '```'
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 5.0)

    def test_yaml_response_scores_high(self):
        text = (
            '```yaml\n'
            'name: Alice\n'
            'age: 30\n'
            'email: alice@example.com\n'
            'roles:\n'
            '  - admin\n'
            '  - editor\n'
            'settings:\n'
            '  theme: dark\n'
            '  notifications: true\n'
            '```'
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 5.0)

    def test_invalid_json_scores_zero(self):
        text = '{"name": "Alice", "age":}'
        score = self.plugin.score(text)
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
