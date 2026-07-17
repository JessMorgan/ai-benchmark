"""Tests for plugin scoring functions."""
import unittest

from plugins import discover_plugins


class TestRateLimiterScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = next(p for p in discover_plugins() if p.id == "rate-limiter")

    def test_empty_response_scores_zero(self):
        self.assertEqual(self.plugin.score(""), 0.0)

    def test_evaluate_returns_rubric(self):
        score, rubric = self.plugin.evaluate("class TokenBucket:\n    pass")
        self.assertIsInstance(score, float)
        self.assertIsInstance(rubric, list)
        self.assertTrue(all("name" in item and "max" in item and "earned" in item and "missed" in item for item in rubric))

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
            "<plan>First check the weather, then search flights, book the hotel, "
            "check the stock price, convert currency, and finally send the email.</plan>\n"
            "<tool_call>{\"name\": \"get_weather\", \"args\": {\"location\": \"Tokyo\", \"unit\": \"celsius\"}}</tool_call>\n"
            "<tool_call>{\"name\": \"search_flights\", \"args\": {\"origin\": \"JFK\", \"destination\": \"Tokyo\", \"date\": \"2024-08-15\"}}</tool_call>\n"
            "<tool_call>{\"name\": \"book_hotel\", \"args\": {\"city\": \"Tokyo\", \"check_in\": \"2024-08-16\", \"check_out\": \"2024-08-20\", \"guests\": 2}}</tool_call>\n"
            "<tool_call>{\"name\": \"get_stock_price\", \"args\": {\"ticker\": \"SONY\"}}</tool_call>\n"
            "<tool_call>{\"name\": \"convert_currency\", \"args\": {\"amount\": 1000, \"from_curr\": \"USD\", \"to_curr\": \"JPY\"}}</tool_call>\n"
            "<tool_call>{\"name\": \"send_email\", \"args\": {\"to\": \"alice@example.com\", \"subject\": \"Tokyo Trip Itinerary\", \"body\": \"Here is your itinerary...\"}}</tool_call>\n"
            "The weather in Tokyo is sunny, the flight from JFK to Tokyo is booked, "
            "the hotel in Tokyo is reserved for 2 guests, SONY stock price is $100, "
            "1000 USD is 150000 JPY, and the itinerary has been emailed to alice@example.com."
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 10.0)

    def test_partial_tool_calls_score_less_than_full(self):
        text = (
            "<plan>Check the weather and stock price.</plan>\n"
            "<tool_call>{\"name\": \"get_weather\", \"args\": {\"location\": \"Tokyo\", \"unit\": \"celsius\"}}</tool_call>\n"
            "<tool_call>{\"name\": \"get_stock_price\", \"args\": {\"ticker\": \"SONY\"}}</tool_call>\n"
            "The weather in Tokyo is sunny and SONY stock price is $100."
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)
        self.assertLess(score, self.plugin.max_score)

    def test_wrong_arguments_score_less_than_full(self):
        text = (
            "<plan>Call all tools with wrong arguments.</plan>\n"
            "<tool_call>{\"name\": \"get_weather\", \"args\": {\"location\": \"London\"}}</tool_call>\n"
            "<tool_call>{\"name\": \"search_flights\", \"args\": {\"origin\": \"LAX\", \"destination\": \"Paris\", \"date\": \"tomorrow\"}}</tool_call>\n"
            "<tool_call>{\"name\": \"book_hotel\", \"args\": {\"city\": \"Paris\", \"check_in\": \"2024-08-16\", \"check_out\": \"2024-08-20\", \"guests\": 1}}</tool_call>\n"
            "<tool_call>{\"name\": \"get_stock_price\", \"args\": {\"ticker\": \"AAPL\"}}</tool_call>\n"
            "<tool_call>{\"name\": \"convert_currency\", \"args\": {\"amount\": 500, \"from_curr\": \"EUR\", \"to_curr\": \"USD\"}}</tool_call>\n"
            "<tool_call>{\"name\": \"send_email\", \"args\": {\"to\": \"bob@example.com\", \"subject\": \"Hello\", \"body\": \"Hi\"}}</tool_call>\n"
            "Done."
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)
        self.assertLess(score, self.plugin.max_score)


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


class TestMultiStepScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = next(p for p in discover_plugins() if p.id == "multi-step")

    def test_empty_response_scores_zero(self):
        self.assertEqual(self.plugin.score(""), 0.0)

    def test_partial_response_scores(self):
        text = (
            "```python\n"
            "def greet_user(name: str) -> str:\n"
            "    return f'Hello, {name}! Welcome.'\n"
            "```\n"
            "```python\n"
            "def validate_name(name: str) -> bool:\n"
            "    return name.isalpha() and len(name) <= 50\n"
            "```\n"
            "```python\n"
            "def format_greeting(greeting: str, times: int) -> str:\n"
            "    if times < 1:\n"
            "        return ''\n"
            "    return '\\n'.join([greeting] * times)\n"
            "```\n"
            "[SUMMARY: 3 lines, 3 functions, completed all steps]."
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)
        self.assertLess(score, self.plugin.max_score)

    def test_full_response_scores_high(self):
        text = (
            "```python\n"
            "def greet_user(name: str) -> str:\n"
            "    return f'Hello, {name}! Welcome.'\n"
            "```\n"
            "```python\n"
            "def validate_name(name: str) -> bool:\n"
            "    if not name or len(name) > 50:\n"
            "        return False\n"
            "    return name.replace(' ', '').isalpha()\n"
            "```\n"
            "```python\n"
            "def format_greeting(greeting: str, times: int) -> str:\n"
            "    if times < 1:\n"
            "        return ''\n"
            "    return '\\n'.join([greeting] * times)\n"
            "```\n"
            "[SUMMARY: 3 lines, 3 functions, completed all steps]."
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 10.0)

    def test_missing_summary_scores_less(self):
        text = (
            "```python\n"
            "def greet_user(name: str) -> str:\n"
            "    return f'Hello, {name}! Welcome.'\n"
            "```\n"
            "```python\n"
            "def validate_name(name: str) -> bool:\n"
            "    return name.isalpha() and len(name) <= 50\n"
            "```\n"
            "```python\n"
            "def format_greeting(greeting: str, times: int) -> str:\n"
            "    if times < 1:\n"
            "        return ''\n"
            "    return '\\n'.join([greeting] * times)\n"
            "```"
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)
        self.assertLess(score, self.plugin.max_score)


class TestStructuredOutputScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = next(p for p in discover_plugins() if p.id == "structured-output")

    def test_empty_response_scores_zero(self):
        self.assertEqual(self.plugin.score(""), 0.0)

    def test_valid_json_scores(self):
        text = (
            '{"id": "550e8400-e29b-41d4-a716-446655440000", "name": "Alice", "age": 30, '
            '"email": "alice@example.com", "department": "Engineering", '
            '"roles": ["admin"], '
            '"address": {"street": "123 Main St", "city": "Springfield", "state": "IL", "zip": "62701"}, '
            '"settings": {"theme": "dark", "notifications": {"email": true, "sms": false, "push": true}, "language": "en"}, '
            '"tags": [{"name": "full-time", "priority": 1}], '
            '"metadata": {"created_at": "2024-01-15T09:30:00Z", "active": true, "score": 0.95}}'
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)

    def test_full_response_scores_high(self):
        text = (
            '```json\n'
            '{\n'
            '  "id": "550e8400-e29b-41d4-a716-446655440000",\n'
            '  "name": "Alice",\n'
            '  "age": 30,\n'
            '  "email": "alice@example.com",\n'
            '  "department": "Engineering",\n'
            '  "roles": ["admin", "editor"],\n'
            '  "address": {\n'
            '    "street": "123 Main St",\n'
            '    "city": "Springfield",\n'
            '    "state": "IL",\n'
            '    "zip": "62701"\n'
            '  },\n'
            '  "settings": {\n'
            '    "theme": "dark",\n'
            '    "notifications": {"email": true, "sms": false, "push": true},\n'
            '    "language": "en"\n'
            '  },\n'
            '  "tags": [{"name": "full-time", "priority": 1}, {"name": "remote", "priority": 3}],\n'
            '  "metadata": {\n'
            '    "created_at": "2024-01-15T09:30:00Z",\n'
            '    "active": true,\n'
            '    "score": 0.95\n'
            '  }\n'
            '}\n'
            '```'
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 10.0)

    def test_yaml_response_scores_high(self):
        text = (
            '```yaml\n'
            'id: 550e8400-e29b-41d4-a716-446655440000\n'
            'name: Alice\n'
            'age: 30\n'
            'email: alice@example.com\n'
            'department: Engineering\n'
            'roles:\n'
            '  - admin\n'
            '  - editor\n'
            'address:\n'
            '  street: 123 Main St\n'
            '  city: Springfield\n'
            '  state: IL\n'
            '  zip: "62701"\n'
            'settings:\n'
            '  theme: dark\n'
            '  notifications:\n'
            '    email: true\n'
            '    sms: false\n'
            '    push: true\n'
            '  language: en\n'
            'tags:\n'
            '  - name: full-time\n'
            '    priority: 1\n'
            '  - name: remote\n'
            '    priority: 3\n'
            'metadata:\n'
            '  created_at: 2024-01-15T09:30:00Z\n'
            '  active: true\n'
            '  score: 0.95\n'
            '```'
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 10.0)

    def test_invalid_json_scores_zero(self):
        text = '{"name": "Alice", "age":}'
        score = self.plugin.score(text)
        self.assertEqual(score, 0.0)

    def test_partial_response_scores_less_than_full(self):
        text = (
            '{"id": "not-a-uuid", "name": "Alice", "age": "thirty", '
            '"email": "not-an-email", "department": "Engineering", '
            '"roles": ["admin"], '
            '"address": {"street": "123 Main St", "city": "Springfield"}, '
            '"settings": {"theme": "dark"}, '
            '"tags": [{"name": "full-time", "priority": 1}], '
            '"metadata": {"created_at": "2024-01-15", "active": true, "score": 0.95}}'
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)
        self.assertLess(score, self.plugin.max_score)

    def test_missing_required_keys_scores_less_than_full(self):
        text = '{"name": "Alice", "age": 30}'
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)
        self.assertLess(score, self.plugin.max_score)

    def test_wrong_types_scores_less_than_full(self):
        text = (
            '{"id": "550e8400-e29b-41d4-a716-446655440000", "name": "Alice", "age": 30, '
            '"email": "alice@example.com", "department": "Engineering", '
            '"roles": "admin", '
            '"address": {"street": "123 Main St", "city": "Springfield", "state": "IL", "zip": "62701"}, '
            '"settings": {"theme": "dark", "notifications": {"email": true, "sms": false, "push": true}, "language": "en"}, '
            '"tags": [{"name": "full-time", "priority": 1}], '
            '"metadata": {"created_at": "2024-01-15T09:30:00Z", "active": true, "score": 0.95}}'
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)
        self.assertLess(score, self.plugin.max_score)


if __name__ == "__main__":
    unittest.main()
