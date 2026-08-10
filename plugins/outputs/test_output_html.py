import unittest

from plugins import discover_plugins
from plugins.outputs.output_html import HTMLOutputPlugin


class TestHTMLOutputPlugin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = HTMLOutputPlugin()
        cls.plugins = discover_plugins()
        cls.sample_results = [
            {
                "model": "test-model",
                "source": "Local",
                "status": "ok",
                "stream_ok": True,
                "ttft": 1.2,
                "rate-limiter_score": 78,
                "rate-limiter_rubric": [
                    {"name": "Interface design", "points": 3, "total": 3},
                    {"name": "Token Bucket", "points": 4, "total": 4},
                ],
                "rate-limiter_response_time": 5.0,
                "rate-limiter_output_tokens": 100,
                "rate-limiter_tps": 20.0,
                "moe-dense_score": 59,
                "moe-dense_rubric": [
                    {"name": "Covers both architectures", "points": 2, "total": 2},
                ],
                "moe-dense_response_time": 3.0,
                "moe-dense_output_tokens": 50,
                "moe-dense_tps": 16.7,
                "total_time": 8.0,
            },
            {
                "model": "fail-model",
                "source": "Local",
                "status": "error",
                "error": "timeout",
                "total_time": 5.0,
            },
        ]

    def test_gen_html_contains_rows(self):
        html = self.plugin.generate(self.sample_results, self.plugins)
        self.assertIn("test-model", html)

    def test_gen_html_includes_session_seed(self):
        html = self.plugin.generate(self.sample_results, self.plugins, session_seed=12345)
        self.assertIn("12345", html)

    def test_gen_html_no_seed_when_session_seed_is_none(self):
        html = self.plugin.generate(self.sample_results, self.plugins, session_seed=None)
        self.assertNotIn("Seed:", html)

    def test_gen_html_includes_rubric_breakdown(self):
        html = self.plugin.generate(self.sample_results, self.plugins)
        self.assertIn("Interface design", html)

    def test_gen_html_includes_response_links_with_output_dir(self):
        html = self.plugin.generate(self.sample_results, self.plugins, output_dir="/tmp/benchmark-results")
        self.assertIn("responses/", html)

    def test_gen_html_includes_empty_reason_column(self):
        """The HTML table header must include a Reason column per plugin, and
        rows with an empty_reason value display it."""
        results = [dict(self.sample_results[0])]
        results[0]["rate-limiter_empty_reason"] = "thinking-truncation"
        html = self.plugin.generate(results, self.plugins)
        self.assertIn("Rate Limiter Reason", html)
        self.assertIn("thinking-truncation", html)

    def test_gen_html_empty_reason_blank_when_unset(self):
        """Rows without empty_reason render an empty cell."""
        html = self.plugin.generate(self.sample_results, self.plugins)
        self.assertIn("Rate Limiter Reason", html)
        # The first data row has no empty_reason set; the reason cell is
        # an empty <td></td> (not a classification label). Avoid checking
        # for "empty" which appears in the CSS class name ".empty-reason".
        for label in ("thinking-truncation", "thinking-only", "max-tokens", "error"):
            self.assertNotIn(label, html)
        # Check the data row has an empty reason cell by looking for the
        # pattern: Rate Limiter Score cell followed by an empty cell.
        self.assertIn("><strong>78</strong></td><td></td>", html)

    def test_gen_html_no_response_links_without_output_dir(self):
        html = self.plugin.generate(self.sample_results, self.plugins)
        self.assertNotIn("responses/", html)

    def test_gen_html_includes_thinking_content_total_token_columns(self):
        """The HTML table breaks token usage into Think Tok / Cont Tok /
        Total Tok header cells, with Total Tok emphasised."""
        results = [dict(self.sample_results[0])]
        results[0]["rate-limiter_thinking_tokens"] = 40
        results[0]["rate-limiter_total_tokens"] = 140
        html = self.plugin.generate(results, self.plugins)
        self.assertIn("Rate Limiter Think Tok", html)
        self.assertIn("Rate Limiter Cont Tok", html)
        self.assertIn("Rate Limiter Total Tok", html)
        self.assertIn("<td>40</td>", html)
        self.assertIn("<td>100</td>", html)
        self.assertIn("<td><strong>140</strong></td>", html)

    def test_output_generators_render_partial_failure(self):
        results = [
            {
                "model": "partial-model",
                "source": "Local",
                "status": "error",
                "error": "rate-limiter failed",
                "total_time": 5.0,
                "stream_ok": False,
                "rate-limiter_score": "fail",
                "rate-limiter_response_time": "fail",
                "rate-limiter_output_tokens": "fail",
                "rate-limiter_tps": "fail",
                "moe-dense_score": 59,
                "moe-dense_response_time": 3.0,
                "moe-dense_output_tokens": 50,
                "moe-dense_tps": 16.7,
            },
        ]
        html = self.plugin.generate(results, self.plugins)
        self.assertIn("partial-model", html)
        self.assertIn("fail", html)
        self.assertIn("59", html)

    def test_output_generators_handle_ok_with_string_score(self):
        results = [
            {
                "model": "weird-model",
                "source": "Local",
                "status": "ok",
                "stream_ok": True,
                "ttft": 1.0,
                "total_time": 5.0,
                "rate-limiter_score": "fail",
                "rate-limiter_response_time": "fail",
                "rate-limiter_output_tokens": "fail",
                "rate-limiter_tps": "fail",
                "moe-dense_score": 59,
                "moe-dense_response_time": 3.0,
                "moe-dense_output_tokens": 50,
                "moe-dense_tps": 16.7,
            },
            {
                "model": "good-model",
                "source": "Local",
                "status": "ok",
                "stream_ok": True,
                "ttft": 2.0,
                "total_time": 4.0,
                "rate-limiter_score": 18.0,
                "rate-limiter_response_time": 1.0,
                "rate-limiter_output_tokens": 100,
                "rate-limiter_tps": 50.0,
                "moe-dense_score": 12.0,
                "moe-dense_response_time": 2.0,
                "moe-dense_output_tokens": 40,
                "moe-dense_tps": 20.0,
            },
        ]
        html = self.plugin.generate(results, self.plugins)
        self.assertIn("weird-model", html)
        self.assertIn("good-model", html)
