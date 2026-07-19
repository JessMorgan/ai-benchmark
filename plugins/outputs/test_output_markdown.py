import unittest

from plugins import discover_plugins
from plugins.outputs.output_markdown import MarkdownOutputPlugin


class TestMarkdownOutputPlugin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = MarkdownOutputPlugin()
        cls.plugins = discover_plugins()
        cls.sample_results = [
            {
                "model": "test-model",
                "source": "Local",
                "status": "ok",
                "stream_ok": True,
                "ttft": 1.2,
                "rate-limiter_score": 15.5,
                "rate-limiter_rubric": [
                    {"name": "Interface design", "max": 3.0, "earned": 3.0, "missed": 0.0},
                    {"name": "Token Bucket", "max": 4.0, "earned": 4.0, "missed": 0.0},
                ],
                "rate-limiter_response_time": 5.0,
                "rate-limiter_output_tokens": 100,
                "rate-limiter_tps": 20.0,
                "moe-dense_score": 10.0,
                "moe-dense_rubric": [
                    {"name": "Covers both architectures", "max": 2.0, "earned": 2.0, "missed": 0.0},
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

    def test_gen_markdown_contains_plugin_names(self):
        md = self.plugin.generate(self.sample_results, self.plugins)
        self.assertIn("**Rate Limiter**", md)

    def test_gen_markdown_includes_session_seed(self):
        md = self.plugin.generate(self.sample_results, self.plugins, session_seed=12345)
        self.assertIn("**Seed:** 12345", md)

    def test_gen_markdown_no_seed_line_when_session_seed_is_none(self):
        md = self.plugin.generate(self.sample_results, self.plugins, session_seed=None)
        self.assertNotIn("**Seed:**", md)

    def test_gen_markdown_includes_rubric_breakdown(self):
        md = self.plugin.generate(self.sample_results, self.plugins)
        self.assertIn("Interface design", md)
        self.assertIn("Token Bucket", md)

    def test_gen_markdown_includes_response_links_with_output_dir(self):
        md = self.plugin.generate(self.sample_results, self.plugins, output_dir="/tmp/benchmark-results")
        self.assertIn("[view]", md)

    def test_gen_markdown_no_response_links_without_output_dir(self):
        md = self.plugin.generate(self.sample_results, self.plugins)
        self.assertNotIn("[view]", md)

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
                "moe-dense_score": 10.0,
                "moe-dense_response_time": 3.0,
                "moe-dense_output_tokens": 50,
                "moe-dense_tps": 16.7,
            },
        ]
        md = self.plugin.generate(results, self.plugins)
        self.assertIn("partial-model", md)
        self.assertIn("fail", md)
        self.assertIn("10.0", md)

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
                "moe-dense_score": 10.0,
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
        md = self.plugin.generate(results, self.plugins)
        self.assertIn("weird-model", md)
        self.assertIn("good-model", md)
