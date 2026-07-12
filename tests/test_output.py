"""Tests for output generators."""
import unittest
from unittest.mock import patch

from plugins import discover_plugins
from tests.utils import load_benchmark_module


class TestOutputGenerators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()
        cls.plugins = discover_plugins()
        cls.sample_results = [
            {
                "model": "test-model",
                "source": "Local",
                "status": "ok",
                "stream_ok": True,
                "ttft": 1.2,
                "rate-limiter_score": 15.5,
                "rate-limiter_response_time": 5.0,
                "rate-limiter_output_tokens": 100,
                "rate-limiter_tps": 20.0,
                "moe-dense_score": 10.0,
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
            },
        ]

    def test_gen_csv_contains_headers(self):
        with patch.object(self.module, "ACTIVE_PLUGINS", self.plugins):
            csv_text = self.module.gen_csv(self.sample_results)
        self.assertIn("Model", csv_text)
        self.assertIn("rate-limiter_Score_20", csv_text)
        self.assertIn("moe-dense_Score_15", csv_text)

    def test_gen_csv_contains_model_data(self):
        with patch.object(self.module, "ACTIVE_PLUGINS", self.plugins):
            csv_text = self.module.gen_csv(self.sample_results)
        self.assertIn("test-model", csv_text)
        self.assertIn("15.5", csv_text)
        self.assertIn("10.0", csv_text)

    def test_gen_markdown_contains_plugin_names(self):
        with patch.object(self.module, "ACTIVE_PLUGINS", self.plugins):
            md = self.module.gen_markdown(self.sample_results)
        self.assertIn("Rate Limiter", md)
        self.assertIn("MoE vs Dense", md)

    def test_gen_html_contains_rows(self):
        with patch.object(self.module, "ACTIVE_PLUGINS", self.plugins):
            html = self.module.gen_html(self.sample_results)
        self.assertIn("test-model", html)
        self.assertIn("<table>", html)


if __name__ == "__main__":
    unittest.main()
