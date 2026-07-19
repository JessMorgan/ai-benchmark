import unittest

from plugins import discover_plugins
from plugins.outputs.output_csv import CSVOutputPlugin


class TestCSVOutputPlugin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = CSVOutputPlugin()
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

    def test_gen_csv_contains_headers(self):
        csv_text = self.plugin.generate(self.sample_results, self.plugins)
        self.assertIn("Model", csv_text)

    def test_gen_csv_contains_model_data(self):
        csv_text = self.plugin.generate(self.sample_results, self.plugins)
        self.assertIn("test-model", csv_text)

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
        csv_text = self.plugin.generate(results, self.plugins)
        self.assertIn("partial-model", csv_text)
        self.assertIn("fail", csv_text)
        self.assertIn("10.0", csv_text)
