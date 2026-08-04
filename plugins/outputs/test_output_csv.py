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

    def test_gen_csv_includes_empty_reason_column(self):
        """The CSV exposes the empty-response classification column; rows
        without a classification render blank."""
        results = [dict(self.sample_results[0])]
        results[0]["rate-limiter_empty_reason"] = "thinking-truncation"
        csv_text = self.plugin.generate(results, self.plugins)
        self.assertIn("rate-limiter_Empty_Reason", csv_text)
        self.assertIn("thinking-truncation", csv_text)

    def test_gen_csv_empty_reason_blank_when_unset(self):
        csv_text = self.plugin.generate(self.sample_results, self.plugins)
        rows = [line.split(",") for line in csv_text.strip().splitlines()]
        headers = rows[0]
        self.assertIn("rate-limiter_Empty_Reason", headers)
        idx = headers.index("rate-limiter_Empty_Reason")
        for row in rows[1:]:
            self.assertEqual(row[idx], "")

    def test_gen_csv_includes_thinking_content_total_token_columns(self):
        """The CSV breaks token usage into thinking / content / total
        columns per plugin (thinking from ``reasoning_content``)."""
        results = [dict(self.sample_results[0])]
        results[0]["rate-limiter_thinking_tokens"] = 40
        results[0]["rate-limiter_total_tokens"] = 140
        csv_text = self.plugin.generate(results, self.plugins)
        rows = [line.split(",") for line in csv_text.strip().splitlines()]
        headers = rows[0]
        for col in ("rate-limiter_Thinking_Tokens",
                    "rate-limiter_Content_Tokens",
                    "rate-limiter_Total_Tokens"):
            self.assertIn(col, headers)
        t_idx = headers.index("rate-limiter_Thinking_Tokens")
        c_idx = headers.index("rate-limiter_Content_Tokens")
        tot_idx = headers.index("rate-limiter_Total_Tokens")
        for row in rows[1:]:
            self.assertEqual(row[t_idx], "40")
            self.assertEqual(row[c_idx], "100")
            self.assertEqual(row[tot_idx], "140")

    def test_gen_csv_derives_total_when_split_absent(self):
        """Legacy results without thinking/total tokens derive total from
        the content-only count (thinking = 0)."""
        csv_text = self.plugin.generate(self.sample_results, self.plugins)
        rows = [line.split(",") for line in csv_text.strip().splitlines()]
        headers = rows[0]
        t_idx = headers.index("rate-limiter_Thinking_Tokens")
        c_idx = headers.index("rate-limiter_Content_Tokens")
        tot_idx = headers.index("rate-limiter_Total_Tokens")
        for row in rows[1:]:
            if row[headers.index("Model")] != "test-model":
                continue
            self.assertEqual(row[t_idx], "0")
            self.assertEqual(row[c_idx], "100")
            self.assertEqual(row[tot_idx], "100")
