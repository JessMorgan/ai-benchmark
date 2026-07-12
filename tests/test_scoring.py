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


if __name__ == "__main__":
    unittest.main()
