"""Tests for storage read-model validation."""
import unittest

from benchmark.storage_validation import compare_read_models


class TestStorageValidation(unittest.TestCase):
    def test_equivalent_models_ignore_presentation_fields(self):
        report = compare_read_models(
            [{"model": "m", "status": "ok", "p_score": 10, "timestamp": "a"}],
            [{"model": "m", "status": "ok", "p_score": 10, "timestamp": "b"}],
        )
        self.assertTrue(report.equivalent)
        self.assertEqual(report.as_dict()["differences"], [])

    def test_reports_score_and_judge_differences(self):
        report = compare_read_models(
            [{"model": "m", "status": "ok", "p_score": 10, "p_judge_score": 8}],
            [{"model": "m", "status": "ok", "p_score": 9, "p_judge_score": 7}],
        )
        self.assertFalse(report.equivalent)
        categories = {difference.category for difference in report.differences}
        self.assertEqual(categories, {"score-status", "judge"})

    def test_reports_missing_rows(self):
        report = compare_read_models([{"model": "a"}], [{"model": "b"}])
        self.assertFalse(report.equivalent)
        self.assertEqual(
            {difference.category for difference in report.differences},
            {"missing-left", "missing-right"},
        )

    def test_per_plugin_presentation_fields_ignored_for_real_plugin_ids(self):
        """Per-plugin presentation fields are keyed by plugin id, not a
        literal ``p_`` prefix.

        Regression: the ignore list once listed literal ``p_*`` keys that
        only matched the plugin id used by the parity fixtures, so a real
        plugin id such as ``code-review`` surfaced false differences for
        presentation-only fields (``code-review_plugin_version`` and
        friends) in ``--compare-storage``.
        """
        report = compare_read_models(
            [{"model": "m", "status": "ok", "code-review_score": 10}],
            [{
                "model": "m", "status": "ok", "code-review_score": 10,
                "code-review_plugin_version": "1.2.0",
                "code-review_rubric": [{"id": "c1", "points": 5}],
                "code-review_diagnostics": {"request_max_tokens": 16384},
                "code-review_attempt_count": 2,
                "code-review_retry_reason": "token_limit",
                "code-review_retry_reasons": ["token_limit"],
                "code-review_judge_models": ["judge-a"],
                "code-review_judge_consensus_by_contract": {"c1": {"score": 9}},
                "code-review_judge_selected_contract": "c1",
                "code-review_judge_queued": False,
            }],
        )
        self.assertTrue(report.equivalent, report.as_dict())

    def test_per_plugin_scoring_fields_still_compared_for_real_plugin_ids(self):
        """The suffix-based ignore must not swallow per-plugin scoring fields."""
        report = compare_read_models(
            [{"model": "m", "status": "ok", "code-review_score": 10}],
            [{"model": "m", "status": "ok", "code-review_score": 9}],
        )
        self.assertFalse(report.equivalent)
        self.assertEqual(
            {difference.category for difference in report.differences},
            {"score-status"},
        )


if __name__ == "__main__":
    unittest.main()
