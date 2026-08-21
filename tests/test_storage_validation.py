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


if __name__ == "__main__":
    unittest.main()
