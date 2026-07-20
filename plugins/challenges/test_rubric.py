"""Tests for the shared Rubric helper."""

from plugins.challenges._rubric import Rubric


class TestRubric:
    def test_empty_rubric_returns_zero(self):
        rubric = Rubric(max_score=10)
        score, criteria = rubric.results()
        assert score == 0.0
        assert criteria == []

    def test_add_criterion_accumulates_score(self):
        rubric = Rubric(max_score=10)
        rubric.add_criterion("completeness", 5, 3.5)
        rubric.add_criterion("clarity", 5, 4.0)
        score, criteria = rubric.results()
        assert score == 7.5
        assert len(criteria) == 2
        assert criteria[0]["name"] == "completeness"
        assert criteria[0]["max"] == 5
        assert criteria[0]["earned"] == 3.5
        assert criteria[0]["missed"] == 1.5
        assert criteria[1]["earned"] == 4.0
        assert criteria[1]["missed"] == 1.0

    def test_earned_points_clamped_to_criterion_max(self):
        rubric = Rubric(max_score=10)
        rubric.add_criterion("overmax", 5, 8.0)
        score, criteria = rubric.results()
        assert criteria[0]["earned"] == 5.0
        assert criteria[0]["missed"] == 0.0
        assert score == 5.0

    def test_final_score_clamped_to_max_score(self):
        rubric = Rubric(max_score=10)
        rubric.add_criterion("one", 10, 8.0)
        rubric.add_criterion("two", 10, 8.0)
        score, _ = rubric.results()
        assert score == 10.0

    def test_eval_regex_sums_matched_patterns(self):
        rubric = Rubric(max_score=10)
        text = "The quick brown fox jumps over the lazy dog."
        patterns = [
            (r"\bquick\b", 2.0),
            (r"\bfake\b", 3.0),
            (r"\blazy\b", 2.5),
        ]
        rubric.eval_regex("matches", 5.0, text, patterns)
        score, criteria = rubric.results()
        assert score == 4.5
        assert criteria[0]["earned"] == 4.5
        assert criteria[0]["missed"] == 0.5

    def test_eval_regex_respects_custom_flags(self):
        rubric = Rubric(max_score=10)
        text = "HELLO World"
        patterns = [(r"hello", 1.0)]
        rubric.eval_regex("case", 1.0, text, patterns, flags=0)
        score, _ = rubric.results()
        assert score == 0.0

    def test_rounding_to_one_decimal(self):
        rubric = Rubric(max_score=10)
        rubric.add_criterion("precise", 3, 1.234)
        score, criteria = rubric.results()
        assert criteria[0]["earned"] == 1.2
        assert criteria[0]["missed"] == 1.8
        assert score == 1.2
