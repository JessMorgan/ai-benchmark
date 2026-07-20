"""Shared rubric helper for challenge plugins."""
import re


class Rubric:
    """Builds a scored rubric for a benchmark task.

    Each criterion is recorded with its name, maximum possible points, and
    earned points. The helper clamps earned points to the criterion max,
    computes missed points, and returns a final score clamped to the task's
    overall max score.
    """

    def __init__(self, max_score: float):
        self.max_score = max_score
        self.criteria: list[dict] = []
        self.total = 0.0

    def add_criterion(self, name: str, max_points: float, earned: float):
        """Add a manually scored criterion.

        Args:
            name: Human-readable criterion name.
            max_points: Maximum points for this criterion.
            earned: Earned points (will be clamped to max_points).
        """
        earned = round(min(earned, max_points), 1)
        missed = round(max_points - earned, 1)
        self.total += earned
        self.criteria.append({
            "name": name,
            "max": max_points,
            "earned": earned,
            "missed": missed,
        })

    def eval_regex(self, name: str, max_points: float, text: str, patterns, flags=re.IGNORECASE):
        """Score a criterion by summing points for each matched regex pattern.

        Args:
            name: Human-readable criterion name.
            max_points: Maximum points for this criterion.
            text: Response text to search.
            patterns: Iterable of (regex_pattern, points) tuples.
            flags: Regex flags to use.
        """
        earned = 0.0
        for pattern, points in patterns:
            if re.search(pattern, text, flags):
                earned += points
        self.add_criterion(name, max_points, earned)

    def results(self) -> tuple[float, list[dict]]:
        """Return the final score and the rubric list."""
        final_score = round(min(self.total, self.max_score), 1)
        return final_score, self.criteria
