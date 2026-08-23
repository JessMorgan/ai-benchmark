"""Validation helpers for comparing JSON and SQLite current read models."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationDifference:
    category: str
    identity: str
    left: Any
    right: Any


@dataclass(frozen=True)
class ValidationReport:
    equivalent: bool
    differences: tuple[ValidationDifference, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "equivalent": self.equivalent,
            "differences": [
                {
                    "category": difference.category,
                    "identity": difference.identity,
                    "left": difference.left,
                    "right": difference.right,
                }
                for difference in self.differences
            ],
        }


def compare_read_models(
    left: Iterable[dict[str, Any]], right: Iterable[dict[str, Any]],
) -> ValidationReport:
    """Compare current result rows while ignoring presentation-only fields."""
    ignored = {
        "timestamp", "total_time", "ttft", "session_seed", "judge_status",
        "judge_models", "is_agent", "p_judge_models", "p_judge_consensus_by_contract",
        "p_judge_selected_contract", "p_judge_queued", "p_attempt_count",
        "p_plugin_version", "p_rubric", "p_diagnostics",
    }
    left_rows = _index(left)
    right_rows = _index(right)
    differences: list[ValidationDifference] = []
    for identity in sorted(set(left_rows) | set(right_rows)):
        if identity not in left_rows:
            differences.append(ValidationDifference("missing-left", identity, None, right_rows[identity]))
            continue
        if identity not in right_rows:
            differences.append(ValidationDifference("missing-right", identity, left_rows[identity], None))
            continue
        left_row = {
            key: _canonical_value(key, value)
            for key, value in left_rows[identity].items()
            if key not in ignored
        }
        right_row = {
            key: _canonical_value(key, value)
            for key, value in right_rows[identity].items()
            if key not in ignored
        }
        for key in sorted(set(left_row) | set(right_row)):
            if left_row.get(key) != right_row.get(key):
                differences.append(ValidationDifference(
                    _category_for_key(key), f"{identity}:{key}",
                    left_row.get(key), right_row.get(key),
                ))
    return ValidationReport(not differences, tuple(differences))


def _index(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        identity = f"{row.get('state_key', row.get('model'))}|{row.get('runner', 'http')}"
        indexed[identity] = row
    return indexed


def _canonical_value(key: str, value: Any) -> Any:
    """Normalize backend scalar and vote-shape differences for comparisons."""
    if key.endswith("_judge_votes") and isinstance(value, list):
        normalized = []
        for vote in value:
            if not isinstance(vote, dict):
                normalized.append(vote)
                continue
            normalized.append({
                field: vote.get(field)
                for field in ("model", "judge_contract_id", "score", "confidence", "rationale", "error", "usable")
                if vote.get(field) is not None
            })
        return normalized
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _category_for_key(key: str) -> str:
    if "judge" in key:
        return "judge"
    if "attempt" in key or "retry" in key:
        return "attempt"
    if key.endswith("_score") or key in {"status", "error"}:
        return "score-status"
    return "metadata"
