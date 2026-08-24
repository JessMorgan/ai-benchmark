"""Shared rubric helper for challenge plugins."""
import re
from collections.abc import Iterable, Sequence
from typing import Any

from benchmark.plugin import EvaluationResult


class Rubric:
    """Builds a scored rubric for a benchmark task.

    Each criterion is recorded with its name, maximum possible points, and
    earned points. The helper clamps earned points to the criterion max,
    computes missed points, and returns a final score clamped to the task's
    overall max score.
    """

    def __init__(self, max_score: float) -> None:
        self.max_score = max_score
        self.criteria: list[dict[str, Any]] = []
        self.total = 0.0
        self.errors: list[str] = []
        self.validations: list[dict[str, Any]] = []

    def add_criterion(
        self,
        name: str,
        max_points: float,
        earned: float,
        *,
        evidence: Iterable[Any] | None = None,
        matched: bool | None = None,
        negative_findings: Iterable[Any] | None = None,
        errors: Iterable[Any] | None = None,
    ) -> None:
        """Add a manually scored criterion and its diagnostic evidence.

        Every criterion carries JSON-safe diagnostic containers. ``matched``
        is a boolean summary of whether the criterion earned any credit unless
        the caller supplies a more precise signal.
        """
        earned = round(max(0.0, min(earned, max_points)), 1)
        missed = round(max_points - earned, 1)
        self.total += earned
        item = {
            "name": name,
            "max": max_points,
            "earned": earned,
            "missed": missed,
        }
        item["evidence"] = list(evidence or [])
        item["matched"] = bool(earned) if matched is None else bool(matched)
        item["negative_findings"] = list(negative_findings or [])
        item["errors"] = [str(error) for error in (errors or [])]
        self.errors.extend(str(error) for error in (errors or []))
        self.criteria.append(item)

    def eval_regex(
        self,
        name: str,
        max_points: float,
        text: str,
        patterns: Sequence[tuple[str, float]],
        flags: int = re.IGNORECASE,
    ) -> None:
        """Score a criterion by summing points for each matched regex pattern."""
        earned = 0.0
        evidence = []
        for pattern, points in patterns:
            match = re.search(pattern, text, flags)
            if match:
                earned += points
                evidence.append({
                    "kind": "regex",
                    "pattern": pattern,
                    "span": match.group(0),
                    "points": points,
                })
        self.add_criterion(
            name, max_points, earned,
            evidence=evidence,
            matched=bool(evidence),
        )

    def credit_criterion(self, name: str, points: float, evidence: Any = None) -> None:
        """Add bounded credit when a stronger validator proves a criterion.

        This is useful when execution or typed validation establishes
        correctness that a lexical sub-check could not recognize. The total
        remains clamped to the criterion maximum, and the diagnostic record
        explains why the credit was added.
        """
        for criterion in reversed(self.criteria):
            if criterion["name"] != name:
                continue
            credit = min(
                max(0.0, points),
                max(0.0, criterion["max"] - criterion["earned"]),
            )
            if credit:
                criterion["earned"] = round(criterion["earned"] + credit, 1)
                criterion["missed"] = round(criterion["max"] - criterion["earned"], 1)
                criterion.setdefault("evidence", []).append({
                    "kind": "validator-credit",
                    "points": credit,
                    "evidence": evidence,
                })
                criterion["matched"] = True
                self.total += credit
            return
        self.errors.append(f"cannot credit unknown criterion {name!r}")

    def penalize_criterion(self, name: str, points: float, finding: str) -> None:
        """Apply a bounded negative finding to an existing criterion."""
        for criterion in reversed(self.criteria):
            if criterion["name"] != name:
                continue
            deduction = min(max(0.0, points), criterion["earned"])
            criterion["earned"] = round(criterion["earned"] - deduction, 1)
            criterion["missed"] = round(criterion["max"] - criterion["earned"], 1)
            criterion.setdefault("negative_findings", []).append({
                "finding": finding,
                "points": deduction,
            })
            self.total = max(0.0, self.total - deduction)
            return
        self.errors.append(f"cannot penalize unknown criterion {name!r}")

    def record_validation(self, validation: Any) -> None:
        """Attach a typed-validator result without changing score totals."""
        if hasattr(validation, "as_evidence"):
            self.validations.append(validation.as_evidence())
            return
        self.validations.append({
            "valid": bool(validation.valid),
            "evidence": list(validation.evidence or []),
            "errors": [str(error) for error in (validation.errors or [])],
        })
        self.errors.extend(str(error) for error in (validation.errors or []))

    def record_execution(
        self,
        execution: Any,
        *,
        criterion: str | None = None,
        penalty: float = 0.0,
        failure_reason: str = "isolated execution check failed",
    ) -> None:
        """Record an execution result and optionally deduct only on failure."""
        self.validations.append(execution.as_evidence())
        if criterion and execution.status in {"failed", "timeout"}:
            self.penalize_criterion(criterion, penalty, failure_reason)

    def results(self) -> EvaluationResult:
        """Return the final score, rubric list, and evaluation diagnostics."""
        final_score = round(min(self.total, self.max_score), 1)
        diagnostics = {
            "criterion_count": len(self.criteria),
            "matched_criterion_count": sum(
                1 for criterion in self.criteria if criterion.get("matched")
            ),
            "errors": list(self.errors),
            "validations": list(self.validations),
        }
        return EvaluationResult(final_score, self.criteria, diagnostics)
