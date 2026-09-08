"""Source-aware code-review challenge."""
from __future__ import annotations

import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from benchmark.types import ConfigMap
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import Validation, parse_structured

# Keyword co-occurrence is polarity-blind: a finding that denies a defect
# ("closed properly", "never raises") matches the same keywords as one that
# asserts it. Shared denials apply to every defect; each check adds its own.
_SHARED_DENIALS: tuple[str, ...] = (
    r"\bis fine\b",
    r"\bworks correctly\b",
    r"\bno problem\b",
    r"\bno issues?\b",
    r"\bcorrect as written\b",
    r"\bno defects?\b",
)


class CodeReviewPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "code-review"

    @property
    def version(self) -> str:
        return "1.2.0"

    @property
    def name(self) -> str:
        return "Code Review"

    @property
    def max_score(self) -> int:
        return int(15.0)

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_prompt(self) -> str:
        return (
            "Review this Python function. Identify the concrete defects, cite the relevant "
            "construct, and give a remediation. Return JSON `{\"issues\": [{\"description\": "
            "\"...\"}]}`; understandable bullet findings are also accepted.\n\n```python\n"
            "import os\nimport time\n\ndef process_user_data(user_ids, db_path=\"/tmp/data.txt\"):\n"
            "    results = []\n    f = open(db_path, \"w\")\n    for i in range(len(user_ids)):\n"
            "        user_id = user_ids[i]\n        if user_id == None:\n            continue\n"
            "        data = fetch_data(user_id)\n        if data:\n            results.append(data)\n"
            "    f.write(str(results))\n    return results\n```"
        )

    def get_temperature(self, global_config: ConfigMap) -> float | None:
        val = global_config.get("code_review_temperature")
        return float(val) if isinstance(val, (int, float)) else None

    @staticmethod
    def _descriptions(text: str, validation: Validation) -> tuple[list[str], bool]:
        """Extract independent findings without requiring valid JSON syntax.

        Uses the shared structured-extraction candidate (the single fenced
        JSON block, or the whole response, as parsed by parse_structured)
        before falling back to bullet extraction. The issue list is the
        ``issues`` array of a JSON object or a top-level JSON array.
        Returns the findings plus a flag for the JSON dead-end: a JSON
        object or top-level array was recognized but yielded no recognized
        findings.
        """
        json_dead_end = False
        value = validation.value
        issues = None
        if isinstance(value, dict):
            candidate = value.get("issues", [])
            if isinstance(candidate, list):
                issues = candidate
        elif isinstance(value, list):
            issues = value
        if issues is not None:
            descriptions = [
                str(
                    item.get("description")
                    or item.get("finding")
                    or item.get("issue")
                    or ""
                ).strip().lower()
                for item in issues
                if isinstance(item, dict)
            ]
            descriptions = [description for description in descriptions if description]
            if descriptions:
                return descriptions, False
            json_dead_end = True
        return [
            match.group(1).strip().lower()
            for match in re.finditer(
                r"(?m)^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$", text
            )
        ], json_dead_end

    @staticmethod
    def _finding_matches(
        findings: list[str],
        groups: tuple[tuple[str, ...], ...],
        denials: tuple[str, ...],
        used: set[int],
    ) -> tuple[bool, str, int]:
        """Require one finding to assert the defect and its remediation.

        A finding matching any of the defect's denial patterns is a
        negation, not an assertion, and does not count. Findings already
        consumed by another defect are skipped: one finding satisfies at
        most one defect.
        """
        for index, finding in enumerate(findings):
            if index in used:
                continue
            if any(re.search(term, finding, re.IGNORECASE) for term in denials):
                continue
            if all(any(re.search(term, finding, re.IGNORECASE) for term in group) for group in groups):
                return True, finding, index
        return False, "", -1

    def evaluate(self, response_text: str) -> EvaluationResult:
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        if not text:
            return EvaluationResult(0.0, [])
        validation = parse_structured(text, fmt="json")
        findings, json_dead_end = self._descriptions(text, validation)
        rubric.record_validation(validation)
        if json_dead_end:
            # A recognized JSON object with no findings is a format-contract
            # violation; name it (or surface the shared multi-candidate
            # rejection verbatim when that is the cause).
            contract = next(
                (
                    error
                    for error in validation.errors
                    if "exactly one structured candidate is required" in error
                ),
                (
                    "no findings recognized from the JSON candidate; expected the format "
                    'contract: a JSON object with an "issues" array of issue objects, a '
                    "top-level JSON array of issue objects, or bullet findings"
                ),
            )
            rubric.record_validation(Validation(False, errors=[contract]))
        if not findings:
            return rubric.results()

        checks = [
            (
                "File handle not closed / resource leak", 3.0,
                ((r"f\s*=|open\(",), (r"close|context\s+manager|with\s+open|leak",)),
                (
                    r"\bno leak", r"does not leak", r"doesn'?t leak",
                    r"closed properly", r"properly closed", r"already closed",
                    r"no resource leak",
                ),
            ),
            (
                "== None instead of is None", 2.0,
                ((r"==\s*none|identity",), (r"is\s+none|comparison|user_id",)),
                (r"correct comparison", r"comparison is correct", r"identity check is fine"),
            ),
            (
                "Hardcoded /tmp path", 2.0,
                ((r"/tmp/data\.txt|hardcoded",), (r"path|parameter|config|inject",)),
                (r"\bacceptable\b", r"standard practice", r"no problem with the path"),
            ),
            (
                "Missing error handling / fetch_data may fail", 3.0,
                ((r"fetch_data",), (r"exception|error|try|except|failure|handling",)),
                (
                    r"never raises", r"\bno exception", r"does not fail", r"doesn'?t fail",
                    r"cannot fail", r"no (error|exception) handling (is )?needed",
                    r"no handling needed",
                ),
            ),
            (
                "Unused imports", 2.0,
                ((r"unused|not used|remove",), (r"\bos\b|\btime\b|import",)),
                (r"\bare used\b", r"both (are )?used", r"\bno unused\b", r"used elsewhere", r"imports? are used"),
            ),
        ]
        used: set[int] = set()
        matched_findings: list[str] = []
        for name, maximum, groups, denials in checks:
            matched, finding, index = self._finding_matches(
                findings, groups, _SHARED_DENIALS + denials, used,
            )
            if matched:
                used.add(index)
                matched_findings.append(finding)
            rubric.add_criterion(
                name, maximum, maximum if matched else 0.0,
                evidence=[{"kind": "finding", "span": finding}] if matched else [],
                negative_findings=[] if matched else [{"finding": "no independent finding with both the defect and its remediation"}],
            )

        remediation_terms = (r"\buse\b", r"replace", r"close", r"context manager", r"is none", r"parameterize", r"inject", r"try", r"except", r"remove", r"validate", r"sanitize")
        actionable = sum(
            bool(re.search(term, finding, re.IGNORECASE))
            for finding in matched_findings
            for term in remediation_terms
        )
        rubric.add_criterion(
            "Actionable / concrete fixes", 2.0,
            min(2.0, float(actionable)),
            negative_findings=[] if actionable else [{"finding": "each defect should include a concrete remediation"}],
        )
        citation_terms = (r"user_ids", r"db_path", r"fetch_data", r"f\.write", r"==\s*None", r"/tmp/data\.txt")
        citations = sum(
            any(re.search(term, finding, re.IGNORECASE) for term in citation_terms)
            for finding in findings
        )
        citations_ok = citations >= 3
        rubric.add_criterion(
            "Source citations", 1.0,
            1.0 if citations_ok else 0.0,
            evidence=[{"kind": "source-citation-count", "count": citations}],
            negative_findings=[] if citations_ok else [
                {"finding": "cite the relevant variable, call, or literal in at least three distinct findings"},
            ],
        )
        return rubric.results()

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
