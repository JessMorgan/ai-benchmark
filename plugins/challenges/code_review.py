"""Source-aware code-review challenge."""
from __future__ import annotations

import json
import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from benchmark.types import ConfigMap
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import parse_structured


class CodeReviewPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "code-review"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def name(self) -> str:
        return "Code Review"

    @property
    def max_score(self) -> int:
        return int(15.0)

    @property
    def supports_streaming(self) -> bool:
        return False

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
    def _descriptions(text: str) -> list[str]:
        """Extract independent findings without requiring valid JSON syntax."""
        try:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                value = json.loads(text[start:end + 1])
                issues = value.get("issues", []) if isinstance(value, dict) else []
                if isinstance(issues, list):
                    return [
                        str(
                            item.get("description")
                            or item.get("finding")
                            or item.get("issue")
                            or ""
                        ).strip().lower()
                        for item in issues
                        if item
                    ]
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        return [
            match.group(1).strip().lower()
            for match in re.finditer(
                r"(?m)^\s*(?:[-*]|\d+[.)])\s+(.+?)\s*$", text
            )
        ]

    @staticmethod
    def _finding_matches(findings: list[str], groups: tuple[tuple[str, ...], ...]) -> tuple[bool, str]:
        """Require one finding to contain every semantic part of an issue."""
        for finding in findings:
            if all(any(re.search(term, finding, re.IGNORECASE) for term in group) for group in groups):
                return True, finding
        return False, ""

    def evaluate(self, response_text: str) -> EvaluationResult:
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        if not text:
            return EvaluationResult(0.0, [])
        findings = self._descriptions(text)
        if not findings:
            return EvaluationResult(0.0, [])
        validation = parse_structured(text, fmt="json")
        rubric.record_validation(validation)

        checks = [
            (
                "File handle not closed / resource leak", 3.0,
                ((r"f\s*=|open\(",), (r"close|context\s+manager|with\s+open|leak",)),
            ),
            (
                "== None instead of is None", 2.0,
                ((r"==\s*none|identity",), (r"is\s+none|comparison|user_id",)),
            ),
            (
                "Hardcoded /tmp path", 2.0,
                ((r"/tmp/data\.txt|hardcoded",), (r"path|parameter|config|inject",)),
            ),
            (
                "Missing error handling / fetch_data may fail", 3.0,
                ((r"fetch_data",), (r"exception|error|try|except|failure|handling",)),
            ),
            (
                "Unused imports", 2.0,
                ((r"unused|not used|remove",), (r"os|time|import",)),
            ),
        ]
        matched_findings: list[str] = []
        for name, maximum, groups in checks:
            matched, finding = self._finding_matches(findings, groups)
            if matched:
                matched_findings.append(finding)
            rubric.add_criterion(
                name, maximum, maximum if matched else 0.0,
                evidence=[{"kind": "finding", "span": finding}] if matched else [],
                negative_findings=[] if matched else [{"finding": "no independent finding with both the defect and its remediation"}],
            )

        remediation_terms = (r"use", r"replace", r"close", r"context manager", r"is none", r"parameterize", r"inject", r"try", r"except", r"remove", r"validate", r"sanitize")
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
        rubric.add_criterion(
            "Source citations", 1.0,
            1.0 if citations >= min(3, len(findings)) else 0.0,
            evidence=[{"kind": "source-citation-count", "count": citations}],
            negative_findings=[] if citations >= min(3, len(findings)) else [{"finding": "cite the relevant variable, call, or literal"}],
        )
        return rubric.results()

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
