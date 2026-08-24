"""Epistemic debugging challenge: detect an inconsistent bug report."""
from __future__ import annotations

import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from benchmark.types import ConfigMap
from plugins.challenges._analysis import first_section, markdown_sections
from plugins.challenges._rubric import Rubric


class DebugConsistencyPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "debug-consistency"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def name(self) -> str:
        return "Debug Report Consistency"

    @property
    def max_score(self) -> int:
        return int(20.0)

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_prompt(self) -> str:
        return (
            "Assess whether this bug report is reproducible; do not invent a defect.\n\n"
            "Bug report: `find_duplicate_users` is said to return [] for duplicate IDs.\n"
            "Implementation:\n```python\n"
            "def find_duplicate_users(log_entries):\n"
            "    counts = {}\n"
            "    for entry in log_entries:\n"
            "        uid = entry.get('user_id')\n"
            "        counts[uid] = counts.get(uid, 0) + 1\n"
            "    return [uid for uid, count in counts.items() if count >= 2]\n```\n"
            "Input: [{'user_id': 'abc'}, {'user_id': 'abc'}].\n\n"
            "Use headings Reproduction, Consistency Check, Diagnosis, Evidence Needed, "
            "Recommendation. State the actual output, explain whether the report follows "
            "from the code, and identify what evidence would be needed if the report persists."
        )

    def get_temperature(self, global_config: ConfigMap) -> float | None:
        val = global_config.get("debug_consistency_temperature")
        return float(val) if isinstance(val, (int, float)) else None

    def evaluate(self, response_text: str) -> EvaluationResult:
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        if not text:
            return rubric.results()
        sections = markdown_sections(text)
        rubric.record_validation(type("Validation", (), {
            "valid": len(sections) >= 5,
            "evidence": [{"kind": "section", "heading": section.heading} for section in sections],
            "errors": [],
        })())
        reproduction = first_section(text, ["Reproduction"])
        consistency = first_section(text, ["Consistency Check"])
        diagnosis = first_section(text, ["Diagnosis"])
        evidence = first_section(text, ["Evidence Needed"])
        recommendation = first_section(text, ["Recommendation"])
        reproduction_ok = bool(
            reproduction
            and re.search(r"abc", reproduction.body, re.IGNORECASE)
            and re.search(r"(?:\[?['\"]?abc['\"]?\]?|list)", reproduction.body, re.IGNORECASE)
            and not re.search(r"(?:empty|\[\])", reproduction.body, re.IGNORECASE)
        )
        rubric.add_criterion("Reproduction trace", 4.0, 4.0 if reproduction_ok else 0.0,
                             negative_findings=[] if reproduction_ok else [{"finding": "trace the supplied input to ['abc']"}])
        consistency_ok = bool(consistency and re.search(r"(?:returns?\s*\[?['\"]?abc|not\s+reproduc|correct|consistent|does\s+not\s+follow|no\s+bug)", consistency.body, re.IGNORECASE))
        rubric.add_criterion("Consistency conclusion", 5.0, 5.0 if consistency_ok else 0.0,
                             negative_findings=[] if consistency_ok else [{"finding": "must conclude that the supplied code returns abc twice as a duplicate"}])
        diagnosis_ok = bool(diagnosis and re.search(r"(?:no\s+(?:code\s+)?bug|inconsistent|cannot\s+confirm|report|environment|input)", diagnosis.body, re.IGNORECASE))
        rubric.add_criterion("Non-hallucinated diagnosis", 4.0, 4.0 if diagnosis_ok else 0.0)
        evidence_ok = bool(evidence and re.search(r"(?:stack|version|actual|input|log|repro|environment|trace)", evidence.body, re.IGNORECASE))
        rubric.add_criterion("Evidence request", 3.0, 3.0 if evidence_ok else 0.0)
        recommendation_ok = bool(recommendation and re.search(r"(?:do not|not enough|collect|reproduce|instrument|verify)", recommendation.body, re.IGNORECASE))
        rubric.add_criterion("Actionable recommendation", 2.0, 2.0 if recommendation_ok else 0.0)
        rubric.add_criterion("Required report structure", 2.0, float(sum(section is not None for section in (reproduction, consistency, diagnosis, evidence, recommendation)) >= 5) * 2.0)
        return rubric.results()

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
