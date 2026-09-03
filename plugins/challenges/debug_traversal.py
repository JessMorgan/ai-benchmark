"""Real-bug debugging and root-cause traversal challenge."""
from __future__ import annotations

import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from benchmark.types import ConfigMap
from plugins.challenges._analysis import first_section, markdown_sections
from plugins.challenges._execution import extract_python_source, run_python_check
from plugins.challenges._rubric import Rubric


class DebugTraversalPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "debug-traversal"

    @property
    def version(self) -> str:
        return "1.1.0"

    @property
    def name(self) -> str:
        return "Debug Traversal"

    @property
    def max_score(self) -> int:
        return int(20.0)

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_prompt(self) -> str:
        return (
            "Debug the following real bug. The function must return user IDs that occur in at "
            "least two different log entries. With the supplied data it should return ['abc123'], "
            "but it returns an empty list.\n\n"
            "```python\n"
            "def find_duplicate_users(log_entries):\n"
            "    user_counts = {}\n"
            "    for entry in log_entries:\n"
            "        user_id = entry.get('user_id')\n"
            "        if user_id:\n"
            "            user_counts[user_id] = user_counts.get(user_id, 0) + 1\n"
            "    result = []\n"
            "    for user_id, count in user_counts.items():\n"
            "        if count > 2:\n"
            "            result.append(user_id)\n"
            "    return result\n"
            "```\n\n"
            "logs = [{'user_id': 'abc123'}, {'user_id': 'abc123'}, {'user_id': 'def456'}]\n\n"
            "Use exactly these headings: Root Cause, Analysis, Fix, Test, Side Effects. "
            "Identify the defective comparison, walk through count=2, provide corrected code, "
            "write a pytest-style test, and discuss ordering/empty-ID side effects."
        )

    def get_temperature(self, global_config: ConfigMap) -> float | None:
        val = global_config.get("debug_traversal_temperature")
        return float(val) if isinstance(val, (int, float)) else None

    def evaluate(self, response_text: str) -> EvaluationResult:
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        if not text:
            return rubric.results()
        sections = markdown_sections(text)
        rubric.record_validation(type("Validation", (), {
            "valid": all(first_section(text, [name]) is not None for name in ("Root Cause", "Analysis", "Fix", "Test", "Side Effects")),
            "evidence": [{"kind": "section", "heading": section.heading} for section in sections],
            "errors": [],
        })())
        root = first_section(text, ["Root Cause"])
        analysis = first_section(text, ["Analysis"])
        fix = first_section(text, ["Fix"])
        test = first_section(text, ["Test"])
        side = first_section(text, ["Side Effects"])

        root_text = root.body if root else ""
        analysis_text = analysis.body if analysis else ""
        fix_text = fix.body if fix else ""
        test_text = test.body if test else ""
        side_text = side.body if side else ""

        trace_hits = sum(bool(re.search(pattern, analysis_text, re.IGNORECASE)) for pattern in (
            r"abc123", r"count\s*(?:=|is)\s*2", r"def456", r"empty|return",
        ))
        rubric.add_criterion("Systematic trace / code walkthrough", 3.0, 3.0 * trace_hits / 4.0)

        correct_diagnosis = bool(re.search(r"(?:>\s*2|greater\s+than\s+2|strict\s+inequality|>=\s*2|at\s+least\s+2)", root_text, re.IGNORECASE))
        rubric.add_criterion(
            "Depth of analysis", 3.0,
            3.0 if correct_diagnosis and re.search(r"count|two|2", analysis_text, re.IGNORECASE) else 0.0,
            negative_findings=[] if correct_diagnosis else [{"finding": "root cause must identify > 2 instead of >= 2"}],
        )

        fix_correct = bool(re.search(r"(?:>=\s*2|count\s*\)\s*>=\s*2|count\s*>=\s*2)", fix_text))
        rubric.add_criterion(
            "Proposed fix / corrected code", 3.0,
            3.0 if fix_correct else 0.0,
            negative_findings=[] if fix_correct else [{"finding": "corrected code must accept exactly two occurrences"}],
        )

        test_correct = bool(
            re.search(r"(?:pytest|def\s+test_|assert)", test_text, re.IGNORECASE)
            and re.search(r"abc123", test_text, re.IGNORECASE)
            and re.search(r"(?:==\s*\[?['\"]?abc123|in\s+)", test_text, re.IGNORECASE)
        )
        rubric.add_criterion("Test code provided", 3.0, 3.0 if test_correct else 0.0)

        side_hits = sum(bool(re.search(pattern, side_text, re.IGNORECASE)) for pattern in (
            r"order|set|determin", r"empty|missing|null", r"duplicate|count",
        ))
        rubric.add_criterion("Side effects analysis", 3.0, side_hits)

        structure_hits = sum(section is not None for section in (root, analysis, fix, test, side))
        rubric.add_criterion("Structured RCA sections", 2.0, float(structure_hits))

        source = extract_python_source(fix_text)
        if source:
            execution = run_python_check(source, """
assert find_duplicate_users([
    {"user_id": "abc123"}, {"user_id": "abc123"}, {"user_id": "def456"}
]) == ["abc123"]
""")
            rubric.add_criterion(
                "Executable fix verification", 3.0,
                3.0 if execution.status == "passed" else 0.0,
                evidence=[{"kind": "execution", "status": execution.status, "isolation": execution.isolation}],
                negative_findings=[] if execution.status == "passed" else [{"finding": execution.error or execution.status}],
            )
        else:
            rubric.add_criterion("Executable fix verification", 3.0, 0.0, negative_findings=[{"finding": "no corrected Python block"}])
        return rubric.results()

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
