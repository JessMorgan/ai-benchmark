"""Instruction-following benchmark task.

This task combines independent filters, a derived transformation, a tie-break,
and exact formatting. The expected output is deterministic so the evaluator
can distinguish small instruction-following failures without an LLM judge.
"""
import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from plugins.challenges._rubric import Rubric


class InstructionFollowingPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "instruction-following"

    @property
    def version(self):
        return "0.1.1"

    @property
    def name(self):
        return "Instruction Following"

    @property
    def max_score(self):
        return 20.0

    @property
    def supports_streaming(self):
        return True

    def get_prompt(self):
        return (
            "Follow every instruction exactly. Do not explain your work and do not use a "
            "code block. Return only the four ORDER lines followed by the SUMMARY line.\n\n"
            "ORDERS:\n"
            "- T-08 | customer Zara | amount 99.90 | status paid | refunded no | region EU | channel web\n"
            "- T-02 | customer Jules | amount 120.00 | status paid | refunded no | region US | channel web\n"
            "- T-05 | customer Noor | amount 120.00 | status paid | refunded no | region EU | channel web\n"
            "- T-06 | customer Asha | amount 50.00 | status paid | refunded no | region APAC | channel web\n"
            "- T-01 | customer Iris | amount 200.00 | status paid | refunded yes | region US | channel web\n"
            "- T-03 | customer Omar | amount 80.00 | status pending | refunded no | region US | channel web\n"
            "- T-04 | customer Pia | amount 75.00 | status paid | refunded no | region EU | channel internal\n"
            "- T-07 | customer Quinn | amount 49.99 | status paid | refunded no | region EU | channel web\n"
            "- T-09 | customer Ravi | amount 65.00 | status paid | refunded no | region EU | channel web\n\n"
            "FILTER: Keep an order only when all four conditions hold: status is exactly paid, "
            "refunded is exactly no, amount is at least 50.00, and region is not APAC. "
            "Also exclude channel internal.\n"
            "SORT: Sort retained orders by amount descending. When amounts tie, sort by "
            "customer name alphabetically, not by the original order or by order ID.\n"
            "TRANSFORM: Uppercase the customer name in the output, but preserve the order ID "
            "and amount. Always show amounts with exactly two decimal places.\n"
            "FORMAT: Write each retained order exactly as `ORDER <id> | CUSTOMER <UPPERCASE_NAME> "
            "| TOTAL <amount>`.\n"
            "SUMMARY: After the order lines, write exactly `[SUMMARY] count=<count>; "
            "total=<sum>; top_order=<first order ID>`. Use two decimal places for the sum.\n\n"
            "The final response must contain no headings, bullets, blank explanatory text, "
            "reasoning, Markdown fences, filtered orders, or extra fields."
        )

    def get_temperature(self, global_config):
        if "instruction_following_temperature" in global_config:
            return global_config["instruction_following_temperature"]
        return None

    @staticmethod
    def _order_ids(text):
        return re.findall(r"(?im)^\s*ORDER\s+(T-\d+)\s*\|", text)

    def evaluate(self, response_text):
        if not response_text or not response_text.strip():
            return EvaluationResult(0.0, [])

        text = response_text.strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        expected_ids = ["T-02", "T-05", "T-08", "T-09"]
        expected_lines = [
            "ORDER T-02 | CUSTOMER JULES | TOTAL 120.00",
            "ORDER T-05 | CUSTOMER NOOR | TOTAL 120.00",
            "ORDER T-08 | CUSTOMER ZARA | TOTAL 99.90",
            "ORDER T-09 | CUSTOMER RAVI | TOTAL 65.00",
        ]
        expected_summary = "[SUMMARY] count=4; total=404.90; top_order=T-02"
        rubric = Rubric(self.max_score)

        ids = self._order_ids(text)
        retained_exactly = set(ids) == set(expected_ids) and len(ids) == len(expected_ids)
        excluded_ids = {"T-01", "T-03", "T-04", "T-06", "T-07"}
        earned = 0.0
        if retained_exactly:
            earned += 3.0
        elif set(ids) == set(expected_ids):
            earned += 1.5
        if not (set(ids) & excluded_ids):
            earned += 1.0
        rubric.add_criterion("All filters applied", 4.0, earned)

        earned = 0.0
        if ids == expected_ids:
            earned = 4.0
        elif ids and set(ids) == set(expected_ids):
            earned = 1.0
        rubric.add_criterion("Sort and tie-break order", 4.0, earned)

        exact_line_matches = sum(line in lines for line in expected_lines)
        earned = round(exact_line_matches, 1)
        rubric.add_criterion(
            "Transformed order lines",
            4.0,
            earned,
            evidence=[
                {"kind": "exact-line", "line": line}
                for line in expected_lines
                if line in lines
            ],
        )

        summary_match = re.search(r"(?im)^\s*(\[SUMMARY\].*)\s*$", text)
        summary = summary_match.group(1) if summary_match else ""
        earned = 0.0
        if re.search(r"\bcount=4\b", summary):
            earned += 1.0
        if re.search(r"\btotal=404\.90\b", summary):
            earned += 1.0
        if re.search(r"\btop_order=T-02\b", summary):
            earned += 1.0
        if summary == expected_summary:
            earned += 1.0
        rubric.add_criterion("Summary arithmetic and format", 4.0, earned)

        forbidden = re.search(
            r"(?im)(?:```|^\s*(?:heading|orders|summary)\s*:|^\s*[-*]\s+|"
            r"\b(?:T-01|T-03|T-04|T-06|T-07)\b)",
            text,
        )
        earned = 0.0
        if text.splitlines() == [*expected_lines, expected_summary]:
            earned = 4.0
        else:
            if len(lines) == 5:
                earned += 1.5
            if not forbidden:
                earned += 1.5
            if summary:
                earned += 1.0
        rubric.add_criterion(
            "Exact response discipline",
            4.0,
            earned,
            negative_findings=(
                [{"finding": "response contains forbidden explanatory or filtered output"}]
                if forbidden
                else []
            ),
        )
        if forbidden:
            rubric.penalize_criterion(
                "Exact response discipline",
                1.0,
                "response contains a heading, Markdown fence, bullet, or filtered order",
            )

        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
