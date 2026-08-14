"""Deterministic instruction-following challenge."""
from __future__ import annotations

import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from plugins.challenges._rubric import Rubric


class InstructionFollowingPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "instruction-following"

    @property
    def version(self):
        return "1.0.0"

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
            "Return only four ORDER lines followed by the SUMMARY line; do not explain.\n"
            "Input orders:\n"
            "T-08 Zara 99.90 paid no EU web; T-02 Jules 120.00 paid no US web; "
            "T-05 Noor 120.00 paid no EU web; T-06 Asha 50.00 paid no APAC web; "
            "T-01 Iris 200.00 paid yes US web; T-03 Omar 80.00 pending no US web; "
            "T-04 Pia 75.00 paid no EU internal; T-07 Quinn 49.99 paid no EU web; "
            "T-09 Ravi 65.00 paid no EU web.\n\n"
            "Keep only paid, not refunded, amount >= 50.00, non-APAC, non-internal orders. "
            "Sort amount descending and break ties by customer name alphabetically. "
            "Uppercase names and preserve IDs/amounts. Format exactly `ORDER <id> | CUSTOMER "
            "<NAME> | TOTAL <amount>`. Then write exactly `[SUMMARY] count=4; total=404.90; top_order=T-02`."
        )

    def get_temperature(self, global_config):
        return global_config.get("instruction_following_temperature")

    _ORDER_RE = re.compile(r"^ORDER (T-\d{2}) \| CUSTOMER ([A-Z]+) \| TOTAL (\d+\.\d{2})$")
    _EXPECTED = (
        ("T-02", "JULES", "120.00"),
        ("T-05", "NOOR", "120.00"),
        ("T-08", "ZARA", "99.90"),
        ("T-09", "RAVI", "65.00"),
    )
    _SUMMARY = "[SUMMARY] count=4; total=404.90; top_order=T-02"

    def evaluate(self, response_text):
        if not response_text or not response_text.strip():
            return EvaluationResult(0.0, [])
        text = response_text.strip()
        lines = text.splitlines()
        order_lines = lines[:-1] if lines and lines[-1] == self._SUMMARY else lines
        records = [self._ORDER_RE.fullmatch(line) for line in order_lines]
        parsed = [match.groups() for match in records if match]
        expected_ids = {record[0] for record in self._EXPECTED}
        parsed_ids = {record[0] for record in parsed}
        filter_score = 0.0
        if parsed_ids == expected_ids and len(parsed) == 4:
            filter_score = 4.0
        elif parsed_ids <= expected_ids and parsed_ids:
            filter_score = 2.0
        rubric = Rubric(self.max_score)
        rubric.add_criterion(
            "All filters applied", 4.0, filter_score,
            evidence=[{"kind": "record", "id": record[0]} for record in parsed],
            negative_findings=[] if filter_score == 4.0 else [{"finding": "retained records do not exactly match all filters"}],
        )
        order_score = 0.0
        if parsed == list(self._EXPECTED):
            order_score = 4.0
        elif parsed and parsed_ids == expected_ids:
            order_score = 1.0
        rubric.add_criterion("Sort and tie-break order", 4.0, order_score)
        transformed = sum(record in self._EXPECTED for record in parsed)
        rubric.add_criterion("Transformed order lines", 4.0, float(transformed),
                             evidence=[{"kind": "exact-record", "record": record} for record in parsed if record in self._EXPECTED])
        rubric.add_criterion("Summary arithmetic and format", 4.0, 4.0 if lines and lines[-1] == self._SUMMARY else 0.0)
        exact = lines == [
            "ORDER T-02 | CUSTOMER JULES | TOTAL 120.00",
            "ORDER T-05 | CUSTOMER NOOR | TOTAL 120.00",
            "ORDER T-08 | CUSTOMER ZARA | TOTAL 99.90",
            "ORDER T-09 | CUSTOMER RAVI | TOTAL 65.00",
            self._SUMMARY,
        ]
        forbidden = [line for line in lines if line != self._SUMMARY and not self._ORDER_RE.fullmatch(line)]
        rubric.add_criterion(
            "Exact response discipline", 4.0, 4.0 if exact else (1.0 if not forbidden else 0.0),
            negative_findings=[] if exact else [{"finding": "extra, malformed, duplicate, or unordered output"}],
        )
        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
