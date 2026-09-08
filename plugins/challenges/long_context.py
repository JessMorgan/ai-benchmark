"""Long-context retrieval and cross-reference challenge."""
from __future__ import annotations

import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from benchmark.types import ConfigMap
from plugins.challenges._rubric import Rubric

_NEGATION = re.compile(
    r"\b(?:not|no|never|without|isn'?t|aren'?t|wasn'?t|weren'?t|can'?t|cannot|"
    r"don'?t|doesn'?t|didn'?t)\s*$",
    re.IGNORECASE,
)


def _positive_ref(text: str, token: str) -> bool:
    """Return True if ``token`` appears in ``text`` as a bounded, non-negated reference.

    A reference immediately preceded by a negation word (``NOT I-17``,
    ``is not I-17``) does not count: the incident criteria must credit a
    positive association with the incident, not bare token presence
    (measured: 5/5 on "NOT I-17"). Matching is word-bounded so ``I-170``
    does not satisfy ``I-17`` and ``P12`` does not satisfy ``P1``.
    """
    pattern = re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 24):match.start()]
        if not _NEGATION.search(prefix):
            return True
    return False


class LongContextPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "long-context"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def name(self) -> str:
        return "Long-Context Retrieval"

    @property
    def max_score(self) -> int:
        return int(20.0)

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_prompt(self) -> str:
        facts = [
            "F01 | incident I-11 | service Beacon | region US | time 09:15 | owner Lena | priority P3",
            "F02 | incident I-17 | service Atlas | region EU | time 14:30 | owner Omar | priority P1",
            "F03 | incident I-19 | service Cedar | region EU | time 15:45 | owner Priya | priority P2",
            "F04 | incident I-23 | service Delta | region APAC | time 14:30 | owner Omar | priority P1",
            "F05 | incident I-17 | status escalated | ticket T-884 | runbook RB-7",
            "F06 | incident I-19 | status resolved | ticket T-885 | runbook RB-2",
            "F07 | service Atlas | primary owner Omar | backup owner Lena",
            "F08 | service Cedar | primary owner Priya | backup owner Omar",
            "F09 | priority P1 | escalation channel PagerDuty | response 5m",
            "F10 | priority P2 | escalation channel OpsGenie | response 15m",
            "F11 | region EU | retention 90d | compliance GDPR",
            "F12 | region US | retention 30d | compliance SOC2",
            "F13 | runbook RB-7 | escalation channel PagerDuty | severity critical",
            "F14 | runbook RB-2 | escalation channel OpsGenie | severity high",
            "F15 | incident I-31 | service Echo | region EU | time 16:00 | owner Chen | priority P4",
            "F16 | incident I-34 | service Atlas | region EU | time 12:00 | owner Omar | priority P2",
            "F17 | incident I-38 | service Foxtrot | region EU | time 14:30 | owner Chen | priority P4",
            "F18 | incident I-42 | service Golf | region EU | time 14:45 | owner Omar | priority P3",
            "F19 | incident I-17 | customer impact 12 percent | mitigation feature flag",
            "F20 | incident I-23 | customer impact 8 percent | mitigation rollback",
        ]
        # Add a deterministic body of near-miss records. They make this a
        # retrieval-and-joining task rather than a twenty-line lookup, while
        # keeping the answer uniquely determined by F02/F05/F09.
        for number in range(21, 121):
            incident = 100 + number
            region = ("US", "APAC", "EU")[number % 3]
            time = ("08:30", "11:15", "14:30", "17:45")[number % 4]
            priority = ("P2", "P3", "P4", "P5")[number % 4]
            facts.append(
                f"F{number:02d} | incident I-{incident} | service Distractor-{number} | "
                f"region {region} | time {time} | owner Owner-{number} | priority {priority} | "
                f"status {'open' if number % 2 else 'resolved'} | ticket T-{900 + number}"
            )
        return (
            "Read all records before answering. Distractors are intentional.\n\n"
            + "\n".join(facts)
            + "\n\nQuestion: Identify the EU incident at 14:30 with P1 priority.\n"
            "Respond with exactly these five lines, one per line, in LABEL: value form:\n"
            "INCIDENT: <incident id>\n"
            "OWNER: <owner name>\n"
            "ESCALATION CHANNEL: <escalation channel>\n"
            "EVIDENCE: <fact ids, e.g. F02 F05 F09>\n"
            "REASONING: <why these facts answer the question; cite at least three fact IDs>"
        )

    def get_temperature(self, global_config: ConfigMap) -> float | None:
        val = global_config.get("long_context_temperature")
        return float(val) if isinstance(val, (int, float)) else None

    def evaluate(self, response_text: str) -> EvaluationResult:
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        if not text:
            return rubric.results()
        values = {}
        for label in ("INCIDENT", "OWNER", "ESCALATION CHANNEL", "EVIDENCE", "REASONING"):
            match = re.search(rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+)$", text)
            values[label] = match.group(1).strip() if match else ""
        # The correct incident (I-17) is the primary criterion. The supporting
        # evidence/cross-ref/owner criteria are gated on it so a wrong incident
        # cannot earn them (measured: wrong-incident + magic tokens 19/20).
        incident_ok = _positive_ref(values["INCIDENT"], "I-17")
        rubric.add_criterion(
            "Incident correctness", 6.0,
            6.0 if incident_ok else 0.0,
            negative_findings=[] if incident_ok else [{"finding": "INCIDENT field does not reference I-17"}],
        )
        # Per-field gating: each sub-check is gated on its own field being
        # present (bool(values[label])), not all on REASONING non-empty
        # (measured: one missing heading 20->7).
        rubric.add_criterion("Exact answer", 4.0, float(sum(bool(values[label]) and expected in values[label].lower() for label, expected in (("OWNER", "omar"), ("ESCALATION CHANNEL", "pagerduty"), ("REASONING", "14:30"), ("REASONING", "p1")))))
        ids = set(re.findall(r"\bF\d{2}\b", values["EVIDENCE"] + " " + values["REASONING"]))
        rubric.add_criterion("Evidence retrieval", 4.0, (4.0 if len(ids) >= 3 and {"F02", "F05", "F09"} <= ids else min(4.0, len(ids))) if incident_ok else 0.0, evidence=[{"kind": "fact-id", "id": value} for value in sorted(ids)])
        cross = incident_ok and _positive_ref(values["REASONING"], "I-17") and all(re.search(pattern, values["REASONING"], re.IGNORECASE) for pattern in (r"EU", r"14:30", r"P1", r"PagerDuty"))
        rubric.add_criterion("Cross-reference reasoning", 3.0, 3.0 if cross else 0.0)
        rubric.add_criterion("Owner/service consistency", 2.0, 2.0 if incident_ok and (_positive_ref(values["EVIDENCE"], "I-17") or _positive_ref(values["EVIDENCE"], "F02")) and re.search(r"Omar", values["EVIDENCE"] + " " + values["OWNER"], re.IGNORECASE) else 0.0)
        exact_headers = all(re.search(rf"(?im)^\s*{re.escape(label)}\s*:", text) for label in ("INCIDENT", "OWNER", "ESCALATION CHANNEL", "EVIDENCE", "REASONING"))
        rubric.add_criterion("Response contract", 1.0, 1.0 if exact_headers else 0.0)
        return rubric.results()

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
