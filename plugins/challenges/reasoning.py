"""Deterministic constrained logic-puzzle challenge."""
from __future__ import annotations

import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from benchmark.types import ConfigMap
from plugins.challenges._rubric import Rubric


class ReasoningPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "reasoning"

    @property
    def version(self) -> str:
        return "1.1.0"

    @property
    def name(self) -> str:
        return "Logical Reasoning"

    @property
    def max_score(self) -> int:
        return int(20.0)

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_prompt(self) -> str:
        return (
            "Solve this exact logic puzzle. Give numbered deductions, then exactly four final lines. "
            "Six services occupy 09:00, 09:15, 09:30, 09:45, 10:00, 10:15: Auth, Billing, Search, "
            "Upload, Notifications, Profile. Owners are Ana, Ben, Chen, Divya, Eli, Farah; priorities "
            "P1-P6 are each used once. Clues: Auth immediately before Search; Profile before Auth; "
            "Upload after Search; Billing after Upload and before Notifications; Ana owns 10:15; Ben owns "
            "Search; Eli owns the incident immediately after Search; Auth is P1; Notifications is P2; "
            "Profile is P4; Upload priority > Search priority > Billing priority. Determine the service, owner, priority, "
            "and time at 09:30. Final lines must be exactly:\nFAILED_SERVICE: <service>\nOWNER: <person>\n"
            "PRIORITY: <P1/P2/P3/P4/P5/P6>\nTIME: <HH:MM>"
        )

    def get_temperature(self, global_config: ConfigMap) -> float | None:
        val = global_config.get("reasoning_temperature")
        return float(val) if isinstance(val, (int, float)) else None

    @staticmethod
    def _has(text: str, pattern: str) -> bool:
        return bool(re.search(pattern, text, re.IGNORECASE))

    def evaluate(self, response_text: str) -> EvaluationResult:
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        if not text:
            return rubric.results()
        answers = {"FAILED_SERVICE": "Search", "OWNER": "Ben", "PRIORITY": "P5", "TIME": "09:30"}
        earned = 0.0
        evidence = []
        # The LAST occurrence of each label wins: a tentative mid-text line
        # must not shadow the final answer lines.
        for label, expected in answers.items():
            values = re.findall(rf"(?m)^\s*{label}:\s*([^\n]+)\s*$", text)
            if values and values[-1].strip().lower() == expected.lower():
                earned += 2.0
                evidence.append({"kind": "final-answer", "field": label, "value": expected})
        rubric.add_criterion("Final answer", 8.0, earned, evidence=evidence)

        # The reasoning-point criteria are gated on a correct final answer:
        # when the final lines are wrong or absent, each is capped at half
        # its max, so restating the clue wording cannot outscore a correct
        # answer.
        final_complete = earned >= 8.0

        def gated(value: float, half_max: float) -> float:
            return value if final_complete else min(value, half_max)

        time_hits = sum(self._has(text, pattern) for pattern in (
            r"Auth.{0,80}immediately before.{0,80}Search",
            r"Profile.{0,80}before.{0,80}Auth",
            r"Upload.{0,80}after.{0,80}Search",
            r"Billing.{0,80}after.{0,80}Upload.{0,80}before.{0,80}Notifications",
        ))
        # The finals contract (exactly four final lines) is enforced against
        # the tail after the last FAILED_SERVICE line: numbered prose there
        # breaks the contract, while mid-text "Step 1:" style deductions do
        # not trigger the penalty.
        service_lines = list(re.finditer(r"(?im)^\s*FAILED_SERVICE:\s*[^\n]*", text))
        tail = text[service_lines[-1].end():] if service_lines else text
        if re.search(r"(?m)^\s*\d+[.)]\s+", tail):
            time_hits = max(0, time_hits - 1)
        rubric.add_criterion("Time-chain deductions", 4.0, gated(float(time_hits), 2.0))
        assignments = sum(self._has(text, pattern) for pattern in (
            r"Profile.{0,60}09:00", r"Auth.{0,60}09:15", r"Search.{0,60}09:30", r"Upload.{0,60}09:45", r"Billing.{0,60}10:00", r"Notifications.{0,60}10:15",
        ))
        rubric.add_criterion("Derived time assignments", 4.0, gated(min(4.0, assignments * 2.0 / 3.0), 2.0))
        ownership = sum(self._has(text, pattern) for pattern in (
            r"Ben.{0,50}(?:owned|owner|owns).{0,30}Search|Search.{0,50}(?:owned|owner|owns).{0,30}Ben",
            r"Eli.{0,60}(?:owned|owner|owns).{0,30}Upload|Upload.{0,60}(?:owned|owner|owns).{0,30}Eli",
            r"Ana.{0,50}(?:owned|owner|owns).{0,30}(?:Notifications|10:15)|(?:Notifications|10:15).{0,50}(?:owned|owner|owns).{0,30}Ana",
        ))
        rubric.add_criterion("Ownership deductions", 2.0, gated(float(ownership) * 2.0 / 3.0, 1.0))
        priorities = sum(self._has(text, pattern) for pattern in (
            r"Auth.{0,30}P1|P1.{0,30}Auth", r"Notifications.{0,30}P2|P2.{0,30}Notifications", r"Upload.{0,60}(?:higher|>).{0,40}Search.{0,60}(?:higher|>).{0,40}Billing",
        ))
        # Profile is pinned to P4 by the clue, leaving P5 for Search, P6 for
        # Upload, and P3 for Billing (B < S < U). The requested 09:30 service
        # is therefore Search/P5; accepting P4 here would reward a
        # plausible-looking but wrong answer.
        # [\s\S] instead of . so the association is found across newlines
        # (a sparse final block like "Search\n09:30\nP5" must match).
        if not self._has(text, r"(?:Search|09:30)[\s\S]{0,40}P5|P5[\s\S]{0,40}(?:Search|09:30)"):
            priorities = max(0, priorities - 1)
        rubric.add_criterion("Priority-chain deductions", 2.0, gated(float(priorities) * 2.0 / 3.0, 1.0))
        # The puzzle has a unique solution, so any service/time pair that
        # disagrees with it (or a final TIME that disagrees with the final
        # FAILED_SERVICE) is a contradiction.
        solution_times = {
            "profile": "09:00",
            "auth": "09:15",
            "search": "09:30",
            "upload": "09:45",
            "billing": "10:00",
            "notifications": "10:15",
        }
        pairs = re.findall(
            r"(?i)\b(Profile|Auth|Search|Upload|Billing|Notifications)\s*(?:is\s+at|at|@|:)?\s*(\d{2}:\d{2})",
            text,
        )
        wrong = any(solution_times[service.lower()] != time for service, time in pairs)
        service_values = re.findall(r"(?im)^\s*FAILED_SERVICE:\s*(\S+)", text)
        time_values = re.findall(r"(?im)^\s*TIME:\s*(\d{2}:\d{2})", text)
        if service_values and time_values:
            failed = service_values[-1].strip().lower()
            if failed in solution_times and time_values[-1] != solution_times[failed]:
                wrong = True
        if wrong:
            rubric.penalize_criterion("Derived time assignments", 1.0, "response contains a contradictory service/time assignment")
        return rubric.results()

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
