"""Deterministic constrained logic-puzzle challenge."""
from __future__ import annotations

import re

from benchmark.plugin import BenchmarkTaskPlugin
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

    def get_prompt(self):
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

    def get_temperature(self, global_config):
        return global_config.get("reasoning_temperature")

    @staticmethod
    def _has(text, pattern):
        return bool(re.search(pattern, text, re.IGNORECASE))

    def evaluate(self, response_text):
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        if not text:
            return rubric.results()
        answers = {"FAILED_SERVICE": "Search", "OWNER": "Ben", "PRIORITY": "P5", "TIME": "09:30"}
        earned = 0.0
        evidence = []
        for label, expected in answers.items():
            match = re.search(rf"(?m)^\s*{label}:\s*([^\n]+)\s*$", text)
            if match and match.group(1).strip().lower() == expected.lower():
                earned += 2.0
                evidence.append({"kind": "final-answer", "field": label, "value": expected})
        rubric.add_criterion("Final answer", 8.0, earned, evidence=evidence)

        time_hits = sum(self._has(text, pattern) for pattern in (
            r"Auth.{0,80}immediately before.{0,80}Search",
            r"Profile.{0,80}before.{0,80}Auth",
            r"Upload.{0,80}after.{0,80}Search",
            r"Billing.{0,80}after.{0,80}Upload.{0,80}before.{0,80}Notifications",
        ))
        if not re.search(r"(?m)^\s*\d+[.)]\s+", text):
            time_hits = max(0, time_hits - 1)
        rubric.add_criterion("Time-chain deductions", 4.0, float(time_hits))
        assignments = sum(self._has(text, pattern) for pattern in (
            r"Profile.{0,60}09:00", r"Auth.{0,60}09:15", r"Search.{0,60}09:30", r"Upload.{0,60}09:45", r"Billing.{0,60}10:00", r"Notifications.{0,60}10:15",
        ))
        rubric.add_criterion("Derived time assignments", 4.0, min(4.0, assignments * 2.0 / 3.0))
        ownership = sum(self._has(text, pattern) for pattern in (
            r"Ben.{0,50}(?:owned|owner).{0,30}Search|Search.{0,50}(?:owned|owner).{0,30}Ben",
            r"Eli.{0,60}(?:owned|owner).{0,30}Upload|Upload.{0,60}(?:owned|owner).{0,30}Eli",
            r"Ana.{0,50}(?:owned|owner).{0,30}(?:Notifications|10:15)|(?:Notifications|10:15).{0,50}(?:owned|owner).{0,30}Ana",
        ))
        rubric.add_criterion("Ownership deductions", 2.0, float(ownership) * 2.0 / 3.0)
        priorities = sum(self._has(text, pattern) for pattern in (
            r"Auth.{0,30}P1|P1.{0,30}Auth", r"Notifications.{0,30}P2|P2.{0,30}Notifications", r"Upload.{0,60}higher.{0,40}Search.{0,60}higher.{0,40}Billing",
        ))
        # Profile is pinned to P4 by the clue, leaving P5 for Search, P6 for
        # Upload, and P3 for Billing (B < S < U). The requested 09:30 service
        # is therefore Search/P5; accepting P4 here would reward a
        # plausible-looking but wrong answer.
        if not self._has(text, r"(?:Search|09:30).{0,40}P5|P5.{0,40}(?:Search|09:30)"):
            priorities = max(0, priorities - 1)
        rubric.add_criterion("Priority-chain deductions", 2.0, float(priorities) * 2.0 / 3.0)
        search_times = re.findall(r"(?i)\bSearch\s+is\s+at\s+(\d{2}:\d{2})", text)
        final_service = re.search(r"(?im)^\s*FAILED_SERVICE:\s*(\S+)", text)
        wrong = any(value != "09:30" for value in search_times) or bool(final_service and final_service.group(1).lower() != "search")
        if wrong:
            rubric.penalize_criterion("Derived time assignments", 1.0, "response contains a contradictory service/time assignment")
        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
