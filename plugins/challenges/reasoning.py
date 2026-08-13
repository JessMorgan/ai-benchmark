"""Reasoning benchmark task based on a constrained incident logic puzzle."""
import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from plugins.challenges._rubric import Rubric


class ReasoningPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "reasoning"

    @property
    def version(self):
        return "0.1.1"

    @property
    def name(self):
        return "Logical Reasoning"

    @property
    def max_score(self):
        return 20.0

    @property
    def supports_streaming(self):
        return True

    def get_prompt(self):
        return (
            "Solve the logic puzzle. Give a concise, numbered deduction before the final "
            "answer. Do not guess: every conclusion must follow from the clues.\n\n"
            "Six services (Auth, Billing, Search, Upload, Notifications, Profile) had one "
            "incident each at exactly one of 09:00, 09:15, 09:30, 09:45, 10:00, and 10:15. "
            "The services were owned by Ana, Ben, Chen, Divya, Eli, and Farah, one owner "
            "per service. Each incident had one priority: P1, P2, P3, P4, P5, or P6, with "
            "P1 higher priority than P2, continuing through P6.\n\n"
            "CLUES:\n"
            "1. Auth happened immediately before Search.\n"
            "2. Profile happened before Auth.\n"
            "3. Upload happened after Search.\n"
            "4. Billing happened after Upload but before Notifications.\n"
            "5. Ana owned the final incident at 10:15.\n"
            "6. Ben owned Search.\n"
            "7. Eli owned the incident immediately after Search.\n"
            "8. Auth had priority P1.\n"
            "9. Notifications had priority P2.\n"
            "10. Upload had a higher priority than Search, and Search had a higher priority "
            "than Billing.\n\n"
            "Determine the service, owner, priority, and time of the incident at 09:30. "
            "Your final answer must use exactly these four lines after your deductions:\n"
            "FAILED_SERVICE: <service>\n"
            "OWNER: <person>\n"
            "PRIORITY: <P1/P2/P3/P4/P5/P6>\n"
            "TIME: <HH:MM>"
        )

    def get_temperature(self, global_config):
        if "reasoning_temperature" in global_config:
            return global_config["reasoning_temperature"]
        return None

    @staticmethod
    def _has(text, pattern):
        return bool(re.search(pattern, text, re.IGNORECASE))

    def evaluate(self, response_text):
        if not response_text or not response_text.strip():
            return EvaluationResult(0.0, [])

        text = response_text.strip()
        rubric = Rubric(self.max_score)

        final_answers = {
            "FAILED_SERVICE": "Search",
            "OWNER": "Ben",
            "PRIORITY": "P4",
            "TIME": "09:30",
        }
        earned = 0.0
        evidence = []
        for label, expected in final_answers.items():
            match = re.search(rf"(?im)^\s*{label}\s*:\s*([^\n]+)\s*$", text)
            if match and match.group(1).strip().lower() == expected.lower():
                earned += 2.0
                evidence.append({"kind": "final-answer", "field": label, "value": expected})
        rubric.add_criterion("Final answer", 8.0, earned, evidence=evidence)

        earned = 0.0
        if self._has(text, r"Auth[^\n]{0,50}immediately before[^\n]{0,50}Search"):
            earned += 1.0
        if self._has(text, r"Profile[^\n]{0,50}before[^\n]{0,50}Auth"):
            earned += 1.0
        if self._has(text, r"Upload[^\n]{0,50}after[^\n]{0,50}Search"):
            earned += 1.0
        if self._has(text, r"Billing[^\n]{0,50}after[^\n]{0,50}Upload[^\n]{0,50}before[^\n]{0,50}Notifications"):
            earned += 1.0
        rubric.add_criterion("Time-chain deductions", 4.0, earned)

        earned = 0.0
        if (
            self._has(text, r"Profile[^\n]{0,60}09:00")
            and self._has(text, r"Auth[^\n]{0,60}09:15")
        ):
            earned += 1.0
        if (
            self._has(text, r"Search[^\n]{0,60}09:30")
            and self._has(text, r"Upload[^\n]{0,60}09:45")
        ):
            earned += 1.0
        if self._has(text, r"Billing[^\n]{0,60}10:00"):
            earned += 1.0
        if self._has(text, r"Notifications[^\n]{0,60}10:15"):
            earned += 1.0
        rubric.add_criterion("Derived time assignments", 4.0, earned)

        earned = 0.0
        if self._has(text, r"Ben[^\n]{0,50}(?:owned|owner of)[^\n]{0,30}Search|Search[^\n]{0,50}(?:owned|owner)[^\n]{0,30}Ben"):
            earned += 1.0
        if self._has(text, r"Eli[^\n]{0,60}(?:owned|owner of)[^\n]{0,30}(?:Upload|incident immediately after Search)|(?:Upload|incident immediately after Search)[^\n]{0,60}(?:owned|owner)[^\n]{0,30}Eli"):
            earned += 0.5
        if self._has(text, r"Ana[^\n]{0,50}(?:owned|owner of)[^\n]{0,30}(?:10:15|Notifications)|(?:Notifications|10:15)[^\n]{0,50}(?:owned|owner)[^\n]{0,30}Ana"):
            earned += 0.5
        rubric.add_criterion("Ownership deductions", 2.0, earned)

        earned = 0.0
        if self._has(text, r"Auth[^\n]{0,30}P1|P1[^\n]{0,30}Auth"):
            earned += 0.5
        if self._has(text, r"Notifications[^\n]{0,30}P2|P2[^\n]{0,30}Notifications"):
            earned += 0.5
        if self._has(text, r"Upload[^\n]{0,50}higher[^\n]{0,30}Search[^\n]{0,50}higher[^\n]{0,30}Billing"):
            earned += 1.0
        rubric.add_criterion("Priority-chain deductions", 2.0, earned)

        if not re.search(r"(?im)^\s*\d+[.)]\s+", text):
            rubric.penalize_criterion(
                "Time-chain deductions",
                1.0,
                "response does not present numbered deductions",
            )

        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
