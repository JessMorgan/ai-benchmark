"""Multi-turn conversation / revision benchmark task.

Tests the model's ability to:
1. Follow an initial creative prompt
2. Accept and incorporate user feedback
3. Iteratively revise based on specific revision requests
4. Track changes and explain what was modified
"""
import re

from benchmark.plugin import BenchmarkTaskPlugin
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import validate_sections


class MultiTurnConversationPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "multi-turn-conversation"

    @property
    def version(self):
        return "0.5.0"

    @property
    def name(self):
        return "Multi-Turn Conversation"

    @property
    def max_score(self):
        return 20.0

    @property
    def supports_streaming(self):
        return True

    def get_prompt(self):
        return (
            "You are a customer-support AI assistant prototyping a new chatbot. "
            "Use the following structure for your response:\n"
            "1. OUTPUT the initial answer to the user's question in a code block.\n"
            "2. After that, REVISE once based on the feedback given (also in a code block).\n"
            "3. REVISE a second time based on further feedback.\n"
            "4. End with a brief summary of what changed between each version.\n\n"
            "Task: Write a polite, professional email declining a job offer "
            "while leaving the door open for future opportunities.\n\n"
            "Feedback 1 (after the initial email): "
            "\"Make it more personal — mention the specific team and the positive impression the interview process left.\"\n\n"
            "Feedback 2 (after the first revision): "
            "\"The tone is still slightly formal. Add a closing that expresses genuine warmth "
            "and a sentence about following the company's future work.\"\n\n"
            "IMPORTANT: You MUST output ALL THREE versions and the summary, "
            "each in its own labeled section. Use clear headings:\n"
            "## Version 1 (Initial)\n"
            "```\n...initial email...\n```\n"
            "## Version 2 (After Feedback 1)\n"
            "```\n...revised email...\n```\n"
            "## Version 3 (After Feedback 2)\n"
            "```\n...final email...\n```\n"
            "## Summary of Changes\n"
            "- Key differences between V1 and V2\n"
            "- Key differences between V2 and V3"
        )

    def get_temperature(self, global_config):
        if "multi_turn_conversation_temperature" in global_config:
            return global_config["multi_turn_conversation_temperature"]
        return None

    def evaluate(self, response_text):
        t = response_text
        rubric = Rubric(self.max_score)
        rubric.record_validation(validate_sections(t, [
            "Version 1", "Version 2", "Version 3", "Summary of Changes",
        ], min_chars=10))

        # Check for all three versions present
        v1 = re.search(r"##?\s*Version\s*1|##?\s*Initial|Version\s*1\s*\(Initial\)", t)
        v2 = re.search(r"##?\s*Version\s*2|After\s+Feedback\s+1", t)
        v3 = re.search(r"##?\s*Version\s*3|After\s+Feedback\s+2", t)

        earned = 0.0
        if v1:
            earned += 1.0
        if v2:
            earned += 1.0
        if v3:
            earned += 1.0
        rubric.add_criterion("All three versions present", 3.0, earned)

        # Check code blocks contain actual email content
        code_blocks = re.findall(r"```(.*?)```", t, re.DOTALL)
        earned = 0.0
        if len(code_blocks) >= 3:
            earned += 2.0
        if len(code_blocks) >= 2:
            earned += 1.0
        if len(code_blocks) >= 1:
            earned += 1.0
        rubric.add_criterion("Email content in code blocks", 4.0, earned)

        # Check for specific content reflecting feedback incorporation
        earned = 0.0
        # Feedback 1 asked for personalization — team name / interview impression
        if re.search(r"(?:team|interview|impression|meet|discuss|learning)", t, re.IGNORECASE):
            earned += 1.5
        if re.search(r"(?:pleasure|enjoy|appreciate|grateful|thank)", t, re.IGNORECASE):
            earned += 1.5
        rubric.add_criterion("Feedback 1 incorporated (personalization)", 3.0, earned)

        earned = 0.0
        # Feedback 2 asked for warmth and following future work
        if re.search(r"(?:warm|genuine|cordial|best wishes|kind regards)", t, re.IGNORECASE):
            earned += 1.0
        if re.search(r"(?:follow|future|opportunity|stay in touch|connection)", t, re.IGNORECASE):
            earned += 1.0
        if re.search(r"(?:excited|hope|wish|sincerely)", t, re.IGNORECASE):
            earned += 1.0
        rubric.add_criterion("Feedback 2 incorporated (warmth, future)", 3.0, earned)

        # Check summary of changes
        summary = re.search(r"##?\s*Summary\s*of\s*Changes", t)
        if summary:
            # Look for specific mentions of what changed
            diff_v1v2 = re.search(r"(?:V1|Version 1|Initial).*?(?:V2|Version 2)", t, re.DOTALL)
            diff_v2v3 = re.search(r"(?:V2|Version 2).*?(?:V3|Version 3)", t, re.DOTALL)
            changes = re.findall(r"(?:changed|added|removed|replaced|modified|softened|tightened|personalize)", t, re.IGNORECASE)
            earned = 0.0
            if diff_v1v2 or diff_v2v3:
                earned += 1.0
            if len(changes) >= 2:
                earned += 1.0
            rubric.add_criterion("Summary of changes", 2.0, earned)

        # Check for evidence of iterative improvement (distinct versions)
        earned = 0.0
        # Each version should be progressively different
        # Count unique sentences/clauses that indicate iteration
        if re.search(r"(?:decline|opportunity|position|offer|role)", t, re.IGNORECASE):
            earned += 1.0
        if re.search(r"(?:future|connection|network|alumni|follow)", t, re.IGNORECASE):
            earned += 1.0
        if re.search(r"(?:specific|personally|valuable|impress|culture)", t, re.IGNORECASE):
            earned += 1.0
        rubric.add_criterion("Evidence of iteration across versions", 3.0, earned)

        # Structure compliance
        earned = 2.0
        if not v1:
            earned -= 0.5
        if not v2:
            earned -= 0.5
        if not summary:
            earned -= 0.5
        if len(code_blocks) < 3:
            earned -= 0.5
        earned = round(max(earned, 0.0), 1)
        rubric.add_criterion("Structure compliance", 2.0, earned)

        sections = validate_sections(t, ["Version 1", "Version 2", "Version 3"], min_chars=10).value or {}
        v1 = next((body for key, body in sections.items() if "version 1" in key), "")
        v2 = next((body for key, body in sections.items() if "version 2" in key), "")
        v3 = next((body for key, body in sections.items() if "version 3" in key), "")
        if v1 and v2 and v1 == v2:
            rubric.penalize_criterion("Evidence of iteration across versions", 1.0, "Version 2 duplicates Version 1")
        if v2 and v3 and v2 == v3:
            rubric.penalize_criterion("Evidence of iteration across versions", 1.0, "Version 3 duplicates Version 2")

        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
