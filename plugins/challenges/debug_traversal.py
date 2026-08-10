"""Debug traversal / root-cause analysis benchmark task.

Tests the model's ability to:
1. Read buggy code and identify the root cause of a reported symptom
2. Trace through logic to find the source of a defect
3. Propose a verified fix
4. Identify potential side effects of the proposed change
"""
import re

from benchmark.plugin import BenchmarkTaskPlugin
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import parse_python


class DebugTraversalPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "debug-traversal"

    @property
    def version(self):
        return "0.3.0"

    @property
    def name(self):
        return "Debug Traversal"

    @property
    def max_score(self):
        return 20.0

    @property
    def supports_streaming(self):
        return True

    def get_prompt(self):
        return (
            "You are a senior debugging engineer. Analyze the bug report and code below, "
            "then produce a structured root-cause analysis.\n\n"
            "---\n"
            "BUG REPORT: The following function is supposed to return a list of unique user IDs "
            "that appear in at least 2 different log entries. However, it always returns an "
            "empty list when there are duplicate user IDs across entries.\n\n"
            "```python\n"
            "from collections import Counter\n\n"
            "def find_duplicate_users(log_entries):\n"
            "    \"\"\"Return user IDs that appear in at least 2 different log entries.\"\"\"\n"
            "    user_counts = {}\n"
            "    for entry in log_entries:\n"
            "        user_id = entry.get('user_id')\n"
            "        if user_id:\n"
            "            user_counts[user_id] = user_counts.get(user_id, 0) + 1\n"
            "    \n"
            "    result = []\n"
            "    for user_id, count in user_counts.items():\n"
            "        if count >= 2:\n"
            "            result.append(user_id)\n"
            "    \n"
            "    # Remove duplicates from the returned list (defensive)\n"
            "    return list(set(result))\n"
            "```\n\n"
            "---\n"
            "CONTEXT: This is called with log entries structured like:\n"
            "```python\n"
            "logs = [\n"
            "    {'user_id': 'abc123', 'action': 'login', 'timestamp': '2024-01-01T10:00:00Z'},\n"
            "    {'user_id': 'abc123', 'action': 'view_page', 'timestamp': '2024-01-01T10:05:00Z'},\n"
            "    {'user_id': 'def456', 'action': 'login', 'timestamp': '2024-01-01T10:10:00Z'},\n"
            "    {'user_id': 'ghi789', 'action': 'purchase', 'timestamp': '2024-01-01T10:15:00Z'},\n"
            "]\n"
            "```\n\n"
            "REQUIRED OUTPUT STRUCTURE:\n"
            "1. **Root Cause**: Exactly which line(s) cause the bug, and why.\n"
            "2. **Analysis**: Walk through the code execution with the given example data.\n"
            "3. **Fix**: Provide corrected code.\n"
            "4. **Test**: Write a pytest test that proves the fix works.\n"
            "5. **Side Effects**: What other parts of the system could be affected by this change?"
        )

    def get_temperature(self, global_config):
        if "debug_traversal_temperature" in global_config:
            return global_config["debug_traversal_temperature"]
        return None

    def evaluate(self, response_text):
        t = response_text
        rubric = Rubric(self.max_score)
        rubric.record_validation(parse_python(t))

        # Check for root cause identification — the model should trace through
        # the logic line by line and identify which line(s) cause the bug.
        # Good answers walk through the execution with the provided example data.
        earned = 0.0
        # The model should trace through the code line by line
        if re.search(r"(?:trace|walk|step|follow|line by line)", t, re.IGNORECASE):
            earned += 1.0
        if re.search(r"(?:execution|code path|logic|flow|control)", t, re.IGNORECASE):
            earned += 1.0
        # Should identify actual code locations
        if re.search(r"(?:line|L\d+|:(\d+))", t):
            earned += 1.0
        rubric.add_criterion("Systematic trace / code walkthrough", 3.0, earned)

        # Analysis depth
        earned = 0.0
        if re.search(r"(?:edge case|null|empty|boundary|condition)", t, re.IGNORECASE):
            earned += 1.0
        if re.search(r"(?:correct|incorrect|bug|issue|problem|error|defect)", t, re.IGNORECASE):
            earned += 1.0
        if re.search(r"(?:user_id|key|hashable|type|string|integer)", t):
            earned += 1.0
        rubric.add_criterion("Depth of analysis", 3.0, earned)

        # Fix correctness
        earned = 0.0
        if re.search(r"(?:fix|corrected|solution|patch|change)", t, re.IGNORECASE):
            earned += 1.0
        if re.search(r"(?:```python)", t):
            earned += 1.0
        if re.search(r"(?:return|def|print|assert)", t):
            earned += 1.0
        rubric.add_criterion("Proposed fix / corrected code", 3.0, earned)

        # Test code
        earned = 0.0
        if re.search(r"(?:def test_|class Test|pytest|unittest)", t):
            earned += 1.5
        if re.search(r"(?:assert|assertEqual|assertTrue|assertFalse)", t):
            earned += 1.5
        rubric.add_criterion("Test code provided", 3.0, earned)

        # Side effects awareness
        earned = 0.0
        if re.search(r"(?:side effect|impact|affect|consequence|regression)", t, re.IGNORECASE):
            earned += 1.0
        if re.search(r"(?:calling code|caller|upstream|downstream|integration)", t, re.IGNORECASE):
            earned += 1.0
        if re.search(r"(?:performance|memory|speed|efficiency|overhead)", t, re.IGNORECASE):
            earned += 1.0
        rubric.add_criterion("Side effects analysis", 3.0, earned)

        # Structure (all 5 sections present)
        earned = 0.0
        required_sections = [
            r"(?:root cause|the bug)",
            r"(?:analysis|walkthrough|trace)",
            r"(?:fix|correction|solution)",
            r"(?:test|verification|pytest|assert)",
            r"(?:side effect|impact|consequence)",
        ]
        hits = sum(1 for section in required_sections if re.search(section, t, re.IGNORECASE))
        earned = min(hits * 1.0, 5.0)
        rubric.add_criterion("Structured RCA sections", 5.0, earned)

        # The supplied implementation is already correct for the example;
        # invented identifiers are a direct contradiction, not a missing
        # keyword. Penalize only the root-cause/analysis criteria.
        root_cause = re.search(r"(?is)(?:root\s+cause|diagnosis).*?(?:\n\s*#{1,4}\s+|\Z)", t)
        root_cause_text = root_cause.group(0) if root_cause else t[:1000]
        if re.search(r"\buser_id[o0]|nonexistent|typo|wrong\s+key", root_cause_text, re.IGNORECASE):
            rubric.penalize_criterion(
                "Depth of analysis", 2.0,
                "response claims an identifier/key defect absent from the supplied code",
            )
            rubric.penalize_criterion(
                "Systematic trace / code walkthrough", 1.0,
                "response's diagnosis contradicts the supplied implementation",
            )

        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
