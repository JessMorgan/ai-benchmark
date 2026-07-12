"""Code review benchmark task."""
import json
import re

from benchmark_plugin import BenchmarkTaskPlugin


class CodeReviewPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "code-review"

    @property
    def version(self):
        return "0.2.0"

    @property
    def name(self):
        return "Code Review"

    @property
    def max_score(self):
        return 15.0

    @property
    def supports_streaming(self):
        return False

    def get_prompt(self):
        return (
            "Review the following Python function. Identify bugs, anti-patterns, "
            "security issues, and maintainability problems. "
            "Return your findings as a JSON object with a single key 'issues', "
            "where each issue has a 'description' field explaining the problem. "
            "Be specific and cite the relevant code. "
            "Do not include any text outside the JSON object.\n\n"
            "```python\n"
            "import os\n"
            "import time\n\n"
            "def process_user_data(user_ids, db_path=\"/tmp/data.txt\"):\n"
            "    results = []\n"
            "    f = open(db_path, \"w\")\n"
            "    for i in range(len(user_ids)):\n"
            "        user_id = user_ids[i]\n"
            "        if user_id == None:\n"
            "            continue\n"
            "        data = fetch_data(user_id)\n"
            "        if data:\n"
            "            results.append(data)\n"
            "    f.write(str(results))\n"
            "    return results\n"
            "```"
        )

    def get_temperature(self, global_config):
        if "code_review_temperature" in global_config:
            return global_config["code_review_temperature"]
        if "general_temperature" in global_config:
            return global_config["general_temperature"]
        return None

    def _extract_descriptions(self, response_text):
        """Extract issue descriptions from JSON or fallback to regex."""
        # Try to find JSON object
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(response_text[start : end + 1])
                issues = data.get("issues", [])
                if isinstance(issues, list):
                    return [str(issue.get("description", "")).lower() for issue in issues]
            except json.JSONDecodeError:
                pass

        # Fallback: look for bullet points or numbered lists
        descriptions = []
        for line in response_text.splitlines():
            line = line.strip()
            if line.startswith(("- ", "* ", "1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. ")):
                descriptions.append(line[2:].lower() if line[0] != "*" else line[2:].lower())
        return descriptions

    def score(self, response_text):
        descriptions = self._extract_descriptions(response_text)
        if not descriptions:
            return 0.0

        combined = " ".join(descriptions)
        s = 0.0

        # 1. File handle not closed / resource leak (0-3)
        # Require mention of the actual open call or close/context manager
        if re.search(r"\b(open\(|\.close\(\)|context\s+manager|with\s+open|file\s+handle)\b", combined):
            s += 3.0

        # 2. == None instead of is None (0-2)
        # Require mention of the actual comparison
        if re.search(r"\b(==\s*none|\bis\s+none)\b", combined):
            s += 2.0

        # 3. Hardcoded /tmp path (0-2)
        # Require mention of the actual path or db_path parameter
        if re.search(r"\b(/tmp/data\.txt|tmp/data|db_path)\b", combined):
            s += 2.0

        # 4. Missing error handling / fetch_data may fail (0-3)
        # Require mention of fetch_data and try/except/error handling
        has_fetch_data = re.search(r"\bfetch_data\b", combined)
        has_error_handling = re.search(r"\b(try|except|error\s+handling|may\s+raise|could\s+fail|exception)\b", combined)
        if has_fetch_data and has_error_handling:
            s += 3.0

        # 5. Unused imports (0-2)
        # Require mention of imports being unused and the actual modules
        has_unused = re.search(r"\b(unused\s+import|not\s+used|remove\s+(?:the\s+)?import)\b", combined)
        has_modules = re.search(r"\b(import\s+os|import\s+time|\bos\b|\btime\b)\b", combined)
        if has_unused and has_modules:
            s += 2.0

        # 6. Actionable / concrete fix or recommendation (0-3)
        # Require specific fix language tied to the code
        if re.search(r"\b(use\s+(?:a\s+)?context\s+manager|with\s+open|remove\s+(?:the\s+)?import|is\s+none|try:|except|parameterize)\b", combined):
            s += 3.0

        return round(min(s, self.max_score), 1)
