"""Code review benchmark task."""
import json
import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import parse_structured


class CodeReviewPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "code-review"

    @property
    def version(self):        return "0.8.1"

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
                    descriptions = []
                    for issue in issues:
                        if isinstance(issue, dict):
                            description = issue.get("description") or issue.get("finding") or issue.get("issue")
                        else:
                            description = issue
                        if description:
                            descriptions.append(str(description).lower())
                    return descriptions
            except json.JSONDecodeError:
                pass

        # Fallback: look for bullet points or numbered lists
        descriptions = []
        for line in response_text.splitlines():
            line = line.strip()
            if line.startswith(("- ", "* ", "1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. ")):
                descriptions.append(line[2:].lower())
        return descriptions

    def evaluate(self, response_text):
        structured_validation = parse_structured(response_text, fmt="json")
        descriptions = self._extract_descriptions(response_text)
        if not descriptions:
            return EvaluationResult(0.0, [])

        combined = " ".join(descriptions)
        source_text = self.get_prompt().lower()
        rubric = Rubric(self.max_score)
        rubric.record_validation(structured_validation)

        resource_evidence = re.search(
            r"\b(?:open\(|\.close\(|context\s+manager|with\s+open|file\s+handle|finally)\b",
            combined,
        ) or (
            re.search(r"\b(?:resource\s+leak|descriptor)\b", combined)
            if re.search(r"\b(?:file|f\.write|db_path|handle)\b", combined)
            else None
        )
        rubric.add_criterion(
            "File handle not closed / resource leak",
            3.0,
            3.0 if resource_evidence else 0.0,
            evidence=[{"kind": "review-evidence", "span": resource_evidence.group(0)}]
            if resource_evidence else [],
        )

        none_evidence = re.search(
            r"\b(==\s*none|\bis\s+none|identity\s+comparison|none\s+comparison|null\s+comparison)\b",
            combined,
        )
        if none_evidence:
            rubric.add_criterion(
                "== None instead of is None",
                2.0,
                2.0,
                evidence=[{"kind": "regex", "span": none_evidence.group(0)}],
            )
        elif re.search(r"\b(?:none|null)\b", source_text) and (
            re.search(r"\b(?:identity|equality|comparison|is\s+none|none\s+check)\b", combined)
        ):
            rubric.add_criterion(
                "== None instead of is None",
                2.0,
                2.0,
                evidence=[{"kind": "semantic-alias", "span": combined[:240]}],
            )
        else:
            rubric.add_criterion("== None instead of is None", 2.0, 0.0)

        path_evidence = re.search(r"\b(?:/tmp/data\.txt|tmp/data|db_path)\b", combined) or (
            re.search(r"\b(?:hardcoded\s+path|absolute\s+path|configur(?:e|able)\s+path)\b", combined)
            if re.search(r"\b(?:path|filename|file)\b", combined)
            else None
        )
        rubric.add_criterion(
            "Hardcoded /tmp path",
            2.0,
            2.0 if path_evidence else 0.0,
            evidence=[{"kind": "review-evidence", "span": path_evidence.group(0)}]
            if path_evidence else [],
        )

        earned = 0.0
        has_fetch_data = re.search(r"\bfetch_data\b", combined)
        has_error_handling = re.search(r"\b(try|except|error\s+handling|may\s+raise|could\s+fail|exception|external\s+call|network\s+failure|failure\s+handling|defensive)\b", combined)
        if (has_fetch_data and has_error_handling) or (
            has_error_handling
            and re.search(r"\b(?:external\s+call|network\s+failure|defensive|failure\s+handling)\b", combined)
            and re.search(r"\b(?:data|provider|request|fetch|user_id)\b", combined)
        ):
            earned = 3.0
        rubric.add_criterion("Missing error handling / fetch_data may fail", 3.0, earned)

        earned = 0.0
        has_unused = re.search(r"\b(unused\s+import|unused\s+module|not\s+used|remove\s+(?:the\s+)?import|dead\s+import|unnecessary\s+import)\b", combined)
        mentions_relevant_import = re.search(r"\b(?:os|time)\b", combined)
        if has_unused and mentions_relevant_import:
            earned = 2.0
        rubric.add_criterion("Unused imports", 2.0, earned)

        rubric.eval_regex(
            "Actionable / concrete fix",
            3.0,
            combined,
            [(r"\b(use\s+(?:a\s+)?context\s+manager|with\s+open|close\s+the\s+file|remove\s+(?:the\s+)?import|is\s+none|compare\s+to\s+none|try:|except|parameterize|validate|sanitize|inject)\b", 3.0)],
        )

        if not re.search(r"\b(?:close|context\s+manager|is\s+none|try|except|finally|validate|sanitize|defensive|parameterize|inject)\b", combined):
            rubric.penalize_criterion(
                "Actionable / concrete fix", 1.0,
                "findings contain no actionable remediation language",
            )
        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
