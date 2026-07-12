"""Structured output (JSON/YAML) benchmark task."""
import json
import re

from benchmark_plugin import BenchmarkTaskPlugin

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


class StructuredOutputPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "structured-output"

    @property
    def version(self):
        return "0.1.0"

    @property
    def name(self):
        return "Structured Output"

    @property
    def max_score(self):
        return 15.0

    @property
    def supports_streaming(self):
        return False

    def get_prompt(self):
        return (
            "Produce a valid JSON or YAML object representing a user profile. "
            "Do not include any explanatory text outside the structured data.\n\n"
            "Required fields:\n"
            "- name (string)\n"
            "- age (integer)\n"
            "- email (string)\n"
            "- roles (array/list of strings)\n"
            "- settings (object/map with theme and notifications keys)\n\n"
            "Example JSON format:\n"
            "{\"name\": \"Alice\", \"age\": 30, \"email\": \"alice@example.com\", "
            "\"roles\": [\"admin\", \"editor\"], \"settings\": {\"theme\": \"dark\", \"notifications\": true}}\n\n"
            "Example YAML format:\n"
            "name: Alice\n"
            "age: 30\n"
            "email: alice@example.com\n"
            "roles:\n"
            "  - admin\n"
            "  - editor\n"
            "settings:\n"
            "  theme: dark\n"
            "  notifications: true"
        )

    def get_temperature(self, global_config):
        if "structured_output_temperature" in global_config:
            return global_config["structured_output_temperature"]
        if "general_temperature" in global_config:
            return global_config["general_temperature"]
        return None

    def _extract_candidate(self, response_text):
        """Extract structured data from markdown fences or raw text."""
        t = response_text.strip()
        match = re.search(r'```(?:json|yaml)?\s*(.*?)\s*```', t, re.DOTALL)
        if match:
            return match.group(1).strip()
        return t

    def _parse_data(self, candidate):
        """Try to parse candidate as JSON then YAML."""
        stripped = candidate.strip()
        looks_like_json = stripped.startswith("{") or stripped.startswith("[")
        if looks_like_json:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None
        if yaml is not None:
            try:
                return yaml.safe_load(candidate)
            except yaml.YAMLError:
                pass
        return None

    def score(self, response_text):
        t = response_text.strip()
        s = 0.0

        candidate = self._extract_candidate(t)

        # Penalize explanatory text outside the code fence
        has_explanatory_text = bool(
            re.search(r'```', t)
            and re.sub(r'```[\s\S]*?```', '', t).strip()
        )

        # 1. Valid JSON/YAML syntax (0-5)
        data = self._parse_data(candidate)
        if data is None:
            return round(s, 1)
        s += 5.0

        if has_explanatory_text:
            s -= 1.0

        # 2. Required top-level fields present (0-5)
        required = {"name", "age", "email", "roles", "settings"}
        present = required & set(data.keys())
        if len(present) == len(required):
            s += 5.0
        else:
            s += len(present)

        # 3. Correct types for fields (0-3)
        type_score = 0.0
        if isinstance(data.get("name"), str):
            type_score += 0.5
        if isinstance(data.get("age"), int):
            type_score += 0.5
        if isinstance(data.get("email"), str):
            type_score += 0.5
        roles = data.get("roles")
        if isinstance(roles, list) and all(isinstance(r, str) for r in roles):
            type_score += 0.5
        settings = data.get("settings")
        if isinstance(settings, dict):
            type_score += 0.5
            if "theme" in settings and "notifications" in settings:
                type_score += 0.5
        s += type_score

        # 4. No hallucinated / non-empty values (0-2)
        if data.get("name") and data.get("email") and data.get("roles"):
            s += 2.0

        return round(min(s, self.max_score), 1)
