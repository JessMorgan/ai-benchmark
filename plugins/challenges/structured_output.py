"""Structured output (JSON/YAML) benchmark task."""
import json
import re

from benchmark_plugin import BenchmarkTaskPlugin
from plugins.challenges._rubric import Rubric

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
        return "0.2.0"

    @property
    def name(self):
        return "Structured Output"

    @property
    def max_score(self):
        return 20.0

    @property
    def supports_streaming(self):
        return False

    def get_prompt(self):
        return (
            "Produce a valid JSON or YAML object representing an employee record. "
            "Do not include any explanatory text outside the structured data.\n\n"
            "Required top-level fields:\n"
            "- id (string, UUID v4 format, e.g. '550e8400-e29b-41d4-a716-446655440000')\n"
            "- name (string)\n"
            "- age (integer, 18-120 inclusive)\n"
            "- email (string, valid email format)\n"
            "- department (string, one of: Engineering, Sales, Marketing, HR)\n"
            "- roles (array of strings, each one of: admin, editor, viewer, auditor)\n"
            "- address (object with keys: street, city, state (2-letter US code uppercase), zip (5-digit string))\n"
            "- settings (object with keys: theme (one of: dark, light, auto), "
            "  notifications (object with keys: email (boolean), sms (boolean), push (boolean)), "
            "  language (2-letter ISO 639-1 code, e.g. 'en', 'es'))\n"
            "- tags (array of objects, each with name (string) and priority (integer 1-5))\n"
            "- metadata (object with created_at (ISO 8601 datetime with timezone, e.g. '2024-01-15T09:30:00Z'), "
            "  active (boolean), score (float 0.0-1.0))\n\n"
            "Example JSON format:\n"
            "{\n"
            "  \"id\": \"550e8400-e29b-41d4-a716-446655440000\",\n"
            "  \"name\": \"Alice\",\n"
            "  \"age\": 30,\n"
            "  \"email\": \"alice@example.com\",\n"
            "  \"department\": \"Engineering\",\n"
            "  \"roles\": [\"admin\", \"editor\"],\n"
            "  \"address\": {\n"
            "    \"street\": \"123 Main St\",\n"
            "    \"city\": \"Springfield\",\n"
            "    \"state\": \"IL\",\n"
            "    \"zip\": \"62701\"\n"
            "  },\n"
            "  \"settings\": {\n"
            "    \"theme\": \"dark\",\n"
            "    \"notifications\": {\"email\": true, \"sms\": false, \"push\": true},\n"
            "    \"language\": \"en\"\n"
            "  },\n"
            "  \"tags\": [{\"name\": \"full-time\", \"priority\": 1}, {\"name\": \"remote\", \"priority\": 3}],\n"
            "  \"metadata\": {\n"
            "    \"created_at\": \"2024-01-15T09:30:00Z\",\n"
            "    \"active\": true,\n"
            "    \"score\": 0.95\n"
            "  }\n"
            "}\n\n"
            "Example YAML format:\n"
            "id: 550e8400-e29b-41d4-a716-446655440000\n"
            "name: Alice\n"
            "age: 30\n"
            "email: alice@example.com\n"
            "department: Engineering\n"
            "roles:\n"
            "  - admin\n"
            "  - editor\n"
            "address:\n"
            "  street: 123 Main St\n"
            "  city: Springfield\n"
            "  state: IL\n"
            "  zip: '62701'\n"
            "settings:\n"
            "  theme: dark\n"
            "  notifications:\n"
            "    email: true\n"
            "    sms: false\n"
            "    push: true\n"
            "  language: en\n"
            "tags:\n"
            "  - name: full-time\n"
            "    priority: 1\n"
            "  - name: remote\n"
            "    priority: 3\n"
            "metadata:\n"
            "  created_at: 2024-01-15T09:30:00Z\n"
            "  active: true\n"
            "  score: 0.95"
        )

    def get_temperature(self, global_config):
        if "structured_output_temperature" in global_config:
            return global_config["structured_output_temperature"]
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

    @staticmethod
    def _is_uuid_v4(value):
        if not isinstance(value, str):
            return False
        pattern = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
        return bool(re.match(pattern, value))

    @staticmethod
    def _is_email(value):
        if not isinstance(value, str):
            return False
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, value))

    @staticmethod
    def _is_iso_datetime(value):
        if not isinstance(value, str):
            return False
        # ISO 8601 with optional timezone, e.g. 2024-01-15T09:30:00Z or +00:00
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$"
        return bool(re.match(pattern, value))

    def evaluate(self, response_text):
        t = response_text.strip()
        rubric = Rubric(self.max_score)

        candidate = self._extract_candidate(t)

        has_explanatory_text = bool(
            re.search(r'```', t)
            and re.sub(r'```[\s\S]*?```', '', t).strip()
        )

        data = self._parse_data(candidate)
        if data is None:
            rubric.add_criterion("Valid JSON/YAML syntax", 4.0, 0.0)
            return rubric.results()
        rubric.add_criterion("Valid JSON/YAML syntax", 4.0, 4.0)

        required = {
            "id", "name", "age", "email", "department", "roles",
            "address", "settings", "tags", "metadata",
        }
        present = required & set(data.keys())
        if len(present) == len(required):
            earned = 4.0
        else:
            earned = (len(present) / len(required)) * 2.0
        rubric.add_criterion("Required top-level fields", 4.0, earned)

        type_score = 0.0

        if isinstance(data.get("name"), str) and data.get("name"):
            type_score += 0.5

        age = data.get("age")
        if isinstance(age, int) and 18 <= age <= 120:
            type_score += 0.5

        if self._is_email(data.get("email")):
            type_score += 0.5

        if self._is_uuid_v4(data.get("id")):
            type_score += 0.5

        department = data.get("department")
        if isinstance(department, str) and department in {"Engineering", "Sales", "Marketing", "HR"}:
            type_score += 0.5

        roles = data.get("roles")
        allowed_roles = {"admin", "editor", "viewer", "auditor"}
        if isinstance(roles, list) and roles and all(isinstance(r, str) and r in allowed_roles for r in roles):
            type_score += 0.5

        address = data.get("address")
        if isinstance(address, dict):
            if all(k in address for k in ("street", "city", "state", "zip")):
                type_score += 0.5
            if isinstance(address.get("state"), str) and re.match(r"^[A-Z]{2}$", address.get("state", "")):
                type_score += 0.5
            if isinstance(address.get("zip"), str) and re.match(r"^\d{5}$", address.get("zip", "")):
                type_score += 0.5

        settings = data.get("settings")
        if isinstance(settings, dict):
            theme = settings.get("theme")
            if theme in {"dark", "light", "auto"}:
                type_score += 0.5
            notifications = settings.get("notifications")
            if isinstance(notifications, dict) and all(k in notifications for k in ("email", "sms", "push")):
                type_score += 0.5
            language = settings.get("language")
            if isinstance(language, str) and re.match(r"^[a-zA-Z]{2}$", language):
                type_score += 0.5

        tags = data.get("tags")
        if isinstance(tags, list) and tags and all(
            isinstance(tag, dict)
            and isinstance(tag.get("name"), str)
            and isinstance(tag.get("priority"), int)
            and 1 <= tag.get("priority", 0) <= 5
            for tag in tags
        ):
            type_score += 0.5

        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            if self._is_iso_datetime(metadata.get("created_at")):
                type_score += 0.5
            if isinstance(metadata.get("active"), bool):
                type_score += 0.5
            score_val = metadata.get("score")
            if isinstance(score_val, (int, float)) and 0.0 <= float(score_val) <= 1.0:
                type_score += 0.5

        rubric.add_criterion("Basic types and constraints", 6.0, type_score)

        complete = (
            data.get("name")
            and data.get("email")
            and isinstance(roles, list)
            and len(roles) > 0
            and isinstance(tags, list)
            and len(tags) > 0
        )
        earned = 2.0 if complete else 0.0
        rubric.add_criterion("Non-empty values / completeness", 4.0, earned)

        if isinstance(data, dict) and set(data.keys()) == required:
            earned = 2.0
        elif isinstance(data, dict):
            earned = max(0.0, 2.0 - abs(len(data.keys()) - len(required)) * 0.5)
        else:
            earned = 0.0
        rubric.add_criterion("Strict format (no extra keys)", 2.0, earned)

        if complete:
            bad_values = {"unknown", "n/a", "none", "null", "", None}
            leaf_values = []
            self._collect_leaf_values(data, leaf_values)
            earned = 2.0 if all(v not in bad_values for v in leaf_values) else 0.0
        else:
            earned = 0.0
        rubric.add_criterion("No placeholder values", 2.0, earned)

        score, criteria = rubric.results()
        if has_explanatory_text:
            score = round(max(score - 0.5, 0.0), 1)
        return score, criteria

    def score(self, response_text):
        return self.evaluate(response_text)[0]

    def _collect_leaf_values(self, obj, out):
        """Recursively collect leaf values for placeholder checking."""
        if isinstance(obj, dict):
            for v in obj.values():
                self._collect_leaf_values(v, out)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_leaf_values(item, out)
        else:
            out.append(obj)
