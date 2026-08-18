"""Strict structured employee-record output challenge."""
from __future__ import annotations

import copy
import re
from datetime import datetime

from benchmark.plugin import BenchmarkTaskPlugin
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import parse_structured

STRUCTURED_OUTPUT_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id", "name", "age", "email", "department", "roles",
        "address", "settings", "tags", "metadata",
    ],
    "properties": {
        "id": {
            "type": "string",
            "pattern": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$",
        },
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 18, "maximum": 120},
        "email": {
            "type": "string",
            "pattern": r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$",
        },
        "department": {
            "type": "string",
            "enum": ["Engineering", "Sales", "Marketing", "HR"],
        },
        "roles": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "enum": ["admin", "editor", "viewer", "auditor"]},
        },
        "address": {
            "type": "object",
            "additionalProperties": False,
            "required": ["street", "city", "state", "zip"],
            "properties": {
                "street": {"type": "string", "minLength": 1},
                "city": {"type": "string", "minLength": 1},
                "state": {"type": "string", "pattern": r"^[A-Z]{2}$"},
                "zip": {"type": "string", "pattern": r"^\d{5}$"},
            },
        },
        "settings": {
            "type": "object",
            "additionalProperties": False,
            "required": ["theme", "notifications", "language"],
            "properties": {
                "theme": {"type": "string", "enum": ["dark", "light", "auto"]},
                "notifications": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["email", "sms", "push"],
                    "properties": {
                        "email": {"type": "boolean"},
                        "sms": {"type": "boolean"},
                        "push": {"type": "boolean"},
                    },
                },
                "language": {
                    "type": "string",
                    "enum": ["en", "es", "fr", "de", "ja", "zh", "pt", "it", "ko", "ar", "hi", "ru"],
                },
            },
        },
        "tags": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "priority"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                },
            },
        },
        "metadata": {
            "type": "object",
            "additionalProperties": False,
            "required": ["created_at", "active", "score"],
            "properties": {
                "created_at": {
                    "type": "string",
                    "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$",
                },
                "active": {"type": "boolean"},
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
        },
    },
}


class StructuredOutputPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "structured-output"

    @property
    def version(self):
        return "1.1.0"

    @property
    def name(self):
        return "Structured Output"

    @property
    def max_score(self):
        return 22.0

    @property
    def supports_streaming(self):
        return False

    def get_prompt(self):
        return (
            "Return exactly one JSON object and no explanatory text. The object must "
            "contain exactly these top-level keys: id (UUID v4 string), name (non-empty string), "
            "age (integer 18-120), email (valid email), department (Engineering/Sales/Marketing/HR), "
            "roles (non-empty array of admin/editor/viewer/auditor), address {street, city, state "
            "(uppercase US two-letter code), zip (five-digit string)}, settings {theme (dark/light/auto), "
            "notifications {email, sms, push booleans}, language (ISO 639-1)}, tags (non-empty array "
            "of {name string, priority integer 1-5}), metadata {created_at ISO-8601 datetime with "
            "timezone, active boolean, score number 0.0-1.0}."
        )

    def get_temperature(self, global_config):
        return global_config.get("structured_output_temperature")

    def get_request_params(self, global_config):
        """Enforce the same employee-record contract at the API boundary."""
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_employee_record",
                    "strict": True,
                    "schema": copy.deepcopy(STRUCTURED_OUTPUT_RESPONSE_SCHEMA),
                },
            },
        }

    _required = frozenset({"id", "name", "age", "email", "department", "roles", "address", "settings", "tags", "metadata"})
    _departments = frozenset({"Engineering", "Sales", "Marketing", "HR"})
    _roles = frozenset({"admin", "editor", "viewer", "auditor"})
    _languages = frozenset({"en", "es", "fr", "de", "ja", "zh", "pt", "it", "ko", "ar", "hi", "ru"})

    @staticmethod
    def _uuid(value):
        return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}", value))

    @staticmethod
    def _email(value):
        return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", value))

    @staticmethod
    def _datetime(value):
        if isinstance(value, datetime):
            return value.tzinfo is not None
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", value):
            return False
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return False
        return True

    @staticmethod
    def _bool(value):
        return type(value) is bool

    def _checks(self, data):
        address = data.get("address")
        settings = data.get("settings")
        notifications = settings.get("notifications") if isinstance(settings, dict) else None
        tags = data.get("tags")
        metadata = data.get("metadata")
        return [
            isinstance(data.get("name"), str) and bool(data["name"].strip()),
            type(data.get("age")) is int and 18 <= data["age"] <= 120,
            self._email(data.get("email")),
            self._uuid(data.get("id")),
            data.get("department") in self._departments,
            isinstance(data.get("roles"), list) and bool(data["roles"]) and all(isinstance(value, str) and value in self._roles for value in data["roles"]),
            isinstance(address, dict) and all(isinstance(address.get(key), str) and bool(address[key].strip()) for key in ("street", "city")),
            isinstance(address, dict) and isinstance(address.get("state"), str) and bool(re.fullmatch(r"[A-Z]{2}", address["state"])),
            isinstance(address, dict) and isinstance(address.get("zip"), str) and bool(re.fullmatch(r"\d{5}", address["zip"])),
            isinstance(settings, dict) and settings.get("theme") in {"dark", "light", "auto"},
            isinstance(notifications, dict) and all(type(notifications.get(key)) is bool for key in ("email", "sms", "push")),
            isinstance(settings, dict) and settings.get("language") in self._languages,
            isinstance(tags, list) and bool(tags) and all(isinstance(tag, dict) and isinstance(tag.get("name"), str) and bool(tag["name"].strip()) and type(tag.get("priority")) is int and 1 <= tag["priority"] <= 5 for tag in tags),
            isinstance(metadata, dict) and self._datetime(metadata.get("created_at")),
            isinstance(metadata, dict) and self._bool(metadata.get("active")),
            isinstance(metadata, dict) and type(metadata.get("score")) in (int, float) and not isinstance(metadata.get("score"), bool) and 0.0 <= float(metadata["score"]) <= 1.0,
        ]

    def _leaf_values(self, value):
        if isinstance(value, dict):
            for child in value.values():
                yield from self._leaf_values(child)
        elif isinstance(value, list):
            for child in value:
                yield from self._leaf_values(child)
        else:
            yield value

    def evaluate(self, response_text):
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        validation = parse_structured(text)
        rubric.record_validation(validation)
        if not validation.valid or not isinstance(validation.value, dict):
            names = [("Valid JSON/YAML syntax", 4.0), ("Required top-level fields", 4.0), ("Basic types and constraints", 6.0), ("Non-empty values / completeness", 4.0), ("Strict format (no extra keys)", 2.0), ("No placeholder values", 2.0)]
            for name, maximum in names:
                rubric.add_criterion(name, maximum, 0.0, negative_findings=[{"finding": "structured object could not be parsed"}] if name == "Valid JSON/YAML syntax" else [])
            return rubric.results()
        data = validation.value
        rubric.add_criterion("Valid JSON/YAML syntax", 4.0, 4.0)
        present = self._required & set(data)
        rubric.add_criterion("Required top-level fields", 4.0, 4.0 if present == self._required else 2.0 * len(present) / len(self._required), negative_findings=[] if present == self._required else [{"finding": f"missing keys: {sorted(self._required - present)}"}])
        checks = self._checks(data)
        rubric.add_criterion("Basic types and constraints", 6.0, round(6.0 * sum(checks) / len(checks), 1))
        complete = present == self._required and all(value not in (None, "", [], {}) for value in (data.get("name"), data.get("email"), data.get("roles"), data.get("tags")))
        rubric.add_criterion("Non-empty values / completeness", 4.0, 4.0 if complete else 0.0)
        exact_keys = set(data) == self._required
        rubric.add_criterion("Strict format (no extra keys)", 2.0, 2.0 if exact_keys else 0.0, negative_findings=[] if exact_keys else [{"finding": f"unexpected top-level keys: {sorted(set(data) - self._required)}"}])
        fenced = ["fence"] if "```" in text else []
        outside = re.sub(r"```[\s\S]*?```", "", text).strip()
        if fenced and outside:
            rubric.penalize_criterion("Strict format (no extra keys)", 0.5, "response contains explanatory text outside structured data")
        bad = {"unknown", "n/a", "none", "null", ""}
        placeholders = [value for value in self._leaf_values(data) if isinstance(value, str) and value.strip().lower() in bad]
        rubric.add_criterion("No placeholder values", 2.0, 2.0 if complete and not placeholders else 0.0, negative_findings=[] if not placeholders else [{"finding": "placeholder value present"}])
        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
