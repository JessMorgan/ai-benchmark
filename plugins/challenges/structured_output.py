"""Strict structured employee-record output challenge."""
from __future__ import annotations

import copy
import re
from datetime import datetime
from typing import ClassVar

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
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 18, "maximum": 120},
        "email": {
            "type": "string",
            "pattern": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+[.][A-Za-z]{2,}$",
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
                "street": {"type": "string"},
                "city": {"type": "string"},
                "state": {"type": "string", "pattern": r"^[A-Z]{2}$"},
                "zip": {"type": "string", "pattern": r"^[0-9]{5}$"},
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
                    "name": {"type": "string"},
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
                    "pattern": r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})$",
                },
                "active": {"type": "boolean"},
                "score": {"type": "number"},
            },
        },
    },
}


STRUCTURED_OUTPUT_SOURCE_PACKET = """SOURCE PACKET

ARCHIVED SNAPSHOT (superseded; do not use these values):
- Record ID: 2e1b4c88-7f90-4a12-8d33-0c5d9a6e7102
- Name: Rivera, Casey
- Email: casey.rivera@old-example.net
- Organization: Product / Sales
- Access labels: sales-read
- Address: 8 Old Harbor Road; Austin, Texas 78701
- Preferences: theme=light; email_digest=disabled; sms_alerts=enabled; push_alerts=disabled; locale=es-MX
- Account state: inactive

CURRENT PROFILE (use this record):
- Record ID: 7f3e9c2a-1d84-4b76-a5c1-0e9d8f6a2b34
- Display name: Rivera, Jordan
- Date of birth: 1990-05-12
- Reference date for age calculation: 2026-08-18
- Primary email:   Jordan.Rivera@Example.COM
- Organization path: Product / Platform Engineering
- Access labels: platform-admin, audit-read
- Shipping address: 24 Market Street; Portland, Oregon 97205
- Preferences: theme=dark; email_digest=enabled; sms_alerts=disabled; push_alerts=enabled; locale=en-US
- Tags: platform (P2), onboarding (P1)
- Created at: 2024-06-01 09:30:00 UTC
- Account state: active
- Quality score: 85/100

TRANSFORMATION RULES:
1. Use CURRENT PROFILE only; the archived snapshot is a decoy.
2. Convert `Family, Given` to `Given Family` for `name`.
3. Calculate age on the reference date; subtract one if the birthday has not occurred yet.
4. Lowercase and trim the email address.
5. Map an organization ending in `Engineering` to department `Engineering`.
6. Map access labels: `platform-admin` → `admin`; `audit-read` → `auditor`.
   Sort roles alphabetically.
7. Expand the state name `Oregon` to `OR` and split the shipping address into fields.
8. Map enabled/disabled preferences to booleans and keep only the first locale
   component (`en-US` → `en`).
9. Convert tag priorities P1–P5 to integers and sort tags by priority, then name.
10. Convert the UTC creation time to ISO-8601 with `Z`; convert quality score
    from a percentage to a decimal.

Return exactly one JSON object matching the requested schema. Do not include
explanations, archived values, source-only fields, or extra keys."""

STRUCTURED_OUTPUT_EXPECTED_RECORD = {
    "id": "7f3e9c2a-1d84-4b76-a5c1-0e9d8f6a2b34",
    "name": "Jordan Rivera",
    "age": 36,
    "email": "jordan.rivera@example.com",
    "department": "Engineering",
    "roles": ["admin", "auditor"],
    "address": {
        "street": "24 Market Street",
        "city": "Portland",
        "state": "OR",
        "zip": "97205",
    },
    "settings": {
        "theme": "dark",
        "notifications": {"email": True, "sms": False, "push": True},
        "language": "en",
    },
    "tags": [
        {"name": "onboarding", "priority": 1},
        {"name": "platform", "priority": 2},
    ],
    "metadata": {
        "created_at": "2024-06-01T09:30:00Z",
        "active": True,
        "score": 0.85,
    },
}


class StructuredOutputPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "structured-output"

    @property
    def version(self):
        return "1.4.0"

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
            "Extract and normalize the current employee profile from the source packet below.\n\n"
            "The output must contain exactly these top-level keys: id (UUID v4 string), "
            "name (non-empty string), age (integer 18-120), email (valid email), "
            "department (Engineering/Sales/Marketing/HR), roles (non-empty array of "
            "admin/editor/viewer/auditor), address {street and city non-empty strings, "
            "state (uppercase US two-letter code), zip (five-digit string)}, settings "
            "{theme (dark/light/auto), notifications {email, sms, push booleans}, "
            "language (ISO 639-1)}, tags (non-empty array of {name non-empty string, "
            "priority integer 1-5}), metadata {created_at ISO-8601 datetime with "
            "timezone, active boolean, score number 0.0-1.0}.\n\n"
            + STRUCTURED_OUTPUT_SOURCE_PACKET
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

    _required = frozenset(STRUCTURED_OUTPUT_EXPECTED_RECORD)
    _departments = frozenset({"Engineering", "Sales", "Marketing", "HR"})
    _roles = frozenset({"admin", "editor", "viewer", "auditor"})
    _languages = frozenset({"en", "es", "fr", "de", "ja", "zh", "pt", "it", "ko", "ar", "hi", "ru"})
    _object_keys: ClassVar[dict[tuple[str, ...], frozenset[str]]] = {
        (): _required,
        ("address",): frozenset({"street", "city", "state", "zip"}),
        ("settings",): frozenset({"theme", "notifications", "language"}),
        ("settings", "notifications"): frozenset({"email", "sms", "push"}),
        ("metadata",): frozenset({"created_at", "active", "score"}),
    }

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

    @staticmethod
    def _value_at(data, path):
        value = data
        for key in path:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    def _unexpected_keys(self, data):
        findings = []
        for path, allowed in self._object_keys.items():
            value = self._value_at(data, path)
            if isinstance(value, dict):
                for key in sorted(set(value) - allowed):
                    location = ".".join(path) or "<root>"
                    findings.append(f"unexpected key {location}.{key}")
        return findings

    def _semantic_criterion(self, rubric, name, maximum, data, paths):
        matched = []
        mismatched = []
        for path in paths:
            actual = self._value_at(data, path)
            expected = self._value_at(STRUCTURED_OUTPUT_EXPECTED_RECORD, path)
            if actual == expected:
                matched.append(".".join(path))
            else:
                mismatched.append({
                    "finding": f"{'.'.join(path)} does not match the current profile",
                    "expected": expected,
                    "actual": actual,
                })
        earned = round(maximum * len(matched) / len(paths), 1) if paths else 0.0
        rubric.add_criterion(
            name,
            maximum,
            earned,
            evidence=[{"kind": "exact-field", "path": path} for path in matched],
            matched=bool(matched),
            negative_findings=mismatched,
        )

    def evaluate(self, response_text):
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        validation = parse_structured(text)
        rubric.record_validation(validation)
        if not validation.valid or not isinstance(validation.value, dict):
            names = [
                ("Valid JSON/YAML syntax", 2.0),
                ("Required top-level fields", 2.0),
                ("Basic types and constraints", 3.0),
                ("Source extraction accuracy", 7.0),
                ("Normalization and derived values", 5.0),
                ("Strict format (no extra keys)", 2.0),
                ("No placeholder values", 1.0),
            ]
            for name, maximum in names:
                rubric.add_criterion(
                    name,
                    maximum,
                    0.0,
                    negative_findings=[{"finding": "structured object could not be parsed"}]
                    if name == "Valid JSON/YAML syntax" else [],
                )
            return rubric.results()

        data = validation.value
        rubric.add_criterion("Valid JSON/YAML syntax", 2.0, 2.0)
        present = self._required & set(data)
        rubric.add_criterion(
            "Required top-level fields",
            2.0,
            2.0 if present == self._required else 2.0 * len(present) / len(self._required),
            negative_findings=[] if present == self._required else [
                {"finding": f"missing keys: {sorted(self._required - present)}"}
            ],
        )
        checks = self._checks(data)
        rubric.add_criterion(
            "Basic types and constraints",
            3.0,
            round(3.0 * sum(checks) / len(checks), 1),
        )
        self._semantic_criterion(
            rubric,
            "Source extraction accuracy",
            7.0,
            data,
            [
                ("id",),
                ("email",),
                ("settings", "theme"),
                ("settings", "notifications", "email"),
                ("settings", "notifications", "sms"),
                ("settings", "notifications", "push"),
                ("metadata", "active"),
            ],
        )
        self._semantic_criterion(
            rubric,
            "Normalization and derived values",
            5.0,
            data,
            [
                ("name",),
                ("age",),
                ("department",),
                ("roles",),
                ("address",),
                ("settings", "language"),
                ("tags",),
                ("metadata", "created_at"),
                ("metadata", "score"),
            ],
        )
        unexpected = self._unexpected_keys(data)
        exact_keys = not unexpected and set(data) == self._required
        rubric.add_criterion(
            "Strict format (no extra keys)",
            2.0,
            2.0 if exact_keys else 0.0,
            negative_findings=[] if exact_keys else [{"finding": item} for item in unexpected]
            or [{"finding": f"unexpected top-level keys: {sorted(set(data) - self._required)}"}],
        )
        fenced = "```" in text
        outside = re.sub(r"```[\s\S]*?```", "", text).strip()
        if fenced and outside:
            rubric.penalize_criterion(
                "Strict format (no extra keys)",
                0.5,
                "response contains explanatory text outside structured data",
            )
        bad = {"unknown", "n/a", "none", "null", ""}
        placeholders = [
            value for value in self._leaf_values(data)
            if isinstance(value, str) and value.strip().lower() in bad
        ]
        rubric.add_criterion(
            "No placeholder values",
            1.0,
            1.0 if not placeholders else 0.0,
            negative_findings=[] if not placeholders else [{"finding": "placeholder value present"}],
        )
        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
