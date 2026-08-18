"""Multi-record filtering, deduplication, normalization, and sorting challenge."""
from __future__ import annotations

import copy
import json
from typing import Any

from jsonschema import Draft202012Validator

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import parse_structured

DATA_TRANSFORMATION_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["records", "summary"],
    "properties": {
        "records": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["order_id", "customer", "total", "rank"],
                "properties": {
                    "order_id": {
                        "type": "string",
                        "pattern": r"^O-[0-9]{3}$",
                    },
                    "customer": {"type": "string"},
                    "total": {"type": "number"},
                    "rank": {"type": "integer", "minimum": 1, "maximum": 5},
                },
            },
        },
        "summary": {
            "type": "object",
            "additionalProperties": False,
            "required": ["count", "total", "top_order_id"],
            "properties": {
                "count": {"type": "integer", "minimum": 1, "maximum": 5},
                "total": {"type": "number"},
                "top_order_id": {
                    "type": "string",
                    "pattern": r"^O-[0-9]{3}$",
                },
            },
        },
    },
}

DATA_TRANSFORMATION_SOURCE_PACKET = """SOURCE PACKET: ORDER FEED

The feed contains historical versions, records that must be filtered out,
and deliberately misleading values. Treat each order_id as one logical order.

RECORDS (the largest version number is current; older versions are superseded):
- O-201 v1 | customer=" alice martin " | total="80.00" | status=paid | refunded=no | region=EU | channel=web
- O-201 v2 | customer="Alice Martin" | total="120.00" | status=paid | refunded=no | region=EU | channel=web
- O-202 v1 | customer="Bob Jones" | total="150.00" | status=pending | refunded=no | region=US | channel=web
- O-202 v2 | customer="Bob Jones" | total="150.00" | status=paid | refunded=no | region=US | channel=web
- O-203 v1 | customer="Carol Smith" | total="200.00" | status=paid | refunded=yes | region=EU | channel=web
- O-204 v1 | customer="Dan Wu" | total="75.00" | status=paid | refunded=no | region=APAC | channel=web
- O-205 v1 | customer="Eve Stone" | total="49.99" | status=paid | refunded=no | region=EU | channel=web
- O-206 v1 | customer="Fiona Green" | total="100.00" | status=paid | refunded=no | region=EU | channel=internal
- O-207 v1 | customer="zara adams" | total="120.00" | status=paid | refunded=no | region=EU | channel=web
- O-208 v1 | customer="Old Name" | total="130.00" | status=paid | refunded=no | region=EU | channel=web
- O-208 v2 | customer="aaron lee" | total="120.00" | status=paid | refunded=no | region=EU | channel=web
- O-209 v1 | customer="  Priya Shah  " | total="60.00" | status=paid | refunded=no | region=EU | channel=web

TRANSFORMATION RULES:
1. Keep only the newest version of each order_id.
2. From those current versions, keep records where status=paid, refunded=no,
   total is at least 50.00, region is not APAC, and channel is not internal.
3. Normalize customer names by trimming whitespace and converting them to
   title case. Convert total strings to JSON numbers.
4. Sort retained records by total descending, then normalized customer name
   ascending. Assign ranks starting at 1 in that final order.
5. Return only order_id, customer, total, and rank for each retained record.
6. In summary, report the retained record count, the sum of retained totals,
   and the order_id of the first ranked record.

Return exactly one JSON object matching the requested schema. Do not include
filtered records, superseded versions, status/region/channel fields, prose,
or extra keys."""

DATA_TRANSFORMATION_EXPECTED_OUTPUT = {
    "records": [
        {"order_id": "O-202", "customer": "Bob Jones", "total": 150.0, "rank": 1},
        {"order_id": "O-208", "customer": "Aaron Lee", "total": 120.0, "rank": 2},
        {"order_id": "O-201", "customer": "Alice Martin", "total": 120.0, "rank": 3},
        {"order_id": "O-207", "customer": "Zara Adams", "total": 120.0, "rank": 4},
        {"order_id": "O-209", "customer": "Priya Shah", "total": 60.0, "rank": 5},
    ],
    "summary": {"count": 5, "total": 570.0, "top_order_id": "O-202"},
}


class DataTransformationPlugin(BenchmarkTaskPlugin):
    """Evaluate deterministic multi-record data processing."""

    @property
    def id(self):
        return "data-transformation"

    @property
    def version(self):
        return "1.0.1"

    @property
    def name(self):
        return "Data Transformation"

    @property
    def max_score(self):
        return 22.0

    @property
    def supports_streaming(self):
        return False

    def get_prompt(self):
        return (
            "Process the order feed according to every transformation rule.\n\n"
            "Return exactly one JSON object with records and summary; do not return "
            "filtered or superseded records. Each record must contain order_id, "
            "customer, total, and rank.\n\n"
            + DATA_TRANSFORMATION_SOURCE_PACKET
        )

    def get_temperature(self, global_config):
        return global_config.get("data_transformation_temperature")

    def get_response_schema(self):
        """Expose the schema for compatibility diagnostics and sentinel tooling."""
        return copy.deepcopy(DATA_TRANSFORMATION_RESPONSE_SCHEMA)

    def get_request_params(self, global_config):
        """Request the strict machine-readable result contract."""
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "data_transformation_result",
                    "strict": True,
                    "schema": copy.deepcopy(DATA_TRANSFORMATION_RESPONSE_SCHEMA),
                },
            },
        }

    @staticmethod
    def _schema_validation(data: Any) -> tuple[bool, list[str]]:
        errors = sorted(
            Draft202012Validator(DATA_TRANSFORMATION_RESPONSE_SCHEMA).iter_errors(data),
            key=lambda error: list(error.path),
        )
        return not errors, [error.message for error in errors]

    @staticmethod
    def _json_format_valid(text: str) -> bool:
        candidate = text.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if len(lines) < 3 or not lines[-1].strip().startswith("```"):
                return False
            candidate = "\n".join(lines[1:-1]).strip()
        try:
            json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            return False
        return True

    @staticmethod
    def _evaluation_with_diagnostics(rubric, schema_valid, schema_errors):
        result = rubric.results()
        diagnostics = dict(result.diagnostics or {})
        diagnostics.update({
            "schema_requested": True,
            "response_schema_valid": schema_valid,
            "response_schema_errors": schema_errors,
            "schema_enforcement_verified": False,
        })
        return EvaluationResult(result.score, result.rubric, diagnostics)

    @staticmethod
    def _records(data):
        records = data.get("records") if isinstance(data, dict) else None
        return records if isinstance(records, list) else []

    @staticmethod
    def _record_map(data):
        return {
            record.get("order_id"): record
            for record in DataTransformationPlugin._records(data)
            if isinstance(record, dict) and isinstance(record.get("order_id"), str)
        }

    @staticmethod
    def _criterion(rubric, name, maximum, earned, evidence=None, findings=None):
        rubric.add_criterion(
            name,
            maximum,
            earned,
            evidence=evidence or [],
            matched=earned > 0,
            negative_findings=findings or [],
        )

    def evaluate(self, response_text):
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        validation = parse_structured(text)
        rubric.record_validation(validation)
        if not validation.valid or not isinstance(validation.value, dict):
            for name, maximum in (
                ("Structured schema contract", 1.0),
                ("Record selection and filtering", 7.0),
                ("Deduplication and latest versions", 4.0),
                ("Normalization", 3.0),
                ("Sorting and ranking", 3.0),
                ("Derived summary", 3.0),
                ("Strict format and no decoys", 1.0),
            ):
                rubric.add_criterion(name, maximum, 0.0)
            return self._evaluation_with_diagnostics(
                rubric, False, list(validation.errors or ["structured object could not be parsed"]),
            )

        data = validation.value
        schema_valid, schema_errors = self._schema_validation(data)
        if not self._json_format_valid(text):
            schema_valid = False
            schema_errors = [*schema_errors, "response is not valid JSON"]
        self._criterion(
            rubric,
            "Structured schema contract",
            1.0,
            1.0 if schema_valid else 0.0,
            findings=[{"finding": error} for error in schema_errors],
        )

        actual_records = self._records(data)
        actual_ids = [record.get("order_id") for record in actual_records if isinstance(record, dict)]
        expected_records = DATA_TRANSFORMATION_EXPECTED_OUTPUT["records"]
        expected_ids = [record["order_id"] for record in expected_records]
        expected_id_set = set(expected_ids)
        actual_id_set = set(actual_ids)
        matched_ids = expected_id_set & actual_id_set
        filtering_findings = []
        if actual_id_set - expected_id_set:
            filtering_findings.append({"finding": "filtered-out or unknown order was returned", "orders": sorted(actual_id_set - expected_id_set)})
        if expected_id_set - actual_id_set:
            filtering_findings.append({"finding": "eligible order was omitted", "orders": sorted(expected_id_set - actual_id_set)})
        self._criterion(
            rubric,
            "Record selection and filtering",
            7.0,
            round(
                7.0 * len(matched_ids) / max(len(expected_id_set), len(actual_id_set), 1),
                1,
            ),
            evidence=[{"kind": "retained-order", "order_id": order_id} for order_id in sorted(matched_ids)],
            findings=filtering_findings,
        )

        actual_map = self._record_map(data)
        historical_expected = {"O-201": 120.0, "O-202": 150.0, "O-208": 120.0}
        historical_matches = [
            order_id for order_id, total in historical_expected.items()
            if actual_map.get(order_id, {}).get("total") == total
        ]
        duplicate_free = len(actual_ids) == len(actual_id_set)
        dedup_earned = 4.0 * (len(historical_matches) / len(historical_expected))
        if not duplicate_free:
            dedup_earned = max(0.0, dedup_earned - 1.0)
        self._criterion(
            rubric,
            "Deduplication and latest versions",
            4.0,
            round(dedup_earned, 1),
            evidence=[{"kind": "latest-version", "order_id": order_id} for order_id in historical_matches],
            findings=[] if duplicate_free else [{"finding": "duplicate order_id appears more than once"}],
        )

        normalized_matches = [
            record["order_id"] for record in expected_records
            if actual_map.get(record["order_id"], {}).get("customer") == record["customer"]
            and actual_map.get(record["order_id"], {}).get("total") == record["total"]
        ]
        self._criterion(
            rubric,
            "Normalization",
            3.0,
            round(3.0 * len(normalized_matches) / len(expected_records), 1),
            evidence=[{"kind": "normalized-record", "order_id": order_id} for order_id in normalized_matches],
            findings=[] if len(normalized_matches) == len(expected_records) else [{"finding": "customer or total was not normalized exactly"}],
        )

        actual_order = actual_ids
        actual_ranks = [record.get("rank") for record in actual_records if isinstance(record, dict)]
        expected_order = [record["order_id"] for record in expected_records]
        order_matches = sum(left == right for left, right in zip(actual_order, expected_order))
        rank_matches = sum(rank == index for index, rank in enumerate(actual_ranks, 1))
        sorting_earned = 1.5 * order_matches / len(expected_order) + 1.5 * rank_matches / len(expected_order)
        self._criterion(
            rubric,
            "Sorting and ranking",
            3.0,
            round(sorting_earned, 1),
            evidence=[{"kind": "ordered-rank", "order_id": order_id, "rank": rank} for order_id, rank in zip(actual_order, actual_ranks)],
            findings=[] if actual_order == expected_order and actual_ranks == list(range(1, 6)) else [{"finding": "records are not sorted and ranked deterministically"}],
        )

        expected_summary = DATA_TRANSFORMATION_EXPECTED_OUTPUT["summary"]
        actual_summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        summary_fields = ["count", "total", "top_order_id"]
        summary_matches = [field for field in summary_fields if actual_summary.get(field) == expected_summary[field]]
        self._criterion(
            rubric,
            "Derived summary",
            3.0,
            round(3.0 * len(summary_matches) / len(summary_fields), 1),
            evidence=[{"kind": "summary-field", "field": field} for field in summary_matches],
            findings=[] if len(summary_matches) == len(summary_fields) else [{"finding": "summary does not match retained records"}],
        )

        strict = set(data) == {"records", "summary"}
        records_strict = all(
            isinstance(record, dict) and set(record) == {"order_id", "customer", "total", "rank"}
            for record in actual_records
        )
        summary_strict = isinstance(actual_summary, dict) and set(actual_summary) == set(summary_fields)
        placeholders = any(
            isinstance(value, str) and value.strip().lower() in {"unknown", "none", "n/a", ""}
            for record in actual_records
            for value in record.values()
        )
        strict_valid = strict and records_strict and summary_strict and not placeholders
        self._criterion(
            rubric,
            "Strict format and no decoys",
            1.0,
            1.0 if strict_valid else 0.0,
            findings=[] if strict_valid else [{"finding": "extra keys or placeholder/decoy values present"}],
        )
        return self._evaluation_with_diagnostics(rubric, schema_valid, schema_errors)

    def score(self, response_text):
        return self.evaluate(response_text).score
