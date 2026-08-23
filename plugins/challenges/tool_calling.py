"""Typed tool-routing and itinerary synthesis challenge."""
from __future__ import annotations

import re
from datetime import date, datetime

from benchmark.plugin import BenchmarkTaskPlugin
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import parse_tool_calls


class ToolCallingPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "tool-calling"

    @property
    def version(self):
        return "1.1.0"

    @property
    def name(self):
        return "Tool Calling Agent"

    @property
    def max_score(self):
        return 25.0

    @property
    def supports_streaming(self):
        return True

    def get_prompt(self):
        return (
            "Plan and call exactly these six tools in this order: get_weather(Tokyo,celsius), "
            "search_flights(JFK,Tokyo,2024-08-15), book_hotel(Tokyo,2024-08-16,2024-08-20,2), "
            "get_stock_price(SONY), convert_currency(1000,USD,JPY), send_email(alice@example.com, "
            "subject Tokyo Trip Itinerary, body). Put the plan in `<plan>...</plan>`, each call in "
            "one valid `<tool_call>{...}</tool_call>`, and after the calls provide a final response "
            "covering weather, flight, hotel, stock, email, and a numeric converted JPY amount."
        )

    def get_temperature(self, global_config):
        return global_config.get("tool_calling_temperature")

    def sanitize_for_judge(self, text):
        """Mask the XML-ish tool-call tags shown to judge models.

        Candidate answers (and the task text) are full of
        ``<tool_call>{...}</tool_call>`` and ``<plan>...</plan>`` blocks. The
        angle-bracket tags mimic the judge's own required JSON output and
        repeatedly hijack judge models into echoing the format instead of
        returning their ``{"score": ...}`` verdict (observed on ~2/3 of
        tool-calling judging attempts). Replace the tags with neutral
        markers, keeping the JSON bodies so the judge can still evaluate the
        arguments.
        """
        text = re.sub(r"<tool_call>", "[TOOL_CALL]", text, flags=re.IGNORECASE)
        text = re.sub(r"</tool_call>", "[/TOOL_CALL]", text, flags=re.IGNORECASE)
        text = re.sub(r"<plan>", "[PLAN]", text, flags=re.IGNORECASE)
        text = re.sub(r"</plan>", "[/PLAN]", text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _date_matches(value, expected):
        if not isinstance(value, str):
            return False
        candidate = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(candidate).date()
        except ValueError:
            try:
                parsed = date.fromisoformat(candidate)
            except ValueError:
                return False
        return parsed.isoformat() == expected

    def evaluate(self, response_text):
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        if not text:
            return rubric.results()
        validation = parse_tool_calls(text)
        rubric.record_validation(validation)
        calls = validation.value if isinstance(validation.value, list) else []
        names = [call.get("name") for call in calls if isinstance(call, dict)]
        expected = ["get_weather", "search_flights", "book_hotel", "get_stock_price", "convert_currency", "send_email"]
        blocks = re.findall(r"<tool_call>.*?</tool_call>", text, re.IGNORECASE | re.DOTALL)
        exact_format = bool(blocks) and validation.valid and len(blocks) == len(calls)
        rubric.add_criterion("Output format compliance", 3.0, 3.0 if exact_format else (1.0 if blocks else 0.0), negative_findings=[] if exact_format else [{"finding": "all tool calls must be valid typed JSON blocks"}])
        plan_end = text.lower().find("<tool_call>")
        plan = text[:plan_end] if plan_end >= 0 else ""
        plan_match = re.fullmatch(r"\s*<plan>\s*([\s\S]*?)\s*</plan>\s*", plan, re.IGNORECASE)
        plan_body = plan_match.group(1).lower() if plan_match else ""
        plan_ok = bool(plan_match) and all(name in plan_body for name in expected)
        rubric.add_criterion("Planning / reasoning", 2.0, 2.0 if plan_ok else 0.0)
        counts_ok = names == expected and len(names) == len(set(names))
        distinct = len(set(names) & set(expected))
        rubric.add_criterion("Required tools present", 5.0, 5.0 if counts_ok else 5.0 * distinct / len(expected), negative_findings=[] if counts_ok else [{"finding": "exactly one call for each required tool is required"}])
        args = [call.get("args", {}) for call in calls if isinstance(call, dict)]
        checks = [
            ("get_weather", lambda a: str(a.get("location", "")).lower() == "tokyo" and str(a.get("unit", "")).lower() in {"celsius", "c"}, 1.0),
            ("search_flights", lambda a: str(a.get("origin", "")).upper() == "JFK" and str(a.get("destination", "")).lower() == "tokyo" and self._date_matches(a.get("date"), "2024-08-15"), 1.5),
            ("book_hotel", lambda a: str(a.get("city", "")).lower() == "tokyo" and self._date_matches(a.get("check_in"), "2024-08-16") and self._date_matches(a.get("check_out"), "2024-08-20") and a.get("guests") == 2, 1.5),
            ("get_stock_price", lambda a: str(a.get("ticker", "")).upper() == "SONY", 1.0),
            ("convert_currency", lambda a: a.get("amount") == 1000 and str(a.get("from_curr")).upper() == "USD" and str(a.get("to_curr")).upper() == "JPY", 1.0),
            ("send_email", lambda a: str(a.get("to", "")).lower() == "alice@example.com" and "tokyo trip itinerary" in str(a.get("subject", "")).lower() and bool(a.get("body")), 2.0),
        ]
        arg_score = sum(weight for name, predicate, weight in checks for call_name, call_args in zip(names, args, strict=False) if call_name == name and isinstance(call_args, dict) and predicate(call_args))
        rubric.add_criterion("Correct arguments", 8.0, arg_score)
        rubric.add_criterion("Correct ordering / dependencies", 3.0, 3.0 if names == expected else 0.0)
        final = text[text.rfind("</tool_call>") + len("</tool_call>"):] if "</tool_call>" in text else ""
        synthesis_hits = sum(bool(re.search(pattern, final, re.IGNORECASE)) for pattern in (r"weather|celsius|degree", r"flight|JFK|Tokyo", r"hotel|reservation|guest", r"stock|SONY|price", r"\b\d+(?:\.\d+)?\s*JPY\b", r"email|itinerary|alice@example\.com"))
        rubric.add_criterion("Synthesis / final response", 4.0, 4.0 if synthesis_hits == 6 else synthesis_hits * 2.0 / 3.0, negative_findings=[] if synthesis_hits == 6 else [{"finding": "final response must include all results and a numeric JPY amount"}])
        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
