"""Targeted regressions for structured scoring behavior."""
import unittest

from plugins.challenges._validators import parse_workflow_graph, validate_sections
from plugins.challenges.code_review import CodeReviewPlugin
from plugins.challenges.multi_step import MultiStepPlugin
from plugins.challenges.orchestration import OrchestrationPlugin
from plugins.challenges.software_architecture import SoftwareArchitecturePlugin
from plugins.challenges.tool_calling import ToolCallingPlugin


class TestTargetedUpdates(unittest.TestCase):
    def test_code_review_accepts_alternative_issue_fields_and_language(self):
        response = '{"issues": [{"finding": "The file handle f leaks; use finally to close it."}, {"issue": "The fetch_data external call needs defensive failure handling."}]}'
        self.assertGreater(CodeReviewPlugin().score(response), 0.0)

    def test_workflow_parser_accepts_mermaid_and_plain_dependencies(self):
        self.assertTrue(parse_workflow_graph("Task 1 --> Task 2\nTask 2 --> Task 3").valid)
        self.assertTrue(parse_workflow_graph("Step 1 is the source. Step 2 depends on Step 1. Step 3 depends on Step 2.").valid)

    def test_workflow_bracket_dependencies_bind_to_local_task(self):
        parsed = parse_workflow_graph("Task 1 [DEPENDS_ON: task 0]\nTask 2 [DEPENDS_ON: task 1]\nTask 3 [DEPENDS_ON: task 1]")
        self.assertEqual(parsed.value["edges"], [("1", "0"), ("2", "1"), ("3", "1")])

    def test_orchestration_requires_all_tasks_in_trace(self):
        response = """## Plan
Task 1 [PARALLEL] process logs.
Task 2 [PARALLEL] perform GeoIP lookup.
Task 3 [SEQUENTIAL] anomaly detection.
Task 4 [SEQUENTIAL] generate the PDF report.
Task 2 [DEPENDS_ON: task 1]
Task 3 [DEPENDS_ON: task 2]
Task 4 [DEPENDS_ON: task 3]
## Trace
Task 1 init running complete.
Task 2 init running complete.
Task 3 init running complete.
Task 4 init running complete.
"""
        self.assertEqual(OrchestrationPlugin().score(response), 16.0)

    def test_multi_step_behavioral_contract(self):
        response = '''```python
def greet_user(name: str) -> str:
    return f"Hello, {name}! Welcome."
```
```python
def validate_name(name: str) -> bool:
    return bool(name.strip()) and all(part.isalpha() for part in name.split()) and len(name) <= 50
```
```python
def format_greeting(greeting: str, times: int) -> str:
    return "\\n".join(greeting for _ in range(max(0, times)))
```
[SUMMARY: 3 functions, 3 code blocks, completed all steps].'''
        self.assertGreaterEqual(MultiStepPlugin().score(response), 15.0)

    def test_tool_calling_rejects_wrong_trip_dates(self):
        response = '<tool_call>{"name":"search_flights","args":{"origin":"JFK","destination":"Tokyo","date":"2025-01-01"}}</tool_call>'
        self.assertEqual(ToolCallingPlugin().evaluate(response).rubric[3]["earned"], 0.0)

    def test_tool_calling_accepts_iso_datetime_arguments(self):
        response = '''<plan>get_weather search_flights book_hotel get_stock_price convert_currency send_email</plan>
<tool_call>{"name":"get_weather","args":{"location":"Tokyo","unit":"celsius"}}</tool_call>
<tool_call>{"name":"search_flights","args":{"origin":"JFK","destination":"Tokyo","date":"2024-08-15T00:00:00Z"}}</tool_call>
<tool_call>{"name":"book_hotel","args":{"city":"Tokyo","check_in":"2024-08-16T00:00:00Z","check_out":"2024-08-20","guests":2}}</tool_call>
<tool_call>{"name":"get_stock_price","args":{"ticker":"SONY"}}</tool_call>
<tool_call>{"name":"convert_currency","args":{"amount":1000,"from_curr":"USD","to_curr":"JPY"}}</tool_call>
<tool_call>{"name":"send_email","args":{"to":"alice@example.com","subject":"Tokyo Trip Itinerary","body":"Itinerary"}}</tool_call>
Weather in Tokyo, flight JFK to Tokyo, hotel for 2 guests, SONY stock, 150000 JPY converted, email itinerary to alice@example.com.'''
        self.assertGreaterEqual(ToolCallingPlugin().score(response), 22.0)

    def test_section_alias_validator_remains_supported(self):
        response = "## Data Design\nPostgres stores users and sessions with replicas, cache, and durable relationships.\n## Threat Model\nOAuth2 authentication, authorization, encryption at rest, and protection of private data."
        validation = validate_sections(response, ["Data Model", "Security"], aliases={"Data Model": ("Data Design",), "Security": ("Threat Model",)})
        self.assertTrue(validation.valid)
        self.assertGreater(SoftwareArchitecturePlugin().score(response), 0.0)


if __name__ == "__main__":
    unittest.main()
