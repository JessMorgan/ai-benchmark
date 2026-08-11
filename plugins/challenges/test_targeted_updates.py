"""Regression tests for targeted false-negative fixes."""
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
        score = CodeReviewPlugin().score(response)
        self.assertGreaterEqual(score, 6.0)

    def test_workflow_parser_accepts_mermaid_and_plain_dependencies(self):
        mermaid = "Task 1 --> Task 2\nTask 2 --> Task 3"
        prose = "Step 1 is the source. Step 2 depends on Step 1. Step 3 depends on Step 2."
        self.assertTrue(parse_workflow_graph(mermaid).valid)
        self.assertTrue(parse_workflow_graph(prose).valid)

    def test_workflow_bracket_dependencies_bind_to_local_task(self):
        parsed = parse_workflow_graph(
            "Task 1 [DEPENDS_ON: task 0]\n"
            "Task 2 [DEPENDS_ON: task 1]\n"
            "Task 3 [DEPENDS_ON: task 1]"
        )
        self.assertEqual(parsed.value["edges"], [("1", "0"), ("2", "1"), ("3", "1")])

    def test_orchestration_accepts_graph_without_bracket_tags(self):
        response = """## Plan
Task 1 [PARALLEL] process logs.
Task 2 [PARALLEL] perform GeoIP lookup.
Task 3 [SEQUENTIAL] generate the PDF report.
Task 1 --> Task 3
Task 2 --> Task 3
## Trace
Task 1 running; Task 2 running; Task 3 complete.
"""
        self.assertGreaterEqual(OrchestrationPlugin().score(response), 12.0)

    def test_multi_step_accepts_equivalent_alpha_validation(self):
        response = '''```python
def greet_user(name):
    return "Hello, " + name + "! Welcome."
```
```python
def validate_name(name):
    return bool(name.strip()) and all(part.isalpha() for part in name.split()) and len(name) < 51
```
```python
def format_greeting(greeting, times):
    return "\\n".join(greeting for _ in range(max(0, times)))
```
[SUMMARY: 3 lines, 3 functions, completed all steps].'''
        self.assertGreaterEqual(MultiStepPlugin().score(response), 15.0)

    def test_tool_calling_rejects_wrong_trip_dates(self):
        response = '<tool_call>{"name":"search_flights","args":{"origin":"JFK","destination":"Tokyo","date":"2025-01-01"}}</tool_call>'
        self.assertEqual(ToolCallingPlugin().evaluate(response).rubric[4]["earned"], 0.0)

    def test_tool_calling_accepts_iso_datetime_arguments(self):
        response = '''<plan>Call the tools in order.</plan>
<tool_call>{"name":"get_weather","args":{"location":"Tokyo","unit":"celsius"}}</tool_call>
<tool_call>{"name":"search_flights","args":{"origin":"JFK","destination":"Tokyo","date":"2024-08-15T00:00:00Z"}}</tool_call>
<tool_call>{"name":"book_hotel","args":{"city":"Tokyo","check_in":"2024-08-16T00:00:00Z","check_out":"2024-08-20","guests":2}}</tool_call>
<tool_call>{"name":"get_stock_price","args":{"ticker":"SONY"}}</tool_call>
<tool_call>{"name":"convert_currency","args":{"amount":1000,"from_curr":"USD","to_curr":"JPY"}}</tool_call>
<tool_call>{"name":"send_email","args":{"to":"alice@example.com","subject":"Tokyo Trip Itinerary","body":"Itinerary"}}</tool_call>
Weather, flight, hotel, stock, converted JPY itinerary emailed to Alice.'''
        self.assertGreaterEqual(ToolCallingPlugin().score(response), 22.0)

    def test_architecture_alias_headings_are_accepted(self):
        response = """## Executive Summary
A complete architecture for FlowState.
## Requirements Summary
Web, mobile, sync, and one million users.
## Architecture Style
A modular monolith with clear boundaries.
## System Components
Auth and planning services.
## Data Design
Postgres and Redis store users, sessions, schedules, and calendar events with durable relationships.
## API Design
REST endpoints.
## Technology Stack
Python and FastAPI.
## Infrastructure
Containers and CI/CD.
## Threat Model
OAuth2 and encryption.
## Capacity Planning
Autoscaling and caching.
## Architectural Decisions
Trade-offs are documented.
"""
        validation = validate_sections(
            response,
            ["Component", "Data Model", "Deployment", "Security", "Scalability", "Trade-offs"],
            aliases={
                "Component": ("System Components",),
                "Data Model": ("Data Design",),
                "Deployment": ("Infrastructure",),
                "Security": ("Threat Model",),
                "Scalability": ("Capacity Planning",),
                "Trade-offs": ("Architectural Decisions",),
            },
        )
        self.assertTrue(validation.valid)
        self.assertNotIn("Data Model", " ".join(validation.errors))
        self.assertGreater(SoftwareArchitecturePlugin().score(response), 0.0)


if __name__ == "__main__":
    unittest.main()
