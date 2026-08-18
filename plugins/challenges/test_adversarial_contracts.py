"""Adversarial scoring regressions for every challenge family."""
import json

import pytest

from plugins.challenges.code_review import CodeReviewPlugin
from plugins.challenges.data_transformation import (
    DATA_TRANSFORMATION_EXPECTED_OUTPUT,
    DataTransformationPlugin,
)
from plugins.challenges.debug_consistency import DebugConsistencyPlugin
from plugins.challenges.debug_traversal import DebugTraversalPlugin
from plugins.challenges.error_recovery import ErrorRecoveryPlugin
from plugins.challenges.event_processor import EventProcessorPlugin
from plugins.challenges.instruction_following import InstructionFollowingPlugin
from plugins.challenges.long_context import LongContextPlugin
from plugins.challenges.moe_dense import MoEDensePlugin
from plugins.challenges.multi_step import MultiStepPlugin
from plugins.challenges.multi_turn_conversation import MultiTurnConversationPlugin
from plugins.challenges.orchestration import OrchestrationPlugin
from plugins.challenges.prd_creation import PRDCreationPlugin
from plugins.challenges.rate_limiter import RateLimiterPlugin
from plugins.challenges.reasoning import ReasoningPlugin
from plugins.challenges.software_architecture import SoftwareArchitecturePlugin
from plugins.challenges.tool_calling import ToolCallingPlugin
from plugins.challenges.wireframes import WireframesPlugin


def test_code_review_cannot_reuse_one_finding_for_every_defect():
    response = '{"issues":[{"description":"the file handle leaks; use a context manager"}]}'
    result = CodeReviewPlugin().evaluate(response)
    assert sum(item["earned"] for item in result.rubric) < 8.0


def test_debug_consistency_rejects_a_patch_for_a_reproducible_report():
    response = """## Reproduction
The output is ['abc'].
## Consistency Check
The report is reproducible.
## Diagnosis
There is a bug.
## Evidence Needed
Collect logs.
## Recommendation
Patch the comparison.
"""
    assert DebugConsistencyPlugin().score(response) < 15.0


def test_debug_traversal_requires_executable_threshold_fix():
    response = """## Root Cause
The threshold should be at least two.
## Analysis
abc123 has count 2.
## Fix
```python
def find_duplicate_users(log_entries):
    return []
```
## Test
pytest assert abc123
## Side Effects
Ordering and empty IDs should be considered.
"""
    assert DebugTraversalPlugin().score(response) < 15.0


def test_instruction_following_wrong_tie_break_does_not_pass():
    response = """ORDER T-05 | CUSTOMER NOOR | TOTAL 120.00
ORDER T-02 | CUSTOMER JULES | TOTAL 120.00
ORDER T-08 | CUSTOMER ZARA | TOTAL 99.90
ORDER T-09 | CUSTOMER RAVI | TOTAL 65.00
[SUMMARY] count=4; total=404.90; top_order=T-02"""
    assert InstructionFollowingPlugin().score(response) < InstructionFollowingPlugin().max_score


def test_reasoning_rejects_the_old_p4_answer():
    response = """1. The time chain places Search at 09:30.
2. Ben owns Search.
3. Upload outranks Search, which outranks Billing.
FAILED_SERVICE: Search
OWNER: Ben
PRIORITY: P4
TIME: 09:30"""
    assert ReasoningPlugin().score(response) < ReasoningPlugin().max_score


def test_long_context_requires_the_joined_evidence_chain():
    response = "INCIDENT: I-17\nOWNER: Omar\nESCALATION CHANNEL: PagerDuty\nEVIDENCE: F02\nREASONING: I guessed this."
    assert LongContextPlugin().score(response) < 15.0


def test_moe_document_keywords_without_local_sections_score_low():
    response = "MoE and dense models use top-k softmax gating, load balancing equations, training, inference, benchmarks, and references."
    assert MoEDensePlugin().score(response) < 10.0


def test_multi_step_requires_one_function_per_block():
    response = """```python
def greet_user(name: str) -> str: return f'Hello, {name}! Welcome.'
def validate_name(name: str) -> bool: return True
```
```python
def format_greeting(greeting: str, times: int) -> str: return greeting
```
```python
x = 1
```
[SUMMARY: 3 functions, 3 code blocks, completed all steps]."""
    assert MultiStepPlugin().score(response) < 18.0


def test_multi_turn_rejects_prose_outside_revision_blocks():
    response = """## Version 1 (Initial)
Prose before block.
```
Dear team, I decline the offer.
```
## Version 2 (After Feedback 1)
```
Dear product team, I enjoyed the interview.
```
## Version 3 (After Feedback 2)
```
Dear team, with warmth I hope to follow your future work.
```
## Summary of Changes
V1 to V2 personalized the team; V2 to V3 warmed the tone and mentioned future work.
"""
    assert MultiTurnConversationPlugin().score(response) < MultiTurnConversationPlugin().max_score


def test_orchestration_does_not_count_dependency_mentions_as_task_work():
    response = """Task 1 [DEPENDS_ON: task 2]
Task 2 [DEPENDS_ON: task 3]
Task 3 [DEPENDS_ON: task 4]
Task 4 [DEPENDS_ON: task 1]
Task 1 init running complete."""
    assert OrchestrationPlugin().score(response) < 10.0


def test_prd_content_in_wrong_heading_does_not_earn_local_credit():
    response = """## Notes
Executive Summary FlowState. Problem pain. Goals 25%. Persona 1 and Persona 2.
As a developer, I want focus, so that I can work.
FR-1 calendar FR-2 music FR-3 AI FR-4 blocks FR-5 metrics.
Performance security reliability scalability. Todoist Notion. Q1 MVP. Risk?
"""
    assert PRDCreationPlugin().score(response) < 8.0


def test_rate_limiter_behavior_rejects_always_allowing_implementations():
    response = """```python
class TokenBucket:
    def __init__(self, limit, window_seconds): pass
    def allow_request(self, client_id, now): return True
    def get_usage_stats(self, client_id): return {}
    def cleanup(self, now): return 0
class SlidingWindowLog(TokenBucket): pass
class FixedWindow(TokenBucket): pass
```"""
    assert RateLimiterPlugin().score(response) < 15.0


def test_architecture_keywords_without_required_sections_score_low():
    response = "microservices API gateway PostgreSQL Redis OAuth2 Kubernetes 1M DAU circuit breaker 99.9%."
    assert SoftwareArchitecturePlugin().score(response) < 8.0


def test_data_transformation_rejects_multiple_candidates():
    payload = DATA_TRANSFORMATION_EXPECTED_OUTPUT
    result = DataTransformationPlugin().evaluate(
        "```json\n" + json.dumps(payload) + "\n```\n```json\n" + json.dumps(payload) + "\n```"
    )
    assert result.score == 0.0
    assert any("multiple structured candidates" in error for error in result.diagnostics["errors"])


def test_tool_calling_rejects_unknown_extra_tool():
    calls = [
            '<tool_call>{"name":"get_weather","args":{"location":"Tokyo","unit":"celsius"}}</tool_call>',
            '<tool_call>{"name":"search_flights","args":{"origin":"JFK","destination":"Tokyo","date":"2024-08-15"}}</tool_call>',
            '<tool_call>{"name":"book_hotel","args":{"city":"Tokyo","check_in":"2024-08-16","check_out":"2024-08-20","guests":2}}</tool_call>',
            '<tool_call>{"name":"get_stock_price","args":{"ticker":"SONY"}}</tool_call>',
            '<tool_call>{"name":"convert_currency","args":{"amount":1000,"from_curr":"USD","to_curr":"JPY"}}</tool_call>',
        '<tool_call>{"name":"unknown","args":{}}</tool_call>',
    ]
    response = "<plan>get_weather search_flights book_hotel get_stock_price convert_currency send_email</plan>\n" + "\n".join(calls)
    assert ToolCallingPlugin().score(response) < 18.0


def test_wireframes_require_distinct_canonical_screens():
    response = "## Focus\nPurpose: timer.\n[Button] Start\n## Focus Session\nPurpose: timer.\n[Button] Start\n## Calendar\nPurpose: events.\n[Button] Sync\n## Calendar Integration\nPurpose: events.\n[Button] Sync\n"
    result = WireframesPlugin().evaluate(response)
    screens = next(item for item in result.rubric if item["name"] == "Multiple screens present")
    assert screens["earned"] < screens["max"]


@pytest.mark.parametrize("plugin", [ErrorRecoveryPlugin, EventProcessorPlugin])
def test_executable_plugins_do_not_credit_stub_sources(plugin):
    assert plugin().score("class Placeholder:\n    pass") < 12.0
