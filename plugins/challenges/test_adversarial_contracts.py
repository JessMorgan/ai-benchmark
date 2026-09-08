"""Adversarial scoring regressions for every challenge family."""
import json
import re

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


def test_code_review_denying_every_defect_scores_low():
    # Measured pre-fix: this all-denial response scored 15/15 because keyword
    # co-occurrence cannot see negation.
    response = json.dumps({"issues": [
        {"description": "open(db_path) is fine; the file handle is closed properly, no leak"},
        {"description": "user_id == None is fine; the comparison works correctly"},
        {"description": "the /tmp/data.txt path is fine and acceptable"},
        {"description": "fetch_data never raises; no exception handling is needed"},
        {"description": "the os and time imports are used; no unused imports"},
    ]})
    result = CodeReviewPlugin().evaluate(response)
    defect_names = {
        "File handle not closed / resource leak",
        "== None instead of is None",
        "Hardcoded /tmp path",
        "Missing error handling / fetch_data may fail",
        "Unused imports",
    }
    assert sum(item["earned"] for item in result.rubric if item["name"] in defect_names) == 0.0
    assert result.score < 5.0


def test_code_review_one_stuffed_line_cannot_score_full():
    # Measured pre-fix: this single keyword-stuffed line scored 15/15 because
    # every defect check reused the same finding and the citations floor
    # degraded to 1. Now one finding satisfies at most one defect and the
    # citations point needs >=3 distinct findings.
    response = json.dumps({"issues": [{
        "description": "open(db_path) is not closed so it leaks; user_id == None should use is None; "
        "the /tmp/data.txt path should be a parameter; fetch_data may raise an exception so add "
        "try/except; the os and time imports are unused, remove them",
    }]})
    result = CodeReviewPlugin().evaluate(response)
    defect_names = {
        "File handle not closed / resource leak",
        "== None instead of is None",
        "Hardcoded /tmp path",
        "Missing error handling / fetch_data may fail",
        "Unused imports",
    }
    defect_earned = [item for item in result.rubric if item["name"] in defect_names and item["earned"] > 0]
    assert len(defect_earned) <= 1
    citations = next(item for item in result.rubric if item["name"] == "Source citations")
    assert citations["earned"] == 0.0
    assert result.score < 8.0


def test_code_review_json_with_unrecognized_keys_falls_back_to_bullets():
    # Measured pre-fix (ornith-nas): valid JSON with unrecognized keys
    # dead-ended at 0/15 while judges said ~90 — the bullet fallback never
    # ran because the JSON branch returned an empty finding list.
    response = (
        "```json\n"
        '{"problems": ["the open(db_path) handle is never closed; use a context manager"]}\n'
        "```\n"
        "- open(db_path) is never closed; use a context manager to avoid the leak\n"
        "- user_id == None should be user_id is None\n"
        "- the /tmp/data.txt path is hardcoded; parameterize db_path\n"
        "- fetch_data may raise; wrap in try/except\n"
        "- the os and time imports are unused; remove them\n"
    )
    result = CodeReviewPlugin().evaluate(response)
    assert result.score >= 13.0
    assert any("format contract" in error for error in result.diagnostics["errors"])


def test_code_review_json_without_recognized_findings_names_the_contract():
    result = CodeReviewPlugin().evaluate('{"problems": ["nothing to see here"]}')
    assert result.score == 0.0
    assert any("format contract" in error for error in result.diagnostics["errors"])


def test_code_review_stray_brace_prose_with_fenced_json_scores():
    # Measured pre-fix: a stray brace in the prose before the fenced JSON
    # made the find("{")..rfind("}") slice span the brace to the JSON's last
    # brace; the slice failed to parse and the response scored 0.
    response = (
        "Here is my review (note: the {brace} in the prompt is a red herring):\n\n"
        "```json\n"
        '{"issues": [\n'
        '  {"description": "open(db_path) is never closed; use a context manager to avoid the leak"},\n'
        '  {"description": "user_id == None should be user_id is None"},\n'
        '  {"description": "the /tmp/data.txt path is hardcoded; parameterize db_path"},\n'
        '  {"description": "fetch_data may raise an exception; wrap it in try/except"},\n'
        '  {"description": "the os and time imports are unused; remove them"}\n'
        "]}\n"
        "```"
    )
    result = CodeReviewPlugin().evaluate(response)
    assert result.score >= 13.0


def test_code_review_non_dict_issue_items_do_not_crash():
    # The candidate is parsed by parse_structured, so the old try/except no
    # longer guards item access: non-dict issue entries must be skipped
    # (dead-end + contract finding), not crash evaluate().
    result = CodeReviewPlugin().evaluate('{"issues": ["a plain string finding", 42]}')
    assert result.score == 0.0
    assert any("format contract" in error for error in result.diagnostics["errors"])


def test_code_review_accepts_unicode_bullets_and_top_level_json_array():
    # Measured pre-fix (CR-5): a •-bulleted review and a top-level JSON
    # array of issue objects both scored 0/15 — the bullet pattern only
    # matched -/* and numbered markers, and only a JSON object was
    # recognized as a structured candidate.
    bullet_response = (
        "• open(db_path) is never closed; use a context manager to avoid the leak\n"
        "• user_id == None should be user_id is None\n"
        "• the /tmp/data.txt path is hardcoded; parameterize db_path\n"
        "• fetch_data may raise an exception; wrap it in try/except\n"
        "• the os and time imports are unused; remove them\n"
    )
    assert CodeReviewPlugin().score(bullet_response) >= 13.0
    array_response = json.dumps([
        {"description": "open(db_path) is never closed; use a context manager to avoid the leak"},
        {"description": "user_id == None should be user_id is None"},
        {"description": "the /tmp/data.txt path is hardcoded; parameterize db_path"},
        {"description": "fetch_data may raise an exception; wrap it in try/except"},
        {"description": "the os and time imports are unused; remove them"},
    ])
    assert CodeReviewPlugin().score(array_response) >= 13.0


def test_code_review_top_level_array_has_no_spurious_not_an_object_error():
    # Regression (CR-5): a top-level JSON array is an accepted issue-list
    # form and scores, but parse_structured reports it as "structured
    # candidate is not an object". That spurious error must not surface in
    # the diagnostics of a response the plugin accepted and scored.
    array_response = json.dumps([
        {"description": "open(db_path) is never closed; use a context manager to avoid the leak"},
        {"description": "user_id == None should be user_id is None"},
        {"description": "the /tmp/data.txt path is hardcoded; parameterize db_path"},
        {"description": "fetch_data may raise an exception; wrap it in try/except"},
        {"description": "the os and time imports are unused; remove them"},
    ])
    result = CodeReviewPlugin().evaluate(array_response)
    assert result.score >= 13.0
    assert not any(
        "structured candidate is not an object" in error
        for error in result.diagnostics["errors"]
    )


def test_code_review_short_keywords_require_word_boundaries():
    # Measured pre-fix (CR-6): the boundary-less use/os/time keywords
    # matched inside longer words — "closed" satisfied the unused-imports
    # defect via "os" (2.0 pre-fix), and "user"/"uses" satisfied the "use"
    # remediation term (1.0 pre-fix).
    plugin = CodeReviewPlugin()
    in_word_os = '{"issues": [{"description": "the unused variable is closed without cleanup"}]}'
    result = plugin.evaluate(in_word_os)
    unused = next(item for item in result.rubric if item["name"] == "Unused imports")
    assert unused["earned"] == 0.0
    in_word_use = '{"issues": [{"description": "the user_id check uses == None"}]}'
    result = plugin.evaluate(in_word_use)
    actionable = next(item for item in result.rubric if item["name"] == "Actionable / concrete fixes")
    assert actionable["earned"] == 0.0
    # Legitimate standalone keywords must still be credited.
    legitimate = '{"issues": [{"description": "the os and time imports are unused; remove them"}]}'
    result = plugin.evaluate(legitimate)
    unused = next(item for item in result.rubric if item["name"] == "Unused imports")
    assert unused["earned"] == 2.0


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


def test_debug_traversal_prose_only_cannot_earn_lexical_criteria():
    # Measured pre-fix (DT-1): a response with fully correct prose but a
    # corrected code block that fails the harness scored 17/20 because the
    # lexical fix/diagnosis/trace criteria were earned from prose alone. After
    # the fix, the execution gate scales those three criteria to 0 and emits a
    # negative finding, so prose-only scores far lower.
    response = (
        "## Root Cause\n"
        "The comparison `count > 2` is a strict inequality; it should be `count >= 2`.\n"
        "## Analysis\n"
        "For abc123, count is 2. Since 2 > 2 is False, abc123 is not added and the "
        "function returns an empty list. def456 has count 1.\n"
        "## Fix\n"
        "Change the comparison to `count >= 2`:\n"
        "```python\n"
        "def find_duplicate_users(log_entries):\n"
        "    user_counts = {}\n"
        "    for entry in log_entries:\n"
        "        user_id = entry.get('user_id')\n"
        "        if user_id:\n"
        "            user_counts[user_id] = user_counts.get(user_id, 0) + 1\n"
        "    result = []\n"
        "    for user_id, count in user_counts.items():\n"
        "        if count > 2:\n"
        "            result.append(user_id)\n"
        "    return result\n"
        "```\n"
        "## Test\n"
        "```python\n"
        "def test_find_duplicate_users():\n"
        "    assert find_duplicate_users([{'user_id': 'abc123'}, {'user_id': 'abc123'}, {'user_id': 'def456'}]) == ['abc123']\n"
        "```\n"
        "## Side Effects\n"
        "Ordering is preserved; empty IDs are skipped; duplicates are counted.\n"
    )
    result = DebugTraversalPlugin().evaluate(response)
    assert result.score < 17.0
    for name in (
        "Systematic trace / code walkthrough",
        "Depth of analysis",
        "Proposed fix / corrected code",
    ):
        item = next(item for item in result.rubric if item["name"] == name)
        assert item["earned"] == 0.0, f"{name} should be scaled to 0, got {item['earned']}"
        assert item["negative_findings"], f"{name} should carry a negative finding"


def test_debug_traversal_correct_executable_fix_scores_full():
    # Positive control: a response with correct prose AND a corrected code
    # block that passes the harness must still earn the full 20/20 (the
    # execution gate must not penalize a valid fix).
    response = (
        "## Root Cause\n"
        "The comparison `count > 2` is a strict inequality; it should be `count >= 2`.\n"
        "## Analysis\n"
        "For abc123, count is 2. Since 2 > 2 is False, abc123 is not added and the "
        "function returns an empty list. def456 has count 1.\n"
        "## Fix\n"
        "Change the comparison to `count >= 2`:\n"
        "```python\n"
        "def find_duplicate_users(log_entries):\n"
        "    user_counts = {}\n"
        "    for entry in log_entries:\n"
        "        user_id = entry.get('user_id')\n"
        "        if user_id:\n"
        "            user_counts[user_id] = user_counts.get(user_id, 0) + 1\n"
        "    result = []\n"
        "    for user_id, count in user_counts.items():\n"
        "        if count >= 2:\n"
        "            result.append(user_id)\n"
        "    return result\n"
        "```\n"
        "## Test\n"
        "```python\n"
        "def test_find_duplicate_users():\n"
        "    assert find_duplicate_users([{'user_id': 'abc123'}, {'user_id': 'abc123'}, {'user_id': 'def456'}]) == ['abc123']\n"
        "```\n"
        "## Side Effects\n"
        "Ordering is preserved; empty IDs are skipped; duplicates are counted.\n"
    )
    assert DebugTraversalPlugin().score(response) == 20.0


def test_debug_traversal_inverted_diagnosis_cannot_earn_depth():
    # Measured pre-fix (DT-2): an inverted diagnosis that identifies the buggy
    # comparison (`count > 2`) but never states the corrective remedy earned
    # full depth credit (17/20). After the fix, full depth credit requires a
    # corrective statement (should/must + the correct comparison), so the
    # depth criterion is 0.0 here even though the fix code passes the harness.
    response = (
        "## Root Cause\n"
        "The comparison `count > 2` is the problem; it filters out users with two entries.\n"
        "## Analysis\n"
        "For abc123, count is 2. Since 2 > 2 is False, abc123 is not added and the "
        "function returns an empty list. def456 has count 1.\n"
        "## Fix\n"
        "```python\n"
        "def find_duplicate_users(log_entries):\n"
        "    user_counts = {}\n"
        "    for entry in log_entries:\n"
        "        user_id = entry.get('user_id')\n"
        "        if user_id:\n"
        "            user_counts[user_id] = user_counts.get(user_id, 0) + 1\n"
        "    result = []\n"
        "    for user_id, count in user_counts.items():\n"
        "        if count >= 2:\n"
        "            result.append(user_id)\n"
        "    return result\n"
        "```\n"
        "## Test\n"
        "```python\n"
        "def test_find_duplicate_users():\n"
        "    assert find_duplicate_users([{'user_id': 'abc123'}, {'user_id': 'abc123'}, {'user_id': 'def456'}]) == ['abc123']\n"
        "```\n"
        "## Side Effects\n"
        "Ordering is preserved; empty IDs are skipped; duplicates are counted.\n"
    )
    result = DebugTraversalPlugin().evaluate(response)
    depth = next(item for item in result.rubric if item["name"] == "Depth of analysis")
    assert depth["earned"] == 0.0
    assert result.score < 20.0


def test_debug_traversal_fix_accepts_equivalent_count_gt_one():
    # DT-3: `count > 1` is the equivalent corrected form (for integer counts it
    # admits exactly count >= 2) and the execution harness already credits it;
    # the lexical fix criterion must accept it too, not only `>= 2`.
    response = (
        "## Root Cause\n"
        "The comparison `count > 2` is too strict; it should be `count > 1`.\n"
        "## Analysis\n"
        "For abc123, count is 2. Since 2 > 2 is False, abc123 is not added and the "
        "function returns an empty list. def456 has count 1.\n"
        "## Fix\n"
        "```python\n"
        "def find_duplicate_users(log_entries):\n"
        "    user_counts = {}\n"
        "    for entry in log_entries:\n"
        "        user_id = entry.get('user_id')\n"
        "        if user_id:\n"
        "            user_counts[user_id] = user_counts.get(user_id, 0) + 1\n"
        "    result = []\n"
        "    for user_id, count in user_counts.items():\n"
        "        if count > 1:\n"
        "            result.append(user_id)\n"
        "    return result\n"
        "```\n"
        "## Test\n"
        "```python\n"
        "def test_find_duplicate_users():\n"
        "    assert find_duplicate_users([{'user_id': 'abc123'}, {'user_id': 'abc123'}, {'user_id': 'def456'}]) == ['abc123']\n"
        "```\n"
        "## Side Effects\n"
        "Ordering is preserved; empty IDs are skipped; duplicates are counted.\n"
    )
    result = DebugTraversalPlugin().evaluate(response)
    fix = next(item for item in result.rubric if item["name"] == "Proposed fix / corrected code")
    assert fix["earned"] == 3.0
    assert result.score == 20.0


def test_debug_traversal_test_in_requires_assertion_context():
    # DT-4: the `in` membership check must sit in a test-assertion context
    # (assert x in y), not any "in " fragment. A loose "abc123 in the list"
    # (e.g. in a comment) must not satisfy the test criterion, while a proper
    # `assert 'abc123' in result` must.
    base_fix = (
        "## Fix\n"
        "```python\n"
        "def find_duplicate_users(log_entries):\n"
        "    user_counts = {}\n"
        "    for entry in log_entries:\n"
        "        user_id = entry.get('user_id')\n"
        "        if user_id:\n"
        "            user_counts[user_id] = user_counts.get(user_id, 0) + 1\n"
        "    return [u for u, c in user_counts.items() if c >= 2]\n"
        "```\n"
    )
    preamble = (
        "## Root Cause\nThe comparison `count > 2` should be `count >= 2`.\n"
        "## Analysis\nFor abc123, count is 2 and the function returns an empty list. def456 has count 1.\n"
    )
    suffix = "## Side Effects\nOrdering and empty IDs are considered; duplicates are counted.\n"
    loose = preamble + base_fix + (
        "## Test\ndef test_check():\n"
        "    # abc123 in the list should be returned\n"
        "    assert True\n"
        + suffix
    )
    result = DebugTraversalPlugin().evaluate(loose)
    test_item = next(item for item in result.rubric if item["name"] == "Test code provided")
    assert test_item["earned"] == 0.0
    proper = preamble + base_fix + (
        "## Test\ndef test_check():\n"
        "    assert 'abc123' in find_duplicate_users(logs)\n"
        + suffix
    )
    result2 = DebugTraversalPlugin().evaluate(proper)
    test_item2 = next(item for item in result2.rubric if item["name"] == "Test code provided")
    assert test_item2["earned"] == 3.0


def test_debug_traversal_trace_return_requires_count_two_coref():
    # DT-5: the trace's empty/return hit must co-reference the specific count=2
    # value, not any "returns" mention. A "returns" far from "count is 2"
    # earns only 3 of 4 trace hits (2.2 after 1-decimal rounding); a
    # co-referenced one earns 4 of 4 (3.0).
    def make(analysis: str) -> str:
        return (
            "## Root Cause\nThe comparison `count > 2` should be `count >= 2`.\n"
            f"## Analysis\n{analysis}\n"
            "## Fix\n```python\ndef find_duplicate_users(log_entries):\n"
            "    user_counts = {}\n    for entry in log_entries:\n"
            "        user_id = entry.get('user_id')\n        if user_id:\n"
            "            user_counts[user_id] = user_counts.get(user_id, 0) + 1\n"
            "    return [u for u, c in user_counts.items() if c >= 2]\n```\n"
            "## Test\n```python\ndef test_check():\n"
            "    assert 'abc123' in find_duplicate_users(logs)\n```\n"
            "## Side Effects\nOrdering and empty IDs are considered; duplicates are counted.\n"
        )

    def trace_earned(response: str) -> float:
        result = DebugTraversalPlugin().evaluate(response)
        return next(
            item for item in result.rubric
            if item["name"] == "Systematic trace / code walkthrough"
        )["earned"]

    no_coref = make(
        "abc123 has count is 2. def456 is present. "
        + "x" * 90
        + " The function returns an empty list."
    )
    assert trace_earned(no_coref) == 2.2
    coref = make(
        "abc123 has count is 2 and the function returns an empty list. def456 is present."
    )
    assert trace_earned(coref) == 3.0


def test_debug_traversal_structure_is_linear_not_clamped():
    # DT-6: the structure criterion is linear (0.4 per section) with full
    # credit (2.0) requiring >=4 of 5 sections. The old float(hits) clamped at
    # 2.0, so any 2+ sections earned full credit.
    def structure_earned(response: str) -> float:
        result = DebugTraversalPlugin().evaluate(response)
        return next(
            item for item in result.rubric
            if item["name"] == "Structured RCA sections"
        )["earned"]

    fix_block = (
        "## Fix\n```python\ndef find_duplicate_users(log_entries):\n"
        "    user_counts = {}\n    for entry in log_entries:\n"
        "        user_id = entry.get('user_id')\n        if user_id:\n"
        "            user_counts[user_id] = user_counts.get(user_id, 0) + 1\n"
        "    return [u for u, c in user_counts.items() if c >= 2]\n```\n"
    )
    three = (
        "## Root Cause\nThe comparison `count > 2` should be `count >= 2`.\n"
        "## Analysis\nabc123 has count is 2 and the function returns an empty list. def456 is present.\n"
        + fix_block
    )
    assert structure_earned(three) == 1.2
    four = three + (
        "## Test\n```python\ndef test_check():\n"
        "    assert 'abc123' in find_duplicate_users(logs)\n```\n"
    )
    assert structure_earned(four) == 2.0


def test_debug_traversal_defective_comparison_is_word_bounded():
    # DT-7: the defective-comparison match is word-bounded, so "count > 20"
    # does not satisfy it (the bare `>\s*2` matched "> 2" as a prefix of
    # "> 20"), while "count > 2" and the correct comparison still do.
    from plugins.challenges.debug_traversal import _DEFECTIVE_COMPARISON_RE
    assert _DEFECTIVE_COMPARISON_RE.search("the comparison count > 2 is wrong")
    assert not _DEFECTIVE_COMPARISON_RE.search("the comparison count > 20 is wrong")
    assert _DEFECTIVE_COMPARISON_RE.search("it should be count >= 2")


def test_debug_traversal_fix_still_accepts_count_ge_two_after_dedup():
    # DT-8: the `count >= 2` form (previously a separate, subsumed alternative)
    # must still be accepted via the `>= 2` alternative after the dedup.
    result = DebugTraversalPlugin().evaluate(
        "## Root Cause\nThe comparison `count > 2` should be `count >= 2`.\n"
        "## Analysis\nFor abc123, count is 2 and the function returns an empty list. def456 is present.\n"
        "## Fix\n```python\ndef find_duplicate_users(log_entries):\n"
        "    user_counts = {}\n    for entry in log_entries:\n"
        "        user_id = entry.get('user_id')\n        if user_id:\n"
        "            user_counts[user_id] = user_counts.get(user_id, 0) + 1\n"
        "    result = []\n    for user_id, count in user_counts.items():\n"
        "        if count >= 2:\n            result.append(user_id)\n"
        "    return result\n```\n"
        "## Test\n```python\ndef test_check():\n"
        "    assert 'abc123' in find_duplicate_users(logs)\n```\n"
        "## Side Effects\nOrdering and empty IDs are considered; duplicates are counted.\n"
    )
    fix = next(item for item in result.rubric if item["name"] == "Proposed fix / corrected code")
    assert fix["earned"] == 3.0


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


def test_long_context_wrong_incident_cannot_earn_the_primary_criterion():
    # Measured before the fix: wrong incident (I-23) + magic tokens scored 19/20.
    # After the fix the incident is the primary 6.0 criterion and the
    # evidence/cross-ref/owner criteria are gated on it, so this scores far lower.
    response = (
        "INCIDENT: I-23\nOWNER: Omar\nESCALATION CHANNEL: PagerDuty\n"
        "EVIDENCE: F02 F05 F09\nREASONING: EU 14:30 P1 I-17 PagerDuty"
    )
    result = LongContextPlugin().evaluate(response)
    assert result.score < 10.0
    incident = next(item for item in result.rubric if item["name"] == "Incident correctness")
    assert incident["earned"] == 0.0


def test_long_context_prompt_shows_the_label_colon_output_shape():
    # Measured before the fix: a correct answer with headings-on-own-lines
    # scored 0/20 (harness parses `LABEL: value` lines the prompt never showed).
    prompt = LongContextPlugin().get_prompt()
    for label in ("INCIDENT", "OWNER", "ESCALATION CHANNEL", "EVIDENCE", "REASONING"):
        assert re.search(rf"^{re.escape(label)}: ", prompt, re.MULTILINE), label


def test_long_context_exact_answer_is_per_field_gated_not_all_on_reasoning():
    # Measured before the fix: one missing heading 20->7 (the outer
    # `if values["REASONING"] else 0.0` gate zeroed the entire Exact answer
    # criterion when REASONING was empty). After the fix, each sub-check is
    # gated on its own field, so a response with REASONING missing but
    # OWNER/ESCALATION present still earns the OWNER and ESCALATION
    # sub-checks (2.0 of 4.0).
    response = (
        "INCIDENT: I-17\n"
        "OWNER: Omar\n"
        "ESCALATION CHANNEL: PagerDuty\n"
        "EVIDENCE: F02 F05 F09\n"
    )
    result = LongContextPlugin().evaluate(response)
    exact = next(item for item in result.rubric if item["name"] == "Exact answer")
    # OWNER "omar" + ESCALATION "pagerduty" = 2.0 (REASONING sub-checks are
    # 0.0 because REASONING is empty, but they do not zero the other checks).
    assert exact["earned"] == 2.0, f"Expected 2.0 (per-field gating), got {exact['earned']}"


def test_long_context_cross_ref_is_negation_aware_for_incident_id():
    # Measured before the fix: 5/5 on "NOT I-17" (cross-ref earned 3.0 when
    # reasoning said "NOT I-17" because the bare `re.search(r"I-17", ...)`
    # matched the substring). After the fix, `_positive_ref` is negation-aware,
    # so the cross-ref earns 0.0 when the reasoning negates the incident ID.
    response = (
        "INCIDENT: I-17\n"
        "OWNER: Omar\n"
        "ESCALATION CHANNEL: PagerDuty\n"
        "EVIDENCE: F02 F05 F09\n"
        "REASONING: EU 14:30 P1 NOT I-17 PagerDuty"
    )
    result = LongContextPlugin().evaluate(response)
    cross = next(item for item in result.rubric if item["name"] == "Cross-reference reasoning")
    assert cross["earned"] == 0.0, f"Expected 0.0 (negation-aware), got {cross['earned']}"


def test_long_context_evidence_is_per_id_credit_for_correct_chain():
    # Measured before the fix: four wrong IDs 4/4 (evidence earned 4.0 when
    # the response cited 4 wrong IDs because the old logic gave partial credit
    # for the number of IDs present). After the fix, per-ID credit is given
    # only for correct-chain membership {F02, F05, F09, F13}, so four wrong
    # IDs earn 0.0.
    response = (
        "INCIDENT: I-17\n"
        "OWNER: Omar\n"
        "ESCALATION CHANNEL: PagerDuty\n"
        "EVIDENCE: F21 F22 F23 F24\n"
        "REASONING: EU 14:30 P1 I-17 PagerDuty"
    )
    result = LongContextPlugin().evaluate(response)
    evidence = next(item for item in result.rubric if item["name"] == "Evidence retrieval")
    assert evidence["earned"] == 0.0, f"Expected 0.0 (per-ID credit), got {evidence['earned']}"


def test_long_context_p1_is_word_bounded_no_p12_leakage():
    # Measured before the fix: P12 leakage (the bare `r"P1"` and
    # `"p1" in reasoning` matched "P12" as a substring, so a response with
    # "P12" in the reasoning earned the Exact answer p1 sub-check and the
    # cross-ref P1 check). After the fix, both checks use word-bounded
    # `\bP1\b` / `\bp1\b`, so "P12" does not satisfy "P1".
    response = (
        "INCIDENT: I-17\n"
        "OWNER: Omar\n"
        "ESCALATION CHANNEL: PagerDuty\n"
        "EVIDENCE: F02 F05 F09 F13\n"
        "REASONING: EU 14:30 P12 I-17 PagerDuty"
    )
    result = LongContextPlugin().evaluate(response)
    cross = next(item for item in result.rubric if item["name"] == "Cross-reference reasoning")
    assert cross["earned"] == 0.0, f"Expected 0.0 (word-bounded P1), got {cross['earned']}"


def test_moe_document_keywords_without_local_sections_score_low():
    response = "MoE and dense models use top-k softmax gating, load balancing equations, training, inference, benchmarks, and references."
    assert MoEDensePlugin().score(response) < 10.0


def test_moe_empty_inference_section_earns_no_inference_points():
    # The inference pattern tuple must be a real tuple. Without the trailing
    # comma it iterates the pattern string per character, and the ``|``
    # characters match the empty string, so an empty inference section body
    # still earned the full 2.0 inference points (5 pipe hits).
    response = (
        "## Gating\ntop-k softmax router g = softmax(Wx).\n"
        "## Load Balancing\nauxiliary loss L = f_i p_i = 0.5.\n"
        "## Training\nexpert collapse and token drop.\n"
        "## Inference\n\n"
        "## Benchmarks\nMoE outperforms dense on MMLU.\n"
        "## References\nShazeer 2017 and Mixtral 8x7B.\n"
    )
    result = MoEDensePlugin().evaluate(response)
    inference = next(item for item in result.rubric if item["name"] == "Inference implications")
    assert inference["earned"] == 0.0


def test_moe_inference_section_with_multiple_concerns_earns_full_points():
    # The inference criterion is worth 2.0 and must be reachable: the pattern
    # is split into two sub-patterns (memory/bandwidth/throughput and
    # parallel/latency/compute) so a response covering two distinct inference
    # concerns earns 2.0, while a single concern earns only 1.0.
    single = (
        "## Gating\ntop-k softmax router g = softmax(Wx).\n"
        "## Load Balancing\nauxiliary loss L = f_i p_i = 0.5.\n"
        "## Training\nexpert collapse and token drop.\n"
        "## Inference\nMoE inference is memory bound.\n"
        "## Benchmarks\nMoE outperforms dense on MMLU.\n"
        "## References\nShazeer 2017 and Mixtral 8x7B.\n"
    )
    single_inference = next(
        item for item in MoEDensePlugin().evaluate(single).rubric
        if item["name"] == "Inference implications"
    )
    assert single_inference["earned"] == 1.0
    multiple = (
        "## Gating\ntop-k softmax router g = softmax(Wx).\n"
        "## Load Balancing\nauxiliary loss L = f_i p_i = 0.5.\n"
        "## Training\nexpert collapse and token drop.\n"
        "## Inference\nMoE inference is memory bound and parallelizes latency.\n"
        "## Benchmarks\nMoE outperforms dense on MMLU.\n"
        "## References\nShazeer 2017 and Mixtral 8x7B.\n"
    )
    multiple_inference = next(
        item for item in MoEDensePlugin().evaluate(multiple).rubric
        if item["name"] == "Inference implications"
    )
    assert multiple_inference["earned"] == 2.0


def test_moe_references_count_distinct_casefolded_names_and_arxiv_ids():
    # References must be counted as distinct casefolded names, so a
    # case-duplicated name ("Mixtral mixtral") is one citation, and a bare
    # "arXiv" keyword with no ID is not a citation. Only a real arXiv ID
    # counts as citation evidence.
    gaming = "## References\nMixtral mixtral arXiv\n"
    result = MoEDensePlugin().evaluate(gaming)
    refs = next(item for item in result.rubric if item["name"] == "Paper references")
    assert refs["earned"] == 1.0
    # A real arXiv ID is legitimate citation evidence and counts toward the two.
    legit = "## References\nMixtral 8x7B (arXiv:2401.04088) and Shazeer et al. (arXiv:1701.03066).\n"
    result2 = MoEDensePlugin().evaluate(legit)
    refs2 = next(item for item in result2.rubric if item["name"] == "Paper references")
    assert refs2["earned"] == 2.0


def test_moe_benchmark_pairs_require_dense_and_dedupe():
    # A MoE-advantage pair must name a dense model (the trailing alternative
    # task/model alone no longer counts), and repeating an identical pair
    # sentence must not inflate the pair count. Measured gaming hit 17/17.
    filler = "x" * 160
    no_dense_bench = (
        "MoE is better for the task on MMLU. " + filler +
        " MoE is better for the task on coding. " + filler +
        " dense is better for the task. " + filler +
        " dense is better for the model."
    )
    response = (
        "## Gating\ntop-k softmax router g = softmax(Wx).\n"
        "## Load Balancing\nauxiliary loss L = f_i p_i = 0.5.\n"
        "## Training\nexpert collapse and token drop.\n"
        "## Inference\nmemory bandwidth latency.\n"
        "## Benchmarks\n" + no_dense_bench + "\n"
        "## References\nShazeer 2017 and Mixtral 8x7B.\n"
    )
    bench = next(
        item for item in MoEDensePlugin().evaluate(response).rubric
        if item["name"] == "Benchmarks/comparison"
    )
    assert bench["earned"] < 2.0
    # The same MoE-vs-dense sentence repeated (spaced) counts once, not twice,
    # while two distinct pairs sharing the prefix up to "dense" still count twice.
    from plugins.challenges.moe_dense import _distinct_pattern_hits
    pair_pattern = r"(?:moe|mixture.of.experts).{0,150}(?:outperform|better|advantage|wins).{0,150}dense"
    repeated = _distinct_pattern_hits(
        "MoE outperforms dense on MMLU. " + filler + " MoE outperforms dense on MMLU.",
        pair_pattern,
    )
    assert repeated == 1
    distinct = _distinct_pattern_hits(
        "MoE outperforms dense on MMLU. " + filler + " MoE outperforms dense on GSM8K.",
        pair_pattern,
    )
    assert distinct == 2
    # Newline-separated identical pairs must dedupe: the terminator char must
    # not leak into the dedup key, or a "\n"-terminated occurrence keys with a
    # trailing space and escapes dedup against the end-of-text occurrence.
    newline_repeat = "MoE outperforms dense on MMLU\nMoE outperforms dense on MMLU"
    assert _distinct_pattern_hits(newline_repeat, pair_pattern) == 1


def test_moe_load_balancing_requires_a_real_variable_not_significant():
    # The f_i variable pattern must be word-bounded: "significant" contains
    # "fi" but is not the f_i load-balancing variable, so it must not satisfy
    # the variable sub-check (measured: a wrong impl passed via "significant").
    response = (
        "## Gating\ntop-k softmax router g = softmax(Wx).\n"
        "## Load Balancing\nThe load balancing term is significant = 0.5.\n"
        "## Training\nexpert collapse and token drop.\n"
        "## Inference\nmemory bandwidth latency.\n"
        "## Benchmarks\nMoE outperforms dense on MMLU.\n"
        "## References\nShazeer 2017 and Mixtral 8x7B.\n"
    )
    load = next(
        item for item in MoEDensePlugin().evaluate(response).rubric
        if item["name"] == "Load-balancing loss"
    )
    assert load["earned"] == 0.0
    # A real f_i variable still satisfies the sub-check.
    legit = "## Load Balancing\nThe auxiliary loss is L_aux = sum f_i * P_i.\n"
    load2 = next(
        item for item in MoEDensePlugin().evaluate(legit).rubric
        if item["name"] == "Load-balancing loss"
    )
    assert load2["earned"] == 3.0


def test_moe_alias_sections_count_for_section_presence():
    # Alias-matched sections ("## Routing" for gating, "## Papers" for
    # references) must count toward section-presence points, not just the
    # content criteria (measured: asymmetric leniency lost the presence point).
    response = (
        "## Routing\ntop-k softmax router g = softmax(Wx).\n"
        "## Load Balancing\nauxiliary loss L = f_i p_i = 0.5.\n"
        "## Training\nexpert collapse and token drop.\n"
        "## Inference\nmemory bandwidth latency.\n"
        "## Benchmarks\nMoE outperforms dense on MMLU.\n"
        "## Papers\nShazeer 2017 and Mixtral 8x7B.\n"
    )
    sections = next(
        item for item in MoEDensePlugin().evaluate(response).rubric
        if item["name"] == "Required comparison sections"
    )
    assert sections["earned"] == 2.0


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


def test_rate_limiter_behavior_rejects_deny_forever_implementations():
    response = """```python
import math
import threading
class _Base:
    def __init__(self, limit, window_seconds):
        if not isinstance(limit, int) or isinstance(limit, bool): raise TypeError("limit")
        if not isinstance(window_seconds, (int, float)) or isinstance(window_seconds, bool): raise TypeError("window")
        if limit <= 0: raise ValueError("limit")
        if not math.isfinite(window_seconds) or window_seconds <= 0: raise ValueError("window")
        self.limit, self.window_seconds, self.counts, self.lock = limit, window_seconds, {}, threading.Lock()
    def allow_request(self, client_id, now):
        with self.lock:
            if self.counts.get(client_id, 0) >= self.limit: return False
            self.counts[client_id] = self.counts.get(client_id, 0) + 1
            return True
    def get_usage_stats(self, client_id):
        with self.lock: return {"count": self.counts.get(client_id, 0), "limit": self.limit}
    def cleanup(self, now):
        with self.lock: return 0
class TokenBucket(_Base): pass
class SlidingWindowLog(_Base): pass
class FixedWindow(_Base): pass
```"""
    result = RateLimiterPlugin().evaluate(response)
    behavior = next(item for item in result.rubric if item["name"] == "Behavioral strategy tests")
    assert behavior["earned"] == 0.0
    assert result.score < 15.0


def test_rate_limiter_behavior_rejects_wall_clock_implementations():
    response = """```python
import math
import threading
import time
class _Base:
    def __init__(self, limit, window_seconds):
        if not isinstance(limit, int) or isinstance(limit, bool): raise TypeError("limit")
        if not isinstance(window_seconds, (int, float)) or isinstance(window_seconds, bool): raise TypeError("window")
        if limit <= 0: raise ValueError("limit")
        if not math.isfinite(window_seconds) or window_seconds <= 0: raise ValueError("window")
        self.limit, self.window_seconds, self.counts, self.lock = limit, window_seconds, {}, threading.Lock()
    def allow_request(self, client_id, now):
        with self.lock:
            now = time.time()
            window, count = self.counts.get(client_id, (now, 0))
            if now - window >= self.window_seconds: window, count = now, 0
            if count >= self.limit: return False
            self.counts[client_id] = (window, count + 1)
            return True
    def get_usage_stats(self, client_id):
        with self.lock: return {"count": self.counts.get(client_id, (0, 0))[1], "limit": self.limit}
    def cleanup(self, now):
        with self.lock:
            now = time.time()
            old = [key for key, (start, _) in self.counts.items() if now - start >= self.window_seconds]
            for key in old: del self.counts[key]
            return len(old)
class TokenBucket(_Base): pass
class SlidingWindowLog(_Base): pass
class FixedWindow(_Base): pass
```"""
    result = RateLimiterPlugin().evaluate(response)
    behavior = next(item for item in result.rubric if item["name"] == "Behavioral strategy tests")
    assert behavior["earned"] == 0.0
    assert result.score < 15.0


def test_architecture_keywords_without_required_sections_score_low():
    response = "microservices API gateway PostgreSQL Redis OAuth2 Kubernetes 1M DAU circuit breaker 99.9%."
    assert SoftwareArchitecturePlugin().score(response) < 8.0


def test_data_transformation_rejects_multiple_candidates():
    payload = DATA_TRANSFORMATION_EXPECTED_OUTPUT
    result = DataTransformationPlugin().evaluate(
        "```json\n" + json.dumps(payload) + "\n```\n```json\n" + json.dumps(payload) + "\n```"
    )
    assert result.score == 0.0
    assert any("exactly one structured candidate is required" in error for error in result.diagnostics["errors"])


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
