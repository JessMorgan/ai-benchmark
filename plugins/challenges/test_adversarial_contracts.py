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
from plugins.challenges.decomposition import DecompositionPlugin
from plugins.challenges.error_recovery import _CONCEPT_PATTERNS, ErrorRecoveryPlugin
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


def test_debug_consistency_negated_forms_do_not_earn_consistency():
    # Measured pre-fix (DC-1): the boundary-less `correct|consistent`
    # alternatives matched `incorrect`/`inconsistent`, so a hallucinated-bug
    # answer's consistency section earned the full 5.0 criterion (20/20 total).
    response = """## Reproduction
Running find_duplicate_users with the supplied input returns ['abc'].
## Consistency Check
The code is incorrect and inconsistent with the specification.
## Diagnosis
There is a bug in the comparison logic.
## Evidence Needed
Collect logs, stack traces, and environment details.
## Recommendation
Verify the fix and patch the comparison.
"""
    result = DebugConsistencyPlugin().evaluate(response)
    consistency = next(item for item in result.rubric if item["name"] == "Consistency conclusion")
    assert consistency["earned"] == 0.0, f"Expected 0.0 (negated forms), got {consistency['earned']}"


def test_debug_consistency_correct_conclusion_earns_consistency():
    # The positive direction must still be credited: a correct conclusion
    # (code is correct / report not reproducible) earns the full criterion.
    response = """## Reproduction
Running find_duplicate_users with the supplied input returns ['abc'].
## Consistency Check
The code is correct and the report is not reproducible.
## Diagnosis
There is no bug in the code.
## Evidence Needed
Collect logs, stack traces, and environment details.
## Recommendation
Verify the fix and patch the comparison.
"""
    result = DebugConsistencyPlugin().evaluate(response)
    consistency = next(item for item in result.rubric if item["name"] == "Consistency conclusion")
    assert consistency["earned"] == 5.0, f"Expected 5.0 (positive signal), got {consistency['earned']}"


def test_debug_consistency_generic_words_do_not_earn_diagnosis():
    # Measured pre-fix (DC-2): the diagnosis criterion matched generic words
    # (`report`, `environment`, `input`), so a hallucinated diagnosis that
    # merely mentioned "the report" and "the input" earned the full 4.0.
    response = """## Reproduction
Running find_duplicate_users with the supplied input returns ['abc'].
## Consistency Check
The code is correct and the report is not reproducible.
## Diagnosis
There is a bug; the report describes the input and environment.
## Evidence Needed
Collect logs, stack traces, and environment details.
## Recommendation
Verify the fix and patch the comparison.
"""
    result = DebugConsistencyPlugin().evaluate(response)
    diagnosis = next(item for item in result.rubric if item["name"] == "Non-hallucinated diagnosis")
    assert diagnosis["earned"] == 0.0, f"Expected 0.0 (generic words), got {diagnosis['earned']}"


def test_debug_consistency_positive_no_bug_signals_earn_diagnosis():
    # Specific positive signals that the code works correctly must still be
    # credited: no bug / cannot reproduce / behaves as specified.
    response = """## Reproduction
Running find_duplicate_users with the supplied input returns ['abc'].
## Consistency Check
The code is correct and the report is not reproducible.
## Diagnosis
There is no bug; the failure cannot be reproduced and the code behaves as specified.
## Evidence Needed
Collect logs, stack traces, and environment details.
## Recommendation
Verify the fix and patch the comparison.
"""
    result = DebugConsistencyPlugin().evaluate(response)
    diagnosis = next(item for item in result.rubric if item["name"] == "Non-hallucinated diagnosis")
    assert diagnosis["earned"] == 4.0, f"Expected 4.0 (positive signals), got {diagnosis['earned']}"


def test_debug_consistency_modified_no_bug_phrasing_earns_diagnosis():
    # A no-bug signal with modifier words ("no demonstrated code bug") is a
    # legitimate positive signal and must be credited, not just the bare
    # "no bug" form.
    response = """## Reproduction
Running find_duplicate_users with the supplied input returns ['abc'].
## Consistency Check
The code is correct and the report is not reproducible.
## Diagnosis
There is no demonstrated code bug; the report and supplied behavior are inconsistent.
## Evidence Needed
If the report persists, collect the actual output (expected ['abc']) and the exact input.
## Recommendation
Verify the fix and patch the comparison.
"""
    result = DebugConsistencyPlugin().evaluate(response)
    diagnosis = next(item for item in result.rubric if item["name"] == "Non-hallucinated diagnosis")
    assert diagnosis["earned"] == 4.0, f"Expected 4.0 (modified no-bug phrasing), got {diagnosis['earned']}"


def test_debug_consistency_reproduction_requires_the_positive_trace():
    # Measured pre-fix (DC-3): a correct answer that quoted the report's
    # claimed `[]` output lost the 4.0 reproduction criterion (16/20 total)
    # because the blanket `empty|\[\]` negation matched the quoted claim.
    # After the fix the criterion requires the positive trace (the actual
    # output ['abc']) and ignores quoted claims.
    response = """## Reproduction
The report claims the output is []. Tracing the code, counts is {'abc': 2},
so the function returns ['abc'].
## Consistency Check
The code is correct and the report is not reproducible.
## Diagnosis
There is no bug in the code.
## Evidence Needed
Collect logs, stack traces, and environment details.
## Recommendation
Verify the fix and patch the comparison.
"""
    result = DebugConsistencyPlugin().evaluate(response)
    reproduction = next(item for item in result.rubric if item["name"] == "Reproduction trace")
    assert reproduction["earned"] == 4.0, f"Expected 4.0 (positive trace present), got {reproduction['earned']}"


def test_debug_consistency_empty_output_claim_earns_no_reproduction():
    # A hallucinated trace that reports the (wrong) empty output must not
    # earn the criterion: the positive trace ['abc'] is absent.
    response = """## Reproduction
Running the code with the input produces an empty list.
## Consistency Check
The code is incorrect and inconsistent with the specification.
## Diagnosis
There is a bug in the comparison logic.
## Evidence Needed
Collect logs, stack traces, and environment details.
## Recommendation
Verify the fix and patch the comparison.
"""
    result = DebugConsistencyPlugin().evaluate(response)
    reproduction = next(item for item in result.rubric if item["name"] == "Reproduction trace")
    assert reproduction["earned"] == 0.0, f"Expected 0.0 (no positive trace), got {reproduction['earned']}"


def test_debug_consistency_evidence_requires_a_trace_reference():
    # Measured pre-fix (DC-4): the evidence criterion credited generic
    # keywords (`log`, `stack`, `environment`) with no reference to the
    # actual trace/output. After the fix the evidence section must also
    # reference the actual output (abc / ['abc']).
    response = """## Reproduction
Running find_duplicate_users with the supplied input returns ['abc'].
## Consistency Check
The code is correct and the report is not reproducible.
## Diagnosis
There is no bug in the code.
## Evidence Needed
Collect logs, stack traces, and environment details.
## Recommendation
Verify the fix and patch the comparison.
"""
    result = DebugConsistencyPlugin().evaluate(response)
    evidence = next(item for item in result.rubric if item["name"] == "Evidence request")
    assert evidence["earned"] == 0.0, f"Expected 0.0 (no trace reference), got {evidence['earned']}"


def test_debug_consistency_evidence_with_trace_reference_earns_full():
    # An evidence request that references the actual output must still be
    # credited in full.
    response = """## Reproduction
Running find_duplicate_users with the supplied input returns ['abc'].
## Consistency Check
The code is correct and the report is not reproducible.
## Diagnosis
There is no bug in the code.
## Evidence Needed
If the report persists, collect the actual output (expected ['abc']) and the exact input.
## Recommendation
Verify the fix and patch the comparison.
"""
    result = DebugConsistencyPlugin().evaluate(response)
    evidence = next(item for item in result.rubric if item["name"] == "Evidence request")
    assert evidence["earned"] == 3.0, f"Expected 3.0 (trace reference), got {evidence['earned']}"


def test_debug_consistency_structure_requires_nonempty_bodies():
    # Measured pre-fix (DC-5): the structure criterion credited heading
    # presence alone, so five empty headings earned the full 2.0. After the
    # fix each section must have content beyond the heading.
    response = """## Reproduction
## Consistency Check
## Diagnosis
## Evidence Needed
## Recommendation
"""
    result = DebugConsistencyPlugin().evaluate(response)
    structure = next(item for item in result.rubric if item["name"] == "Required report structure")
    assert structure["earned"] == 0.0, f"Expected 0.0 (empty bodies), got {structure['earned']}"


def test_debug_consistency_structure_with_bodies_earns_full():
    # Five sections with substantive bodies must still earn the criterion.
    response = """## Reproduction
Running find_duplicate_users with the supplied input returns ['abc'].
## Consistency Check
The code is correct and the report is not reproducible.
## Diagnosis
There is no bug in the code.
## Evidence Needed
If the report persists, collect the actual output (expected ['abc']) and the exact input.
## Recommendation
Do not patch the code; verify by reproducing the input first.
"""
    result = DebugConsistencyPlugin().evaluate(response)
    structure = next(item for item in result.rubric if item["name"] == "Required report structure")
    assert structure["earned"] == 2.0, f"Expected 2.0 (non-empty bodies), got {structure['earned']}"


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


def test_debug_traversal_early_exit_does_not_pass_execution_gate():
    # DT-9 (SH-1 wiring): a response that exits 0 before the harness (sys.exit)
    # reports status "passed" but harness_ok False; it must not pass the
    # execution gate, and the lexical criteria must be scaled down.
    response = (
        "## Root Cause\nThe comparison `count > 2` should be `count >= 2`.\n"
        "## Analysis\nFor abc123, count is 2 and the function returns an empty list. def456 is present.\n"
        "## Fix\nChange the comparison to `count >= 2`:\n"
        "```python\nimport sys\ndef find_duplicate_users(log_entries):\n"
        "    sys.exit(0)\n```\n"
        "## Test\n```python\ndef test_check():\n"
        "    assert 'abc123' in find_duplicate_users(logs)\n```\n"
        "## Side Effects\nOrdering and empty IDs are considered; duplicates are counted.\n"
    )
    result = DebugTraversalPlugin().evaluate(response)
    exec_item = next(item for item in result.rubric if item["name"] == "Executable fix verification")
    assert exec_item["earned"] == 0.0
    assert any("early exit" in finding["finding"] for finding in exec_item["negative_findings"])
    # The lexical fix criterion is scaled down even though the prose names the
    # correct comparison (the harness never ran to completion).
    fix = next(item for item in result.rubric if item["name"] == "Proposed fix / corrected code")
    assert fix["earned"] == 0.0
    assert any("withheld" in finding["finding"] for finding in fix["negative_findings"])
def test_decomposition_appended_domain_mapping_cannot_override_task_domains():
    # Measured pre-fix: this degenerate plan scored 20/20 because the appended
    # "Domain mapping" section re-declared every task ID and the last-line-wins
    # domain_by_task binding let it override the tasks' own descriptions.
    response = """Task 1: set up the project and define the data schema
Task 2 [DEPENDS_ON: 1]: implement the core processing loop
Task 3 [DEPENDS_ON: 2]: add the detection logic
Task 4 [DEPENDS_ON: 3]: wire up the user-facing output
Task 5 [DEPENDS_ON: 2]: produce the final output
Task 6: add dashboards
Domain mapping:
Task 1: ingestion
Task 2: enrich
Task 3: anomaly
Task 4: alert
Task 5: report
Task 6: observe
Parallel stages: 4, 5 and 6 in parallel.
Sequential stages: 1 then 2 then 3.
Ordering rationale: data flow and prerequisite order.
"""
    result = DecompositionPlugin().evaluate(response)
    direction = next(c for c in result.rubric if c["name"] == "Semantic dependency direction")
    assert direction["earned"] == 0.0
    assert result.score < 16.0


def test_decomposition_domain_of_uses_most_keyword_hits_with_position_tiebreak():
    # Measured pre-fix: earliest-keyword-wins misclassified a correct line to
    # the wrong domain (15/20 on a correct plan). Now the domain with the most
    # distinct keyword hits wins, and ties fall to the earliest first hit.
    plugin = DecompositionPlugin()
    assert plugin._domain_of("Task 4: anomaly alerts and real-time feed for operators") == "alert"
    assert plugin._domain_of("Task 3: anomaly detection over the normalized stream") == "anomaly"
    assert plugin._domain_of("Task 3: normalized stream for anomaly detection") == "enrich"


def test_decomposition_penalizes_every_declared_forbidden_edge():
    # Measured pre-fix: a reversed plan scored 15-16/20 because any reversed
    # edge applied a single global half-marks cap (min(points, 3.0)) instead
    # of penalizing each declared forbidden edge. Two required edges present
    # plus one declared forbidden edge now earn 2.0, not the old 3.0 cap.
    response = """Task 1: Accept and buffer log batches over HTTP ingestion
Task 2 [DEPENDS_ON: 1] [DEPENDS_ON: 3]: GeoIP enrich and normalize each line
Task 3 [DEPENDS_ON: 1]: anomaly detection over the normalized stream
Task 4 [DEPENDS_ON: 3]: real-time alert feed for anomalies
Task 5: nightly aggregate report
Task 6: export metrics for observability
Parallel stages: 5 and 6 can run in parallel; they are independent.
Sequential stages: 1 then 2 then 3 then 4.
Ordering rationale: Task 1 before Task 2, data flows from ingestion to enrichment.
"""
    result = DecompositionPlugin().evaluate(response)
    direction = next(c for c in result.rubric if c["name"] == "Semantic dependency direction")
    assert direction["earned"] == 2.0


def test_decomposition_parallelization_requires_independence_language():
    # Bare "parallel" presence (which the prompt itself elicits) no longer
    # earns the full criterion: the plan must justify the parallel stages
    # with independence language.
    bare = """Task 1: Accept and buffer log batches over HTTP ingestion
Task 2 [DEPENDS_ON: 1]: GeoIP enrich and normalize each line
Task 3 [DEPENDS_ON: 2]: anomaly detection over the normalized stream
Task 4 [DEPENDS_ON: 3]: real-time alert feed for anomalies
Task 5 [DEPENDS_ON: 2]: nightly aggregate report
Task 6: export metrics for observability
Parallel stages: 4, 5 and 6.
Sequential stages: 1 then 2 then 3.
Ordering rationale: Task 1 before Task 2, data flows from ingestion to enrichment.
"""
    result = DecompositionPlugin().evaluate(bare)
    parallel = next(c for c in result.rubric if c["name"] == "Parallelization reasoning")
    assert parallel["earned"] == 1.0
    justified = bare.replace(
        "Parallel stages: 4, 5 and 6.",
        "Parallel stages: 4, 5 and 6 are independent and can run in parallel.",
    )
    result = DecompositionPlugin().evaluate(justified)
    parallel = next(c for c in result.rubric if c["name"] == "Parallelization reasoning")
    assert parallel["earned"] == 2.0


def test_decomposition_rationale_requires_task_references():
    # Ordering vocabulary alone (data flow, prerequisite, order, before...)
    # with no reference to the plan's specific tasks earns no rationale
    # points.
    response = """Task 1: Accept and buffer log batches over HTTP ingestion
Task 2 [DEPENDS_ON: 1]: GeoIP enrich and normalize each line
Task 3 [DEPENDS_ON: 2]: anomaly detection over the normalized stream
Task 4 [DEPENDS_ON: 3]: real-time alert feed for anomalies
Task 5 [DEPENDS_ON: 2]: nightly aggregate report
Task 6: export metrics for observability
Parallel stages: 4, 5 and 6 are independent and can run in parallel.
Sequential stages: 1 then 2 then 3.
Ordering rationale: data flow, prerequisite, depends on, order, before, after, first, then.
"""
    result = DecompositionPlugin().evaluate(response)
    rationale = next(c for c in result.rubric if c["name"] == "Ordering rationale")
    assert rationale["earned"] == 0.0


def test_decomposition_accepts_t_style_task_ids():
    # Measured pre-fix: a correct plan named T1..T6 scored 10/20 because the
    # task-ID regex and the shared graph parser only recognized "Task N", so
    # the graph validity and semantic direction criteria both dead-ended at 0.
    response = """T1: Accept and buffer log batches over HTTP ingestion
T2 [DEPENDS_ON: T1]: GeoIP enrich and normalize each line
T3 [DEPENDS_ON: T2]: anomaly detection over the normalized stream
T4 [DEPENDS_ON: T3]: real-time alert feed for anomalies
T5 [DEPENDS_ON: T2]: nightly aggregate report
T6: export metrics for observability
Parallel stages: T4, T5 and T6 can run in parallel; they are independent.
Sequential stages: T1 then T2 then T3 then T4.
Ordering rationale: T1 before T2 because data flows from ingestion to enrichment.
"""
    result = DecompositionPlugin().evaluate(response)
    assert result.score == 20.0


def test_decomposition_geo_and_feed_keywords_are_word_bounded():
    # The boundary-less "geo"/"feed" keywords matched inside longer words
    # ("geography", "feedback"), crediting the wrong domains.
    plugin = DecompositionPlugin()
    assert plugin._domain_of("Task 4: track user geography and location") is None
    assert plugin._domain_of("Task 5: gather operator feedback") is None
    # Legitimate uses still resolve.
    assert plugin._domain_of("Task 2: GeoIP enrich each line") == "enrich"
    assert plugin._domain_of("Task 4: real-time alert feed for anomalies") == "alert"


def test_decomposition_reversed_edge_finding_reads_declared_direction():
    # The diagnostic must name the declared (wrong) edge direction, not the
    # correct one: declaring "ingestion depends on enrich" is reported as
    # "reversed dependency ingestion -> enrich", matching the arrow
    # convention of the "missing dependency" findings.
    response = """Task 1: Accept and buffer log batches over HTTP ingestion
Task 2: GeoIP enrich and normalize each line
Task 3: anomaly detection over the normalized stream
Task 4: real-time alert feed for anomalies
Task 5: nightly aggregate report
Task 6: export metrics for observability
Dependencies: Task 1 [DEPENDS_ON: 2]
"""
    result = DecompositionPlugin().evaluate(response)
    direction = next(c for c in result.rubric if c["name"] == "Semantic dependency direction")
    assert any(
        "reversed dependency ingestion -> enrich" in f["finding"]
        for f in direction["negative_findings"]
    )


def test_instruction_following_wrong_tie_break_does_not_pass():
    response = """ORDER T-05 | CUSTOMER NOOR | TOTAL 120.00
ORDER T-02 | CUSTOMER JULES | TOTAL 120.00
ORDER T-08 | CUSTOMER ZARA | TOTAL 99.90
ORDER T-09 | CUSTOMER RAVI | TOTAL 65.00
[SUMMARY] count=4; total=404.90; top_order=T-02"""
    assert InstructionFollowingPlugin().score(response) < InstructionFollowingPlugin().max_score


def test_instruction_following_duplicate_summary_line_is_forbidden():
    # IF-1: a repeated [SUMMARY] line is duplicate output and must count as
    # forbidden (the discipline criterion's own finding text names duplicates),
    # consistent with the duplicate-ORDER penalty. Pre-fix the duplicate summary
    # was excluded from `forbidden`, so this scored 17/20 (discipline 1.0).
    response = """ORDER T-02 | CUSTOMER JULES | TOTAL 120.00
ORDER T-05 | CUSTOMER NOOR | TOTAL 120.00
ORDER T-08 | CUSTOMER ZARA | TOTAL 99.90
ORDER T-09 | CUSTOMER RAVI | TOTAL 65.00
[SUMMARY] count=4; total=404.90; top_order=T-02
[SUMMARY] count=4; total=404.90; top_order=T-02"""
    result = InstructionFollowingPlugin().evaluate(response)
    discipline = next(item for item in result.rubric if item["name"] == "Exact response discipline")
    assert discipline["earned"] == 0.0
    assert result.score < 17.0


def test_instruction_following_summary_requires_at_least_one_order_line():
    # IF-2: a zero-work response (no ORDER lines) must not earn the summary
    # criterion. Pre-fix the bare [SUMMARY] line alone scored 5/20 (summary
    # 4.0 + discipline 1.0). After the fix the summary points are gated on
    # having at least one parsed ORDER line, so this scores 1/20.
    response = "[SUMMARY] count=4; total=404.90; top_order=T-02"
    result = InstructionFollowingPlugin().evaluate(response)
    summary = next(item for item in result.rubric if item["name"] == "Summary arithmetic and format")
    assert summary["earned"] == 0.0
    assert result.score < 5.0


def test_instruction_following_case_error_in_id_does_not_double_penalty_filter():
    # IF-3: a case error in a task ID (t-05 vs T-05) must not cascade into a
    # double-penalty in the "All filters applied" criterion. The filter
    # criterion case-normalizes ID membership, so all four IDs are recognized;
    # the order/transformed criteria keep the raw IDs and still penalize the
    # case error. Pre-fix the lowercase line failed the ORDER regex entirely,
    # dropping the filter to 2.0 and marking the line forbidden (discipline 0).
    response = """ORDER T-02 | CUSTOMER JULES | TOTAL 120.00
ORDER t-05 | CUSTOMER NOOR | TOTAL 120.00
ORDER T-08 | CUSTOMER ZARA | TOTAL 99.90
ORDER T-09 | CUSTOMER RAVI | TOTAL 65.00
[SUMMARY] count=4; total=404.90; top_order=T-02"""
    result = InstructionFollowingPlugin().evaluate(response)
    filter_item = next(item for item in result.rubric if item["name"] == "All filters applied")
    assert filter_item["earned"] == 4.0
    # The case error is still penalized in the order criterion (raw IDs).
    order_item = next(item for item in result.rubric if item["name"] == "Sort and tie-break order")
    assert order_item["earned"] == 0.0


def test_reasoning_rejects_the_old_p4_answer():
    response = """1. The time chain places Search at 09:30.
2. Ben owns Search.
3. Upload outranks Search, which outranks Billing.
FAILED_SERVICE: Search
OWNER: Ben
PRIORITY: P4
TIME: 09:30"""
    assert ReasoningPlugin().score(response) < ReasoningPlugin().max_score


def test_reasoning_clue_restatement_with_wrong_answer_scores_low():
    # RE-1: a response that restates the clue wording measured 14.0 while a
    # naturally-phrased correct answer measured 13.3 (6 of 8 reasoning points
    # were earnable by copying the clues). The four reasoning-point criteria
    # are now capped at half their max when the final answer lines are wrong
    # or absent.
    response = """1. Auth is immediately before Search, Profile is before Auth, Upload is after Search, and Billing is after Upload but before Notifications.
2. Therefore Profile is at 09:00, Auth is at 09:15, Search is at 09:30, Upload is at 09:45, Billing is at 10:00, and Notifications is at 10:15.
3. Ben owned Search, Eli owned Upload, and Ana owned Notifications at 10:15.
4. Auth is P1, Notifications is P2, and Upload has higher priority than Search, which has higher priority than Billing; therefore Search is P5.
FAILED_SERVICE: Profile
OWNER: Ana
PRIORITY: P4
TIME: 09:00"""
    result = ReasoningPlugin().evaluate(response)
    assert result.score < 8.0
    for name in (
        "Time-chain deductions",
        "Derived time assignments",
        "Ownership deductions",
        "Priority-chain deductions",
    ):
        item = next(item for item in result.rubric if item["name"] == name)
        assert item["earned"] <= item["max"] / 2.0
    # The restatement no longer beats a naturally-phrased correct answer.
    correct = response.replace(
        "FAILED_SERVICE: Profile\nOWNER: Ana\nPRIORITY: P4\nTIME: 09:00",
        "FAILED_SERVICE: Search\nOWNER: Ben\nPRIORITY: P5\nTIME: 09:30",
    )
    assert result.score < ReasoningPlugin().score(correct)


def test_reasoning_accepts_owns_phrasing_and_gt_priority_chains():
    # RE-2: "Ben owns Search" (the ownership patterns required the literal
    # words "owned"/"owner") and "Upload > Search > Billing" (the chain
    # pattern required the literal word "higher") were phrasing traps; a
    # table-format correct answer measured 10.7, the same as a
    # wrong-priority answer. Both phrasings now earn their criteria.
    response = """1. Auth is immediately before Search, Profile is before Auth, Upload is after Search, and Billing is after Upload but before Notifications.
2. Profile 09:00, Auth 09:15, Search 09:30, Upload 09:45, Billing 10:00, Notifications 10:15.
3. Ben owns Search, Eli owns Upload, and Ana owns Notifications at 10:15.
4. Auth is P1, Notifications is P2, and Upload > Search > Billing, so Search is P5.
FAILED_SERVICE: Search
OWNER: Ben
PRIORITY: P5
TIME: 09:30"""
    result = ReasoningPlugin().evaluate(response)
    ownership = next(item for item in result.rubric if item["name"] == "Ownership deductions")
    assert ownership["earned"] == ownership["max"]
    priorities = next(item for item in result.rubric if item["name"] == "Priority-chain deductions")
    assert priorities["earned"] == priorities["max"]
    assert result.score == ReasoningPlugin().max_score


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


def test_orchestration_scales_breakdown_by_operation_coverage():
    # Measured pre-fix: this zero-operation keyword response scored 16/16
    # because the fallback credit min(4.0, len(declared_ids)) ignored the
    # operation count entirely.
    response = """Task 1 [PARALLEL]
Task 2 [PARALLEL]
Task 3 [SEQUENTIAL]
Task 4 [SEQUENTIAL]
Task 2 [DEPENDS_ON: task 1]
Task 3 [DEPENDS_ON: task 2]
Task 4 [DEPENDS_ON: task 3]
Task 1 init running complete.
Task 2 init running complete.
Task 3 init running complete.
Task 4 init running complete."""
    result = OrchestrationPlugin().evaluate(response)
    breakdown = next(item for item in result.rubric if item["name"] == "Task breakdown presence")
    assert breakdown["earned"] == 0.0
    assert result.score < 16.0


def test_orchestration_partial_operation_coverage_earns_partial_breakdown():
    response = """Task 1 [PARALLEL] process logs.
Task 2 [PARALLEL] perform GeoIP lookup.
Task 3 [SEQUENTIAL]
Task 4 [SEQUENTIAL]
Task 2 [DEPENDS_ON: task 1]
Task 3 [DEPENDS_ON: task 2]
Task 4 [DEPENDS_ON: task 3]
Task 1 init running complete.
Task 2 init running complete.
Task 3 init running complete.
Task 4 init running complete."""
    result = OrchestrationPlugin().evaluate(response)
    breakdown = next(item for item in result.rubric if item["name"] == "Task breakdown presence")
    assert breakdown["earned"] == 2.0


def test_orchestration_accepts_numbered_list_task_declarations():
    # Measured pre-fix: this numbered-list response scored 0.0/16 because
    # only "task"/"step"-prefixed lines were recognized as task
    # declarations, although the prompt never mandates that prefix.
    response = """1. Process 1TB server logs (parallel)
2. GeoIP lookup (sequential) [DEPENDS_ON: 1]
3. Anomaly detection (sequential) [DEPENDS_ON: 2]
4. Generate PDF report (sequential) [DEPENDS_ON: 3]
1: init running complete
2: init running complete
3: init running complete
4: init running complete"""
    assert OrchestrationPlugin().score(response) == 16.0


def test_orchestration_penalizes_more_than_four_declared_tasks():
    # Measured pre-fix: the extra Task 5 was ignored by the
    # min(4.0, len(declared_ids)) credit, so this response scored 16/16
    # despite the prompt requiring exactly four tasks.
    response = """Task 1 [PARALLEL] process logs.
Task 2 [PARALLEL] perform GeoIP lookup.
Task 3 [SEQUENTIAL] anomaly detection.
Task 4 [SEQUENTIAL] generate the PDF report.
Task 5 [SEQUENTIAL] send the report by email.
Task 2 [DEPENDS_ON: task 1]
Task 3 [DEPENDS_ON: task 2]
Task 4 [DEPENDS_ON: task 3]
Task 5 [DEPENDS_ON: task 4]
Task 1 init running complete.
Task 2 init running complete.
Task 3 init running complete.
Task 4 init running complete.
Task 5 init running complete."""
    result = OrchestrationPlugin().evaluate(response)
    breakdown = next(item for item in result.rubric if item["name"] == "Task breakdown presence")
    assert breakdown["earned"] < 4.0
    assert any(finding["finding"] == "declares more than four tasks" for finding in breakdown["negative_findings"])
    assert result.score < 16.0


def test_orchestration_flags_cross_line_label_contradiction():
    # Measured pre-fix: criterion 3 only flagged lines carrying both
    # "parallel" and "sequential", so a cross-line contradiction in one
    # task's block passed even though parse_workflow_graph flags the same
    # task as labeled both ways.
    response = """Task 1 [PARALLEL] process logs.
Task 2 [PARALLEL] perform GeoIP lookup.
Task 3 [SEQUENTIAL] anomaly detection.
Task 4 [SEQUENTIAL] generate the PDF report.
Task 4 actually runs in parallel with task 3.
Task 2 [DEPENDS_ON: task 1]
Task 3 [DEPENDS_ON: task 2]
Task 4 [DEPENDS_ON: task 3]
Task 1 init running complete.
Task 2 init running complete.
Task 3 init running complete.
Task 4 init running complete."""
    result = OrchestrationPlugin().evaluate(response)
    labels = next(item for item in result.rubric if item["name"] == "Parallel vs sequential logic")
    assert labels["earned"] == 0.0


def test_orchestration_edge_partial_credit_requires_valid_graph():
    # Measured pre-fix: a cyclic (invalid) graph with edges still earned
    # the 2.0 partial credit because only the full-credit branch required
    # graph.valid.
    response = """Task 1 [PARALLEL] process logs.
Task 2 [PARALLEL] perform GeoIP lookup.
Task 3 [SEQUENTIAL] anomaly detection.
Task 4 [SEQUENTIAL] generate the PDF report.
Task 2 [DEPENDS_ON: task 3]
Task 3 [DEPENDS_ON: task 2]
Task 1 init running complete.
Task 2 init running complete.
Task 3 init running complete.
Task 4 init running complete."""
    result = OrchestrationPlugin().evaluate(response)
    tagging = next(item for item in result.rubric if item["name"] == "Explicit dependency tagging")
    assert tagging["earned"] == 0.0


def test_orchestration_numbered_list_cycle_earns_no_tagging_credit():
    # The local numbered-list graph pass must apply the same validity
    # rules as the shared parser: a cyclic numbered plan earns no
    # dependency-tagging credit.
    response = """1. Process 1TB server logs (parallel)
2. GeoIP lookup (sequential) [DEPENDS_ON: 3]
3. Anomaly detection (sequential) [DEPENDS_ON: 2]
4. Generate PDF report (sequential) [DEPENDS_ON: 1]
1: init running complete
2: init running complete
3: init running complete
4: init running complete"""
    result = OrchestrationPlugin().evaluate(response)
    tagging = next(item for item in result.rubric if item["name"] == "Explicit dependency tagging")
    assert tagging["earned"] == 0.0
    assert result.score < 16.0


def test_orchestration_numbered_list_prose_dependencies():
    # Plain-language dependencies between numbered tasks ("2. ... depends
    # on 1") are as explicit as bracket tags.
    response = """1. Process 1TB server logs (parallel)
2. GeoIP lookup (sequential) depends on 1
3. Anomaly detection (sequential) depends on 2
4. Generate PDF report (sequential) depends on 3
1: init running complete
2: init running complete
3: init running complete
4: init running complete"""
    assert OrchestrationPlugin().score(response) == 16.0


def test_orchestration_numbered_plan_with_task_prefixed_trace():
    # A numbered plan whose trace lines use the "Task N" prefix must not be
    # hijacked by the shared parser (which sees the task IDs but no edges,
    # because its bracket tags only bind to task/step mentions): the local
    # numbered pass binds the DEPENDS_ON brackets and the plan scores 16/16.
    # Measured against the state just before the fallback-gate fix:
    # this response scored 12.0/16 (breakdown 4.0, tags 0.0, labels 4.0,
    # trace 4.0); the original pre-OR-1 code scored it 8.0.
    response = """1. Process 1TB server logs (parallel)
2. GeoIP lookup (sequential) [DEPENDS_ON: 1]
3. Anomaly detection (sequential) [DEPENDS_ON: 2]
4. Generate PDF report (sequential) [DEPENDS_ON: 3]
Task 1 init running complete
Task 2 init running complete
Task 3 init running complete
Task 4 init running complete"""
    assert OrchestrationPlugin().score(response) == 16.0


def test_orchestration_numbered_plan_with_task_prefixed_brackets():
    # The prompt's canonical bracket format is "[DEPENDS_ON: task_id]", and
    # models routinely write the ID with its "task" prefix. Pre-fix,
    # _task_blocks bound each numbered declaration line to the task ID
    # inside its own bracket (the first "task N" mention on the line), so
    # this fully valid plan scored 11.0/16: breakdown 3.0 (task 2's
    # operation counted under block 1) and labels 0.0 (block 1 held both
    # "parallel" and "sequential" -> false contradiction).
    response = """1. Process 1TB server logs (parallel)
2. GeoIP lookup (sequential) [DEPENDS_ON: task 1]
3. Anomaly detection (sequential) [DEPENDS_ON: task 2]
4. Generate PDF report (sequential) [DEPENDS_ON: task 3]
1: init running complete
2: init running complete
3: init running complete
4: init running complete"""
    assert OrchestrationPlugin().score(response) == 16.0


def test_orchestration_label_contradiction_costs_the_same_in_both_formats():
    # A task labeled both parallel and sequential must cost the same in
    # numbered format as in task/step format: the local graph pass applies
    # the same label check as parse_workflow_graph, so both formats lose
    # the tagging and label criteria (8.0/16 each) instead of the numbered
    # format escaping with 12.0/16.
    task_step = """Task 1 [PARALLEL] process 1TB server logs.
Task 2 [PARALLEL] perform GeoIP lookup.
Task 3 [SEQUENTIAL] run anomaly detection.
Task 4 [SEQUENTIAL] generate the PDF report.
Task 4 actually runs in parallel with task 3.
Task 2 [DEPENDS_ON: task 1]
Task 3 [DEPENDS_ON: task 2]
Task 4 [DEPENDS_ON: task 3]
Task 1 init running complete.
Task 2 init running complete.
Task 3 init running complete.
Task 4 init running complete."""
    numbered = """1. Process 1TB server logs (parallel)
2. GeoIP lookup (parallel) [DEPENDS_ON: task 1]
3. Anomaly detection (sequential) [DEPENDS_ON: task 2]
4. Generate PDF report (sequential) [DEPENDS_ON: task 3]
4 actually runs in parallel with 3.
1: init running complete
2: init running complete
3: init running complete
4: init running complete"""
    for response in (task_step, numbered):
        result = OrchestrationPlugin().evaluate(response)
        tagging = next(item for item in result.rubric if item["name"] == "Explicit dependency tagging")
        labels = next(item for item in result.rubric if item["name"] == "Parallel vs sequential logic")
        assert tagging["earned"] == 0.0
        assert labels["earned"] == 0.0
    assert OrchestrationPlugin().score(task_step) == OrchestrationPlugin().score(numbered)


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


def test_tool_calling_synthesis_window_is_case_insensitive():
    # TC-1: the blocks regex counts closing tags case-insensitively, but the
    # synthesis window used a case-sensitive rfind for the lowercase closing
    # tag, so a response whose closing tags are all uppercase (a
    # </TOOL_CALL>-only response) lost all 4.0 Synthesis points.
    calls = [
        '<TOOL_CALL>{"name":"get_weather","args":{"location":"Tokyo","unit":"celsius"}}</TOOL_CALL>',
        '<TOOL_CALL>{"name":"search_flights","args":{"origin":"JFK","destination":"Tokyo","date":"2024-08-15"}}</TOOL_CALL>',
        '<TOOL_CALL>{"name":"book_hotel","args":{"city":"Tokyo","check_in":"2024-08-16","check_out":"2024-08-20","guests":2}}</TOOL_CALL>',
        '<TOOL_CALL>{"name":"get_stock_price","args":{"ticker":"SONY"}}</TOOL_CALL>',
        '<TOOL_CALL>{"name":"convert_currency","args":{"amount":1000,"from_curr":"USD","to_curr":"JPY"}}</TOOL_CALL>',
        '<TOOL_CALL>{"name":"send_email","args":{"to":"alice@example.com","subject":"Tokyo Trip Itinerary","body":"All set"}}</TOOL_CALL>',
    ]
    response = (
        "<plan>get_weather search_flights book_hotel get_stock_price convert_currency send_email</plan>\n"
        + "\n".join(calls)
        + "\nFinal: 22 celsius weather in Tokyo, flight from JFK, hotel reservation for 2 guests, "
        "SONY stock price 120.50, email sent to alice@example.com, converted 155000 JPY."
    )
    result = ToolCallingPlugin().evaluate(response)
    synthesis = next(item for item in result.rubric if item["name"] == "Synthesis / final response")
    assert synthesis["earned"] == 4.0
    assert result.score == 25.0
def test_tool_calling_plan_with_intro_prose_earns_planning_credit():
    # TC-2: intro prose before <plan> broke the fullmatch, so the 2.0
    # Planning credit was lost even when the plan named all six tools.
    calls = [
        '<tool_call>{"name":"get_weather","args":{"location":"Tokyo","unit":"celsius"}}</tool_call>',
        '<tool_call>{"name":"search_flights","args":{"origin":"JFK","destination":"Tokyo","date":"2024-08-15"}}</tool_call>',
        '<tool_call>{"name":"book_hotel","args":{"city":"Tokyo","check_in":"2024-08-16","check_out":"2024-08-20","guests":2}}</tool_call>',
        '<tool_call>{"name":"get_stock_price","args":{"ticker":"SONY"}}</tool_call>',
        '<tool_call>{"name":"convert_currency","args":{"amount":1000,"from_curr":"USD","to_curr":"JPY"}}</tool_call>',
        '<tool_call>{"name":"send_email","args":{"to":"alice@example.com","subject":"Tokyo Trip Itinerary","body":"All set"}}</tool_call>',
    ]
    response = (
        "I will plan first, then execute the six calls.\n"
        "<plan>get_weather search_flights book_hotel get_stock_price convert_currency send_email</plan>\n"
        + "\n".join(calls)
        + "\nFinal: 22 celsius weather in Tokyo, flight from JFK, hotel reservation for 2 guests, "
        "SONY stock price 120.50, email sent to alice@example.com, converted 155000 JPY."
    )
    result = ToolCallingPlugin().evaluate(response)
    planning = next(item for item in result.rubric if item["name"] == "Planning / reasoning")
    assert planning["earned"] == 2.0
    assert result.score == 25.0


def test_tool_calling_no_negative_finding_when_full_required_tools_credit():
    # TC-3: when all six required tools are called exactly once (here in the
    # wrong order), the Required-tools criterion earns the full 5.0, so the
    # "exactly one call" negative finding must not appear alongside full
    # credit (contradictory rubric evidence).
    wrong_order = [
        '<tool_call>{"name":"search_flights","args":{"origin":"JFK","destination":"Tokyo","date":"2024-08-15"}}</tool_call>',
        '<tool_call>{"name":"get_weather","args":{"location":"Tokyo","unit":"celsius"}}</tool_call>',
        '<tool_call>{"name":"book_hotel","args":{"city":"Tokyo","check_in":"2024-08-16","check_out":"2024-08-20","guests":2}}</tool_call>',
        '<tool_call>{"name":"get_stock_price","args":{"ticker":"SONY"}}</tool_call>',
        '<tool_call>{"name":"convert_currency","args":{"amount":1000,"from_curr":"USD","to_curr":"JPY"}}</tool_call>',
        '<tool_call>{"name":"send_email","args":{"to":"alice@example.com","subject":"Tokyo Trip Itinerary","body":"All set"}}</tool_call>',
    ]
    response = (
        "<plan>get_weather search_flights book_hotel get_stock_price convert_currency send_email</plan>\n"
        + "\n".join(wrong_order)
    )
    result = ToolCallingPlugin().evaluate(response)
    required = next(item for item in result.rubric if item["name"] == "Required tools present")
    assert required["earned"] == 5.0
    assert required["negative_findings"] == []

    # Negative control: dropping a required tool reduces credit and the
    # finding is emitted again.
    missing = wrong_order[:5]
    result_missing = ToolCallingPlugin().evaluate(
        "<plan>get_weather search_flights book_hotel get_stock_price convert_currency send_email</plan>\n"
        + "\n".join(missing)
    )
    required_missing = next(item for item in result_missing.rubric if item["name"] == "Required tools present")
    assert required_missing["earned"] < 5.0
    assert any("exactly one call" in finding["finding"] for finding in required_missing["negative_findings"])


def test_wireframes_require_distinct_canonical_screens():
    response = "## Focus\nPurpose: timer.\n[Button] Start\n## Focus Session\nPurpose: timer.\n[Button] Start\n## Calendar\nPurpose: events.\n[Button] Sync\n## Calendar Integration\nPurpose: events.\n[Button] Sync\n"
    result = WireframesPlugin().evaluate(response)
    screens = next(item for item in result.rubric if item["name"] == "Multiple screens present")
    assert screens["earned"] < screens["max"]


# A fully correct error-recovery response: concurrent provider attempts,
# error-payload/timeout/malformed handling, per-provider failure logging, and
# an AllProvidersFailedError carrying every provider's details.
ER_CORRECT_RESPONSE = '''```python
import asyncio
import logging

logger = logging.getLogger("weather")


class AllProvidersFailedError(Exception):
    """Raised when every weather provider fails."""


class WeatherClient:
    """Fetches weather from a single provider."""

    async def fetch(self, provider: str, city: str) -> dict:
        """Fetch one provider's weather payload."""
        raise NotImplementedError


async def get_weather_resilient(city: str, client: WeatherClient) -> dict:
    """Try every provider concurrently and return the first success.

    Treats an exception, a timeout, a malformed response, or a 200 response
    containing an error field as failure, logs every failure with the
    provider and reason, and raises AllProvidersFailedError with per-provider
    details when all providers fail.
    """
    providers = ("WeatherAPI", "OpenMeteo", "VisualCrossing")
    failures: dict[str, str] = {}

    async def attempt(provider: str) -> dict | None:
        try:
            value = await asyncio.wait_for(client.fetch(provider, city), timeout=5)
            if not isinstance(value, dict) or "error" in value:
                raise ValueError("malformed response or error payload")
            return value
        except Exception as exc:
            logger.error("provider %s failed: %s", provider, exc)
            failures[provider] = str(exc)
            return None

    results = await asyncio.gather(*(attempt(provider) for provider in providers))
    for provider, value in zip(providers, results, strict=False):
        if value is not None:
            return value
    raise AllProvidersFailedError("; ".join(f"{provider}: {failures[provider]}" for provider in providers))


async def demo() -> None:
    """Show all providers succeed, one provider fails, and all providers fail."""

    class DemoClient(WeatherClient):
        def __init__(self, mode: str) -> None:
            self.mode = mode

        async def fetch(self, provider: str, city: str) -> dict:
            if self.mode == "all-fail" or (self.mode == "partial" and provider == "WeatherAPI"):
                raise RuntimeError(provider + " is down")
            if self.mode == "payload" and provider == "WeatherAPI":
                return {"error": "rate limited"}
            return {"city": city, "temperature": 20}

    for mode in ("all-success", "partial", "all-fail"):
        try:
            print(mode, await get_weather_resilient("Paris", DemoClient(mode)))
        except AllProvidersFailedError as exc:
            print(mode, "failed:", exc)
```'''


def test_error_recovery_correct_response_scores_full():
    # Positive control for the split behavioral block (ER-1): a fully correct
    # response must still earn 20/20 — four 2.5-pt mode sub-criteria plus the
    # lexical criteria.
    assert ErrorRecoveryPlugin().score(ER_CORRECT_RESPONSE) == 20.0


def test_error_recovery_missing_one_mode_earns_partial_behavioral_credit():
    # Measured pre-fix (ER-1): the 10-pt behavioral block was all-or-nothing,
    # so a response correct in every mode except the all-failure exception
    # details (provider names missing from the message) lost all 10 points.
    # After the split only the all-failure sub-criterion is 0.0 and the other
    # three modes keep their 2.5 each (17.5 total, not ~10).
    response = ER_CORRECT_RESPONSE.replace(
        'raise AllProvidersFailedError("; ".join(f"{provider}: {failures[provider]}" for provider in providers))',
        'raise AllProvidersFailedError("all providers failed")',
    )
    result = ErrorRecoveryPlugin().evaluate(response)
    all_fail = next(item for item in result.rubric if item["name"] == "Behavioral all-failure mode")
    assert all_fail["earned"] == 0.0
    assert all_fail["negative_findings"]
    for name in ("Behavioral all-success mode", "Behavioral partial-failure mode", "Behavioral error-payload mode"):
        item = next(item for item in result.rubric if item["name"] == name)
        assert item["earned"] == 2.5, f"{name} should keep full credit, got {item['earned']}"
    assert result.score == 17.5


def test_error_recovery_no_executable_source_records_four_zero_mode_criteria():
    # The no-source branch must record the four split sub-criteria (not the
    # old single 10-pt "Behavioral provider tests" criterion), each 0.0 with
    # the "no executable source" finding.
    result = ErrorRecoveryPlugin().evaluate("```python\n```")
    behavioral = [item for item in result.rubric if item["name"].startswith("Behavioral ")]
    assert len(behavioral) == 4
    for item in behavioral:
        assert item["max"] == 2.5
        assert item["earned"] == 0.0
        assert any("no executable source" in finding["finding"] for finding in item["negative_findings"])
    assert result.score == 0.0


def test_error_recovery_concept_regex_covers_ensure_future_and_bare_gather():
    # Measured pre-fix (ER-2): concurrent implementations using ensure_future
    # or a bare gather (``from asyncio import gather``) were lexically
    # penalized — the concept regex only matched asyncio.-prefixed forms.
    pattern = _CONCEPT_PATTERNS["concurrent provider calls"]
    assert re.search(pattern, "from asyncio import gather\nawait gather(*tasks)", re.IGNORECASE)
    assert re.search(pattern, "from asyncio import ensure_future\nensure_future(attempt())", re.IGNORECASE)
    # The old asyncio.-prefixed forms must still match.
    assert re.search(pattern, "asyncio.gather(*tasks)", re.IGNORECASE)
    assert re.search(pattern, "asyncio.ensure_future(attempt())", re.IGNORECASE)
    # A non-concurrent mention of the word must not match.
    assert not re.search(pattern, "the providers gather their data sequentially", re.IGNORECASE)


def test_error_recovery_gather_only_concurrency_earns_full_recovery_design():
    # A response that imports gather bare (from asyncio import gather) must
    # earn the full Recovery design criterion (5/5 concepts), not 4 of 5.
    response = (
        "from asyncio import gather\n"
        "import logging\n"
        "logger = logging.getLogger()\n"
        "class AllProvidersFailedError(Exception):\n    pass\n"
        "class WeatherClient:\n    async def fetch(self, provider: str, city: str) -> dict:\n        ...\n"
        "async def get_weather_resilient(city: str, client: WeatherClient) -> dict:\n"
        '    """Try providers with a timeout; treat a malformed error payload as failure."""\n'
        "    try:\n"
        '        return await gather(client.fetch("WeatherAPI", city))\n'
        "    except Exception:\n"
        '        raise AllProvidersFailedError("fallback exhausted")\n'
        "async def demo() -> None:\n    ...\n"
    )
    result = ErrorRecoveryPlugin().evaluate(response)
    design = next(item for item in result.rubric if item["name"] == "Recovery design")
    assert design["earned"] == 2.0


def test_error_recovery_module_level_demo_network_call_is_blocked_not_hanging():
    # Measured pre-fix (ER-3): a module-level demo that makes a live network
    # call (``if __name__ == "__main__"`` fires inside the check script) hung
    # the local-restricted check until the 5s execution timeout — a 10-pt
    # swing for an otherwise correct implementation. The exec preamble now
    # blocks real sockets before the response source runs, so the call fails
    # fast with the sandbox marker and the harness still executes.
    guard = (
        'if __name__ == "__main__":\n'
        "    import urllib.request\n"
        "    try:\n"
        '        urllib.request.urlopen("http://weather.example.invalid/", timeout=1)\n'
        "    except Exception as exc:\n"
        '        print("demo probe failed:", exc)\n'
    )
    response = ER_CORRECT_RESPONSE.removesuffix("```") + guard + "```\n"
    result = ErrorRecoveryPlugin().evaluate(response)
    behavioral = [item for item in result.rubric if item["name"].startswith("Behavioral ")]
    assert sum(item["earned"] for item in behavioral) == 10.0
    # The sandbox socket block fired (not a real network failure): the
    # harness evidence carries the block's error marker.
    evidence = behavioral[0]["evidence"][0]
    assert "network access is disabled in the benchmark sandbox" in evidence["output"]
    # Positive control: a response without network side effects is unaffected
    # by the preamble (full behavioral credit, no block marker in evidence).
    base = ErrorRecoveryPlugin().evaluate(ER_CORRECT_RESPONSE)
    base_behavioral = [item for item in base.rubric if item["name"].startswith("Behavioral ")]
    assert sum(item["earned"] for item in base_behavioral) == 10.0
    assert "network access is disabled in the benchmark sandbox" not in base_behavioral[0]["evidence"][0]["output"]


def _demo_scenario_response(labels: str) -> str:
    # A minimal response whose demo() docstring carries the given scenario
    # labels; the "Demo scenarios" criterion is text-based, so this isolates
    # the marker regex from the execution/behavioral criteria.
    return (
        "class AllProvidersFailedError(Exception):\n    pass\n"
        "class WeatherClient:\n    async def fetch(self, provider, city):\n        return {}\n"
        "async def get_weather_resilient(city: str, client: WeatherClient) -> dict:\n    return {}\n"
        # No ``-> None`` hint: the pre-existing partial marker ``one`` would
        # otherwise match the "one" inside "None" and mask the negative case.
        f"async def demo():\n    \"\"\"Demo: {labels}.\"\"\"\n"
    )


def test_error_recovery_hyphenated_demo_labels_earn_full_demo_credit():
    # Measured pre-fix (ER-4): the prompt tells responses to show
    # "all-success, partial-failure, and all-failure" scenarios, but the
    # success/failure marker regexes required whitespace + verb forms
    # (succeed/fail), so a response using the prompt's own hyphenated noun
    # labels earned only the partial marker (1/3, stored as 0.3 after the
    # rubric's 1-decimal rounding) instead of 1.0.
    result = ErrorRecoveryPlugin().evaluate(_demo_scenario_response("all-success, partial-failure, all-failure"))
    demo = next(item for item in result.rubric if item["name"] == "Demo scenarios")
    assert demo["earned"] == 1.0


def test_error_recovery_whitespace_demo_labels_still_earn_full_demo_credit():
    # Positive control: the original whitespace + verb-form labels must keep
    # earning full credit (no regression from the hyphenated extension).
    result = ErrorRecoveryPlugin().evaluate(_demo_scenario_response("all succeed, partial, all fail"))
    demo = next(item for item in result.rubric if item["name"] == "Demo scenarios")
    assert demo["earned"] == 1.0


def test_error_recovery_no_demo_labels_earn_no_demo_credit():
    # Negative control: a demo with none of the scenario labels earns 0.
    result = ErrorRecoveryPlugin().evaluate(_demo_scenario_response("happy path only"))
    demo = next(item for item in result.rubric if item["name"] == "Demo scenarios")
    assert demo["earned"] == 0.0


def _signature_response(client_annotation: str) -> str:
    # A minimal response with all four signature hits; only the client
    # annotation form varies.
    return (
        "class AllProvidersFailedError(Exception):\n    pass\n"
        "class WeatherClient:\n    async def fetch(self, provider, city):\n        return {}\n"
        f"async def get_weather_resilient(city: str, client: {client_annotation}) -> dict:\n    return {{}}\n"
        "async def demo():\n    ...\n"
    )


def test_error_recovery_string_literal_forward_ref_earns_full_signature_credit():
    # Measured pre-fix (ER-5): a string-literal forward reference
    # (``client: "WeatherClient"``) parsed as a string Constant, not a Name,
    # so the signature check (isinstance(annotation, ast.Name)) failed and the
    # response lost the gr_sig hit (3/4 = 1.5 instead of 2.0).
    result = ErrorRecoveryPlugin().evaluate(_signature_response('"WeatherClient"'))
    sig = next(item for item in result.rubric if item["name"] == "Typed injectable signatures")
    assert sig["earned"] == 2.0


def test_error_recovery_name_annotation_still_earns_full_signature_credit():
    # Positive control: a plain Name annotation must keep earning full credit.
    result = ErrorRecoveryPlugin().evaluate(_signature_response("WeatherClient"))
    sig = next(item for item in result.rubric if item["name"] == "Typed injectable signatures")
    assert sig["earned"] == 2.0


@pytest.mark.parametrize("annotation", ["int", '"int"'])
def test_error_recovery_wrong_annotation_loses_signature_credit(annotation):
    # Negative control: a wrong annotation type must not earn the gr_sig hit.
    # The string form (``"int"``) exercises the new string-Constant branch.
    result = ErrorRecoveryPlugin().evaluate(_signature_response(annotation))
    sig = next(item for item in result.rubric if item["name"] == "Typed injectable signatures")
    assert sig["earned"] == 1.5


def test_error_recovery_fake_mode_markers_without_harness_earn_no_behavioral_credit():
    # Measured pre-fix (ER-6): a response that prints fake MODE_RESULT PASS
    # markers at module level and then exits early (sys.exit) was scored on its
    # own markers -- the harness never ran, but _mode_results parsed the fake
    # markers and awarded the full 10-pt behavioral block. The behavioral
    # criteria are now gated on execution.harness_ok (the completion sentinel),
    # so an early exit that skips the harness earns no behavioral credit.
    response = (
        "import sys\n"
        'print("MODE_RESULT all-success PASS")\n'
        'print("MODE_RESULT partial PASS")\n'
        'print("MODE_RESULT payload PASS")\n'
        'print("MODE_RESULT all-fail PASS")\n'
        "sys.exit(0)\n"
    )
    result = ErrorRecoveryPlugin().evaluate(response)
    behavioral = [item for item in result.rubric if item["name"].startswith("Behavioral ")]
    assert len(behavioral) == 4
    assert sum(item["earned"] for item in behavioral) == 0.0
    # Every criterion names the missing completion sentinel (not a per-mode
    # failure); ``all`` is the stronger pin since each 0.0 criterion carries one.
    assert all(
        "harness did not complete" in finding["finding"]
        for item in behavioral
        for finding in item["negative_findings"]
    )


@pytest.mark.parametrize("plugin", [ErrorRecoveryPlugin, EventProcessorPlugin])
def test_executable_plugins_do_not_credit_stub_sources(plugin):
    assert plugin().score("class Placeholder:\n    pass") < 12.0
