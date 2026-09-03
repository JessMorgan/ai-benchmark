"""Tests for the rewritten debugging and conversation challenges."""
from plugins.challenges.debug_consistency import DebugConsistencyPlugin
from plugins.challenges.debug_traversal import DebugTraversalPlugin
from plugins.challenges.error_recovery import ErrorRecoveryPlugin
from plugins.challenges.multi_turn_conversation import MultiTurnConversationPlugin


def test_debug_traversal_prompt_contains_real_threshold_bug():
    plugin = DebugTraversalPlugin()
    assert plugin.version == "1.2.0"
    assert "count > 2" in plugin.get_prompt()


def test_debug_consistency_prompt_requires_evidence_not_a_patch():
    plugin = DebugConsistencyPlugin()
    assert plugin.version == "0.1.0"
    assert "do not invent" in plugin.get_prompt().lower()


def test_debug_consistency_correct_response_scores_high():
    response = """## Reproduction
The two records for abc produce ['abc'].
## Consistency Check
The report is not reproducible: the code correctly returns abc as a duplicate.
## Diagnosis
There is no demonstrated code bug; the report and supplied behavior are inconsistent.
## Evidence Needed
Collect the exact input, runtime version, logs, and a reproduction trace.
## Recommendation
Do not patch code until the observed input and environment are verified.
"""
    assert DebugConsistencyPlugin().score(response) >= 18.0


def test_debug_traversal_correct_response_scores_high():
    response = """## Root Cause
The comparison uses > 2 instead of >= 2, so exactly two abc123 entries are rejected.
## Analysis
abc123 has count=2, def456 has count=1, so the buggy code returns an empty list.
## Fix
```python
def find_duplicate_users(log_entries):
    counts = {}
    for entry in log_entries:
        uid = entry.get('user_id')
        counts[uid] = counts.get(uid, 0) + 1
    return [uid for uid, count in counts.items() if count >= 2]
```
## Test
```python
def test_duplicates():
    assert find_duplicate_users([{'user_id': 'abc123'}, {'user_id': 'abc123'}]) == ['abc123']
```
## Side Effects
A list preserves deterministic order; empty IDs should be validated or ignored explicitly.
"""
    assert DebugTraversalPlugin().score(response) >= 18.0


def test_error_recovery_missing_injection_is_not_full_credit():
    response = """```python
class AllProvidersFailedError(Exception):
    pass
async def get_weather_resilient(city: str) -> dict:
    raise RuntimeError('bad')
```"""
    assert ErrorRecoveryPlugin().score(response) < 15.0


def test_multi_turn_requires_typed_state_transitions():
    response = """## Turn 1
```json
{"start":"09:00","duration_minutes":25,"music":true,"calendar_event":true,"labels":[],"notification_minutes":null,"changes":[]}
```
## Turn 2
```json
{"start":"09:00","duration_minutes":25,"music":false,"calendar_event":true,"labels":["deep-work"],"notification_minutes":null,"changes":["disabled music","added deep-work label"]}
```
## Turn 3
```json
{"start":"09:00","duration_minutes":50,"music":false,"calendar_event":true,"labels":["deep-work"],"notification_minutes":5,"changes":["changed duration to 50","added five-minute notification"]}
```
## State Summary
Turn 1 to Turn 2 disabled music and added the deep-work label. Turn 2 to Turn 3 changed duration to 50 and added a 5 minutes notification.
"""
    assert MultiTurnConversationPlugin().score(response) == 20.0
