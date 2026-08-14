"""Tests for structured wireframe scoring."""
from plugins.challenges.wireframes import WireframesPlugin


def full_response():
    return """## Dashboard
Purpose: show today's schedule.
```text
[Header] FlowState [Card] Focus [Button] Start [Nav] Focus
```
Note: tapping Start opens Focus Session.
## Focus Session
Purpose: run the focus timer.
```text
[Timer] 25:00 [Button] Pause [Slider] Music
```
Note: tapping Pause stops the session.
## Calendar Integration
Purpose: manage calendar events.
```text
[List] Events [Toggle] Sync [Button] Import
```
Note: tapping Import loads events.
## AI Planning
Purpose: generate tomorrow's schedule.
```text
[Card] Plan [Button] Apply [List] Blocks
```
Note: tapping Apply writes the schedule.
## Settings
Purpose: configure music and notifications.
```text
[Toggle] Notifications [Button] Connect [Nav] Calendar
```
Note: tapping Connect opens Calendar Integration.
## Navigation
Dashboard -> Focus Session
Dashboard -> Calendar Integration
Dashboard -> AI Planning
Settings -> Calendar Integration
"""


def test_empty_and_whitespace_score_zero():
    plugin = WireframesPlugin()
    assert plugin.score("") == 0.0
    assert plugin.score(" \n") == 0.0


def test_complete_distinct_wireframes_score_high():
    assert WireframesPlugin().score(full_response()) >= 18.0


def test_four_empty_or_duplicate_screens_do_not_get_full_screen_credit():
    response = "## Dashboard\n## Dashboard\n## Dashboard\n## Dashboard\n"
    result = WireframesPlugin().evaluate(response)
    screens = next(item for item in result.rubric if item["name"] == "Multiple screens present")
    assert screens["earned"] < screens["max"]


def test_navigation_requires_known_screen_edges():
    result = WireframesPlugin().evaluate("## Dashboard\nPurpose: home.\n```text\n[Button] Start\n```")
    navigation = next(item for item in result.rubric if item["name"] == "Navigation flows")
    assert navigation["earned"] == 0.0
