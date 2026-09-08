"""Tests for structured wireframe scoring."""
from plugins.challenges.wireframes import WireframesPlugin


def full_response():
    return """## Dashboard
Purpose: show today's schedule.
+--------------------------+
| [Header] FlowState       |
| [Card] Focus [Button]    |
+--------------------------+
Note: tapping Start opens Focus Session.
## Focus Session
Purpose: run the focus timer.
+--------------------------+
| [Timer] 25:00            |
| [Button] Pause           |
+--------------------------+
Note: tapping Pause stops the session.
## Calendar Integration
Purpose: manage calendar events.
+--------------------------+
| [List] Events            |
| [Toggle] Sync            |
+--------------------------+
Note: tapping Import loads events.
## AI Planning
Purpose: generate tomorrow's schedule.
+--------------------------+
| [Card] Plan              |
| [Button] Apply           |
+--------------------------+
Note: tapping Apply writes the schedule.
## Settings
Purpose: configure app settings for music and notifications.
+--------------------------+
| [Toggle] Notifications   |
| [Button] Connect         |
+--------------------------+
Note: tapping Connect opens Calendar Integration.
## Navigation
Dashboard -> Focus Session
Dashboard -> Calendar Integration
Dashboard -> AI Planning
Settings -> Calendar Integration
"""


def soup_response():
    """Keyword-soup gaming response: stuffed component/feature keywords,
    no real wireframe structure, and position "matches" that only exist as
    substrings ("s**top**s")."""
    return """## Dashboard
Purpose: overview of the app.
[Button] Start [Card] Summary [List] Items [Nav] Menu
Note: tapping Start stops the flow.
## Focus Session
Purpose: run a focus session.
[Timer] 25:00 [Slider] Level [Icon] Dot
Note: tapping Pause stops the timer.
## Calendar Integration
Purpose: sync the calendar.
[Toggle] Sync [Input] Date [Modal] Picker
Note: tapping Sync stops the refresh.
## AI Planning
Purpose: plan the schedule.
planning blocks and schedule items with a timer session.
Note: when user taps apply.
## Settings
Purpose: tune settings.
options for notifications and preferences.
Note: on tap to save.
## Navigation
Dashboard -> Focus Session
Dashboard -> Calendar Integration
Settings -> AI Planning
"""


def test_empty_and_whitespace_score_zero():
    plugin = WireframesPlugin()
    assert plugin.score("") == 0.0
    assert plugin.score(" \n") == 0.0


def test_complete_distinct_wireframes_score_high():
    # Box-drawing evidence lifts the correct response to 20/20; 19 is the
    # HIGH floor, strictly above the pre-fix 18/20 ceiling.
    assert WireframesPlugin().score(full_response()) >= 19.0


def test_correct_wireframe_scores_higher_than_keyword_soup():
    plugin = WireframesPlugin()
    correct = plugin.score(full_response())
    soup = plugin.score(soup_response())
    assert correct > soup


def test_box_drawing_lines_count_as_wireframe_evidence():
    response = (
        "## Dashboard\n"
        "Purpose: overview.\n"
        "+------------------+\n"
        "| FlowState Dashboard |\n"
        "+------------------+\n"
        "## Focus Session\n"
        "Purpose: timer.\n"
        "┌──────────────────┐\n"
        "│ [Timer] 25:00    │\n"
        "└──────────────────┘\n"
    )
    result = WireframesPlugin().evaluate(response)
    visual = next(item for item in result.rubric if item["name"] == "Visual/structural wireframe")
    assert visual["earned"] >= 2.0


def test_stops_substring_does_not_match_top_position():
    result = WireframesPlugin().evaluate(
        "## Dashboard\nPurpose: home.\n[Button] Start\nNote: tapping Pause stops the session.\n"
    )
    visual = next(item for item in result.rubric if item["name"] == "Visual/structural wireframe")
    assert visual["earned"] == 0.0


def test_four_empty_or_duplicate_screens_do_not_get_full_screen_credit():
    response = "## Dashboard\n## Dashboard\n## Dashboard\n## Dashboard\n"
    result = WireframesPlugin().evaluate(response)
    screens = next(item for item in result.rubric if item["name"] == "Multiple screens present")
    assert screens["earned"] < screens["max"]


def test_navigation_requires_known_screen_edges():
    result = WireframesPlugin().evaluate("## Dashboard\nPurpose: home.\n```text\n[Button] Start\n```")
    navigation = next(item for item in result.rubric if item["name"] == "Navigation flows")
    assert navigation["earned"] == 0.0


def test_self_loop_edges_earn_nothing():
    response = (
        "## Dashboard\nPurpose: home.\n"
        "## Focus Session\nPurpose: timer.\n"
        "## Calendar Integration\nPurpose: events.\n"
        "## AI Planning\nPurpose: plan.\n"
        "Dashboard -> Dashboard\n"
        "Dashboard -> Dashboard\n"
        "Dashboard -> Dashboard\n"
    )
    result = WireframesPlugin().evaluate(response)
    navigation = next(item for item in result.rubric if item["name"] == "Navigation flows")
    assert navigation["earned"] == 0.0


def test_duplicate_edges_count_once():
    response = (
        "## Dashboard\nPurpose: home.\n"
        "## Focus Session\nPurpose: timer.\n"
        "## Calendar Integration\nPurpose: events.\n"
        "## AI Planning\nPurpose: plan.\n"
        "Dashboard -> Focus Session\n"
        "Dashboard -> Focus Session\n"
        "Dashboard -> Focus Session\n"
        "Dashboard -> Calendar Integration\n"
    )
    result = WireframesPlugin().evaluate(response)
    navigation = next(item for item in result.rubric if item["name"] == "Navigation flows")
    assert navigation["earned"] == 2.0


def test_prd_feature_criterion_ignores_features_the_prompt_does_not_name():
    # The prompt names focus, calendar, planning/schedule, session/timer, and
    # settings — not music. Mentioning music must not substitute for a
    # missing named feature (calendar here).
    response = (
        "## Dashboard\nPurpose: focus and planning.\n"
        "## Focus Session\nPurpose: timer session.\n"
        "## AI Planning\nPurpose: schedule.\n"
        "## Settings\nPurpose: settings and music.\n"
    )
    result = WireframesPlugin().evaluate(response)
    features = next(item for item in result.rubric if item["name"] == "Coverage of PRD features")
    assert features["earned"] == 0.5


def test_navigation_and_tabletop_do_not_match_component_keywords():
    response = (
        "## Dashboard\nPurpose: the navigation panel.\n"
        "## Focus Session\nPurpose: a tabletop surface.\n"
        "## Calendar Integration\nPurpose: navigation again.\n"
        "## AI Planning\nPurpose: tabletop again.\n"
    )
    result = WireframesPlugin().evaluate(response)
    components = next(item for item in result.rubric if item["name"] == "Key UI components")
    assert components["earned"] == 0.0


def test_partial_credit_cannot_reach_full_without_passing_gate():
    # Five screens, but only four carry a purpose and visual evidence: the
    # partial-credit path must stay strictly below the full score.
    lines = []
    for index, name in enumerate(("Dashboard", "Focus Session", "Calendar Integration", "AI Planning", "Settings")):
        lines.append(f"## {name}")
        if index < 4:
            lines.append("Purpose: the top bar.")
            lines.append("[Button] Action")
        else:
            lines.append("Plain text only.")
    result = WireframesPlugin().evaluate("\n".join(lines))
    by_name = {item["name"]: item for item in result.rubric}
    assert by_name["Screen names and purposes"]["earned"] < by_name["Screen names and purposes"]["max"]
    assert by_name["Visual/structural wireframe"]["earned"] < by_name["Visual/structural wireframe"]["max"]
