"""Regression tests for the multi-turn-conversation challenge plugin."""
import json

from plugins.challenges.multi_turn_conversation import MultiTurnConversationPlugin

CORRECT_SUMMARY = (
    "Turn 1 to Turn 2 disabled music and added the deep-work label. "
    "Turn 2 to Turn 3 changed duration to 50 and added a 5 minutes notification."
)

TURN_1 = {
    "start": "09:00", "duration_minutes": 25, "music": True, "calendar_event": True,
    "labels": [], "notification_minutes": None, "changes": [],
}
TURN_2 = {
    "start": "09:00", "duration_minutes": 25, "music": False, "calendar_event": True,
    "labels": ["deep-work"], "notification_minutes": None,
    "changes": ["disabled music", "added deep-work label"],
}
TURN_3 = {
    "start": "09:00", "duration_minutes": 50, "music": False, "calendar_event": True,
    "labels": ["deep-work"], "notification_minutes": 5,
    "changes": ["changed duration to 50", "added five-minute notification"],
}


def build_response(turn1: dict, turn2: dict, turn3: dict, summary: str) -> str:
    def block(state: dict) -> str:
        return "```json\n" + json.dumps(state) + "\n```"
    return (
        "## Turn 1\n" + block(turn1) + "\n"
        "## Turn 2\n" + block(turn2) + "\n"
        "## Turn 3\n" + block(turn3) + "\n"
        "## State Summary\n" + summary + "\n"
    )


def test_multi_turn_correct_response_scores_full():
    assert MultiTurnConversationPlugin().score(
        build_response(TURN_1, TURN_2, TURN_3, CORRECT_SUMMARY)
    ) == 20.0


def test_multi_turn_verbatim_final_state_scores_low():
    # Measured gaming case: the verbatim Turn-3 state copied into every turn
    # scored 18.3/20 because the rubric graded every turn against the final
    # state. Each turn must be graded against its own expected state (Turn 1
    # has music on, 25 minutes, and no labels).
    response = build_response(TURN_3, TURN_3, TURN_3, CORRECT_SUMMARY)
    assert MultiTurnConversationPlugin().score(response) < 15.0
