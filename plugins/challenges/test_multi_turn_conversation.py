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


def test_multi_turn_false_summary_earns_no_summary_credit():
    # Measured gaming case: a summary that negates the actual transitions
    # ("did not disable music", "music was enabled", "shortened the
    # duration") still contained every required marker and scored 20/20.
    false_summary = (
        "Turn 1 to Turn 2 did not disable music; music was enabled instead. "
        "Turn 2 to Turn 3 shortened the duration to 50 and added a 5 minutes notification."
    )
    response = build_response(TURN_1, TURN_2, TURN_3, false_summary)
    result = MultiTurnConversationPlugin().evaluate(response)
    summary_criterion = next(
        item for item in result.rubric if item["name"] == "State-change summary"
    )
    assert summary_criterion["earned"] == 0.0
    assert result.score < 20.0


def test_multi_turn_true_negated_statements_not_flagged():
    # Regression (MC-2 review, [major]): the enable/shortened guards were
    # negation-blind and flagged TRUE statements — "did not enable music"
    # (the music was disabled) and "did not shorten the duration" (the
    # duration was lengthened) each cost the 2.0 summary credit, so a
    # correct response scored 18.0 instead of 20.0.
    true_negated_summary = (
        "Turn 1 to Turn 2 did not enable music; the music was disabled and "
        "the deep-work label was added. Turn 2 to Turn 3 did not shorten the "
        "duration; it changed to 50 and added a 5 minutes notification."
    )
    response = build_response(TURN_1, TURN_2, TURN_3, true_negated_summary)
    result = MultiTurnConversationPlugin().evaluate(response)
    summary_criterion = next(
        item for item in result.rubric if item["name"] == "State-change summary"
    )
    assert summary_criterion["earned"] == 2.0
    assert result.score == 20.0


def test_multi_turn_negation_guard_is_object_aware():
    # Regression (MC-2 review, [major]): the negation guard flagged any
    # negated disabl word, so a true "did not disable the calendar event"
    # (the event was kept) was flagged as a music-disable negation. The
    # guard must require "music" near the disabl word.
    object_aware_summary = (
        "Turn 1 to Turn 2 disabled music and added the deep-work label; "
        "it did not disable the calendar event. Turn 2 to Turn 3 changed "
        "duration to 50 and added a 5 minutes notification."
    )
    response = build_response(TURN_1, TURN_2, TURN_3, object_aware_summary)
    result = MultiTurnConversationPlugin().evaluate(response)
    summary_criterion = next(
        item for item in result.rubric if item["name"] == "State-change summary"
    )
    assert summary_criterion["earned"] == 2.0
    assert result.score == 20.0


def test_multi_turn_accepts_five_minute_paraphrase():
    # Measured: a correct response that writes the notification as
    # "5-minute" in the summary lost the 2.0 summary credit because the
    # marker was the literal string "5 minutes". The 5[\s-]minutes?
    # paraphrase must be accepted as equivalent.
    paraphrase_summary = (
        "Turn 1 to Turn 2 disabled music and added the deep-work label. "
        "Turn 2 to Turn 3 changed duration to 50 and added a 5-minute notification."
    )
    response = build_response(TURN_1, TURN_2, TURN_3, paraphrase_summary)
    assert MultiTurnConversationPlugin().score(response) == 20.0
