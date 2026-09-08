"""Typed multi-turn conversation and state-transition challenge."""
from __future__ import annotations

import json
import re
from typing import Any

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from benchmark.types import ConfigMap
from plugins.challenges._analysis import exact_section, fenced_blocks, markdown_sections
from plugins.challenges._rubric import Rubric


class MultiTurnConversationPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "multi-turn-conversation"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def name(self) -> str:
        return "Multi-Turn Conversation"

    @property
    def max_score(self) -> int:
        return int(20.0)

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_prompt(self) -> str:
        return (
            "Simulate this three-turn assistant conversation as a stateful agent. Return exactly "
            "one fenced JSON object under each heading and no prose outside the objects.\n\n"
            "## Turn 1\nUser: Schedule a focus block at 09:00 for 25 minutes, with music enabled "
            "and a calendar event.\n\n"
            "## Turn 2\nUser: Keep the 09:00 calendar event, disable music, and add the `deep-work` label.\n\n"
            "## Turn 3\nUser: Keep all prior constraints, change the duration to 50 minutes, and notify me "
            "5 minutes before it starts.\n\n"
            "## State Summary\nSummarize exactly what changed from Turn 1 to Turn 2 and Turn 2 to Turn 3.\n\n"
            "Each state object must contain exactly these keys: start (HH:MM string), "
            "duration_minutes (integer), music (boolean), calendar_event (boolean), labels "
            "(array of strings), notification_minutes (integer or null), and changes (array of strings)."
        )

    def get_temperature(self, global_config: ConfigMap) -> float | None:
        val = global_config.get("multi_turn_conversation_temperature")
        return float(val) if isinstance(val, (int, float)) else None

    @staticmethod
    def _state(section: Any | None) -> dict[str, Any] | None:
        if section is None:
            return None
        blocks = fenced_blocks(section.body, "json")
        if len(blocks) != 1:
            return None
        try:
            value = json.loads(blocks[0])
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _negated_before(text: str, start: int) -> bool:
        # A negation token within ~20 chars before the claim makes it a
        # true statement ("did not enable music"), not a false claim.
        window = text[max(0, start - 20):start]
        return bool(re.search(r"\b(?:not|no|never|without)\b|n't", window))

    def evaluate(self, response_text: str) -> EvaluationResult:
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        if not text:
            return rubric.results()
        sections = markdown_sections(text)
        rubric.record_validation(type("Validation", (), {
            "valid": len(sections) == 4,
            "evidence": [{"kind": "section", "heading": section.heading} for section in sections],
            "errors": [] if len(sections) == 4 else ["exactly four conversation sections are required"],
        })())
        turns = [
            exact_section(text, "Turn 1"),
            exact_section(text, "Turn 2"),
            exact_section(text, "Turn 3"),
        ]
        summary = exact_section(text, "State Summary")
        states = [self._state(section) for section in turns]
        required_keys = {
            "start", "duration_minutes", "music", "calendar_event", "labels",
            "notification_minutes", "changes",
        }
        typed = 0
        for state in states:
            if not isinstance(state, dict) or set(state) != required_keys:
                continue
            if (
                isinstance(state["start"], str)
                and isinstance(state["duration_minutes"], int)
                and type(state["music"]) is bool
                and type(state["calendar_event"]) is bool
                and isinstance(state["labels"], list)
                and all(isinstance(label, str) for label in state["labels"])
                and (state["notification_minutes"] is None or isinstance(state["notification_minutes"], int))
                and isinstance(state["changes"], list)
                and all(isinstance(change, str) for change in state["changes"])
            ):
                typed += 1
        rubric.add_criterion(
            "Typed state objects", 5.0, 5.0 * typed / 3.0,
            evidence=[{"kind": "state", "turn": index + 1} for index, state in enumerate(states) if state],
            negative_findings=[] if typed == 3 else [{"finding": "each turn needs exactly the typed state keys"}],
        )
        expected = [
            {"start": "09:00", "duration_minutes": 25, "music": True, "calendar_event": True, "notification_minutes": None},
            {"start": "09:00", "duration_minutes": 25, "music": False, "calendar_event": True, "notification_minutes": None},
            {"start": "09:00", "duration_minutes": 50, "music": False, "calendar_event": True, "notification_minutes": 5},
        ]
        # Per-turn label expectation: Turn 1 must not carry the deep-work
        # label yet; Turns 2 and 3 must carry it.
        label_required = [False, True, True]
        state_values: list[dict[str, Any]] = []
        for state in states:
            if isinstance(state, dict):
                state_values.append(state)
        transition_hits = 0
        for state, target, require_label in zip(state_values, expected, label_required, strict=False):
            hits = sum(state.get(key) == value for key, value in target.items())
            labels = state.get("labels")
            if isinstance(labels, list) and ("deep-work" in labels) is require_label:
                hits += 1
            transition_hits += hits
        rubric.add_criterion(
            "Requested state values", 5.0, 5.0 * transition_hits / 18.0,
            negative_findings=[] if transition_hits == 18 else [
                {"finding": "each turn must carry its own expected state, not the final state"}
            ],
        )
        preserved = (
            len(state_values) == 3
            and state_values[0].get("start") == state_values[1].get("start") == state_values[2].get("start") == "09:00"
            and state_values[0].get("calendar_event") is state_values[1].get("calendar_event") is state_values[2].get("calendar_event") is True
            and state_values[0].get("music") is True
            and state_values[1].get("music") is False
            and state_values[2].get("music") is False
            and isinstance(state_values[0].get("labels"), list)
            and "deep-work" not in state_values[0].get("labels", [])
            and "deep-work" in state_values[1].get("labels", [])
            and "deep-work" in state_values[2].get("labels", [])
        )
        rubric.add_criterion("State preservation and updates", 5.0, 5.0 if preserved else 0.0,
                             negative_findings=[] if preserved else [{"finding": "later turns must preserve prior state while applying only requested changes"}])
        summary_text = (summary.body if summary else "").lower()
        summary_patterns = (
            r"turn\s*1", r"turn\s*2", r"disabl\w*", r"music", r"turn\s*3", r"50", r"5 minutes",
        )
        summary_ok = all(re.search(pattern, summary_text) for pattern in summary_patterns)
        # The summary must match the transitions that actually happened:
        # music was disabled (Turn 1 -> 2) and the duration was lengthened
        # from 25 to 50 minutes (Turn 2 -> 3). A negated or inverted claim
        # voids the summary credit even when every marker is present.
        # The guards are negation-aware (a true negated statement such as
        # "did not enable music" is not a false claim) and the negation
        # guard is object-aware (the disabl word must be about the music).
        false_claims: list[str] = []
        for match in re.finditer(
            r"\b(?:not|no|never|without|didn'?t|don'?t|doesn'?t|wasn'?t|isn'?t|cannot|can'?t)\b[^.\n]{0,15}\b(?P<disabl>disabl\w*)",
            summary_text,
        ):
            # Object-aware: "music" must sit within ~20 chars of the disabl
            # word, so "did not disable the calendar event" is not counted.
            start = match.start("disabl")
            around = summary_text[max(0, start - 20):start + 20]
            if re.search(r"\bmusic\b", around):
                false_claims.append("the music-disable transition is negated")
                break
        for match in re.finditer(
            r"\benabl\w*\s+(?:the\s+)?music\b|\bmusic\s+(?:was\s+|is\s+|were\s+)?enabl\w*",
            summary_text,
        ):
            if not self._negated_before(summary_text, match.start()):
                false_claims.append("music was enabled, not disabled")
                break
        for match in re.finditer(
            r"\b(?:shorten\w*|reduc\w*|decreas\w*|cut\w*)\b[^.\n]{0,40}\b(?:duration|length|minute\w*)",
            summary_text,
        ):
            if not self._negated_before(summary_text, match.start()):
                false_claims.append("the duration was lengthened, not shortened")
                break
        summary_ok = summary_ok and not false_claims
        rubric.add_criterion(
            "State-change summary", 2.0, 2.0 if summary_ok else 0.0,
            negative_findings=[{"finding": claim} for claim in false_claims],
        )
        structure_ok = (
            len(sections) == 4
            and all(section is not None and len(fenced_blocks(section.body, "json")) == 1 for section in turns)
            and summary is not None
            and all(
                (section.body.strip().lower().startswith("```json")  # type: ignore[union-attr]
                and section.body.strip().endswith("```"))  # type: ignore[union-attr]
                for section in turns
            )
        )
        rubric.add_criterion("Conversation response contract", 3.0, 3.0 if structure_ok else 0.0)
        return rubric.results()

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
