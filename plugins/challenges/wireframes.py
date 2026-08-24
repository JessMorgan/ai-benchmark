"""Structured text-wireframe challenge."""
from __future__ import annotations

import re
from typing import Any

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from benchmark.types import ConfigMap
from plugins.challenges._analysis import markdown_sections, normalize_heading
from plugins.challenges._rubric import Rubric


class WireframesPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "wireframes"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def name(self) -> str:
        return "Wireframes"

    @property
    def max_score(self) -> int:
        return int(20.0)

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_prompt(self) -> str:
        return (
            "Create at least four distinct mobile wireframe sections for FlowState: Dashboard, "
            "Focus Session, Calendar Integration, AI Planning, and Settings. Each section must "
            "have a heading naming the screen, a purpose, structural ASCII/component-list content, "
            "and key controls. Include explicit navigation edges between named screens and notes "
            "for important interactions."
        )

    def get_temperature(self, global_config: ConfigMap) -> float | None:
        val = global_config.get("wireframes_temperature")
        return float(val) if isinstance(val, (int, float)) else None

    @staticmethod
    def _screen_name(heading: str) -> str | None:
        normalized = normalize_heading(heading)
        for name in ("dashboard", "focus session", "calendar integration", "ai planning", "settings"):
            if re.search(rf"\b{re.escape(name)}\b", normalized):
                return name
        aliases = {
            "focus": "focus session",
            "calendar": "calendar integration",
            "planning": "ai planning",
        }
        for alias, canonical in aliases.items():
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                return canonical
        return None

    def evaluate(self, response_text: str) -> EvaluationResult:
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        if not text:
            return rubric.results()
        sections = markdown_sections(text)
        screens = [(self._screen_name(section.heading), section) for section in sections]
        screens = [(name, section) for name, section in screens if name]
        unique: dict[str, Any] = {}
        for name, section in screens:
            assert name is not None
            unique.setdefault(name, section)
        screen_count = len(unique)
        rubric.record_validation(type("Validation", (), {
            "valid": screen_count >= 4,
            "evidence": [{"kind": "screen", "name": name, "chars": len(section.body)} for name, section in unique.items()],
            "errors": [] if screen_count >= 4 else ["fewer than four distinct named screens"],
        })())
        rubric.add_criterion("Multiple screens present", 3.0, 3.0 if screen_count >= 4 else screen_count * 0.75)
        purpose_count = sum(bool(re.search(r"purpose|goal|shows|used to", section.body, re.IGNORECASE)) for section in unique.values())
        rubric.add_criterion("Screen names and purposes", 3.0, 3.0 if screen_count >= 4 and purpose_count == screen_count else min(3.0, purpose_count * 0.75))
        visual_count = sum(bool(re.search(r"```|[┌┐└┘├┤┬┴│─]|\[[^\]]+\]", section.body)) and bool(re.search(r"top|bottom|header|footer|left|right|middle|position", section.body, re.IGNORECASE)) for section in unique.values())
        rubric.add_criterion("Visual/structural wireframe", 4.0, 4.0 if screen_count and visual_count == screen_count else min(4.0, visual_count))
        component_count = sum(bool(re.search(r"button|card|list|nav|menu|tab|modal|input|icon|timer|slider|toggle", section.body, re.IGNORECASE)) for section in unique.values())
        rubric.add_criterion("Key UI components", 4.0, 4.0 if screen_count >= 4 and component_count >= 4 else min(4.0, component_count))
        edges = re.findall(r"([A-Za-z][A-Za-z ]{1,30})\s*(?:->|→|=>)\s*([A-Za-z][A-Za-z ]{1,30})", text)
        known = set(unique)
        valid_edges = [(left.strip().lower(), right.strip().lower()) for left, right in edges if any(name in left.lower() for name in known) and any(name in right.lower() for name in known)]
        rubric.add_criterion("Navigation flows", 3.0, 3.0 if len(valid_edges) >= 3 else float(len(valid_edges)))
        notes = sum(bool(re.search(r"annotation|note:|interaction|on tap|on click|when user|behavior", section.body, re.IGNORECASE)) for section in unique.values())
        rubric.add_criterion("Annotations and interaction notes", 2.0, 2.0 if notes >= 2 else float(notes))
        screen_text = " ".join(section.body for section in unique.values())
        feature_hits = sum(bool(re.search(pattern, screen_text, re.IGNORECASE)) for pattern in (r"focus", r"calendar", r"music", r"schedule|planning", r"timer|session", r"settings"))
        rubric.add_criterion("Coverage of PRD features", 1.0, 1.0 if feature_hits >= 5 else 0.5 if feature_hits >= 3 else 0.0)
        return rubric.results()

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
