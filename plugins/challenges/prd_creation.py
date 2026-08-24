"""Structured Product Requirements Document challenge."""
from __future__ import annotations

import re

from benchmark.plugin import BenchmarkTaskPlugin
from plugins.challenges._analysis import (
    exact_section,
    markdown_sections,
    numbered_or_bulleted_items,
)
from plugins.challenges._rubric import Rubric


class PRDCreationPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "prd-creation"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def name(self) -> str:
        return "PRD Creation"

    @property
    def max_score(self) -> int:
        return int(22.0)

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_prompt(self):
        return (
            "Create a detailed FlowState PRD with exactly these sections: Executive Summary, "
            "Problem Statement, Goals & Objectives, Target Users & Personas, User Stories, "
            "Functional Requirements, Non-Functional Requirements, Success Metrics / KPIs, "
            "Competitive Analysis, Timeline / Milestones, Open Questions / Risks. Include at "
            "least 3 measurable goals, 2 distinct personas, 3 lines in `As a ... I want ... so "
            "that ...` format, 5 distinct functional requirements, performance/security/"
            "reliability/scalability NFRs, 3 quantitative KPIs with targets, 2 distinct competitors, "
            "milestones, and risks/questions. Use section-local content."
        )

    def get_temperature(self, global_config):
        return global_config.get("prd_creation_temperature")

    def evaluate(self, response_text):
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        if not text:
            return rubric.results()
        sections = markdown_sections(text)
        rubric.record_validation(type("Validation", (), {
            "valid": len(sections) >= 11,
            "evidence": [{"kind": "section", "heading": section.heading, "chars": len(section.body)} for section in sections],
            "errors": [],
        })())
        def body(names):
            section = exact_section(text, names[0], names[1:])
            return section.body if section else ""
        executive = body(["Executive Summary"])
        problem = body(["Problem Statement"])
        goals = body(["Goals & Objectives", "Goals", "Objectives"])
        personas = body(["Target Users & Personas", "Target Users", "Personas"])
        stories = body(["User Stories"])
        functional = body(["Functional Requirements"])
        nfr = body(["Non-Functional Requirements", "NFR"])
        metrics = body(["Success Metrics / KPIs", "Success Metrics", "KPIs", "Key Performance Indicators"])
        competitors = body(["Competitive Analysis", "Competitors"])
        timeline = body(["Timeline / Milestones", "Timeline", "Milestones", "Roadmap"])
        risks = body(["Open Questions / Risks", "Open Questions", "Risks"])
        rubric.add_criterion("Executive Summary", 2.0, 2.0 if len(executive) >= 80 and re.search(r"flowstate|productivity|focus", executive, re.IGNORECASE) else 0.0)
        rubric.add_criterion("Problem Statement", 2.0, 2.0 if len(problem) >= 60 and re.search(r"problem|pain|distraction|scheduling|focus", problem, re.IGNORECASE) else 0.0)
        goal_items = numbered_or_bulleted_items(goals)
        measurable = [item for item in goal_items if re.search(r"\d+\s*%|\d+\s*(?:users?|days?|weeks?|months?|seconds?)|increase|reduce|improve", item, re.IGNORECASE)]
        rubric.add_criterion("Goals & Objectives", 2.0, 2.0 if len(measurable) >= 3 else (1.0 if len(measurable) >= 1 else 0.0))
        persona_items = numbered_or_bulleted_items(personas)
        rubric.add_criterion("Target Users & Personas", 2.0, 2.0 if len(persona_items) >= 2 and len({item.split(":", 1)[0].strip().lower() for item in persona_items}) >= 2 else 0.0)
        story_lines = [line.strip() for line in stories.splitlines() if re.match(r"(?:[-*]\s*)?As an?\s+.+?\s*,?\s+I want\s+.+?\s*,?\s+so that\s+.+", line, re.IGNORECASE)]
        rubric.add_criterion("User Stories", 2.0, 2.0 if len(story_lines) >= 3 else (1.0 if story_lines else 0.0))
        req_items = numbered_or_bulleted_items(functional)
        req_items += re.findall(r"(?im)^\s*FR[- ]?\d+\s*:\s*(.+)$", functional)
        rubric.add_criterion("Functional Requirements", 3.0, 3.0 if len({item.lower() for item in req_items}) >= 5 else (1.5 if len(req_items) >= 3 else 0.0))
        nfr_hits = sum(bool(re.search(rf"\b{topic}\b", nfr, re.IGNORECASE)) for topic in ("performance", "security", "reliability", "scalability"))
        rubric.add_criterion("Non-Functional Requirements", 2.0, 2.0 if nfr_hits == 4 else nfr_hits / 2.0)
        metric_items = numbered_or_bulleted_items(metrics)
        quantified = [item for item in metric_items if re.search(r"\d+\s*%|\d+\s*(?:users?|minutes?|seconds?|hours?)", item, re.IGNORECASE)]
        rubric.add_criterion("Success Metrics / KPIs", 2.0, 2.0 if len(quantified) >= 3 else (1.0 if quantified else 0.0))
        names = set(re.findall(r"\b(?:Todoist|Notion|Trello|Asana|Forest|Rescue Time|Focusmate)\b", competitors, re.IGNORECASE))
        rubric.add_criterion("Competitive Analysis", 2.0, 2.0 if len(names) >= 2 and re.search(r"(?:lacks|strength|weakness|different|advantage|comparison)", competitors, re.IGNORECASE) else float(min(len(names), 2) / 2.0))
        rubric.add_criterion("Timeline / Milestones", 2.0, 2.0 if len(numbered_or_bulleted_items(timeline)) >= 2 and re.search(r"MVP|beta|launch|Q[1-4]|phase", timeline, re.IGNORECASE) else 0.0)
        rubric.add_criterion("Open Questions / Risks", 1.0, 1.0 if re.search(r"\?|risk|mitigation|assumption|dependency", risks, re.IGNORECASE) else 0.0)
        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
