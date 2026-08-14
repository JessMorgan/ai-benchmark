#!/usr/bin/env python3
"""HISTORICAL MIGRATION: retained for audit context to add KPI guard / NFR section
to benchmark-agents.yml. Re-runs are safe (each edit is idempotent on the
NEW strings; new edits to benchmark-agents.yml will break the oldString
match, returning exit code 1 — that's a deliberate sanity signal, not a bug).

The two changes we want:
  - Product Manager: drop the interactive collect_data halt, instead commit
    to industry-standard assumptions.  Strengthen the User Stories and KPI
    specifications (priority tag + Gherkin-style AC + edge-case AC + baseline
    + target + measurement method).
  - Architect: insert a Non-Functional Requirements section between Data
    Model and Implementation Roadmap, and explicitly require Security +
    Resiliency + Observability coverage.  Trim the API scope hint.
"""
from __future__ import annotations

import sys
from pathlib import Path

CONFIG = Path("benchmark-agents.yml")


# Phrase-level sanity assertions that DO NOT require pyyaml.  Each function
# returns True on success; False (after printing) on failure (and the caller
# returns 1).
PM_BANNED_PHRASE = "must not hallucinate details. Instead, use the collect_data tool"
PM_REQUIRED_PHRASES = ("baseline", "target", "measurement")
ARCH_REQUIRED_PHRASES = ("Non-Functional Requirements", "Failure Modes", "Security", "Observability")
ARCH_API_CAP = "5-10 endpoint"


def pm_section_check(new_src: str) -> bool:
    # The most reliable signal for "PM is updated" is the *absence* of the
    # old banned phrase and the *presence* of every required KPI phrase.
    # Walk every `Product Manager:` block to its next agent key.
    block = _extract_agent_block(new_src, "Product Manager")
    if block is None:
        print(f"ERROR: cannot locate Product Manager block in {CONFIG}", file=sys.stderr)
        return False
    if PM_BANNED_PHRASE in block:
        print(
            "ERROR: PM prompt still contains the old 'must not hallucinate / use collect_data' instruction",
            file=sys.stderr,
        )
        return False
    block_lower = block.lower()
    for needle in PM_REQUIRED_PHRASES:
        if needle not in block_lower:
            print(f"ERROR: PM prompt missing required phrase: '{needle}'", file=sys.stderr)
            return False
    return True


def arch_section_check(new_src: str) -> bool:
    block = _extract_agent_block(new_src, "Architect")
    if block is None:
        print(f"ERROR: cannot locate Architect block in {CONFIG}", file=sys.stderr)
        return False
    for needle in ARCH_REQUIRED_PHRASES:
        if needle not in block:
            print(f"ERROR: Architect prompt missing required phrase: '{needle}'", file=sys.stderr)
            return False
    if ARCH_API_CAP not in block:
        print(f"ERROR: Architect prompt missing API scope cap: '{ARCH_API_CAP}'", file=sys.stderr)
        return False
    return True


def _extract_agent_block(text: str, agent_name: str) -> str | None:
    """Return the substring of *text* that contains the agent's YAML block.

    Looks for a line whose first non-space characters match `  {Name}:` and
    returns everything up to the next `  <OtherName>:` header at the same
    indent.  Returns None if the agent name isn't present.
    """
    needle = f"  {agent_name}:"
    start = text.find(needle)
    if start == -1:
        return None
    tail = text[start + len(needle):]
    # Find the next top-level agent key (two-space indent, ends with colon,
    # followed by newline and a four-space-indented key under it).  Walking
    # line by line keeps this robust against YAML literal-block scalars.
    lines = tail.splitlines(keepends=True)
    end_offset = len(tail)
    for i, line in enumerate(lines):
        # Top-level agent header pattern: "  SomeName:\n"
        if (
            line.startswith("  ")
            and not line.startswith("   ")
            and line.rstrip().endswith(":")
        ):
            end_offset = sum(len(L) for L in lines[:i])
            break
    return tail[:end_offset]

# Exact PM system_prompt content (6-space indented YAML literal block).
# Pulled verbatim from the existing file - whitespace matters because YAML
# literal block scalars rely on consistent indentation.
OLD_PM_PROMPT = (
    "      When drafting User Stories, follow the standard format: 'As a "
    "[persona], I want to [action] so that [benefit].' Each story must "
    "be accompanied by clear, testable Acceptance Criteria (AC). Use the "
    "sequentialthinking tool to map out complex features, identify edge "
    "cases, and ensure no logic gaps exist before generating final "
    "content.\n"
    "      \n"
    "      If the user provides insufficient information or vague "
    "requirements for a specific feature, you must not hallucinate "
    "details. Instead, use the collect_data tool to present a form to the "
    "user, gathering necessary specifics (e.g., target audience, priority "
    "level, technical constraints) before proceeding with the "
    "documentation.\n"
    "      \n"
    "      You may also use spawn_subtask to delegate specialized "
    "components of a large PRD\u2014such as drafting detailed API "
    "specifications or creating an extensive backlog of edge-case "
    "scenarios\u2014to sub-agents, ensuring that each part of the "
    "document is meticulously reviewed and polished.\n"
)

NEW_PM_PROMPT = (
    "      When drafting User Stories, follow the standard format: 'As a "
    "[persona], I want [action], so that [benefit].' Every story MUST "
    "include all four of: (a) explicit persona name, (b) priority tag "
    "(High/Medium/Low) tied to a release milestone, (c) at least 3 "
    "bullet-pointed Acceptance Criteria in Given/When/Then or otherwise "
    "testable form, and (d) at least one edge-case or negative-path "
    "Acceptance Criterion (e.g., 'when the calendar API is unreachable', "
    "'when the user denies permissions', 'when sync conflicts occur'). "
    "Aim for 4-6 stories per PRD - quantity is rewarded, but only when "
    "each story is concretely scoped. Use the sequentialthinking tool to "
    "map out complex features, identify edge cases, and ensure no logic "
    "gaps exist before generating final content.\n"
    "      \n"
    "      When drafting Goals & Objectives and Success Metrics / KPIs, "
    "treat every metric as a quantitative hypothesis a stakeholder could "
    "disagree with. Each KPI MUST specify: (1) a numeric baseline or "
    "'current state' (e.g., 'average user completes 60% of manually-"
    "planned tasks'), (2) a numeric target with measurement window "
    "(e.g., 'increase to 80% within 90 days of v1.0 launch'), and (3) an "
    "explicit measurement method or data source (e.g., 'tracked via in-"
    "app task completion events, reported weekly in Mixpanel cohort'). "
    "Prefer baseline-and-delta framing over standalone percentages - a "
    "number like 'reduce context-switches from 12/day (baseline) to "
    "8/day (33% reduction)' is stronger than 'reduce context-switches "
    "by 30%'. Aim for at least 4 KPIs covering acquisition, engagement, "
    "retention, and a leading indicator of long-term value (NPS, paid "
    "conversion, or session depth).\n"
    "      \n"
    "      When requirements are open-ended or under-specified (a common "
    "case in benchmark or briefing scenarios), you MUST confidently "
    "invent industry-standard, defensible details rather than halting or "
    "hedging. State any assumptions explicitly in the Open Questions / "
    "Risks section as 'Assumed: ...' entries so they can be challenged "
    "downstream. NEVER use placeholder literals like '[Insert Date]', "
    "'[Your Name]', or 'TBD' for required PRD fields (Version, Date, "
    "Author, Approvers). Pick reasonable defaults from the brief's "
    "domain (a cross-platform productivity app launching in 2026, "
    "targeting knowledge workers, etc.) and state them. Do NOT call "
    "collect_data or any interactive form tool - these evaluations are "
    "non-interactive and you are expected to commit to specifics.\n"
    "      \n"
    "      You may also use spawn_subtask to delegate specialized "
    "components of a large PRD - such as drafting detailed API "
    "specifications or creating an extensive backlog of edge-case "
    "scenarios - to sub-agents, ensuring that each part of the document "
    "is meticulously reviewed and polished.\n"
)

# Architect OUTPUT FORMAT block - we want to insert NFR section and tighten
# the API guidance. The original output block uses 6-space indentation
# inside the |-block and 10-space (sub-list) for the numbered list.
OLD_ARCH_OUTPUT = (
    "      - **Structure:**\n"
    "          1. **Executive Summary:** A brief overview of the proposed solution.\n"
    "          2. **Technical Stack:** A table or list of selected technologies and justifications.\n"
    "          3. **Architecture Diagram (Text/Mermaid):** A representation of system components and their interactions.\n"
    "          4. **Data Model:** Definitions of key entities and relationships.\n"
    "          5. **Implementation Roadmap:** A numbered list of decomposed tasks for a development team.\n"
)
NEW_ARCH_OUTPUT = (
    "      - **Structure:**\n"
    "          1. **Executive Summary:** Brief overview of the proposed solution.\n"
    "          2. **Technical Stack:** Table or list of selected technologies and justifications.\n"
    "          3. **Architecture Diagram (Text/Mermaid):** Representation of system components and their interactions.\n"
    "          4. **Data Model:** Definitions of key entities, relationships, and storage choices.\n"
    "          5. **Non-Functional Requirements:** Dedicated, first-class section covering:\n"
    "              - **Security:** authentication, authorization, data encryption at rest/in transit, secrets management, threat model. Include specific mechanisms (e.g., OAuth2 PKCE, JWT with refresh rotation, AES-256, KMS).\n"
    "              - **Resiliency & Failure Modes:** circuit breakers, retry/backoff, dead-letter queues, multi-region failover, blast radius analysis. Explicitly enumerate what happens when Google Calendar, Spotify, or Microsoft Graph are degraded or unreachable.\n"
    "              - **Observability & SLOs:** SLO/SLI targets with concrete numbers (e.g., p99 < 200ms for `/planner/daily`, 99.9% monthly uptime with error budget), distributed tracing strategy, log/metrics tooling.\n"
    "          6. **Implementation Roadmap:** Numbered list of decomposed tasks for a development team.\n"
)

# Also tighten the API design line in the Architect METHODOLOGY block so the
# agent doesn't blow its token budget on a full GraphQL schema.
OLD_ARCH_METHODOLOGY_API = (
    "      - **Tool vs. Knowledge:** Use internal knowledge for general software engineering principles; use `execute_python` to validate logic/math; use web tools only when current industry standards or specific API capabilities need verification.\n"
)
NEW_ARCH_METHODOLOGY_API = (
    "      - **Tool vs. Knowledge:** Use internal knowledge for general software engineering principles; use `execute_python` to validate logic/math; use web tools only when current industry standards or specific API capabilities need verification. Cap API design at 5-10 endpoint summaries or a small GraphQL sketch - do NOT generate full schema resolvers; redirect that depth into the Non-Functional Requirements section instead.\n"
)


def main() -> int:
    src = CONFIG.read_text(encoding="utf-8")

    if OLD_PM_PROMPT not in src:
        print(f"ERROR: PM prompt block not found verbatim in {CONFIG}", file=sys.stderr)
        return 1
    if OLD_ARCH_OUTPUT not in src:
        print(f"ERROR: Architect OUTPUT block not found verbatim in {CONFIG}", file=sys.stderr)
        return 1
    if OLD_ARCH_METHODOLOGY_API not in src:
        print(f"ERROR: Architect METHODOLOGY line not found verbatim in {CONFIG}", file=sys.stderr)
        return 1

    new_src = src.replace(OLD_PM_PROMPT, NEW_PM_PROMPT)
    new_src = new_src.replace(OLD_ARCH_OUTPUT, NEW_ARCH_OUTPUT)
    new_src = new_src.replace(OLD_ARCH_METHODOLOGY_API, NEW_ARCH_METHODOLOGY_API)

    # Cheap string-based sanity checks first - these run unconditionally so a
    # host without pyyaml still gets post-write validation.  We treat the
    # *absence* of the OLD sentinel strings as success, and the *presence*
    # of the NEW required phrases as success.
    if pm_section_check(new_src) is False:
        return 1
    if arch_section_check(new_src) is False:
        return 1

    # Stronger structural check if pyyaml is available.
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None  # type: ignore
    if yaml is not None:
        data = yaml.safe_load(new_src)
        agents = data.get("agents", {}) if isinstance(data, dict) else {}
        for name in ("Product Manager", "Architect"):
            if name not in agents:
                print(f"ERROR: agent '{name}' missing after patch", file=sys.stderr)
                return 1
            prompt = agents[name].get("system_prompt", "")
            if not prompt or len(prompt) < 200:
                print(f"ERROR: agent '{name}' prompt is too short after patch", file=sys.stderr)
                return 1

    CONFIG.write_text(new_src, encoding="utf-8")
    print(f"Patched {CONFIG} successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
