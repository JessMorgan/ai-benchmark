"""Deterministic parsing helpers shared by challenge evaluators.

The challenge suite deliberately avoids treating a document as one unstructured
bag of keywords.  These helpers keep section boundaries and repeated headings
visible so individual plugins can score the requirement where it was actually
written.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    """One Markdown heading and the body that follows it."""

    heading: str
    normalized: str
    body: str
    index: int


def normalize_heading(value: str) -> str:
    """Normalize Markdown decoration and punctuation for heading matching."""
    value = re.sub(r"[*_`]+", "", value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def markdown_sections(text: str) -> list[Section]:
    """Return every Markdown section, excluding headings inside code fences."""
    matches = []
    # A model cannot satisfy a document section by putting a fake heading in
    # an example block.
    fence_ranges = [
        (match.start(), match.end())
        for match in re.finditer(r"(?ms)^\s{0,3}(?:```|~~~).*?^\s{0,3}(?:```|~~~)\s*$", text)
    ]
    for match in re.finditer(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$", text):
        if not any(start <= match.start() < end for start, end in fence_ranges):
            matches.append(match)
    sections: list[Section] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        heading = match.group(1).strip()
        sections.append(Section(
            heading=heading,
            normalized=normalize_heading(heading),
            body=text[match.end():end].strip(),
            index=index,
        ))
    return sections


def exact_section(text: str, name: str, aliases: Iterable[str] = ()) -> Section | None:
    """Return the first section whose normalized heading exactly matches."""
    wanted = {normalize_heading(name), *(normalize_heading(alias) for alias in aliases)}
    return next((section for section in markdown_sections(text) if section.normalized in wanted), None)


def section_bodies(text: str, names: Iterable[str]) -> list[str]:
    """Return bodies for exact canonical headings and aliases."""
    wanted = {normalize_heading(name) for name in names}
    return [section.body for section in markdown_sections(text) if section.normalized in wanted]


def mermaid_graph(text: str) -> tuple[set[str], set[tuple[str, str]]]:
    """Parse the small Mermaid flowchart subset used by architecture prompts."""
    nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()
    for block in fenced_blocks(text, "mermaid"):
        for line in block.splitlines():
            match = re.search(r"(?:^|\s)([A-Za-z][\w-]*)\s*(?:\[[^]]+\]|\([^)]*\))?\s*--+>\s*([A-Za-z][\w-]*)", line)
            if match:
                source, target = match.groups()
                nodes.update((source, target))
                edges.add((source, target))
            else:
                for node in re.findall(r"\b([A-Za-z][\w-]*)\s*(?:\[[^]]+\]|\([^)]*\))", line):
                    if node.lower() not in {"graph", "flowchart", "td", "lr"}:
                        nodes.add(node)
    return nodes, edges


def matching_sections(text: str, names: Iterable[str]) -> list[Section]:
    """Find sections whose normalized heading contains one of ``names``."""
    wanted = [normalize_heading(name) for name in names]
    return [
        section for section in markdown_sections(text)
        if any(name in section.normalized for name in wanted)
    ]


def first_section(text: str, names: Iterable[str]) -> Section | None:
    """Return the first matching section, or ``None``."""
    matches = matching_sections(text, names)
    return matches[0] if matches else None


def section_has_content(section: Section | None, minimum: int = 20) -> bool:
    """Return whether a section exists and has substantive body text."""
    return bool(section and len(section.body.strip()) >= minimum)


def numbered_or_bulleted_items(body: str) -> list[str]:
    """Extract non-empty numbered or bulleted list items from a section."""
    return [
        match.group(1).strip()
        for match in re.finditer(r"(?m)^\s*(?:[-*]|\d+[.)])\s+(.+?)\s*$", body)
        if match.group(1).strip()
    ]


def distinct_normalized(values: Iterable[str]) -> list[str]:
    """Return values with case/whitespace normalization and stable ordering."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", value.strip().lower())
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def has_real_code_block(text: str, language: str = "python") -> bool:
    """Return whether a fenced block for ``language`` contains code."""
    pattern = rf"```\s*{re.escape(language)}\s*\n([\s\S]*?)```"
    return any(block.strip() for block in re.findall(pattern, text, re.IGNORECASE))


def fenced_blocks(text: str, language: str | None = None) -> list[str]:
    """Extract fenced blocks, optionally restricted by language."""
    pattern = r"```([^\n`]*)\n(.*?)```"
    blocks = []
    wanted = language.lower() if language else None
    for match in re.finditer(pattern, text, re.DOTALL):
        label = match.group(1).strip().lower()
        if wanted and label != wanted:
            continue
        blocks.append(match.group(2))
    return blocks


def text_without_fences(text: str) -> str:
    """Remove fenced blocks when checking for forbidden prose."""
    return re.sub(r"```[\s\S]*?```", "", text)
