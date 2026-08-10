"""Dependency-free typed validators shared by challenge plugins."""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

_YAML_ERROR = yaml.YAMLError if yaml is not None else ()


class StructuredParseError(ValueError):
    """Raised internally for a structured parser failure."""


@dataclass
class Validation:
    """A typed validation result suitable for rubric evidence."""

    valid: bool
    evidence: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    value: Any = None


def extract_fenced_blocks(text: str, language: str | None = None) -> list[str]:
    """Return fenced code blocks, optionally filtered by language."""
    wanted = language.lower() if language else None
    blocks = []
    for match in re.finditer(r"```([^\n`]*)\n(.*?)```", text, re.DOTALL):
        label = match.group(1).strip().lower()
        if wanted and label and label != wanted:
            continue
        blocks.append(match.group(2))
    return blocks


def parse_python(text: str, *, require_block: bool = False) -> Validation:
    """Extract Python and validate its syntax with ``ast.parse``.

    ``require_block`` is retained for callers that want to distinguish the
    source-selection policy, but raw Python remains valid input because the
    code-generation tasks do not uniformly require Markdown fences.
    """
    blocks = extract_fenced_blocks(text, "python")
    source = "\n\n".join(blocks) if blocks else text
    if not source.strip():
        return Validation(False, errors=["no Python source found"])
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return Validation(False, errors=[f"SyntaxError at line {exc.lineno}: {exc.msg}"])
    return Validation(
        True,
        evidence=[{"kind": "python-ast", "nodes": len(list(ast.walk(tree)))}],
        value=tree,
    )


def find_definitions(tree: ast.AST) -> dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Index function definitions by name."""
    definitions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.setdefault(node.name, []).append(node)
    return definitions


_TOOL_SCHEMAS: dict[str, dict[str, dict[str, type | tuple[type, ...]]]] = {
    "get_weather": {"required": {"location": str}, "optional": {"unit": str}},
    "search_flights": {"required": {"origin": str, "destination": str, "date": str}, "optional": {}},
    "book_hotel": {"required": {"city": str, "check_in": str, "check_out": str, "guests": int}, "optional": {}},
    "get_stock_price": {"required": {"ticker": str}, "optional": {}},
    "convert_currency": {"required": {"amount": (int, float), "from_curr": str, "to_curr": str}, "optional": {}},
    "send_email": {"required": {"to": str, "subject": str, "body": str}, "optional": {}},
}


def parse_tool_calls(text: str) -> Validation:
    """Parse tool calls and validate names, required arguments, and types."""
    calls = []
    errors = []
    for index, match in enumerate(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL | re.IGNORECASE), 1):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            errors.append(f"tool call {index} is invalid JSON: {exc.msg}")
            continue
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            errors.append(f"tool call {index} must be an object with a string name")
            continue
        name = value["name"]
        schema = _TOOL_SCHEMAS.get(name)
        args = value.get("args")
        if schema is None:
            errors.append(f"tool call {index} uses unknown tool {name!r}")
        elif not isinstance(args, dict):
            errors.append(f"tool call {index} args must be an object")
        else:
            for argument, expected_type in schema["required"].items():
                if argument not in args:
                    errors.append(f"tool call {index} is missing argument {argument!r}")
                elif not isinstance(args[argument], expected_type) or (
                    expected_type is int and isinstance(args[argument], bool)
                ):
                    errors.append(f"tool call {index} argument {argument!r} has the wrong type")
            for argument, expected_type in schema["optional"].items():
                if argument in args and not isinstance(args[argument], expected_type):
                    errors.append(f"tool call {index} argument {argument!r} has the wrong type")
        calls.append(value)
    if not calls:
        errors.append("no typed tool calls found")
    return Validation(
        not errors,
        evidence=[{"kind": "tool-call", "index": i, "name": call.get("name")} for i, call in enumerate(calls, 1)],
        errors=errors,
        value=calls,
    )


def parse_structured(text: str, *, fmt: str | None = None) -> Validation:
    """Parse one JSON/YAML candidate and return its typed value."""
    candidates = extract_fenced_blocks(text, fmt) if fmt else extract_fenced_blocks(text)
    source = candidates[0] if candidates else text.strip()
    if not source:
        return Validation(False, errors=["no structured candidate found"])
    try:
        if (fmt or "").lower() == "yaml" or (fmt is None and not source.lstrip().startswith(("{", "["))):
            if yaml is None:
                return Validation(False, errors=["PyYAML is unavailable"])
            value = yaml.safe_load(source)
        else:
            value = json.loads(source)
    except (json.JSONDecodeError, AttributeError, KeyError, TypeError, ValueError) as exc:
        return Validation(False, errors=[f"structured parse error: {type(exc).__name__}: {exc}"])
    except _YAML_ERROR as exc:
        return Validation(False, errors=[f"structured parse error: {type(exc).__name__}: {exc}"])
    if not isinstance(value, dict):
        return Validation(False, errors=["structured candidate is not an object"], value=value)
    return Validation(
        True,
        evidence=[{"kind": "structured-parse", "format": fmt or "auto", "keys": sorted(value)}],
        value=value,
    )


def section_map(text: str) -> dict[str, str]:
    """Split Markdown into normalized heading-to-content sections."""
    matches = list(re.finditer(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$", text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        heading = re.sub(r"[*_`]+", "", match.group(1)).strip().lower()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[heading] = text[match.end():end].strip()
    return sections


def validate_sections(text: str, required: list[str], *, min_chars: int = 20) -> Validation:
    """Require each named section to contain substantive content."""
    sections = section_map(text)
    missing = []
    evidence = []
    for heading in required:
        normalized = heading.lower()
        content = next((body for key, body in sections.items() if normalized in key), "")
        if len(content.strip()) < min_chars:
            missing.append(f"section {heading!r} is missing or too short")
        else:
            evidence.append({"kind": "section", "heading": heading, "chars": len(content)})
    return Validation(not missing, evidence=evidence, errors=missing, value=sections)


def parse_workflow_graph(text: str) -> Validation:
    """Parse task IDs and dependency tags, rejecting cycles/references."""
    all_matches = list(re.finditer(r"\b(?:task|step)[ _-]?(\d+)\b", text, re.IGNORECASE))
    referenced_positions = set()
    for reference in re.finditer(r"\[DEPENDS_ON\s*:\s*(?:task|step)?[_ -]?(\d+)\]", text, re.IGNORECASE):
        referenced_positions.update(
            match.start() for match in all_matches
            if reference.start() <= match.start() < reference.end()
        )
    task_ids = {
        match.group(1) for match in all_matches if match.start() not in referenced_positions
    }
    edges = []
    for match in re.finditer(r"\[DEPENDS_ON\s*:\s*(?:task|step)?[_ -]?(\d+)\]", text, re.IGNORECASE):
        before = text[:match.start()]
        current = re.findall(r"\b(?:task|step)[ _-]?(\d+)\b", before, re.IGNORECASE)
        if current:
            edges.append((current[-1], match.group(1)))
    errors = []
    for source, target in edges:
        if target not in task_ids:
            errors.append(f"dependency references unknown task {target}")
    adjacency: dict[str, set[str]] = {task_id: set() for task_id in task_ids}
    for source, target in edges:
        adjacency.setdefault(source, set()).add(target)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            errors.append("dependency graph contains a cycle")
            return
        if node in visited:
            return
        visiting.add(node)
        for child in adjacency.get(node, ()):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for task_id in adjacency:
        visit(task_id)
    if not edges:
        errors.append("no dependency edges found")
    return Validation(
        not errors and len(task_ids) >= 2,
        evidence=[{"kind": "workflow-graph", "tasks": sorted(task_ids), "edges": edges}],
        errors=errors or ([] if len(task_ids) >= 2 else ["fewer than two task IDs found"]),
        value={"tasks": task_ids, "edges": edges},
    )
