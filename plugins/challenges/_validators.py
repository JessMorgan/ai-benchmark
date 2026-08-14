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


def stub_definitions(tree: ast.AST, names: set[str]) -> list[str]:
    """Return required definitions whose bodies contain only stubs."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in names:
            continue
        meaningful = [
            item for item in node.body
            if not (
                isinstance(item, ast.Expr)
                and isinstance(item.value, ast.Constant)
                and isinstance(item.value.value, str)
            )
        ]
        if meaningful and all(
            isinstance(item, ast.Pass)
            or (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and item.value.value is Ellipsis)
            for item in meaningful
        ):
            found.append(node.name)
    return found


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
    if len(candidates) > 1:
        return Validation(
            False,
            evidence=[{"kind": "structured-candidate-count", "count": len(candidates)}],
            errors=["multiple structured candidates found; exactly one is required"],
        )
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
        evidence=[{"kind": "structured-parse", "format": fmt or "auto", "keys": sorted(value), "candidate_count": len(candidates) or 1}],
        value=value,
    )


def heading_occurrences(text: str) -> list[tuple[str, str]]:
    """Return every normalized Markdown heading and its body."""
    matches = list(re.finditer(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$", text))
    occurrences = []
    for index, match in enumerate(matches):
        heading = re.sub(r"[*_`]+", "", match.group(1)).strip().lower()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        occurrences.append((heading, text[match.end():end].strip()))
    return occurrences


def section_map(text: str) -> dict[str, str]:
    """Split Markdown into normalized heading-to-content sections."""
    return dict(heading_occurrences(text))


def validate_sections(
    text: str,
    required: list[str],
    *,
    min_chars: int = 20,
    aliases: dict[str, tuple[str, ...]] | None = None,
) -> Validation:
    """Require each named section to contain substantive content.

    ``aliases`` lets a document use an equivalent heading without turning the
    evaluator into a fixed-vocabulary test. The canonical heading remains in
    diagnostics so reports retain a stable rubric name.
    """
    sections = section_map(text)
    aliases = aliases or {}
    missing = []
    evidence = []
    for heading in required:
        candidates = (heading, *aliases.get(heading, ()))
        matched_key = next(
            (
                key
                for key in sections
                if any(candidate.lower() in key for candidate in candidates)
            ),
            None,
        )
        content = sections.get(matched_key, "") if matched_key is not None else ""
        if len(content.strip()) < min_chars:
            missing.append(f"section {heading!r} is missing or too short")
        else:
            assert matched_key is not None
            matched_heading = next(
                candidate for candidate in candidates if candidate.lower() in matched_key
            )
            evidence.append({
                "kind": "section",
                "heading": heading,
                "matched_heading": matched_heading,
                "chars": len(content),
            })
    return Validation(not missing, evidence=evidence, errors=missing, value=sections)


def parse_workflow_graph(text: str) -> Validation:
    """Parse explicit, Mermaid, and plain-language workflow dependencies."""
    task_pattern = r"\b(?:task|step)[ _-]?(\d+)\b"
    bracket_pattern = r"\[DEPENDS_ON\s*:\s*(?:task|step)?[_ -]?(\d+)\]"
    all_matches = list(re.finditer(task_pattern, text, re.IGNORECASE))
    edges: list[tuple[str, str]] = []
    referenced_positions: set[int] = set()

    # The benchmark's original notation means "this task depends on X".
    # Bind each tag to the nearest declared task, never to a task ID inside a
    # preceding dependency reference. This matters when compact plans put
    # several dependency tags on one line.
    bracket_matches = list(re.finditer(bracket_pattern, text, re.IGNORECASE))
    for dependency in bracket_matches:
        current = [
            item.group(1)
            for item in all_matches
            if item.start() < dependency.start()
            and not any(
                reference.start() <= item.start() < reference.end()
                for reference in bracket_matches
            )
        ]
        if current:
            edges.append((current[-1], dependency.group(1)))
        referenced_positions.update(
            item.start()
            for item in all_matches
            if dependency.start() <= item.start() < dependency.end()
        )

    # Accept common graph renderings without making bracket tags mandatory.
    for match in re.finditer(
        r"(?:task|step)[ _-]?(\d+)\s*-->?\s*(?:task|step)[ _-]?(\d+)",
        text,
        re.IGNORECASE,
    ):
        edges.append((match.group(1), match.group(2)))

    # Plain prose such as "Task 3 depends on Step 1" is equally explicit.
    for match in re.finditer(
        r"(?:task|step)[ _-]?(\d+)\s+(?:depends on|requires|after)\s+"
        r"(?:task|step)[ _-]?(\d+)",
        text,
        re.IGNORECASE,
    ):
        edges.append((match.group(1), match.group(2)))

    task_ids = {
        match.group(1)
        for match in all_matches
        if match.start() not in referenced_positions
    }
    edges = list(dict.fromkeys(edges))
    errors = []
    for source, target in edges:
        if source not in task_ids or target not in task_ids:
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
    labels_by_task: dict[str, set[str]] = {}
    for line in text.splitlines():
        task_match = re.search(task_pattern, line, re.IGNORECASE)
        if task_match:
            labels = labels_by_task.setdefault(task_match.group(1), set())
            if re.search(r"(?:\[\s*)?parallel\b", line, re.IGNORECASE):
                labels.add("parallel")
            if re.search(r"(?:\[\s*)?sequential\b", line, re.IGNORECASE):
                labels.add("sequential")
    if any(labels == {"parallel", "sequential"} for labels in labels_by_task.values()):
        errors.append("a task is labeled both parallel and sequential")
    if not edges:
        errors.append("no dependency edges found")
    if len(task_ids) < 2:
        errors.append("fewer than two task IDs found")
    return Validation(
        not errors,
        evidence=[{"kind": "workflow-graph", "tasks": sorted(task_ids), "edges": edges}],
        errors=errors,
        value={"tasks": task_ids, "edges": edges},
    )
