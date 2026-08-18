"""Static compatibility probes for JSON schemas sent to model grammars.

The benchmark sends these schemas through OpenAI-compatible endpoints. Local
Ollama deployments and llama.cpp both ultimately turn the schema into a
constrained JSON grammar, so this test intentionally checks a conservative
intersection of their supported behavior rather than merely validating Draft
2020-12 syntax.
"""
from __future__ import annotations

from collections.abc import Iterator

import pytest
from jsonschema import Draft202012Validator

from benchmark.core import JUDGE_RESPONSE_SCHEMA
from plugins.challenges.data_transformation import DATA_TRANSFORMATION_RESPONSE_SCHEMA

SCHEMAS = {
    "judge": JUDGE_RESPONSE_SCHEMA,
    "data-transformation": DATA_TRANSFORMATION_RESPONSE_SCHEMA,
}

# This is deliberately narrower than all of JSON Schema. Ollama documents a
# schema object for ``format`` but does not promise full Draft 2020-12
# semantics; llama.cpp explicitly documents this subset and silently skips
# unsupported keywords. Keep request schemas in the intersection so a provider
# cannot appear to enforce a contract that the grammar backend ignored.
SUPPORTED_SCHEMA_KEYWORDS = frozenset({
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "enum",
    "pattern",
    "minimum",
    "maximum",
})


def _schema_nodes(schema: object, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], dict]]:
    """Yield schema nodes, following only JSON Schema structural positions."""
    if not isinstance(schema, dict):
        return
    yield path, schema
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, child in properties.items():
            yield from _schema_nodes(child, path + ("properties", str(name)))
    for key in ("items", "additionalProperties"):
        child = schema.get(key)
        if isinstance(child, dict):
            yield from _schema_nodes(child, path + (key,))
    for key in ("oneOf", "anyOf", "allOf", "prefixItems"):
        children = schema.get(key)
        if isinstance(children, list):
            for index, child in enumerate(children):
                yield from _schema_nodes(child, path + (key, str(index)))


def _string_schema_nodes(schema: object) -> Iterator[tuple[tuple[str, ...], dict]]:
    """Yield every nested JSON-schema node whose type is string."""
    yield from (
        (path, node)
        for path, node in _schema_nodes(schema)
        if node.get("type") == "string"
    )


def _pattern_nodes(schema: object) -> Iterator[tuple[tuple[str, ...], dict]]:
    """Yield every string node constrained by a regular-expression pattern."""
    yield from (
        (path, node)
        for path, node in _string_schema_nodes(schema)
        if "pattern" in node
    )


def _numeric_nodes(schema: object) -> Iterator[tuple[tuple[str, ...], dict]]:
    """Yield every numeric schema node."""
    yield from (
        (path, node)
        for path, node in _schema_nodes(schema)
        if node.get("type") in {"integer", "number"}
    )


@pytest.mark.parametrize("schema_name", sorted(SCHEMAS))
def test_prompt_schema_is_valid_json_schema(schema_name):
    """The schemas accepted by the API are valid Draft 2020-12 schemas."""
    Draft202012Validator.check_schema(SCHEMAS[schema_name])


@pytest.mark.parametrize("schema_name", sorted(SCHEMAS))
def test_prompt_schema_uses_conservative_provider_keyword_subset(schema_name):
    """Reject keywords Ollama/llama.cpp may ignore or cannot compile."""
    violations = []
    for path, node in _schema_nodes(SCHEMAS[schema_name]):
        unsupported = sorted(set(node) - SUPPORTED_SCHEMA_KEYWORDS)
        if unsupported:
            violations.append(f"{'.'.join(path) or '<root>'}: {unsupported}")
    assert violations == []


@pytest.mark.parametrize("schema_name", sorted(SCHEMAS))
def test_prompt_schema_has_no_bounded_strings_for_llama_grammar(schema_name):
    """Avoid repeated-character grammar rules in llama.cpp.

    String emptiness and practical length are communicated in prompts and
    checked after generation. This avoids grammar expansion limits for escaped
    Unicode-aware strings.
    """
    violations = [
        ".".join(path) or "<root>"
        for path, node in _string_schema_nodes(SCHEMAS[schema_name])
        if "minLength" in node or "maxLength" in node
    ]
    assert violations == [], (
        f"{schema_name} contains llama.cpp-hostile string bounds at: "
        + ", ".join(violations)
    )


@pytest.mark.parametrize("schema_name", sorted(SCHEMAS))
def test_prompt_schema_patterns_are_anchored_and_gbnf_safe(schema_name):
    """Use explicit character classes instead of regex escapes/shorthand.

    llama.cpp requires anchored patterns and its grammar converter does not
    accept common regex shorthand such as ``\\d``. Avoiding all backslashes in
    these small patterns also avoids provider-specific escape translation.
    """
    violations = []
    for path, node in _pattern_nodes(SCHEMAS[schema_name]):
        pattern = node["pattern"]
        if not isinstance(pattern, str) or not pattern.startswith("^") or not pattern.endswith("$"):
            violations.append(f"{'.'.join(path)}: pattern must be anchored")
        if isinstance(pattern, str) and "\\" in pattern:
            violations.append(f"{'.'.join(path)}: pattern contains a backslash escape")
        if isinstance(pattern, str) and "(?" in pattern:
            violations.append(f"{'.'.join(path)}: pattern contains a grouped/lookaround extension")
    assert violations == [], "\n".join(violations)


@pytest.mark.parametrize("schema_name", sorted(SCHEMAS))
def test_prompt_schema_numeric_bounds_target_llama_supported_integers(schema_name):
    """Do not imply that llama.cpp enforces bounds on JSON ``number`` values."""
    violations = [
        ".".join(path) or "<root>"
        for path, node in _numeric_nodes(SCHEMAS[schema_name])
        if node.get("type") == "number"
        and any(key in node for key in ("minimum", "maximum"))
    ]
    assert violations == [], (
        "llama.cpp only grammar-enforces minimum/maximum for integer nodes; "
        "validate bounded fractional values after generation at: "
        + ", ".join(violations)
    )


@pytest.mark.parametrize("schema_name", sorted(SCHEMAS))
def test_prompt_schema_objects_explicitly_disallow_extra_properties(schema_name):
    """Keep object grammars finite and aligned with strict structured output."""
    violations = [
        ".".join(path) or "<root>"
        for path, node in _schema_nodes(SCHEMAS[schema_name])
        if node.get("type") == "object" and node.get("additionalProperties") is not False
    ]
    assert violations == [], "objects must explicitly set additionalProperties=false: " + ", ".join(violations)
