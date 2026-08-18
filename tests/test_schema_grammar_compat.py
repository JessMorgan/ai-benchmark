"""Static compatibility probes for JSON schemas sent to llama.cpp grammars."""
from __future__ import annotations

from collections.abc import Iterator

import pytest
from jsonschema import Draft202012Validator

from benchmark.core import JUDGE_RESPONSE_SCHEMA
from plugins.challenges.structured_output import STRUCTURED_OUTPUT_RESPONSE_SCHEMA

SCHEMAS = {
    "judge": JUDGE_RESPONSE_SCHEMA,
    "structured-output": STRUCTURED_OUTPUT_RESPONSE_SCHEMA,
}


def _string_schema_nodes(schema: object, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], dict]]:
    """Yield every nested JSON-schema node whose type is string."""
    if isinstance(schema, dict):
        if schema.get("type") == "string":
            yield path, schema
        for key, value in schema.items():
            yield from _string_schema_nodes(value, path + (str(key),))
    elif isinstance(schema, list):
        for index, value in enumerate(schema):
            yield from _string_schema_nodes(value, path + (str(index),))


@pytest.mark.parametrize("schema_name", sorted(SCHEMAS))
def test_prompt_schema_is_valid_json_schema(schema_name):
    """The schemas accepted by the API are valid Draft 2020-12 schemas."""
    Draft202012Validator.check_schema(SCHEMAS[schema_name])


@pytest.mark.parametrize("schema_name", sorted(SCHEMAS))
def test_prompt_schema_has_no_bounded_strings_for_llama_grammar(schema_name):
    """Avoid large repeated-character grammar rules in llama.cpp.

    llama.cpp expands string length bounds into grammar repetition rules. Even
    a modest-looking bound can become expensive when the character rule also
    handles escaping and Unicode. String emptiness and practical length are
    therefore communicated in the prompt and checked after generation.
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
