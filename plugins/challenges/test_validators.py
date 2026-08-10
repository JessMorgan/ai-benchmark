"""Tests for shared typed scoring validators."""
from plugins.challenges._validators import (
    parse_python,
    parse_structured,
    parse_tool_calls,
    parse_workflow_graph,
    validate_sections,
)


def test_parse_python_accepts_raw_and_rejects_syntax_errors():
    assert parse_python("def f():\n    return 1").valid
    invalid = parse_python("def f(:\n    pass")
    assert not invalid.valid
    assert "SyntaxError" in invalid.errors[0]


def test_parse_structured_requires_object():
    assert parse_structured('{"name": "Alice"}', fmt="json").valid
    assert not parse_structured("[1, 2]", fmt="json").valid


def test_parse_tool_calls_validates_schema():
    valid = '<tool_call>{"name":"get_weather","args":{"location":"Tokyo","unit":"celsius"}}</tool_call>'
    assert parse_tool_calls(valid).valid
    invalid = '<tool_call>{"name":"unknown","args":{}}</tool_call>'
    result = parse_tool_calls(invalid)
    assert not result.valid
    assert any("unknown tool" in error for error in result.errors)


def test_workflow_graph_rejects_unknown_and_cyclic_dependencies():
    unknown = parse_workflow_graph("Step 1 [DEPENDS_ON: step99]")
    assert not unknown.valid
    assert any("unknown task" in error for error in unknown.errors)
    cyclic = parse_workflow_graph(
        "Step 1 [DEPENDS_ON: step2]\nStep 2 [DEPENDS_ON: step1]"
    )
    assert not cyclic.valid
    assert any("cycle" in error for error in cyclic.errors)


def test_sections_require_substantive_content():
    result = validate_sections("## One\nshort\n## Two\nadequate content here", ["One", "Two"])
    assert not result.valid
    assert any("One" in error for error in result.errors)
