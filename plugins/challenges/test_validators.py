"""Tests for shared typed scoring validators."""
import ast

from plugins.challenges._validators import (
    parse_python,
    parse_structured,
    parse_tool_calls,
    parse_workflow_graph,
    stub_definitions,
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


def test_stub_definitions_flags_docstring_only_bodies():
    func = stub_definitions(ast.parse('def foo():\n    """Doc."""\n'), {"foo"})
    assert "foo" in func
    klass = stub_definitions(ast.parse('class Bar:\n    """Doc."""\n'), {"Bar"})
    assert "Bar" in klass


def test_stub_definitions_ignores_real_bodies():
    real = stub_definitions(ast.parse("def foo():\n    return 1\n"), {"foo"})
    assert "foo" not in real


def test_stub_definitions_still_flags_pass_only_bodies():
    stubs = stub_definitions(ast.parse("def foo():\n    pass\n"), {"foo"})
    assert "foo" in stubs


def test_stub_definitions_exempts_exception_classes():
    docstring_only = stub_definitions(
        ast.parse('class AllProvidersFailedError(Exception):\n    """Doc."""\n'),
        {"AllProvidersFailedError"},
    )
    assert "AllProvidersFailedError" not in docstring_only
    ellipsis_only = stub_definitions(ast.parse("class E(Exception):\n    ...\n"), {"E"})
    assert "E" not in ellipsis_only


def test_stub_definitions_docstring_plus_real_body_not_flagged():
    real = stub_definitions(ast.parse('def foo():\n    """Doc."""\n    return 1\n'), {"foo"})
    assert "foo" not in real


def test_stub_definitions_error_recovery_ab():
    source = (
        'class AllProvidersFailedError(Exception):\n'
        '    """Raised when every provider fails."""\n'
        "\n"
        "class WeatherClient:\n"
        "    def __init__(self):\n"
        "        self.providers = {}\n"
        "\n"
        "    async def fetch(self, provider, city):\n"
        "        return {\"temp\": 1}\n"
        "\n"
        "async def get_weather_resilient(city: str, client: WeatherClient) -> dict:\n"
        "    return await client.fetch(\"WeatherAPI\", city)\n"
        "\n"
        "def demo() -> None:\n"
        '    print("ok")\n'
    )
    assert stub_definitions(ast.parse(source), {"AllProvidersFailedError"}) == []
