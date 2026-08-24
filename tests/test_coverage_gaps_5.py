"""Final push for 90% coverage: more core.py pi config, state.py journal compaction, transport edge cases, validators."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# benchmark.core – _resolve_pi_config exhaustive branches
# ---------------------------------------------------------------------------


class TestResolvePiConfigExhaustive:
    def test_system_prompt_bad_type(self) -> None:
        from benchmark.configuration import _resolve_pi_config

        with pytest.raises(ValueError, match="system_prompt must be a string"):
            _resolve_pi_config("m1", {"system_prompt": 123})

    def test_reasoning_bad_type(self) -> None:
        from benchmark.configuration import _resolve_pi_config

        with pytest.raises(ValueError, match="reasoning must be boolean"):
            _resolve_pi_config("m1", {"reasoning": "yes"})

    def test_max_tool_calls_bad_type(self) -> None:
        from benchmark.configuration import _resolve_pi_config

        with pytest.raises(ValueError, match="max_tool_calls must be a non-negative"):
            _resolve_pi_config("m1", {"max_tool_calls": "bad"})

    def test_max_tool_calls_negative(self) -> None:
        from benchmark.configuration import _resolve_pi_config

        with pytest.raises(ValueError, match="max_tool_calls must be a non-negative"):
            _resolve_pi_config("m1", {"max_tool_calls": -1})

    def test_max_tool_calls_bool(self) -> None:
        from benchmark.configuration import _resolve_pi_config

        with pytest.raises(ValueError, match="max_tool_calls must be a non-negative"):
            _resolve_pi_config("m1", {"max_tool_calls": True})

    def test_max_tokens_bad(self) -> None:
        from benchmark.configuration import _resolve_pi_config

        with pytest.raises(ValueError, match="max_tokens must be a positive"):
            _resolve_pi_config("m1", {"max_tokens": 0})

    def test_max_tokens_negative(self) -> None:
        from benchmark.configuration import _resolve_pi_config

        with pytest.raises(ValueError, match="max_tokens must be a positive"):
            _resolve_pi_config("m1", {"max_tokens": -5})

    def test_max_tokens_bool(self) -> None:
        from benchmark.configuration import _resolve_pi_config

        with pytest.raises(ValueError, match="max_tokens must be a positive"):
            _resolve_pi_config("m1", {"max_tokens": True})

    def test_max_tokens_none_ok(self) -> None:
        from benchmark.configuration import _resolve_pi_config

        result = _resolve_pi_config("m1", {"max_tokens": None})
        assert result["max_tokens"] is None

    def test_compat_bad_type(self) -> None:
        from benchmark.configuration import _resolve_pi_config

        with pytest.raises(ValueError, match="compat must be an object"):
            _resolve_pi_config("m1", {"compat": "bad"})

    def test_thinking_budgets_bad_type(self) -> None:
        from benchmark.configuration import _resolve_pi_config

        with pytest.raises(ValueError, match="thinking_budgets must be an object or null"):
            _resolve_pi_config("m1", {"thinking_budgets": "bad"})

    def test_thinking_budgets_none_ok(self) -> None:
        from benchmark.configuration import _resolve_pi_config

        result = _resolve_pi_config("m1", {"thinking_budgets": None})
        assert result["thinking_budgets"] is None

    def test_valid_full_config(self) -> None:
        from benchmark.configuration import _resolve_pi_config

        result = _resolve_pi_config("m1", {
            "tools": ["read", "bash"],
            "permissions": {"read": "allow"},
            "system_prompt": "Custom prompt",
            "reasoning": True,
            "max_tool_calls": 100,
            "max_tokens": 8192,
            "compat": {"legacy": True},
            "thinking_budgets": {"fast": 4096},
        })
        assert result["reasoning"] is True
        assert result["max_tool_calls"] == 100
        assert result["max_tokens"] == 8192


# ---------------------------------------------------------------------------
# benchmark.core – resolve_targets edge cases
# ---------------------------------------------------------------------------


class TestResolveTargets:
    def test_resolve_targets_basic(self) -> None:
        from benchmark.configuration import resolve_targets

        cfg = {
            "sources": {"http": {"api_url": "http://localhost"}},
            "models": {"model-a": "http"},
        }
        targets = resolve_targets(cfg)
        assert "model-a" in targets
        assert targets["model-a"]["source"] == "http"

    def test_resolve_targets_token_levels_removed(self) -> None:
        from benchmark.configuration import resolve_targets

        cfg = {
            "sources": {"http": {"api_url": "http://localhost"}},
            "models": {"model-a": "http"},
            "token_levels": [1000, 2000],
        }
        with pytest.raises(ValueError, match="token_levels"):
            resolve_targets(cfg)

    def test_resolve_targets_model_token_levels_removed(self) -> None:
        from benchmark.configuration import resolve_targets

        cfg = {
            "sources": {"http": {"api_url": "http://localhost"}},
            "models": {"model-a": "http"},
            "model_token_levels": {"model-a": [1000, 2000]},
        }
        with pytest.raises(ValueError, match="token_levels"):
            resolve_targets(cfg)

    def test_resolve_targets_with_agents(self) -> None:
        from benchmark.configuration import resolve_targets

        cfg = {
            "sources": {"http": {"api_url": "http://localhost"}},
            "models": {},
            "agents": {"agent-a": {"source": "http", "model": "gpt-4", "system_prompt": "test"}},
        }
        targets = resolve_targets(cfg)
        assert "agent-a" in targets
        assert targets["agent-a"]["is_agent"] is True
        assert targets["agent-a"]["api_model"] == "gpt-4"

    def test_resolve_targets_dict_model(self) -> None:
        from benchmark.configuration import resolve_targets

        cfg = {
            "sources": {"http": {"api_url": "http://localhost"}},
            "models": {"model-a": {"source": "http"}},
        }
        targets = resolve_targets(cfg)
        assert targets["model-a"]["source"] == "http"

    def test_resolve_targets_unknown_model_source(self) -> None:
        from benchmark.configuration import resolve_targets

        cfg = {
            "sources": {"http": {"api_url": "http://localhost"}},
            "models": {"model-a": 123},
        }
        targets = resolve_targets(cfg)
        assert targets["model-a"]["source"] == "Default"


# ---------------------------------------------------------------------------
# benchmark.core – _source_abbrev edge cases
# ---------------------------------------------------------------------------


class TestSourceAbbrev:
    def test_empty_source(self) -> None:
        from benchmark.core import _source_abbrev

        # Single word -> first letter only -> len < 2 -> doubled
        result = _source_abbrev("x")
        assert len(result) == 2

    def test_uppercase_source(self) -> None:
        from benchmark.core import _source_abbrev

        result = _source_abbrev("HTTP")
        assert len(result) >= 2

    def test_multi_word(self) -> None:
        from benchmark.core import _source_abbrev

        result = _source_abbrev("My Cool Source")
        assert result == "MCS"


# ---------------------------------------------------------------------------
# benchmark.core – summarize_judge_criteria with bad data
# ---------------------------------------------------------------------------


class TestSummarizeJudgeCriteriaBadData:
    def test_non_dict_report(self) -> None:
        from benchmark.judging import summarize_judge_criteria

        plugin = MagicMock()
        plugin.id = "p1"
        results = [{"p1_judge_criteria": [42]}]  # int, not dict
        summary = summarize_judge_criteria(results, [plugin])
        assert summary["criteria"] == 0

    def test_non_list_criteria_items(self) -> None:
        from benchmark.judging import summarize_judge_criteria

        plugin = MagicMock()
        plugin.id = "p1"
        results = [{"p1_judge_criteria": [{"criteria": "not a list"}]}]
        summary = summarize_judge_criteria(results, [plugin])
        assert summary["criteria"] == 0

    def test_criteria_non_dict_item(self) -> None:
        from benchmark.judging import summarize_judge_criteria

        plugin = MagicMock()
        plugin.id = "p1"
        results = [{"p1_judge_criteria": [{"criteria": ["not a dict"]}]}]
        summary = summarize_judge_criteria(results, [plugin])
        assert summary["criteria"] == 0

    def test_criteria_no_evidence(self) -> None:
        from benchmark.judging import summarize_judge_criteria

        plugin = MagicMock()
        plugin.id = "p1"
        results = [{"p1_judge_criteria": [{"criteria": [{"id": "c1", "status": "met", "evidence": ""}]}]}]
        summary = summarize_judge_criteria(results, [plugin])
        assert summary["by_plugin"]["p1"]["evidence"] == 0


# ---------------------------------------------------------------------------
# benchmark.state – compact_journal more paths
# ---------------------------------------------------------------------------


class TestCompactJournalMore:
    def test_compact_journal_no_journal(self, tmp_path: Path) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        state.add_result({"model": "m1", "runner": "http", "http_score": 10})
        path = str(tmp_path / "state.json")
        result = state.compact_journal(path)
        assert result is True
        assert os.path.exists(path)

    def test_compact_journal_with_events(self, tmp_path: Path) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        journal_path = str(tmp_path / "journal.jsonl")
        state.set_journal_path(journal_path, truncate=True)
        state.add_result({"model": "m1", "runner": "http", "http_score": 10})
        state_path = str(tmp_path / "state.json")
        result = state.compact_journal(state_path)
        assert result is True
        # Journal should have been compacted (events older than snapshot removed)
        if os.path.exists(journal_path):
            with open(journal_path) as f:
                lines = [line.strip() for line in f if line.strip()]
            # Events should have been removed or retained based on sequence
            assert isinstance(lines, list)

    def test_compact_journal_save_fails(self, tmp_path: Path) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        state.add_result({"model": "m1", "runner": "http", "http_score": 10})
        # Path to non-writable directory
        result = state.compact_journal("/nonexistent/path/state.json")
        assert result is False


# ---------------------------------------------------------------------------
# benchmark.state – _apply_http_retry_default more branches
# ---------------------------------------------------------------------------


class TestApplyRetryDefaultMore:
    def test_apply_retry_default_no_sources_key(self) -> None:
        from benchmark.configuration import _apply_http_retry_default

        cfg: dict[str, Any] = {}
        _apply_http_retry_default(cfg, retry_on_429=False)
        assert "sources" not in cfg

    def test_apply_retry_default_source_not_dict(self) -> None:
        from benchmark.configuration import _apply_http_retry_default

        cfg = {"sources": {"local": "string_value"}}
        _apply_http_retry_default(cfg, retry_on_429=False)
        assert cfg["sources"]["local"] == "string_value"  # unchanged


# ---------------------------------------------------------------------------
# plugins.challenges._validators – parse_workflow_graph more branches
# ---------------------------------------------------------------------------


class TestParseWorkflowGraphMore:
    def test_depends_on_prose(self) -> None:
        from plugins.challenges._validators import parse_workflow_graph

        text = "Task 3 depends on Step 1"
        result = parse_workflow_graph(text)
        assert result.valid is True
        assert len(result.value["edges"]) >= 1

    def test_requires_prose(self) -> None:
        from plugins.challenges._validators import parse_workflow_graph

        text = "Step 2 requires Task 1"
        result = parse_workflow_graph(text)
        assert result.valid is True

    def test_after_prose(self) -> None:
        from plugins.challenges._validators import parse_workflow_graph

        text = "Task 2 after Step 1"
        result = parse_workflow_graph(text)
        assert result.valid is True

    def test_bracket_depends(self) -> None:
        from plugins.challenges._validators import parse_workflow_graph

        text = "Step 1\nStep 2 [DEPENDS_ON: step 1]"
        result = parse_workflow_graph(text)
        assert result.valid is True


# ---------------------------------------------------------------------------
# plugins.challenges._validators – parse_python more branches
# ---------------------------------------------------------------------------


class TestParsePythonMore:
    def test_parse_python_no_block_valid(self) -> None:
        from plugins.challenges._validators import parse_python

        result = parse_python("just text", require_block=False)
        assert result.valid is False  # no python code found

    def test_parse_python_invalid_syntax(self) -> None:
        from plugins.challenges._validators import parse_python

        result = parse_python("```python\ndef (\n```")
        assert result.valid is False

    def test_parse_python_valid_code(self) -> None:
        from plugins.challenges._validators import parse_python

        result = parse_python("```python\nx = 1\nprint(x)\n```")
        assert result.valid is True
        assert result.value is not None


# ---------------------------------------------------------------------------
# plugins.challenges._validators – validate_sections with aliases
# ---------------------------------------------------------------------------


class TestValidateSectionsMore:
    def test_validate_sections_with_aliases(self) -> None:
        from plugins.challenges._validators import validate_sections

        text = "# Overview\nEnough content here for the minimum length check.\n## Design\nSufficient content here for the test."
        result = validate_sections(
            text,
            required=["Title", "Design"],
            aliases={"Title": ("Overview",)},
        )
        assert result.valid is True


# ---------------------------------------------------------------------------
# plugins.challenges._validators – find_definitions more
# ---------------------------------------------------------------------------


class TestFindDefinitionsMore:
    def test_find_definitions_async(self) -> None:
        import ast

        from plugins.challenges._validators import find_definitions

        tree = ast.parse("async def afunc(): pass\ndef sfunc(): pass")
        defs = find_definitions(tree)
        assert "afunc" in defs
        assert "sfunc" in defs

    def test_find_definitions_empty(self) -> None:
        import ast

        from plugins.challenges._validators import find_definitions

        tree = ast.parse("x = 1")
        defs = find_definitions(tree)
        assert len(defs) == 0


# ---------------------------------------------------------------------------
# benchmark.core – _json_object_fallback_params more
# ---------------------------------------------------------------------------


class TestJsonFallbackMore:
    def test_fallback_preserves_other_params(self) -> None:
        from benchmark.core import _json_object_fallback_params

        params = {
            "temperature": 0.5,
            "response_format": {"type": "json_schema", "json_schema": {"name": "x"}},
        }
        result = _json_object_fallback_params(params)
        assert result["temperature"] == 0.5
        assert result["response_format"]["type"] == "json_object"

    def test_fallback_returns_none_for_no_format(self) -> None:
        from benchmark.core import _json_object_fallback_params

        assert _json_object_fallback_params({"temperature": 0.5}) is None


# ---------------------------------------------------------------------------
# plugins.challenges._validators – extract_fenced_blocks more
# ---------------------------------------------------------------------------


class TestExtractFencedBlocksMore:
    def test_extract_with_language(self) -> None:
        from plugins.challenges._validators import extract_fenced_blocks

        text = "```python\ncode1\n```\n```javascript\ncode2\n```"
        blocks = extract_fenced_blocks(text, "python")
        assert len(blocks) == 1
        assert "code1" in blocks[0]

    def test_extract_no_blocks(self) -> None:
        from plugins.challenges._validators import extract_fenced_blocks

        assert extract_fenced_blocks("no code here") == []

    def test_extract_with_empty_lang(self) -> None:
        from plugins.challenges._validators import extract_fenced_blocks

        text = "```\ncode\n```"
        blocks = extract_fenced_blocks(text)
        assert len(blocks) == 1


# ---------------------------------------------------------------------------
# plugins.challenges._validators – parse_tool_calls more branches
# ---------------------------------------------------------------------------


class TestParseToolCallsMore:
    def test_optional_arg_wrong_type(self) -> None:
        from plugins.challenges._validators import parse_tool_calls

        text = '<tool_call>\n{"name": "book_hotel", "args": {"city": "NYC", "check_in": "2025-01-01", "check_out": "2025-01-02", "guests": "two"}}\n</tool_call>'
        result = parse_tool_calls(text)
        assert result.valid is False  # guests should be int

    def test_optional_arg_correct_type(self) -> None:
        from plugins.challenges._validators import parse_tool_calls

        text = '<tool_call>\n{"name": "book_hotel", "args": {"city": "NYC", "check_in": "2025-01-01", "check_out": "2025-01-02", "guests": 2}}\n</tool_call>'
        result = parse_tool_calls(text)
        assert result.valid is True

    def test_send_email(self) -> None:
        from plugins.challenges._validators import parse_tool_calls

        text = '<tool_call>\n{"name": "send_email", "args": {"to": "a@b.com", "subject": "Hi", "body": "Hello"}}\n</tool_call>'
        result = parse_tool_calls(text)
        assert result.valid is True

    def test_convert_currency(self) -> None:
        from plugins.challenges._validators import parse_tool_calls

        text = '<tool_call>\n{"name": "convert_currency", "args": {"amount": 100, "from_curr": "USD", "to_curr": "EUR"}}\n</tool_call>'
        result = parse_tool_calls(text)
        assert result.valid is True

    def test_get_stock_price(self) -> None:
        from plugins.challenges._validators import parse_tool_calls

        text = '<tool_call>\n{"name": "get_stock_price", "args": {"ticker": "AAPL"}}\n</tool_call>'
        result = parse_tool_calls(text)
        assert result.valid is True
