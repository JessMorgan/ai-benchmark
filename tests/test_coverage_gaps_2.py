"""Targeted tests for remaining coverage gaps in core.py, _analysis.py, state.py, and _execution.py."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# benchmark.core – pure helpers (no network)
# ---------------------------------------------------------------------------


class TestCoreHelpers:
    def test_is_repeating_false(self) -> None:
        from benchmark.core import is_repeating

        assert is_repeating("short text") is False

    def test_is_repeating_true(self) -> None:
        from benchmark.core import is_repeating

        seq = "x" * 80
        assert is_repeating(seq * 3) is True

    def test_is_repeating_threshold(self) -> None:
        from benchmark.core import is_repeating

        seq = "a" * 80
        assert is_repeating(seq * 2) is False
        assert is_repeating(seq * 3) is True

    def test_response_nature_completed(self) -> None:
        from benchmark.core import response_nature

        assert response_nature(text="hello", error=None, finish_reason="stop") == "completed"

    def test_response_nature_empty(self) -> None:
        from benchmark.core import response_nature

        assert response_nature(text="", error=None, finish_reason="stop") == "empty"
        assert response_nature(text="   ", error=None, finish_reason="stop") == "empty"

    def test_response_nature_cancelled(self) -> None:
        from benchmark.core import response_nature

        assert response_nature(text="", error="cancelled", finish_reason="stop") == "cancelled"
        assert response_nature(text="", error=None, finish_reason="stop", cancelled=True) == "cancelled"

    def test_response_nature_timeout(self) -> None:
        from benchmark.core import response_nature

        assert response_nature(text="", error="ReadTimeout", finish_reason="stop") == "timeout"
        assert response_nature(text="", error="timed out", finish_reason="stop") == "timeout"

    def test_response_nature_token_limit(self) -> None:
        from benchmark.core import response_nature

        assert response_nature(text="content", error=None, finish_reason="length") == "token_limit"
        assert response_nature(text="content", error="token limit exceeded", finish_reason="stop") == "token_limit"

    def test_response_nature_repetition(self) -> None:
        from benchmark.core import response_nature

        assert response_nature(text="x" * 300, error=None, finish_reason="stop", repeating=True) == "repetition_abort"

    def test_response_nature_transport_error(self) -> None:
        from benchmark.core import response_nature

        assert response_nature(text="", error="500 Internal Server Error", finish_reason="stop") == "transport_error"

    def test_count_tokens(self) -> None:
        from benchmark.core import count_tokens

        assert count_tokens("") == 0
        assert count_tokens("x" * 4) == 1
        assert count_tokens("x" * 8) == 2

    def test_classify_empty_reason(self) -> None:
        from benchmark.core import classify_empty_reason

        assert classify_empty_reason("real text") is None
        assert classify_empty_reason("", finish_reason="stop") == "empty"
        assert classify_empty_reason("", error="timeout") == "error"
        assert classify_empty_reason("", finish_reason="length", think_text="x" * 100) == "thinking-truncation"
        assert classify_empty_reason("", think_text="x" * 100) == "thinking-only"
        assert classify_empty_reason("", finish_reason="length") == "max-tokens"

    def test_parse_judge_response_valid(self) -> None:
        from benchmark.core import parse_judge_response

        text = json.dumps({
            "score": 85, "confidence": "high", "rationale": "Good response",
            "criteria": [{"id": "c1", "criterion": "Test", "status": "met", "evidence": "It met"}],
        })
        result = parse_judge_response(text)
        assert result.score == 85
        assert result.confidence == "high"
        assert result.error is None

    def test_parse_judge_response_fenced(self) -> None:
        from benchmark.core import parse_judge_response

        text = "```json\n" + json.dumps({
            "score": 70, "confidence": "medium", "rationale": "OK",
            "criteria": [{"id": "c1", "criterion": "Test", "status": "partial", "evidence": "Sort of"}],
        }) + "\n```"
        result = parse_judge_response(text)
        assert result.score == 70
        assert result.error is None

    def test_parse_judge_response_malformed_fence(self) -> None:
        from benchmark.core import parse_judge_response

        result = parse_judge_response("```json\nno closing")
        assert result.error is not None

    def test_parse_judge_response_invalid_json(self) -> None:
        from benchmark.core import parse_judge_response

        result = parse_judge_response("not json at all")
        assert "invalid judge JSON" in result.error

    def test_parse_judge_response_non_dict(self) -> None:
        from benchmark.core import parse_judge_response

        result = parse_judge_response("[1, 2, 3]")
        assert "JSON object" in result.error

    def test_parse_judge_response_bad_score(self) -> None:
        from benchmark.core import parse_judge_response

        result = parse_judge_response(json.dumps({
            "score": "high", "confidence": "high", "rationale": "x",
            "criteria": [{"id": "c1", "criterion": "T", "status": "met", "evidence": "e"}],
        }))
        assert "score must be numeric" in result.error

    def test_parse_judge_response_score_out_of_range(self) -> None:
        from benchmark.core import parse_judge_response

        result = parse_judge_response(json.dumps({
            "score": 150, "confidence": "high", "rationale": "x",
            "criteria": [{"id": "c1", "criterion": "T", "status": "met", "evidence": "e"}],
        }))
        assert "score must be numeric" in result.error

    def test_parse_judge_response_score_bool(self) -> None:
        from benchmark.core import parse_judge_response

        result = parse_judge_response(json.dumps({
            "score": True, "confidence": "high", "rationale": "x",
            "criteria": [{"id": "c1", "criterion": "T", "status": "met", "evidence": "e"}],
        }))
        assert "score must be numeric" in result.error

    def test_parse_judge_response_bad_confidence(self) -> None:
        from benchmark.core import parse_judge_response

        result = parse_judge_response(json.dumps({
            "score": 50, "confidence": "very_high", "rationale": "x",
            "criteria": [{"id": "c1", "criterion": "T", "status": "met", "evidence": "e"}],
        }))
        assert "confidence must be" in result.error

    def test_parse_judge_response_empty_rationale(self) -> None:
        from benchmark.core import parse_judge_response

        result = parse_judge_response(json.dumps({
            "score": 50, "confidence": "high", "rationale": "",
            "criteria": [{"id": "c1", "criterion": "T", "status": "met", "evidence": "e"}],
        }))
        assert "rationale" in result.error

    def test_parse_judge_response_no_criteria(self) -> None:
        from benchmark.core import parse_judge_response

        result = parse_judge_response(json.dumps({
            "score": 50, "confidence": "high", "rationale": "x",
            "criteria": [],
        }))
        assert "criteria" in result.error

    def test_parse_judge_response_bad_criterion(self) -> None:
        from benchmark.core import parse_judge_response

        result = parse_judge_response(json.dumps({
            "score": 50, "confidence": "high", "rationale": "x",
            "criteria": ["not a dict"],
        }))
        assert "criterion 1 must be an object" in result.error

    def test_parse_judge_response_criterion_no_id(self) -> None:
        from benchmark.core import parse_judge_response

        result = parse_judge_response(json.dumps({
            "score": 50, "confidence": "high", "rationale": "x",
            "criteria": [{"criterion": "T", "status": "met", "evidence": "e"}],
        }))
        assert "criterion 1 must have a non-empty id" in result.error

    def test_parse_judge_response_criterion_bad_status(self) -> None:
        from benchmark.core import parse_judge_response

        result = parse_judge_response(json.dumps({
            "score": 50, "confidence": "high", "rationale": "x",
            "criteria": [{"id": "c1", "criterion": "T", "status": "maybe", "evidence": "e"}],
        }))
        assert "invalid status" in result.error

    def test_parse_judge_response_criterion_no_evidence(self) -> None:
        from benchmark.core import parse_judge_response

        result = parse_judge_response(json.dumps({
            "score": 50, "confidence": "high", "rationale": "x",
            "criteria": [{"id": "c1", "criterion": "T", "status": "met", "evidence": ""}],
        }))
        assert "evidence" in result.error

    def test_judge_vote_identity(self) -> None:
        from benchmark.core import judge_vote_identity

        vote = {"model": "judge-1", "judge_contract_id": "contract-1"}
        assert judge_vote_identity(vote) == ("judge-1", "contract-1")
        assert judge_vote_identity({}) == (None, None)
        assert judge_vote_identity("not a dict") == (None, None)

    def test_judge_votes_for_contract(self) -> None:
        from benchmark.core import judge_votes_for_contract

        votes = [
            {"model": "j1", "judge_contract_id": "c1"},
            {"model": "j2", "judge_contract_id": "c2"},
            {"model": "j3", "judge_contract_id": "c1"},
        ]
        result = judge_votes_for_contract(votes, "c1")
        assert len(result) == 2
        assert all(v["judge_contract_id"] == "c1" for v in result)

    def test_merge_judge_vote(self) -> None:
        from benchmark.core import merge_judge_vote

        votes = [
            {"model": "j1", "judge_contract_id": "c1", "score": 10},
            {"model": "j1", "judge_contract_id": "c2", "score": 20},
        ]
        new_vote = {"model": "j1", "judge_contract_id": "c1", "score": 15}
        result = merge_judge_vote(votes, new_vote)
        assert len(result) == 2
        c1_votes = [v for v in result if v["judge_contract_id"] == "c1"]
        assert len(c1_votes) == 1
        assert c1_votes[0]["score"] == 15

    def test_parse_plugin_temperatures(self) -> None:
        from benchmark.core import parse_plugin_temperatures

        cfg = {
            "rate-limiter_temperature": 0.5,
            "tool-calling_temperature": 0.3,
            "other_key": "value",
        }
        result = parse_plugin_temperatures(cfg)
        assert result["rate-limiter"] == 0.5
        assert result["tool-calling"] == 0.3
        assert "other_key" not in result

    def test_resolve_model_sources(self) -> None:
        from benchmark.core import resolve_model_sources

        models = {
            "model-a": "http",
            "model-b": {"source": "ollama", "drop_params": []},
            "model-c": 123,
        }
        result = resolve_model_sources(models)
        assert result["model-a"] == "http"
        assert result["model-b"] == "ollama"
        assert result["model-c"] == "Default"

    def test_get_target_plugins_blacklist(self) -> None:
        from benchmark.core import get_target_plugins_blacklist

        targets = {"m1": {"plugins_blacklist": ["p1", "p2"]}}
        assert get_target_plugins_blacklist(targets, "m1") == ["p1", "p2"]
        assert get_target_plugins_blacklist(targets, "m2") == []

    def test_is_schema_grammar_error(self) -> None:
        from benchmark.core import _is_schema_grammar_error

        assert _is_schema_grammar_error("Failed to initialize samplers") is True
        assert _is_schema_grammar_error("grammar sampler error") is True
        assert _is_schema_grammar_error("error initializing grammar") is True
        assert _is_schema_grammar_error("failed to parse grammar") is True
        assert _is_schema_grammar_error("normal error") is False
        assert _is_schema_grammar_error(None) is False

    def test_schema_probe_error_status(self) -> None:
        from benchmark.core import _schema_probe_error_status

        assert _schema_probe_error_status("HTTP 400 schema error") == "schema_rejected"
        assert _schema_probe_error_status("HTTP 422 grammar error") == "schema_rejected"
        assert _schema_probe_error_status("failed to parse grammar") == "schema_rejected"
        assert _schema_probe_error_status("connection reset") == "schema_transport_error"
        assert _schema_probe_error_status(None) == "schema_transport_error"

    def test_json_object_fallback_params(self) -> None:
        from benchmark.core import _json_object_fallback_params

        params = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "test", "schema": {}},
            }
        }
        result = _json_object_fallback_params(params)
        assert result is not None
        assert result["response_format"]["type"] == "json_object"
        # Original should be unchanged
        assert params["response_format"]["type"] == "json_schema"

    def test_json_object_fallback_params_non_dict(self) -> None:
        from benchmark.core import _json_object_fallback_params

        assert _json_object_fallback_params("not a dict") is None

    def test_json_object_fallback_params_no_schema(self) -> None:
        from benchmark.core import _json_object_fallback_params

        assert _json_object_fallback_params({"response_format": {"type": "json_object"}}) is None

    def test_source_abbrev(self) -> None:
        from benchmark.core import _source_abbrev

        assert _source_abbrev("http") == "HT"
        assert _source_abbrev("My Custom Source") == "MCS"

    def test_response_reasoning_tokens(self) -> None:
        from benchmark.core import _response_reasoning_tokens

        response = MagicMock()
        response.usage = {"completion_tokens_details": {"reasoning_tokens": 42}}
        response.think_text = ""
        assert _response_reasoning_tokens(response) == 42

    def test_response_reasoning_tokens_think_text(self) -> None:
        from benchmark.core import _response_reasoning_tokens

        response = MagicMock()
        response.usage = {}
        response.think_text = "x" * 40  # 10 tokens at char/4
        assert _response_reasoning_tokens(response) == 10

    def test_response_reasoning_tokens_none(self) -> None:
        from benchmark.core import _response_reasoning_tokens

        response = MagicMock()
        response.usage = {}
        response.think_text = ""
        assert _response_reasoning_tokens(response) is None

    def test_resolve_pi_config_none(self) -> None:
        from benchmark.core import _resolve_pi_config

        assert _resolve_pi_config("m1", None) == {}

    def test_resolve_pi_config_invalid_type(self) -> None:
        from benchmark.core import _resolve_pi_config

        with pytest.raises(ValueError, match="must be an object"):
            _resolve_pi_config("m1", "bad")

    def test_resolve_pi_config_unknown_key(self) -> None:
        from benchmark.core import _resolve_pi_config

        with pytest.raises(ValueError, match="unsupported"):
            _resolve_pi_config("m1", {"unknown_key": True})

    def test_resolve_pi_config_invalid_tools(self) -> None:
        from benchmark.core import _resolve_pi_config

        with pytest.raises(ValueError, match="list of strings"):
            _resolve_pi_config("m1", {"tools": "not a list"})

    def test_resolve_pi_config_unsupported_tool(self) -> None:
        from benchmark.core import _resolve_pi_config

        with pytest.raises(ValueError, match="unsupported tool"):
            _resolve_pi_config("m1", {"tools": ["read", "dangerous_tool"]})

    def test_resolve_pi_config_invalid_permissions(self) -> None:
        from benchmark.core import _resolve_pi_config

        with pytest.raises(ValueError, match="must map"):
            _resolve_pi_config("m1", {"permissions": "bad"})

    def test_resolve_pi_config_valid(self) -> None:
        from benchmark.core import _resolve_pi_config

        result = _resolve_pi_config("m1", {"tools": ["read", "bash"], "permissions": {"read": "allow"}})
        assert result["tools"] == ["read", "bash"]
        assert result["permissions"] == {"read": "allow"}


class TestCoreSchemaClassification:
    def test_schema_request_metadata_not_requested(self) -> None:
        from benchmark.core import _schema_request_metadata

        plugin = MagicMock()
        plugin.get_response_schema.return_value = None
        result = _schema_request_metadata(plugin, {}, response_schema_valid=None)
        assert result["schema_request_status"] == "schema_not_requested"

    def test_schema_request_metadata_not_applied(self) -> None:
        from benchmark.core import _schema_request_metadata

        plugin = MagicMock()
        plugin.get_response_schema.return_value = {"type": "object"}
        result = _schema_request_metadata(
            plugin, {}, request_applied=False,
        )
        assert result["schema_request_status"] == "schema_not_applied_by_runner"

    def test_schema_request_metadata_fallback_valid(self) -> None:
        from benchmark.core import _schema_request_metadata

        plugin = MagicMock()
        plugin.get_response_schema.return_value = {"type": "object"}
        result = _schema_request_metadata(
            plugin, {}, schema_fallback_used=True, response_schema_valid=True,
        )
        assert result["schema_request_status"] == "schema_fallback_json_object_valid"

    def test_schema_request_metadata_fallback_invalid(self) -> None:
        from benchmark.core import _schema_request_metadata

        plugin = MagicMock()
        plugin.get_response_schema.return_value = {"type": "object"}
        result = _schema_request_metadata(
            plugin, {}, schema_fallback_used=True, response_schema_valid=False,
        )
        assert result["schema_request_status"] == "schema_fallback_json_object_invalid"

    def test_schema_request_metadata_fallback_unknown(self) -> None:
        from benchmark.core import _schema_request_metadata

        plugin = MagicMock()
        plugin.get_response_schema.return_value = {"type": "object"}
        result = _schema_request_metadata(
            plugin, {}, schema_fallback_used=True, response_schema_valid=None)
        assert result["schema_request_status"] == "schema_fallback_json_object_unknown"

    def test_schema_request_metadata_fallback_failed(self) -> None:
        from benchmark.core import _schema_request_metadata

        plugin = MagicMock()
        plugin.get_response_schema.return_value = {"type": "object"}
        result = _schema_request_metadata(
            plugin, {}, schema_fallback_used=True, error="fail")
        assert result["schema_request_status"] == "schema_fallback_json_object_failed"

    def test_schema_request_metadata_accepted_valid(self) -> None:
        from benchmark.core import _schema_request_metadata

        plugin = MagicMock()
        plugin.get_response_schema.return_value = {"type": "object"}
        result = _schema_request_metadata(
            plugin, {}, response_schema_valid=True,
        )
        assert result["schema_request_status"] == "schema_accepted_valid"

    def test_schema_request_metadata_accepted_invalid(self) -> None:
        from benchmark.core import _schema_request_metadata

        plugin = MagicMock()
        plugin.get_response_schema.return_value = {"type": "object"}
        result = _schema_request_metadata(
            plugin, {}, response_schema_valid=False)
        assert result["schema_request_status"] == "schema_accepted_invalid"

    def test_schema_request_metadata_accepted_unknown(self) -> None:
        from benchmark.core import _schema_request_metadata

        plugin = MagicMock()
        plugin.get_response_schema.return_value = {"type": "object"}
        result = _schema_request_metadata(
            plugin, {}, response_schema_valid=None)
        assert result["schema_request_status"] == "schema_accepted_unknown"

    def test_schema_request_metadata_rejected(self) -> None:
        from benchmark.core import _schema_request_metadata

        plugin = MagicMock()
        plugin.get_response_schema.return_value = {"type": "object"}
        result = _schema_request_metadata(
            plugin, {}, error="HTTP 400 schema rejected")
        assert result["schema_request_status"] == "schema_rejected"

    def test_schema_request_metadata_accepted_invalid_error(self) -> None:
        from benchmark.core import _schema_request_metadata

        plugin = MagicMock()
        plugin.get_response_schema.return_value = {"type": "object"}
        result = _schema_request_metadata(
            plugin, {}, error="invalid completion response")
        assert result["schema_request_status"] == "schema_accepted_invalid"

    def test_schema_request_metadata_transport_error(self) -> None:
        from benchmark.core import _schema_request_metadata

        plugin = MagicMock()
        plugin.get_response_schema.return_value = {"type": "object"}
        result = _schema_request_metadata(
            plugin, {}, error="connection reset")
        assert result["schema_request_status"] == "schema_transport_error"


# ---------------------------------------------------------------------------
# plugins.challenges._analysis
# ---------------------------------------------------------------------------


class TestAnalysis:
    def test_normalize_heading(self) -> None:
        from plugins.challenges._analysis import normalize_heading

        assert normalize_heading("## **Hello World**") == "hello world"
        assert normalize_heading("# Section 1.2:") == "section 1 2"
        assert normalize_heading("  PLAINTEXT  ") == "plaintext"

    def test_markdown_sections(self) -> None:
        from plugins.challenges._analysis import markdown_sections

        text = "# Intro\nBody 1\n## Section A\nBody 2\n### Deep\nBody 3"
        sections = markdown_sections(text)
        assert len(sections) == 3
        assert sections[0].heading == "Intro"
        assert sections[1].heading == "Section A"
        assert sections[2].heading == "Deep"

    def test_markdown_sections_skips_code_fences(self) -> None:
        from plugins.challenges._analysis import markdown_sections

        text = "# Real\nContent\n```\n# Fake Heading\nIn code\n```\n## Also Real\nMore"
        sections = markdown_sections(text)
        assert len(sections) == 2
        assert sections[0].heading == "Real"
        assert sections[1].heading == "Also Real"

    def test_exact_section(self) -> None:
        from plugins.challenges._analysis import exact_section

        text = "# Architecture\nThe system uses...\n## Details\nMore info"
        section = exact_section(text, "Architecture")
        assert section is not None
        assert "system uses" in section.body

    def test_exact_section_with_alias(self) -> None:
        from plugins.challenges._analysis import exact_section

        text = "# **Design**\nContent"
        section = exact_section(text, "SomethingElse", aliases=["Design"])
        assert section is not None

    def test_exact_section_not_found(self) -> None:
        from plugins.challenges._analysis import exact_section

        assert exact_section("# Title\nBody", "Nonexistent") is None

    def test_section_bodies(self) -> None:
        from plugins.challenges._analysis import section_bodies

        text = "# A\nBody A\n# B\nBody B\n# C\nBody C"
        bodies = section_bodies(text, ["A", "C"])
        assert len(bodies) == 2
        assert "Body A" in bodies[0]
        assert "Body C" in bodies[1]

    def test_matching_sections(self) -> None:
        from plugins.challenges._analysis import matching_sections

        text = "# Security Architecture\nSec content\n## Network Design\nNet content"
        sections = matching_sections(text, ["architecture"])
        assert len(sections) == 1

    def test_first_section(self) -> None:
        from plugins.challenges._analysis import first_section

        text = "# First\nBody1\n# Second\nBody2"
        section = first_section(text, ["First"])
        assert section is not None
        assert "Body1" in section.body

    def test_first_section_not_found(self) -> None:
        from plugins.challenges._analysis import first_section

        assert first_section("# Title\nBody", ["Missing"]) is None

    def test_section_has_content(self) -> None:
        from plugins.challenges._analysis import Section, section_has_content

        assert section_has_content(None) is False
        assert section_has_content(Section("h", "h", "", 0), minimum=20) is False
        assert section_has_content(Section("h", "h", "x" * 30, 0), minimum=20) is True

    def test_numbered_or_bulleted_items(self) -> None:
        from plugins.challenges._analysis import numbered_or_bulleted_items

        body = "1. First item\n2. Second item\n- Bullet one\n* Bullet two\nRegular text"
        items = numbered_or_bulleted_items(body)
        assert len(items) == 4
        assert "First item" in items[0]

    def test_distinct_normalized(self) -> None:
        from plugins.challenges._analysis import distinct_normalized

        result = distinct_normalized(["Hello", "hello", "HELLO", "World", "  World  "])
        assert result == ["hello", "world"]

    def test_has_real_code_block(self) -> None:
        from plugins.challenges._analysis import has_real_code_block

        text = "```python\nprint('hello')\n```"
        assert has_real_code_block(text, "python") is True
        assert has_real_code_block(text, "javascript") is False
        assert has_real_code_block("no code", "python") is False

    def test_fenced_blocks(self) -> None:
        from plugins.challenges._analysis import fenced_blocks

        text = "Text\n```python\ncode1\n```\nMore\n```javascript\ncode2\n```"
        blocks = fenced_blocks(text)
        assert len(blocks) == 2

    def test_fenced_blocks_filtered(self) -> None:
        from plugins.challenges._analysis import fenced_blocks

        text = "```python\ncode1\n```\n```javascript\ncode2\n```"
        blocks = fenced_blocks(text, "python")
        assert len(blocks) == 1

    def test_text_without_fences(self) -> None:
        from plugins.challenges._analysis import text_without_fences

        text = "Before\n```python\ncode\n```\nAfter"
        result = text_without_fences(text)
        assert "code" not in result
        assert "Before" in result
        assert "After" in result

    def test_mermaid_graph(self) -> None:
        from plugins.challenges._analysis import mermaid_graph

        text = "```mermaid\nA[Box] --> B[Box2]\nC --> D\n```"
        nodes, edges = mermaid_graph(text)
        assert "A" in nodes
        assert "B" in nodes
        assert ("A", "B") in edges

    def test_mermaid_graph_no_edges(self) -> None:
        from plugins.challenges._analysis import mermaid_graph

        text = "```mermaid\nA[Box]\n```"
        nodes, edges = mermaid_graph(text)
        assert "A" in nodes
        assert len(edges) == 0


# ---------------------------------------------------------------------------
# plugins.challenges._execution
# ---------------------------------------------------------------------------


class TestExecution:
    def test_execution_result_dataclass(self) -> None:
        from plugins.challenges._execution import ExecutionResult

        result = ExecutionResult("passed", passed=True, output="ok")
        assert result.status == "passed"
        assert result.passed is True
        assert result.output == "ok"
        assert result.error is None

    def test_execution_result_default(self) -> None:
        from plugins.challenges._execution import ExecutionResult

        result = ExecutionResult("failed")
        assert result.passed is False
        assert result.output == ""

    def test_run_python_check_local(self) -> None:
        from plugins.challenges._execution import run_python_check

        result = run_python_check("x = 1\nassert x == 1", "assert True", timeout=5.0)
        assert result.passed is True

    def test_run_python_check_failure(self) -> None:
        from plugins.challenges._execution import run_python_check

        result = run_python_check("x = 1\nassert x == 2", "assert True", timeout=5.0)
        assert result.passed is False


# ---------------------------------------------------------------------------
# benchmark.state – recovery and hydration
# ---------------------------------------------------------------------------


class TestStateRecovery:
    def test_prepare_state_recovery_valid(self, tmp_path: Path) -> None:
        from benchmark.state import prepare_state_recovery

        state = {
            "model_info": {"m1": {"status": "completed"}},
            "results": [{"model": "m1", "http_score": 10}],
            "active_plugins": ["http"],
            "plugin_versions": {},
        }
        path = tmp_path / "state.json"
        path.write_text(json.dumps(state))
        recovery = prepare_state_recovery(str(path))
        assert recovery["kind"] in ("known", "partial")
        assert recovery["recoverable_results"] == 1
        assert recovery["results_found"] is True

    def test_apply_state_recovery(self, tmp_path: Path) -> None:
        from benchmark.state import apply_state_recovery, prepare_state_recovery

        state = {
            "model_info": {"m1": {"status": "completed"}},
            "results": [{"model": "m1", "http_score": 10}],
            "active_plugins": ["http"],
            "plugin_versions": {},
        }
        path = tmp_path / "state.json"
        path.write_text(json.dumps(state))
        recovery = prepare_state_recovery(str(path))
        backup = apply_state_recovery(str(path), recovery)
        assert os.path.exists(backup)
        # State file should still be valid
        with open(path) as f:
            data = json.load(f)
        assert len(data["results"]) == 1

    def test_repair_state_file_corrupted(self, tmp_path: Path) -> None:
        from benchmark.state import repair_state_file

        # Write a file with the exact known corrupted byte pattern
        # The corruption must be inside the model_info region
        raw_parts = [
            '{\n  "model_info": {\n    "m1": {},\n    "moe-dense_first_chunk_see: : false,\n    "m2": {}\n  },\n',
            '  "results": [{"model": "m1", "http_score": 10}],\n',
            '  "active_plugins": ["http"],\n  "plugin_versions": {}\n}',
        ]
        raw = "".join(raw_parts)
        path = tmp_path / "state.json"
        path.write_text(raw)
        backup = repair_state_file(str(path))
        assert backup is not None

    def test_repair_state_file_not_repairable(self, tmp_path: Path) -> None:
        from benchmark.state import repair_state_file

        path = tmp_path / "state.json"
        path.write_text("{invalid json")
        assert repair_state_file(str(path)) is None

    def test_top_level_value(self) -> None:
        from benchmark.state import _top_level_value

        data = '{"session_seed": 42, "results": []}'
        assert _top_level_value(data.encode(), "session_seed") == 42

    def test_top_level_value_missing(self) -> None:
        from benchmark.state import _top_level_value

        assert _top_level_value(b'{"a": 1}', "missing") is None

    def test_valid_state_data(self) -> None:
        from benchmark.state import _valid_state_data

        assert _valid_state_data({
            "model_info": {}, "results": [], "active_plugins": [],
        }) is True
        assert _valid_state_data({}) is False
        assert _valid_state_data({"model_info": {}, "results": "bad", "active_plugins": []}) is False


# ---------------------------------------------------------------------------
# benchmark.storage – SQLiteRunStore edge cases
# ---------------------------------------------------------------------------


class TestSQLiteRunStoreEdgeCases:
    def test_prepare_run_before_start(self) -> None:
        from benchmark.runtime_records import TargetRecord
        from benchmark.storage import SQLiteRunStore

        store = SQLiteRunStore(":memory:")
        with pytest.raises(RuntimeError, match="must be started"):
            store.prepare_run([TargetRecord("m", "http", "local", "m", "s")], [])

    def test_register_target_before_start(self) -> None:
        from benchmark.runtime_records import TargetRecord
        from benchmark.storage import SQLiteRunStore

        store = SQLiteRunStore(":memory:")
        with pytest.raises(RuntimeError, match="must be started"):
            store.register_target(TargetRecord("m", "http", "local", "m", "s"))

    def test_register_plugin_before_start(self) -> None:
        from benchmark.runtime_records import PluginRecord
        from benchmark.storage import SQLiteRunStore

        store = SQLiteRunStore(":memory:")
        with pytest.raises(RuntimeError, match="must be started"):
            store.register_plugin(PluginRecord("p", "1", "P", 20, True))

    def test_ensure_cell_before_start(self) -> None:
        from benchmark.runtime_records import PluginRecord, TargetRecord
        from benchmark.storage import SQLiteRunStore

        store = SQLiteRunStore(":memory:")
        with pytest.raises(RuntimeError, match="target/run must be registered"):
            store.ensure_cell(TargetRecord("m", "http", "local", "m", "s"), PluginRecord("p", "1", "P", 20, True))

    def test_record_benchmark_attempt_before_start(self) -> None:
        from benchmark.runtime_records import BenchmarkAttemptRecord
        from benchmark.storage import SQLiteRunStore

        store = SQLiteRunStore(":memory:")
        with pytest.raises(RuntimeError, match="must be started"):
            store.record_benchmark_attempt(1, BenchmarkAttemptRecord(attempt_number=1))

    def test_record_judge_attempt_before_start(self) -> None:
        from benchmark.runtime_records import JudgeAttemptRecord
        from benchmark.storage import SQLiteRunStore

        store = SQLiteRunStore(":memory:")
        with pytest.raises(RuntimeError, match="must be started"):
            store.record_judge_attempt(1, JudgeAttemptRecord(judge_model="j", contract_id="c", attempt_number=1))

    def test_register_judge_before_start(self) -> None:
        from benchmark.storage import SQLiteRunStore

        store = SQLiteRunStore(":memory:")
        with pytest.raises(RuntimeError, match="must be started"):
            store.register_judge("judge", "http")

    def test_register_contract_before_start(self) -> None:
        from benchmark.storage import SQLiteRunStore

        store = SQLiteRunStore(":memory:")
        with pytest.raises(RuntimeError, match="must be started"):
            store.register_contract("c", plugin_id="p", plugin_version="1",
                                    prompt_version="1", instructions_version="1")

    def test_latest_results_empty_revision(self) -> None:
        from benchmark.storage import SQLiteRunStore

        store = SQLiteRunStore(":memory:")
        store.start_run = MagicMock()
        store._revision_id = None
        store._results = {("m1", "http"): {"model": "m1"}}
        assert store.latest_results() == [{"model": "m1"}]


# ---------------------------------------------------------------------------
# benchmark.logs – redaction edge cases
# ---------------------------------------------------------------------------


class TestLogsRedaction:
    def test_redact_password_header(self) -> None:
        from benchmark.logs import redact_log_text

        text = "password: hunter2"
        result, changed = redact_log_text(text)
        assert changed is True
        assert "hunter2" not in result

    def test_redact_command_line_api_key(self) -> None:
        from benchmark.logs import redact_log_text

        text = 'api_key="sk-1234567890"'
        result, changed = redact_log_text(text)
        assert changed is True
        assert "sk-1234567890" not in result

    def test_redact_set_cookie(self) -> None:
        from benchmark.logs import redact_log_text

        text = "set-cookie: session=abc123"
        result, changed = redact_log_text(text)
        assert changed is True
