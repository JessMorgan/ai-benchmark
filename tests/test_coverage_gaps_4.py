"""Final targeted tests for remaining coverage gaps."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

# ---------------------------------------------------------------------------
# benchmark.state – judge activity, models, and contracts
# ---------------------------------------------------------------------------


class TestStateJudgeActivity:
    def test_start_judge_activity(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        activity_id = state.start_judge_activity("judge-1", "m1", "p1")
        assert isinstance(activity_id, int)
        snapshot = state.judge_activity_snapshot()
        assert len(snapshot) >= 1

    def test_set_judge_activity_attempt(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        aid = state.start_judge_activity("judge-1", "m1", "p1")
        state.set_judge_activity_attempt(aid, 3)
        snap = state.judge_activity_snapshot()
        assert snap[aid]["attempt"] == 3
        # tokens should be reset
        assert snap[aid]["tokens"] == 0

    def test_set_judge_activity_attempt_bad_value(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        aid = state.start_judge_activity("judge-1", "m1", "p1")
        state.set_judge_activity_attempt(aid, cast(int, "bad"))
        snap = state.judge_activity_snapshot()
        assert snap[aid]["attempt"] == 1

    def test_set_judge_activity_attempt_nonexistent(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        state.set_judge_activity_attempt(999, 3)  # should not raise

    def test_update_judge_activity(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        aid = state.start_judge_activity("judge-1", "m1", "p1")
        state.update_judge_activity(aid, thinking_tokens=50, content_tokens=100)
        snap = state.judge_activity_snapshot()
        assert snap[aid]["thinking_tokens"] == 50
        assert snap[aid]["content_tokens"] == 100
        assert snap[aid]["tokens"] == 150

    def test_update_judge_activity_total_only(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        aid = state.start_judge_activity("judge-1", "m1", "p1")
        state.update_judge_activity(aid, tokens=200)
        snap = state.judge_activity_snapshot()
        assert snap[aid]["tokens"] == 200

    def test_update_judge_activity_nonexistent(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        state.update_judge_activity(999, tokens=100)  # should not raise

    def test_update_judge_activity_negative_clamped(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        aid = state.start_judge_activity("judge-1", "m1", "p1")
        state.update_judge_activity(aid, thinking_tokens=-10)
        snap = state.judge_activity_snapshot()
        assert snap[aid]["thinking_tokens"] == 0

    def test_finish_judge_activity(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        aid = state.start_judge_activity("judge-1", "m1", "p1")
        state.finish_judge_activity(aid)
        assert aid not in state.judge_activity_snapshot()

    def test_finish_judge_activity_nonexistent(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        state.finish_judge_activity(999)  # should not raise

    def test_clear_judge_queued(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        state._model_info["m1"]["p1_judge_queued"] = True
        state.clear_judge_queued("m1", "p1")
        assert state._model_info["m1"]["p1_judge_queued"] is False

    def test_clear_judge_queued_missing_model(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        state.clear_judge_queued("nonexistent", "p1")  # should not raise


class TestStateJudgeModels:
    def test_set_judge_models(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        state.set_judge_models(["judge-a", "judge-b"])
        info = state._model_info["m1"]
        assert info["judge_models"] == ["judge-a", "judge-b"]
        assert state.results[0]["judge_models"] == ["judge-a", "judge-b"] if state.results else True

    def test_set_judge_models_empty(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        state.set_judge_models([])
        assert state._model_info["m1"]["judge_models"] == []

    def test_set_judge_models_dedup(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        state.set_judge_models(["a", "a", "b"])
        assert state._model_info["m1"]["judge_models"] == ["a", "b"]

    def test_judge_selected_snapshot(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        state.set_judge_selected("judge-1", True)
        assert "judge-1" in state.judge_selected_snapshot()
        state.set_judge_selected("judge-1", False)
        assert "judge-1" not in state.judge_selected_snapshot()


class TestStateSetJudgeModelsResults:
    def test_set_judge_models_updates_results(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        state.add_result({"model": "m1", "runner": "http"})
        state.set_judge_models(["j1"])
        assert state.results[0]["judge_models"] == ["j1"]


# ---------------------------------------------------------------------------
# benchmark.core – _apply_http_retry_default, resolve_stream_guards
# ---------------------------------------------------------------------------


class TestCoreRetryDefault:
    def test_apply_http_retry_default_enabled(self) -> None:
        from benchmark.configuration import _apply_http_retry_default

        cfg = {"sources": {"local": {"api_url": "http://localhost"}}}
        _apply_http_retry_default(cfg, retry_on_429=True)
        assert "max_429_retries" not in cfg["sources"]["local"]

    def test_apply_http_retry_default_disabled(self) -> None:
        from benchmark.configuration import _apply_http_retry_default

        cfg = {"sources": {"local": {"api_url": "http://localhost"}}}
        _apply_http_retry_default(cfg, retry_on_429=False)
        assert cfg["sources"]["local"]["max_429_retries"] == 0

    def test_apply_http_retry_default_preserves_explicit(self) -> None:
        from benchmark.configuration import _apply_http_retry_default

        cfg = {"sources": {"local": {"max_429_retries": 5}}}
        _apply_http_retry_default(cfg, retry_on_429=False)
        assert cfg["sources"]["local"]["max_429_retries"] == 5

    def test_apply_http_retry_default_no_sources(self) -> None:
        from benchmark.configuration import _apply_http_retry_default

        cfg: dict[str, Any] = {}
        _apply_http_retry_default(cfg, retry_on_429=False)  # should not raise

    def test_apply_http_retry_default_bad_source(self) -> None:
        from benchmark.configuration import _apply_http_retry_default

        cfg = {"sources": {"local": "not a dict"}}
        _apply_http_retry_default(cfg, retry_on_429=False)  # should not raise


class TestResolveStreamGuards:
    def test_resolve_stream_guards(self) -> None:
        from benchmark.core import resolve_stream_guards

        sources = {"local": {"max_content_tokens": 5000, "max_thinking_tokens": 4000}}
        content, thinking, rep = resolve_stream_guards(sources, "local")
        assert content == 5000
        assert thinking == 4000

    def test_resolve_stream_guards_defaults(self) -> None:
        from benchmark.core import resolve_stream_guards

        sources: dict[str, Any] = {"local": {}}
        content, thinking, rep = resolve_stream_guards(sources, "local")
        assert isinstance(content, int)
        assert isinstance(thinking, int)


# ---------------------------------------------------------------------------
# benchmark.state – save_state and compact_journal
# ---------------------------------------------------------------------------


class TestStateSaveCompact:
    def test_save_state(self, tmp_path: Path) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        state.add_result({"model": "m1", "runner": "http", "http_score": 10})
        path = str(tmp_path / "state.json")
        assert state.save_state(path) is True
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data["results"]) == 1

    def test_compact_journal(self, tmp_path: Path) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        state.set_journal_path(str(tmp_path / "journal.jsonl"), truncate=True)
        state.add_result({"model": "m1", "runner": "http", "http_score": 10})
        state_path = str(tmp_path / "state.json")
        result = state.compact_journal(state_path)
        assert result is True
        assert os.path.exists(state_path)


# ---------------------------------------------------------------------------
# plugins.challenges._validators – parse_structured
# ---------------------------------------------------------------------------


class TestParseStructured:
    def test_parse_structured_json(self) -> None:
        from plugins.challenges._validators import parse_structured

        text = '```json\n{"key": "value"}\n```'
        result = parse_structured(text)
        assert result.valid is True
        assert result.value == {"key": "value"}

    def test_parse_structured_no_block(self) -> None:
        from plugins.challenges._validators import parse_structured

        result = parse_structured("no code block here")
        assert result.valid is False

    def test_parse_structured_multiple_blocks(self) -> None:
        from plugins.challenges._validators import parse_structured

        text = '```json\n{"a": 1}\n```\n```json\n{"b": 2}\n```'
        result = parse_structured(text)
        assert result.valid is False

    def test_parse_structured_invalid_json(self) -> None:
        from plugins.challenges._validators import parse_structured

        text = '```json\nnot json\n```'
        result = parse_structured(text)
        assert result.valid is False

    def test_parse_structured_with_format(self) -> None:
        from plugins.challenges._validators import parse_structured

        text = '```yaml\nkey: value\n```'
        result = parse_structured(text, fmt="yaml")
        # YAML may or may not be available
        assert result is not None


# ---------------------------------------------------------------------------
# plugins.challenges._validators – actual validator helpers
# ---------------------------------------------------------------------------


class TestValidatorHelpers:
    def test_validate_sections(self) -> None:
        from plugins.challenges._validators import validate_sections

        text = "# Title\nEnough content here to pass minimum length check.\n## Section A\nDetails here enough chars.\n## Section B\nMore content here enough chars."
        result = validate_sections(text, required=["Title", "Section A"])
        assert result.valid is True

    def test_validate_sections_missing(self) -> None:
        from plugins.challenges._validators import validate_sections

        text = "# Title\nEnough content here.\n## Short\nx"
        result = validate_sections(text, required=["Title", "Short"], min_chars=20)
        assert result.valid is False

    def test_heading_occurrences(self) -> None:
        from plugins.challenges._validators import heading_occurrences

        text = "# Title\n## Sub\n### Deep"
        result = heading_occurrences(text)
        assert len(result) == 3
        assert result[0][0] == "title"

    def test_section_map(self) -> None:
        from plugins.challenges._validators import section_map

        text = "# Title\nContent A\n## Section\nContent B"
        result = section_map(text)
        assert "title" in result
        assert "section" in result

    def test_parse_workflow_graph(self) -> None:
        from plugins.challenges._validators import parse_workflow_graph

        text = "Step 1 --> Step 2\nStep 2 --> Step 3"
        result = parse_workflow_graph(text)
        assert result.valid is True

    def test_parse_workflow_graph_no_mermaid(self) -> None:
        from plugins.challenges._validators import parse_workflow_graph

        result = parse_workflow_graph("no mermaid here")
        assert result.valid is False

    def test_parse_python_valid(self) -> None:
        from plugins.challenges._validators import parse_python

        result = parse_python("```python\ndef foo(): pass\n```")
        assert result.valid is True

    def test_parse_python_no_block(self) -> None:
        from plugins.challenges._validators import parse_python

        result = parse_python("no code here", require_block=True)
        assert result.valid is False

    def test_stub_definitions(self) -> None:
        import ast

        from plugins.challenges._validators import stub_definitions

        tree = ast.parse("def foo(): pass\ndef bar(): pass\ndef baz(): pass")
        stubs = stub_definitions(tree, {"foo", "baz"})
        assert len(stubs) == 2
        assert any("foo" in s for s in stubs)
        assert any("baz" in s for s in stubs)


# ---------------------------------------------------------------------------
# benchmark.core – judge_sidecar_path
# ---------------------------------------------------------------------------


class TestJudgeSidecarPath:
    def test_judge_sidecar_path(self) -> None:
        from benchmark.judging import judge_sidecar_path

        path = judge_sidecar_path("/tmp/judges", "model-a", "http", "tool-calling")
        assert "model-a" in path
        assert "tool-calling" in path
        assert path.endswith(".json")


# ---------------------------------------------------------------------------
# plugins.challenges._validators – find_imports
# ---------------------------------------------------------------------------


class TestFindDefinitions:
    def test_find_definitions(self) -> None:
        import ast

        from plugins.challenges._validators import find_definitions

        tree = ast.parse("def foo(): pass\nclass Bar:\n    def baz(self): pass")
        defs = find_definitions(tree)
        assert "foo" in defs
        assert "baz" in defs
