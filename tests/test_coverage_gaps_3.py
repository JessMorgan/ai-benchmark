"""Targeted tests for remaining coverage gaps: state.py, core.py, _rubric, _validators, pi, sqlite."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# benchmark.state – journal, hydration, and edge cases
# ---------------------------------------------------------------------------


class TestStateJournal:
    def test_replay_journal_events_empty(self) -> None:
        from benchmark.state import BenchmarkState

        assert BenchmarkState.replay_journal_events("") == []
        assert BenchmarkState.replay_journal_events("/nonexistent") == []

    def test_replay_journal_events_valid(self, tmp_path: Path) -> None:
        from benchmark.state import BenchmarkState

        path = tmp_path / "journal.jsonl"
        events = [
            json.dumps({"seq": 1, "type": "result", "data": {"model": "m1"}}),
            json.dumps({"seq": 2, "type": "judge", "data": {"state_key": "m1", "runner": "http", "fields": {"score": 10}}}),
            json.dumps({"seq": 3, "type": "result", "data": {"model": "m2"}}),
        ]
        path.write_text("\n".join(events))
        result = BenchmarkState.replay_journal_events(str(path))
        assert len(result) == 3
        assert result[0]["type"] == "result"
        assert result[1]["type"] == "judge"

    def test_replay_journal_events_legacy(self, tmp_path: Path) -> None:
        from benchmark.state import BenchmarkState

        path = tmp_path / "journal.jsonl"
        # Legacy format: bare dict with "model" key, no event envelope
        path.write_text(json.dumps({"model": "m1", "http_score": 5}) + "\n")
        result = BenchmarkState.replay_journal_events(str(path))
        assert len(result) == 1
        assert result[0]["type"] == "result"

    def test_replay_journal_events_corrupt_line(self, tmp_path: Path) -> None:
        from benchmark.state import BenchmarkState

        path = tmp_path / "journal.jsonl"
        path.write_text("not json\n" + json.dumps({"seq": 1, "type": "result", "data": {"model": "m1"}}) + "\n")
        result = BenchmarkState.replay_journal_events(str(path))
        assert len(result) == 1  # corrupt line skipped

    def test_replay_journal(self, tmp_path: Path) -> None:
        from benchmark.state import BenchmarkState

        path = tmp_path / "journal.jsonl"
        path.write_text(
            json.dumps({"seq": 1, "type": "result", "data": {"model": "m1", "http_score": 10}}) + "\n"
            + json.dumps({"seq": 2, "type": "judge", "data": {"state_key": "m1", "fields": {}}}) + "\n"
        )
        results = BenchmarkState.replay_journal(str(path))
        assert len(results) == 1
        assert results[0]["model"] == "m1"

    def test_apply_journal_events_to_data(self) -> None:
        from benchmark.state import BenchmarkState

        data = {"model_info": {"m1": {}}, "results": [{"model": "m1", "http_score": 5}]}
        events = [
            {"seq": 1, "type": "result", "data": {"model": "m2", "http_score": 20}},
            {"seq": 2, "type": "judge", "data": {"state_key": "m1", "runner": "http", "fields": {"http_judge_score": 88}}},
            {"seq": 3, "type": "judge", "data": {"state_key": "m_missing", "fields": {}}},
            {"seq": 4, "type": "unknown_type", "data": None},
        ]
        result = BenchmarkState.apply_journal_events_to_data(data, events)
        assert len(result["results"]) == 2  # one added
        assert result["model_info"]["m1"]["http_judge_score"] == 88
        assert result["journal_sequence"] == 4

    def test_apply_journal_events_to_data_bad_input(self) -> None:
        from benchmark.state import BenchmarkState

        assert BenchmarkState.apply_journal_events_to_data("not a dict", []) == "not a dict"

    def test_set_journal_path(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w") as f:
            state.set_journal_path(f.name, truncate=True)
            assert state._journal_path == f.name

    def test_set_journal_path_no_path(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        state.set_journal_path(None)
        assert state._journal_path is None

    def test_consume_journal_failures_empty(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        assert state.consume_journal_failures() == []


class TestStateHydrate:
    def test_hydrate_results_marks_completed(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["p1", "p2"])
        rows = [
            {"state_key": "m1", "runner": "http", "p1_score": 15, "p2_score": 10},
        ]
        state.hydrate_results(rows)
        assert len(state.results) == 1
        assert state._model_info["m1"]["status"] == "completed"

    def test_hydrate_results_marks_pending_for_partial(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["p1", "p2"])
        rows = [
            {"state_key": "m1", "runner": "http", "p1_score": 15, "p2_score": "fail"},
        ]
        state.hydrate_results(rows)
        # Partial completion: one valid score, one "fail" → pending (not failed)
        assert state._model_info["m1"]["status"] == "pending"
        # The valid score is preserved, the "fail" string is not overwriting it
        assert state._model_info["m1"]["p1_score"] == 15

    def test_hydrate_results_unknown_model(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["p1"])
        rows = [{"state_key": "m_unknown", "runner": "http", "p1_score": 10}]
        state.hydrate_results(rows)  # should not raise


class TestStateHasLiveWork:
    def test_has_live_work(self) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}, "m2": {}}, ["http"])
        assert state.has_live_work() is False
        state._model_info["m1"]["running_pids"] = ["http"]
        assert state.has_live_work() is True


class TestStateReplayTail:
    def test_replay_journal_tail(self, tmp_path: Path) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        state._journal_sequence = 0
        path = tmp_path / "journal.jsonl"
        path.write_text(
            json.dumps({"seq": 1, "type": "result", "data": {"model": "m1", "http_score": 10}}) + "\n"
            + json.dumps({"seq": 2, "type": "result", "data": {"model": "m2", "http_score": 20}}) + "\n"
        )
        applied = state.replay_journal_tail(str(path))
        assert applied == 2
        assert len(state.results) == 2

    def test_replay_journal_tail_skips_old(self, tmp_path: Path) -> None:
        from benchmark.state import BenchmarkState

        state = BenchmarkState({"m1": {}}, ["http"])
        state._journal_sequence = 5  # already applied up to seq 5
        path = tmp_path / "journal.jsonl"
        path.write_text(json.dumps({"seq": 3, "type": "result", "data": {"model": "m1"}}) + "\n")
        applied = state.replay_journal_tail(str(path))
        assert applied == 0


# ---------------------------------------------------------------------------
# benchmark.core – summarizers, judge_contract_id, judge_instructions_version
# ---------------------------------------------------------------------------


class TestCoreSummarizers:
    def test_summarize_judge_criteria(self) -> None:
        from benchmark.core import summarize_judge_criteria

        plugin = MagicMock()
        plugin.id = "p1"
        results = [
            {"p1_judge_criteria": [{"criteria": [
                {"id": "c1", "status": "met", "evidence": "good"},
                {"id": "c2", "status": "not_met", "evidence": "bad"},
            ]}]},
        ]
        summary = summarize_judge_criteria(results, [plugin])
        assert summary["criterion_reports"] == 1
        assert summary["criteria"] == 2
        assert summary["by_plugin"]["p1"]["status_counts"]["met"] == 1
        assert summary["by_plugin"]["p1"]["evidence"] == 2

    def test_summarize_judge_criteria_empty(self) -> None:
        from benchmark.core import summarize_judge_criteria

        plugin = MagicMock()
        plugin.id = "p1"
        summary = summarize_judge_criteria([], [plugin])
        assert summary["criterion_reports"] == 0
        assert summary["criteria"] == 0

    def test_summarize_judge_criteria_bad_data(self) -> None:
        from benchmark.core import summarize_judge_criteria

        plugin = MagicMock()
        plugin.id = "p1"
        results = [{"p1_judge_criteria": "not a list"}]
        summary = summarize_judge_criteria(results, [plugin])
        assert summary["criteria"] == 0

    def test_summarize_schema_compatibility(self) -> None:
        from benchmark.core import summarize_schema_compatibility

        plugin = MagicMock()
        plugin.id = "p1"
        results = [
            {"p1_schema_requested": True, "p1_schema_request_status": "schema_accepted_valid",
             "p1_response_schema_valid": True, "p1_schema_enforcement_verified": True},
            {"p1_schema_requested": True, "p1_schema_request_status": "schema_rejected",
             "p1_response_schema_valid": False},
            {"p1_schema_requested": False},
        ]
        summary = summarize_schema_compatibility(results, [plugin])
        assert summary["requested_cells"] == 2
        assert summary["response_valid_cells"] == 1
        assert summary["enforcement_verified_cells"] == 1
        assert summary["by_plugin"]["p1"]["requested_cells"] == 2

    def test_summarize_schema_compatibility_empty(self) -> None:
        from benchmark.core import summarize_schema_compatibility

        plugin = MagicMock()
        plugin.id = "p1"
        summary = summarize_schema_compatibility([], [plugin])
        assert summary["requested_cells"] == 0

    def test_judge_instructions_version(self) -> None:
        from benchmark.core import judge_instructions_version

        plugin = MagicMock()
        plugin.judge_instructions_version = "2.0.0"
        assert judge_instructions_version(plugin) == "2.0.0"

    def test_judge_instructions_version_none(self) -> None:
        from benchmark.core import judge_instructions_version

        plugin = MagicMock()
        plugin.judge_instructions_version = None
        assert judge_instructions_version(plugin) == "1.0.0"

    def test_judge_instructions_version_default(self) -> None:
        from benchmark.core import judge_instructions_version

        plugin = MagicMock(spec=[])  # no judge_instructions_version attr
        assert judge_instructions_version(plugin) == "1.0.0"

    def test_judge_contract_id(self) -> None:
        from benchmark.core import judge_contract_id

        plugin = MagicMock()
        plugin.id = "p1"
        plugin.version = "1.0"
        plugin.get_judge_instructions.return_value = "instructions text"
        plugin.judge_instructions_version = "1.0.0"
        contract_id = judge_contract_id(plugin)
        assert contract_id.startswith("judge-contract-v1:")
        assert len(contract_id) > 20

    def test_judge_contract_id_no_instructions(self) -> None:
        from benchmark.core import judge_contract_id

        plugin = MagicMock()
        plugin.id = "p1"
        plugin.version = "1.0"
        plugin.get_judge_instructions = None
        del plugin.judge_instructions_version
        contract_id = judge_contract_id(plugin)
        assert contract_id.startswith("judge-contract-v1:")


# ---------------------------------------------------------------------------
# plugins.challenges._rubric
# ---------------------------------------------------------------------------


class TestRubric:
    def test_add_criterion(self) -> None:
        from plugins.challenges._rubric import Rubric

        rubric = Rubric(max_score=20)
        rubric.add_criterion("c1", 10, 8, evidence=[{"kind": "test"}])
        assert rubric.total == 8.0
        assert rubric.criteria[0]["matched"] is True
        assert rubric.criteria[0]["missed"] == 2.0

    def test_add_criterion_unmatched(self) -> None:
        from plugins.challenges._rubric import Rubric

        rubric = Rubric(max_score=20)
        rubric.add_criterion("c1", 10, 0)
        assert rubric.total == 0.0
        assert rubric.criteria[0]["matched"] is False

    def test_add_criterion_clamped(self) -> None:
        from plugins.challenges._rubric import Rubric

        rubric = Rubric(max_score=20)
        rubric.add_criterion("c1", 10, 15)  # exceeds max
        assert rubric.total == 10.0  # clamped

    def test_add_criterion_with_errors(self) -> None:
        from plugins.challenges._rubric import Rubric

        rubric = Rubric(max_score=20)
        rubric.add_criterion("c1", 10, 5, errors=["some error"])
        assert "some error" in rubric.errors

    def test_credit_criterion(self) -> None:
        from plugins.challenges._rubric import Rubric

        rubric = Rubric(max_score=20)
        rubric.add_criterion("c1", 10, 3)
        rubric.credit_criterion("c1", 5, evidence="execution passed")
        assert rubric.criteria[0]["earned"] == 8.0
        assert rubric.total == 8.0

    def test_credit_criterion_clamped(self) -> None:
        from plugins.challenges._rubric import Rubric

        rubric = Rubric(max_score=20)
        rubric.add_criterion("c1", 10, 8)
        rubric.credit_criterion("c1", 10)  # would go to 18 but max is 10
        assert rubric.criteria[0]["earned"] == 10.0

    def test_credit_criterion_unknown(self) -> None:
        from plugins.challenges._rubric import Rubric

        rubric = Rubric(max_score=20)
        rubric.add_criterion("c1", 10, 5)
        rubric.credit_criterion("nonexistent", 5)
        assert any("cannot credit unknown criterion" in e for e in rubric.errors)

    def test_penalize_criterion(self) -> None:
        from plugins.challenges._rubric import Rubric

        rubric = Rubric(max_score=20)
        rubric.add_criterion("c1", 10, 8)
        rubric.penalize_criterion("c1", 3, "missing test")
        assert rubric.criteria[0]["earned"] == 5.0
        assert rubric.total == 5.0

    def test_penalize_criterion_clamped(self) -> None:
        from plugins.challenges._rubric import Rubric

        rubric = Rubric(max_score=20)
        rubric.add_criterion("c1", 10, 2)
        rubric.penalize_criterion("c1", 10, "overpenalize")  # would go negative but clamped to 0
        assert rubric.criteria[0]["earned"] == 0.0

    def test_penalize_criterion_unknown(self) -> None:
        from plugins.challenges._rubric import Rubric

        rubric = Rubric(max_score=20)
        rubric.add_criterion("c1", 10, 5)
        rubric.penalize_criterion("nonexistent", 5, "bad")
        assert any("cannot penalize unknown criterion" in e for e in rubric.errors)

    def test_record_validation(self) -> None:
        from plugins.challenges._rubric import Rubric

        rubric = Rubric(max_score=20)
        validation = MagicMock()
        validation.as_evidence.return_value = {"valid": True}
        rubric.record_validation(validation)
        assert rubric.validations == [{"valid": True}]

    def test_record_validation_plain(self) -> None:
        from plugins.challenges._rubric import Rubric

        rubric = Rubric(max_score=20)
        rubric.record_validation(MagicMock(valid=True))  # no as_evidence
        assert len(rubric.validations) == 1

    def test_eval_regex(self) -> None:
        from plugins.challenges._rubric import Rubric

        rubric = Rubric(max_score=20)
        rubric.eval_regex("patterns", 10, "Hello World", [
            (r"Hello", 5),
            (r"World", 5),
        ])
        assert rubric.total == 10.0

    def test_eval_regex_partial(self) -> None:
        from plugins.challenges._rubric import Rubric

        rubric = Rubric(max_score=20)
        rubric.eval_regex("patterns", 10, "Hello", [
            (r"Hello", 5),
            (r"World", 5),
        ])
        assert rubric.total == 5.0


# ---------------------------------------------------------------------------
# plugins.challenges._validators
# ---------------------------------------------------------------------------


class TestValidators:
    def test_parse_tool_calls_valid(self) -> None:
        from plugins.challenges._validators import parse_tool_calls

        text = '<tool_call>\n{"name": "get_weather", "args": {"location": "NYC"}}\n</tool_call>'
        result = parse_tool_calls(text)
        assert result.valid is True
        assert len(result.value) == 1

    def test_parse_tool_calls_invalid_json(self) -> None:
        from plugins.challenges._validators import parse_tool_calls

        text = "<tool_call>\nnot json\n</tool_call>"
        result = parse_tool_calls(text)
        assert result.valid is False

    def test_parse_tool_calls_no_name(self) -> None:
        from plugins.challenges._validators import parse_tool_calls

        text = '<tool_call>\n{"args": {}}\n</tool_call>'
        result = parse_tool_calls(text)
        assert result.valid is False

    def test_parse_tool_calls_unknown_tool(self) -> None:
        from plugins.challenges._validators import parse_tool_calls

        text = '<tool_call>\n{"name": "unknown_tool", "args": {}}\n</tool_call>'
        result = parse_tool_calls(text)
        assert result.valid is False

    def test_parse_tool_calls_missing_required(self) -> None:
        from plugins.challenges._validators import parse_tool_calls

        text = '<tool_call>\n{"name": "get_weather", "args": {}}\n</tool_call>'
        result = parse_tool_calls(text)
        assert result.valid is False

    def test_parse_tool_calls_bad_type(self) -> None:
        from plugins.challenges._validators import parse_tool_calls

        text = '<tool_call>\n{"name": "get_weather", "args": {"location": 123}}\n</tool_call>'
        result = parse_tool_calls(text)
        assert result.valid is False

    def test_parse_tool_calls_no_args(self) -> None:
        from plugins.challenges._validators import parse_tool_calls

        text = '<tool_call>\n{"name": "get_weather"}\n</tool_call>'
        result = parse_tool_calls(text)
        assert result.valid is False

    def test_parse_tool_calls_none_args(self) -> None:
        from plugins.challenges._validators import parse_tool_calls

        text = '<tool_call>\n{"name": "get_weather", "args": null}\n</tool_call>'
        result = parse_tool_calls(text)
        assert result.valid is False

    def test_parse_tool_calls_multiple(self) -> None:
        from plugins.challenges._validators import parse_tool_calls

        text = (
            '<tool_call>\n{"name": "get_weather", "args": {"location": "NYC"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "search_flights", "args": {"origin": "NYC", "destination": "LAX", "date": "2025-01-01"}}\n</tool_call>'
        )
        result = parse_tool_calls(text)
        assert result.valid is True
        assert len(result.value) == 2

    def test_parse_tool_calls_optional_arg_wrong_type(self) -> None:
        from plugins.challenges._validators import parse_tool_calls

        text = '<tool_call>\n{"name": "get_weather", "args": {"location": "NYC", "unit": 123}}\n</tool_call>'
        result = parse_tool_calls(text)
        assert result.valid is False


class TestFindDefinitions:
    def test_find_definitions(self) -> None:
        import ast
        from plugins.challenges._validators import find_definitions

        tree = ast.parse("def foo(): pass\ndef bar(): pass\nclass Foo:\n    def baz(self): pass")
        defs = find_definitions(tree)
        assert "foo" in defs
        assert "bar" in defs
        assert "baz" in defs


# ---------------------------------------------------------------------------
# benchmark.pi – helper functions
# ---------------------------------------------------------------------------


class TestPiHelpers:
    def test_worker_path_default(self) -> None:
        from benchmark.pi import _worker_path

        path = _worker_path()
        assert path.name == "worker.mjs"

    def test_worker_path_custom(self, tmp_path: Path) -> None:
        from benchmark.pi import _worker_path

        custom = tmp_path / "custom.mjs"
        custom.write_text("// custom")
        assert _worker_path(str(custom)) == custom

    def test_source_payload(self) -> None:
        from benchmark.pi import _source_payload

        source_config = {"ollama": {"api_url": "http://localhost:11434", "api_key": "key123"}}
        payload = _source_payload(source_config, "ollama", {"tools": ["read"]})
        assert payload["name"] == "ollama"
        assert payload["api_url"] == "http://localhost:11434"
        assert payload["pi"] == {"tools": ["read"]}

    def test_source_payload_no_pi(self) -> None:
        from benchmark.pi import _source_payload

        source_config = {"ollama": {"api_url": "http://localhost:11434"}}
        payload = _source_payload(source_config, "ollama", None)
        assert "pi" not in payload

    def test_source_payload_bad_type(self) -> None:
        from benchmark.pi import _source_payload

        with pytest.raises(TypeError, match="must be an object"):
            _source_payload({"ollama": "bad"}, "ollama", None)

    def test_source_payload_with_api_key_field(self) -> None:
        from benchmark.pi import _source_payload

        source_config = {"ollama": {"api_url": "http://localhost", "apiKey": "key456"}}
        payload = _source_payload(source_config, "ollama", None)
        assert payload["api_key"] == "key456"

    def test_pi_process_result_defaults(self) -> None:
        from benchmark.pi import PiProcessResult

        result = PiProcessResult(text="hello", stderr="", elapsed=1.0, error=None, returncode=0)
        assert result.think_text == ""
        assert result.tool_called is False
        assert result.tools == ()


# ---------------------------------------------------------------------------
# benchmark.sqlite_writer – more error paths
# ---------------------------------------------------------------------------


class TestSQLiteWriterMore:
    def test_execute_batch_failure(self, tmp_path: Path) -> None:
        from benchmark.sqlite_writer import SQLiteWriteQueue

        q = SQLiteWriteQueue(str(tmp_path / "test.db"))
        q.start()

        def bad_op(conn):
            raise sqlite3.OperationalError("test failure")

        future = q.submit(bad_op)
        with pytest.raises(sqlite3.OperationalError):
            future.result(timeout=5)

        assert len(q.failures) >= 1
        q.close(timeout=5)

    def test_callback_exception_suppressed(self, tmp_path: Path) -> None:
        from benchmark.sqlite_writer import SQLiteWriteQueue

        def bad_callback(exc):
            raise RuntimeError("callback error")

        q = SQLiteWriteQueue(
            str(tmp_path / "test.db"),
            failure_callback=bad_callback,
        )
        q.start()

        def bad_op(conn):
            raise sqlite3.OperationalError("test")

        future = q.submit(bad_op)
        with pytest.raises(sqlite3.OperationalError):
            future.result(timeout=5)
        q.close(timeout=5)
        # Should not raise despite bad callback


# ---------------------------------------------------------------------------
# benchmark.storage – more edge cases
# ---------------------------------------------------------------------------


class TestStorageMore:
    def test_json_report_source_loads_from_file(self, tmp_path: Path) -> None:
        from benchmark.storage import JsonReportSource

        state = {
            "results": [{"model": "m1", "http_score": 5}],
            "active_plugins": ["http"],
            "session_seed": 99,
        }
        path = tmp_path / "state.json"
        path.write_text(json.dumps(state))
        results, plugins, seed = JsonReportSource().load_results(str(path))
        assert seed == 99

    def test_latest_result_rows_dedup(self) -> None:
        from benchmark.storage import latest_result_rows

        rows = [
            {"state_key": "m1", "runner": "http", "score": 1},
            {"state_key": "m1", "runner": "http", "score": 2},
        ]
        result = latest_result_rows(rows)
        assert len(result) == 1
        assert result[0]["score"] == 2

    def test_json_debug_log_stores_bytes(self, tmp_path: Path) -> None:
        from benchmark.storage import JsonDebugLogStore

        store = JsonDebugLogStore()
        log_path = str(tmp_path / "debug.log")
        store.append(log_path, b"binary data\n")
        assert os.path.exists(log_path)

    def test_json_payload_store_not_implemented(self) -> None:
        from benchmark.storage import JsonPayloadStore

        with pytest.raises(NotImplementedError):
            JsonPayloadStore().put("x", b"y")
        with pytest.raises(NotImplementedError):
            JsonPayloadStore().get(1)

    def test_run_store_protocol_compliance(self) -> None:
        from benchmark.storage import JsonRunStore, RunStore, SQLiteRunStore

        mock_state = MagicMock()
        json_store = JsonRunStore(mock_state)
        assert isinstance(json_store, RunStore)

    def test_payload_store_protocol_compliance(self) -> None:
        from benchmark.sqlite_payloads import SQLitePayloadStore
        from benchmark.storage import PayloadStore

        conn = sqlite3.connect(":memory:")
        from benchmark.sqlite_schema import configure_connection, initialize_schema
        configure_connection(conn)
        initialize_schema(conn)
        store = SQLitePayloadStore(conn)
        assert isinstance(store, PayloadStore)

    def test_debug_log_store_protocol_compliance(self) -> None:
        from benchmark.storage import DebugLogStore, JsonDebugLogStore

        assert isinstance(JsonDebugLogStore(), DebugLogStore)


# ---------------------------------------------------------------------------
# plugins/challenges/_validators – extract_fenced_blocks
# ---------------------------------------------------------------------------


class TestExtractFencedBlocks:
    def test_extract_fenced_blocks(self) -> None:
        from plugins.challenges._validators import extract_fenced_blocks

        text = "Before\n```python\ncode\n```\nAfter\n```\nother\n```"
        blocks = extract_fenced_blocks(text)
        assert len(blocks) == 2
        assert "code" in blocks[0]

    def test_extract_fenced_blocks_empty(self) -> None:
        from plugins.challenges._validators import extract_fenced_blocks

        assert extract_fenced_blocks("no code here") == []
