"""Comprehensive tests for coverage gaps across storage, sqlite, logs, evaluation, and judge_analysis."""
from __future__ import annotations

import gzip
import json
import os
import sqlite3
from concurrent.futures import Future
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# benchmark.storage – JsonReportSource / JsonPayloadStore / JsonDebugLogStore
# ---------------------------------------------------------------------------


class TestJsonReportSource:
    def test_load_results_from_directory(self, tmp_path: Path) -> None:
        from benchmark.storage import JsonReportSource

        state = {
            "results": [{"model": "a", "http_score": 5}],
            "active_plugins": ["http"],
            "session_seed": 42,
        }
        (tmp_path / "benchmark_state.json").write_text(json.dumps(state))
        results, plugins, seed = JsonReportSource().load_results(str(tmp_path))
        assert len(results) == 1
        assert plugins == ["http"]
        assert seed == 42

    def test_load_results_from_file(self, tmp_path: Path) -> None:
        from benchmark.storage import JsonReportSource

        state: dict[str, Any] = {"results": [], "active_plugins": []}
        path = tmp_path / "state.json"
        path.write_text(json.dumps(state))
        results, plugins, seed = JsonReportSource().load_results(str(path))
        assert results == []
        assert seed is None

    def test_load_results_rejects_sqlite(self, tmp_path: Path) -> None:
        from benchmark.storage import JsonReportSource

        path = tmp_path / "run.sqlite3"
        path.write_bytes(b"fake")
        with pytest.raises(RuntimeError, match="SQLiteReportSource"):
            JsonReportSource().load_results(str(path))

    def test_load_results_missing_results(self, tmp_path: Path) -> None:
        from benchmark.storage import JsonReportSource

        path = tmp_path / "state.json"
        path.write_text(json.dumps({"active_plugins": []}))
        with pytest.raises(TypeError, match="does not contain a results list"):
            JsonReportSource().load_results(str(path))

    def test_load_results_invalid_plugins(self, tmp_path: Path) -> None:
        from benchmark.storage import JsonReportSource

        path = tmp_path / "state.json"
        path.write_text(json.dumps({"results": [], "active_plugins": "bad"}))
        with pytest.raises(TypeError, match="invalid active_plugins"):
            JsonReportSource().load_results(str(path))


class TestJsonPayloadStore:
    def test_put_raises(self) -> None:
        from benchmark.storage import JsonPayloadStore

        with pytest.raises(NotImplementedError):
            JsonPayloadStore().put("x", b"data")

    def test_get_raises(self) -> None:
        from benchmark.storage import JsonPayloadStore

        with pytest.raises(NotImplementedError):
            JsonPayloadStore().get(1)


class TestJsonDebugLogStore:
    def test_append_creates_file(self, tmp_path: Path) -> None:
        from benchmark.storage import JsonDebugLogStore

        store = JsonDebugLogStore()
        log_path = str(tmp_path / "logs" / "debug.log")
        store.append(log_path, "hello world\n")
        assert os.path.exists(log_path)
        with open(log_path) as handle:
            assert handle.read() == "hello world\n"

    def test_append_bytes(self, tmp_path: Path) -> None:
        from benchmark.storage import JsonDebugLogStore

        store = JsonDebugLogStore()
        log_path = str(tmp_path / "debug.log")
        store.append(log_path, b"bytes data\n")
        with open(log_path, "rb") as handle:
            assert handle.read() == b"bytes data\n"

    def test_close_noop(self) -> None:
        from benchmark.storage import JsonDebugLogStore

        JsonDebugLogStore().close()


class TestLatestResultRows:
    def test_latest_result_rows(self) -> None:
        from benchmark.storage import latest_result_rows

        rows = [
            {"state_key": "m1", "runner": "http", "score": 1},
            {"state_key": "m1", "runner": "http", "score": 2},
            {"state_key": "m2", "runner": "opencode", "score": 3},
        ]
        result = latest_result_rows(rows)
        assert len(result) == 2
        by_key = {(r["state_key"], r["runner"]): r for r in result}
        assert by_key[("m1", "http")]["score"] == 2
        assert by_key[("m2", "opencode")]["score"] == 3

    def test_latest_result_rows_empty(self) -> None:
        from benchmark.storage import latest_result_rows

        assert latest_result_rows([]) == []


class TestJsonRunStoreFacade:
    def test_unsupported_normalized_methods_are_explicit(self) -> None:
        from benchmark.storage import (
            BenchmarkAttemptRecord,
            JsonRunStore,
            JudgeAttemptRecord,
            PluginRecord,
            RunIdentity,
            TargetRecord,
        )

        mock_state = MagicMock()
        store = JsonRunStore(mock_state)
        store.start_run(RunIdentity("run1", 1))
        assert store.identity is not None and store.identity.run_id == "run1"

        store.prepare_run([], [])
        target = TargetRecord(logical_name="m", runner="http", source="local", api_model="m", target_signature="s")
        plugin = PluginRecord(plugin_id="p", plugin_version="1", name="P", max_score=20, supports_streaming=True)
        unsupported = [
            lambda: store.register_target(target),
            lambda: store.register_plugin(plugin),
            lambda: store.ensure_cell(target, plugin),
            lambda: store.record_benchmark_attempt(1, BenchmarkAttemptRecord(attempt_number=1)),
            lambda: store.record_judge_attempt(1, JudgeAttemptRecord(judge_model="j", contract_id="c", attempt_number=1)),
            lambda: store.register_judge("judge", "http"),
            lambda: store.register_contract("c", plugin_id="p", plugin_version="1",
                                            prompt_version="1", instructions_version="1"),
        ]
        for operation in unsupported:
            with pytest.raises(NotImplementedError, match="normalized operation"):
                operation()
        assert store.get_cell_id("m", "http", "p") is None
        assert store.close() is True

    def test_save_snapshot_none_path_raises(self) -> None:
        from benchmark.storage import JsonRunStore

        store = JsonRunStore(MagicMock())
        with pytest.raises(ValueError, match="state path"):
            store.save_snapshot(path=None)


# ---------------------------------------------------------------------------
# benchmark.sqlite_payloads
# ---------------------------------------------------------------------------


class TestSQLitePayloadStore:
    def _make_store(self) -> tuple[sqlite3.Connection, Any]:
        from benchmark.sqlite_payloads import SQLitePayloadStore
        from benchmark.sqlite_schema import configure_connection, initialize_schema

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        configure_connection(conn)
        initialize_schema(conn)
        return conn, SQLitePayloadStore(conn)

    def test_put_and_get(self) -> None:
        conn, store = self._make_store()
        pid = store.put("prompt", b"hello world")
        assert pid > 0
        assert store.get(pid) == b"hello world"

    def test_deduplication(self) -> None:
        conn, store = self._make_store()
        p1 = store.put("prompt", b"data")
        p2 = store.put("prompt", b"data")
        assert p1 == p2
        assert store.count() == 1

    def test_put_text(self) -> None:
        conn, store = self._make_store()
        pid = store.put_text("prompt", "unicode: 你好")
        assert store.get_text(pid) == "unicode: 你好"

    def test_get_text_non_utf8(self) -> None:
        conn, store = self._make_store()
        # Insert raw bytes that aren't valid UTF-8
        import hashlib

        from benchmark.sqlite_payloads import PayloadIntegrityError
        data = b"\xff\xfe"
        digest = hashlib.sha256(data).hexdigest()
        compressed = gzip.compress(data, mtime=0)
        conn.execute(
            "INSERT INTO payloads(sha256, kind, compression, uncompressed_bytes, stored_bytes, data, created_at) "
            "VALUES (?, ?, 'gzip', ?, ?, ?, strftime('%s','now'))",
            (digest, "raw", len(data), len(compressed), compressed),
        )
        conn.commit()
        pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        with pytest.raises(PayloadIntegrityError, match="not valid UTF-8"):
            store.get_text(pid)

    def test_get_unknown_key(self) -> None:
        conn, store = self._make_store()
        with pytest.raises(KeyError, match="unknown payload ID"):
            store.get(9999)

    def test_metadata(self) -> None:
        conn, store = self._make_store()
        pid = store.put("response", b"test")
        meta = store.metadata(pid)
        assert meta["kind"] == "response"
        assert meta["compression"] == "gzip"

    def test_metadata_unknown(self) -> None:
        conn, store = self._make_store()
        with pytest.raises(KeyError, match="unknown payload ID"):
            store.metadata(9999)

    def test_count_empty(self) -> None:
        conn, store = self._make_store()
        assert store.count() == 0

    def test_put_invalid_kind(self) -> None:
        conn, store = self._make_store()
        with pytest.raises(ValueError, match="non-empty string"):
            store.put("", b"data")

    def test_put_invalid_kind_whitespace(self) -> None:
        conn, store = self._make_store()
        with pytest.raises(ValueError, match="non-empty string"):
            store.put("   ", b"data")

    def test_put_non_bytes(self) -> None:
        conn, store = self._make_store()
        with pytest.raises(TypeError, match="must be bytes"):
            store.put("kind", "not bytes")  # type: ignore

    def test_put_text_non_str(self) -> None:
        conn, store = self._make_store()
        with pytest.raises(TypeError, match="must be str"):
            store.put_text("kind", b"not str")  # type: ignore

    def test_build_payload_only_judge_input(self) -> None:
        from benchmark.sqlite_payloads import build_payload_only_judge_input

        conn, store = self._make_store()
        item = {"prompt": "judge prompt", "response": "model response", "other": "data"}
        result = build_payload_only_judge_input(store, item)
        assert "prompt" not in result
        assert "response" not in result
        assert result["prompt_payload_id"] > 0
        assert result["response_payload_id"] > 0
        assert result["other"] == "data"

    def test_build_payload_only_judge_input_bad_type(self) -> None:
        from benchmark.sqlite_payloads import build_payload_only_judge_input

        conn, store = self._make_store()
        with pytest.raises(TypeError, match="string prompt and response"):
            build_payload_only_judge_input(store, {"prompt": 123, "response": "ok"})

    def test_integrity_error_on_bad_data(self) -> None:
        from benchmark.sqlite_payloads import PayloadIntegrityError

        conn, store = self._make_store()
        pid = store.put("prompt", b"good data")
        # Tamper with the stored data
        conn.execute("UPDATE payloads SET data = X'deadbeef' WHERE payload_id = ?", (pid,))
        conn.commit()
        with pytest.raises(PayloadIntegrityError):
            store.get(pid)


# ---------------------------------------------------------------------------
# benchmark.sqlite_writer
# ---------------------------------------------------------------------------


class TestSQLiteWriteQueue:
    def test_start_and_close(self, tmp_path: Path) -> None:
        from benchmark.sqlite_writer import SQLiteWriteQueue

        db = str(tmp_path / "test.db")
        q = SQLiteWriteQueue(db)
        q.start()
        assert q.is_alive
        q.flush()
        result = q.close(timeout=5)
        assert result is True
        assert not q.is_alive

    def test_batch_size_must_be_positive(self) -> None:
        from benchmark.sqlite_writer import SQLiteWriteQueue

        with pytest.raises(ValueError, match="batch_size"):
            SQLiteWriteQueue("/tmp/x.db", batch_size=0)

    def test_flush_interval_negative(self) -> None:
        from benchmark.sqlite_writer import SQLiteWriteQueue

        with pytest.raises(ValueError, match="flush_interval"):
            SQLiteWriteQueue("/tmp/x.db", flush_interval=-1)

    def test_submit_non_callable(self, tmp_path: Path) -> None:
        from benchmark.sqlite_writer import SQLiteWriteQueue

        q = SQLiteWriteQueue(str(tmp_path / "test.db"))
        q.start()
        with pytest.raises(TypeError, match="callable"):
            q.submit("not a function")  # type: ignore[arg-type]
        q.close(timeout=5)

    def test_submit_after_close(self, tmp_path: Path) -> None:
        from benchmark.sqlite_writer import SQLiteWriteQueue

        db = str(tmp_path / "test.db")
        q = SQLiteWriteQueue(db)
        q.start()
        q.flush()
        result = q.close(timeout=5)
        assert result is True
        # After close, submit should return a future with RuntimeError
        future: Future = q.submit(lambda conn: None)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="closed"):
            future.result(timeout=5)

    def test_failure_callback(self, tmp_path: Path) -> None:
        from benchmark.sqlite_writer import SQLiteWriteQueue

        errors: list[Exception] = []
        q = SQLiteWriteQueue(
            str(tmp_path / "test.db"),
            failure_callback=lambda e: errors.append(e),
        )
        q.start()

        def bad_op(conn: sqlite3.Connection) -> None:
            raise sqlite3.OperationalError("test error")

        future = q.submit(bad_op)
        with pytest.raises(sqlite3.OperationalError):
            future.result(timeout=5)
        q.close(timeout=5)
        assert len(errors) >= 1

    def test_failures_property(self, tmp_path: Path) -> None:
        from benchmark.sqlite_writer import SQLiteWriteQueue

        q = SQLiteWriteQueue(str(tmp_path / "test.db"))
        assert q.failures == []

    def test_close_without_start(self) -> None:
        from benchmark.sqlite_writer import SQLiteWriteQueue

        q = SQLiteWriteQueue("/tmp/x.db")
        result = q.close(timeout=1)
        assert result is True

    def test_batch_operation(self, tmp_path: Path) -> None:
        from benchmark.sqlite_writer import SQLiteWriteQueue

        q = SQLiteWriteQueue(str(tmp_path / "test.db"))
        q.start()
        results = []
        for i in range(10):
            future = q.submit(lambda conn, _i=i: _i * 2)
            results.append(future)
        q.flush()
        for i, f in enumerate(results):
            assert f.result(timeout=5) == i * 2
        q.close(timeout=5)


# ---------------------------------------------------------------------------
# benchmark.sqlite_reports
# ---------------------------------------------------------------------------


class TestSQLiteReportSource:
    def _make_db_with_data(self) -> tuple[sqlite3.Connection, int]:
        from benchmark.sqlite_benchmarks import SQLiteBenchmarkStore
        from benchmark.sqlite_schema import configure_connection, initialize_schema

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        configure_connection(conn)
        initialize_schema(conn)

        store = SQLiteBenchmarkStore(conn)
        rev_id = store.create_run(
            "test-run", score_schema="v1", storage_profile="compact",
            runner_mode="http", config={}
        )
        target_id = store.register_target(
            rev_id, run_id="test-run", logical_name="m1", runner="http",
            source="local", api_model="m1", target_signature="s1",
            is_agent=False, system_prompt=None, target_config={}, order_index=0,
        )
        store.register_plugin("p1", "1.0", name="P1", max_score=20,
                              supports_streaming=True, metadata={})
        store.activate_plugin(rev_id, "p1", "1.0")
        cell_id = store.ensure_cell(rev_id, target_id, "p1", "1.0")
        store.record_attempt(rev_id, cell_id, {
            "attempt_number": 1, "prompt": "p", "content": "c", "thinking": "",
            "max_tokens": 4096, "output_tokens": 50, "thinking_tokens": 0,
            "total_tokens": 50, "tps": 10.0, "finish_reason": "stop",
            "response_nature": "ok", "retry_reason": None, "prompt_altered": None,
            "truncated": False, "truncated_due_to_time": False,
            "failure_cause": None, "stream_ok": True, "repeating": False,
            "empty_reason": None, "error": None, "score": 10.0,
            "rubric": [], "diagnostics": {}, "status": "ok",
        }, selected=True)
        conn.commit()
        return conn, rev_id

    def test_load_results(self) -> None:
        from benchmark.sqlite_reports import SQLiteReportSource

        conn, rev_id = self._make_db_with_data()
        source = SQLiteReportSource(conn)
        rows, active, seed, rev = source.load_results(revision=rev_id)
        assert len(rows) == 1
        assert rows[0]["model"] == "m1"
        assert "p1_score" in rows[0]
        assert rows[0]["p1_score"] == 10.0
        assert active == ["p1"]

    def test_resolve_revision_none(self) -> None:
        from benchmark.sqlite_reports import SQLiteReportSource

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        from benchmark.sqlite_schema import configure_connection, initialize_schema
        configure_connection(conn)
        initialize_schema(conn)

        source = SQLiteReportSource(conn)
        with pytest.raises(ValueError, match="no current revision"):
            source._resolve_revision(None)
        conn.close()

    def test_resolve_revision_not_found(self) -> None:
        from benchmark.sqlite_reports import SQLiteReportSource

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        from benchmark.sqlite_schema import configure_connection, initialize_schema
        configure_connection(conn)
        initialize_schema(conn)

        source = SQLiteReportSource(conn)
        with pytest.raises(ValueError, match="does not exist"):
            source._resolve_revision(9999)
        conn.close()

    def test_load_results_bad_revision(self) -> None:
        from benchmark.sqlite_reports import SQLiteReportSource

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        from benchmark.sqlite_schema import configure_connection, initialize_schema
        configure_connection(conn)
        initialize_schema(conn)

        source = SQLiteReportSource(conn)
        with pytest.raises(ValueError, match="does not exist"):
            source.load_results(revision=9999)
        conn.close()

    def test_json_load_none(self) -> None:
        from benchmark.sqlite_reports import SQLiteReportSource

        assert SQLiteReportSource._json_load(None) is None

    def test_json_load_invalid(self) -> None:
        from benchmark.sqlite_reports import SQLiteReportSource

        assert SQLiteReportSource._json_load("not json {{") is None

    def test_json_load_valid(self) -> None:
        from benchmark.sqlite_reports import SQLiteReportSource

        assert SQLiteReportSource._json_load('{"a": 1}') == {"a": 1}


class TestSqlitePathFromReportPath:
    def test_directory_with_sqlite(self, tmp_path: Path) -> None:
        from benchmark.sqlite_reports import sqlite_path_from_report_path

        (tmp_path / "run.sqlite3").write_bytes(b"")
        result = sqlite_path_from_report_path(str(tmp_path))
        assert result is not None
        assert result.endswith("run.sqlite3")

    def test_directory_without_sqlite(self, tmp_path: Path) -> None:
        from benchmark.sqlite_reports import sqlite_path_from_report_path

        assert sqlite_path_from_report_path(str(tmp_path)) is None

    def test_direct_sqlite_file(self, tmp_path: Path) -> None:
        from benchmark.sqlite_reports import sqlite_path_from_report_path

        path = tmp_path / "run.db"
        path.write_bytes(b"")
        assert sqlite_path_from_report_path(str(path)) == str(path)

    def test_non_sqlite_file(self, tmp_path: Path) -> None:
        from benchmark.sqlite_reports import sqlite_path_from_report_path

        path = tmp_path / "state.json"
        path.write_bytes(b"")
        assert sqlite_path_from_report_path(str(path)) is None

    def test_open_classmethod(self, tmp_path: Path) -> None:
        from benchmark.sqlite_reports import SQLiteReportSource

        db = tmp_path / "test.db"
        source = SQLiteReportSource.open(str(db))
        try:
            assert source.connection is not None
        finally:
            source.close()

    def test_close_no_owned(self) -> None:
        from benchmark.sqlite_reports import SQLiteReportSource

        conn = sqlite3.connect(":memory:")
        source = SQLiteReportSource(conn)
        source._owned_connection = None
        source.close()  # should not raise
        conn.close()


# ---------------------------------------------------------------------------
# benchmark.evaluation
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_evaluate_saved_response(self, tmp_path: Path) -> None:
        from benchmark.evaluation import evaluate_saved_response

        response_file = tmp_path / "response.txt"
        response_file.write_text("Hello world test response content")
        result = evaluate_saved_response("reasoning", str(response_file))
        assert "diagnostic_schema" in result
        assert result["plugin"] == "reasoning"
        assert result["response_length"] > 0

    def test_evaluate_unknown_plugin(self, tmp_path: Path) -> None:
        from benchmark.evaluation import evaluate_saved_response

        response_file = tmp_path / "response.txt"
        response_file.write_text("test")
        with pytest.raises(ValueError, match="Unknown plugin"):
            evaluate_saved_response("nonexistent-plugin-xyz", str(response_file))


# ---------------------------------------------------------------------------
# benchmark.judge_analysis
# ---------------------------------------------------------------------------


class TestJudgeAnalysis:
    def test_empty_data(self) -> None:
        from benchmark.judge_analysis import judge_statistics

        result = judge_statistics({"results": []})
        assert result["per_judge"] == []
        assert result["pairwise"] == []

    def test_validate_threshold_negative(self) -> None:
        from benchmark.judge_analysis import _validate_threshold

        with pytest.raises(ValueError, match="finite non-negative"):
            _validate_threshold("test", -1.0)

    def test_validate_threshold_nan(self) -> None:
        from benchmark.judge_analysis import _validate_threshold

        with pytest.raises(ValueError, match="finite non-negative"):
            _validate_threshold("test", float("nan"))

    def test_validate_threshold_none(self) -> None:
        from benchmark.judge_analysis import _validate_threshold

        _validate_threshold("test", None)  # should not raise

    def test_mean_or_none_empty(self) -> None:
        from benchmark.judge_analysis import _mean_or_none

        assert _mean_or_none([]) is None

    def test_sample_sd_single(self) -> None:
        from benchmark.judge_analysis import _sample_sd

        assert _sample_sd([5.0]) is None

    def test_pearson_short(self) -> None:
        from benchmark.judge_analysis import _pearson

        assert _pearson([1.0], [2.0]) is None

    def test_pearson_constant(self) -> None:
        from benchmark.judge_analysis import _pearson

        assert _pearson([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) is None

    def test_numeric(self) -> None:
        from benchmark.judge_analysis import _numeric

        assert _numeric(5) is True
        assert _numeric(3.14) is True
        assert _numeric(True) is False
        assert _numeric("5") is False


# ---------------------------------------------------------------------------
# benchmark.logs
# ---------------------------------------------------------------------------


class TestLogs:
    def test_redact_log_text(self) -> None:
        from benchmark.logs import redact_log_text

        text = "Authorization: Bearer secret123"
        result, changed = redact_log_text(text)
        assert changed is True
        assert "secret123" not in result
        assert "REDACTED" in result

    def test_redact_log_text_no_match(self) -> None:
        from benchmark.logs import redact_log_text

        text = "Normal log line with no secrets"
        result, changed = redact_log_text(text)
        assert changed is False
        assert result == text

    def test_redact_json_field(self) -> None:
        from benchmark.logs import redact_log_text

        text = '"api_key": "sk-12345"'
        result, changed = redact_log_text(text)
        assert changed is True
        assert "sk-12345" not in result

    def test_normalise_data_str(self) -> None:
        from benchmark.logs import _normalise_data

        data, changed = _normalise_data("hello", False)
        assert data == b"hello"
        assert changed is False

    def test_normalise_data_bytes(self) -> None:
        from benchmark.logs import _normalise_data

        data, changed = _normalise_data(b"hello", False)
        assert data == b"hello"

    def test_normalise_data_bad_type(self) -> None:
        from benchmark.logs import _normalise_data

        with pytest.raises(TypeError, match="must be str or bytes"):
            _normalise_data(123, False)  # type: ignore

    def test_normalise_data_bytes_redact(self) -> None:
        from benchmark.logs import _normalise_data

        data, changed = _normalise_data(b"authorization: secret", True)
        assert changed is True

    def test_normalise_data_bytes_not_utf8(self) -> None:
        from benchmark.logs import _normalise_data

        data, changed = _normalise_data(b"\xff\xfe", True)
        assert changed is False  # can't decode, no redaction

    def test_append_only_gzip_log_basic(self, tmp_path: Path) -> None:
        from benchmark.logs import AppendOnlyGzipLog

        path = str(tmp_path / "test.log")
        log = AppendOnlyGzipLog(path, member_target_bytes=10, redact=False)
        log.append("line 1\n")
        log.append("line 2\n")
        log.flush()
        log.close()
        # Should be readable
        from benchmark.logs import iter_log_members

        members = list(iter_log_members(path))
        assert len(members) >= 1
        content = b"".join(members)
        assert b"line 1" in content

    def test_append_record(self, tmp_path: Path) -> None:
        from benchmark.logs import AppendOnlyGzipLog

        path = str(tmp_path / "record.log")
        log = AppendOnlyGzipLog(path, redact=False)
        log.append_record(["record one\n", "record two\n"])
        log.close()
        from benchmark.logs import iter_log_members

        members = list(iter_log_members(path))
        assert len(members) == 1
        assert b"record one" in members[0]

    def test_recover_log_nonexistent(self, tmp_path: Path) -> None:
        from benchmark.logs import recover_log

        r = recover_log(str(tmp_path / "nonexistent.log"))
        assert r.complete_members == 0

    def test_recover_log_repair(self, tmp_path: Path) -> None:
        from benchmark.logs import AppendOnlyGzipLog, recover_log

        path = str(tmp_path / "repair.log")
        log = AppendOnlyGzipLog(path, member_target_bytes=1024, redact=False)
        log.append("good data\n")
        log.close()

        r = recover_log(path, repair=True)
        assert r.complete_members >= 1

    def test_member_target_bytes_zero(self) -> None:
        from benchmark.logs import AppendOnlyGzipLog

        with pytest.raises(ValueError, match="member_target_bytes"):
            AppendOnlyGzipLog("/tmp/x.log", member_target_bytes=0)

    def test_flush_interval_negative(self) -> None:
        from benchmark.logs import AppendOnlyGzipLog

        with pytest.raises(ValueError, match="flush_interval"):
            AppendOnlyGzipLog("/tmp/x.log", flush_interval=-1)

    def test_sync_policy_invalid(self) -> None:
        from benchmark.logs import AppendOnlyGzipLog

        with pytest.raises(ValueError, match="sync_policy"):
            AppendOnlyGzipLog("/tmp/x.log", sync_policy="bad")

    def test_close_twice(self, tmp_path: Path) -> None:
        from benchmark.logs import AppendOnlyGzipLog

        log = AppendOnlyGzipLog(str(tmp_path / "x.log"), redact=False)
        log.close()
        log.close()  # second close is a no-op

    def test_append_after_close_raises(self, tmp_path: Path) -> None:
        from benchmark.logs import AppendOnlyGzipLog

        log = AppendOnlyGzipLog(str(tmp_path / "x.log"), redact=False)
        log.close()
        with pytest.raises(RuntimeError, match="closed"):
            log.append("data")

    def test_recover_on_init(self, tmp_path: Path) -> None:
        from benchmark.logs import AppendOnlyGzipLog

        path = str(tmp_path / "r.log")
        # Create a log, close it, then open with recover
        log1 = AppendOnlyGzipLog(path, redact=False)
        log1.append("first\n")
        log1.close()
        log2 = AppendOnlyGzipLog(path, recover_tail=True, redact=False)
        log2.append("second\n")
        log2.close()
        from benchmark.logs import iter_log_members

        members = list(iter_log_members(path))
        assert len(members) >= 2


# ---------------------------------------------------------------------------
# benchmark.storage_validation
# ---------------------------------------------------------------------------


class TestStorageValidation:
    def test_compare_read_models_identical(self) -> None:
        from benchmark.storage_validation import compare_read_models

        rows = [
            {"state_key": "m1", "runner": "http", "http_score": 15.0},
        ]
        report = compare_read_models(rows, list(rows))
        assert report.equivalent
        assert len(report.differences) == 0

    def test_compare_read_models_different(self) -> None:
        from benchmark.storage_validation import compare_read_models

        left = [{"state_key": "m1", "runner": "http", "http_score": 15.0}]
        right = [{"state_key": "m1", "runner": "http", "http_score": 10.0}]
        report = compare_read_models(left, right)
        assert not report.equivalent
        assert len(report.differences) > 0

    def test_compare_read_models_missing(self) -> None:
        from benchmark.storage_validation import compare_read_models

        left = [{"state_key": "m1", "runner": "http", "http_score": 15.0}]
        report = compare_read_models(left, [])
        assert not report.equivalent
        assert report.differences[0].category == "missing-right"

    def test_validation_report_as_dict(self) -> None:
        from benchmark.storage_validation import compare_read_models

        left = [{"state_key": "m1", "runner": "http", "http_score": 15.0}]
        right = [{"state_key": "m1", "runner": "http", "http_score": 10.0}]
        report = compare_read_models(left, right)
        d = report.as_dict()
        assert "equivalent" in d
        assert "differences" in d


# ---------------------------------------------------------------------------
# benchmark.sqlite_schema
# ---------------------------------------------------------------------------


class TestSQLiteSchema:
    def test_configure_connection(self) -> None:
        from benchmark.sqlite_schema import configure_connection

        conn = sqlite3.connect(":memory:")
        configure_connection(conn)
        row = conn.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1
        conn.close()

    def test_configure_connection_invalid_sync(self) -> None:
        from benchmark.sqlite_schema import configure_connection

        conn = sqlite3.connect(":memory:")
        with pytest.raises(ValueError, match="synchronous"):
            configure_connection(conn, synchronous="INVALID")
        conn.close()

    def test_initialize_schema_idempotent(self, tmp_path: Path) -> None:
        from benchmark.sqlite_schema import (
            configure_connection,
            connect_database,
            initialize_schema,
        )

        db = str(tmp_path / "test.db")
        conn = connect_database(db)
        configure_connection(conn)
        initialize_schema(conn)
        initialize_schema(conn)  # should not raise
        conn.close()

    def test_connect_database(self, tmp_path: Path) -> None:
        from benchmark.sqlite_schema import connect_database

        db = str(tmp_path / "test.db")
        conn = connect_database(db)
        conn.close()
        assert os.path.exists(db)


# ---------------------------------------------------------------------------
# benchmark.sqlite_benchmarks edge cases
# ---------------------------------------------------------------------------


class TestSQLiteBenchmarkStore:
    def _make_store(self) -> tuple[sqlite3.Connection, Any]:
        from benchmark.sqlite_benchmarks import SQLiteBenchmarkStore
        from benchmark.sqlite_schema import configure_connection, initialize_schema

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        configure_connection(conn)
        initialize_schema(conn)
        return conn, SQLiteBenchmarkStore(conn)

    def test_create_run(self) -> None:
        conn, store = self._make_store()
        rev_id = store.create_run(
            "run-1", score_schema="v1", storage_profile="compact",
            runner_mode="http", config={}
        )
        assert rev_id > 0

    def test_register_target(self) -> None:
        conn, store = self._make_store()
        rev_id = store.create_run(
            "run-1", score_schema="v1", storage_profile="compact",
            runner_mode="http", config={}
        )
        target_id = store.register_target(
            rev_id, run_id="run-1", logical_name="m1", runner="http",
            source="local", api_model="m1", target_signature="s1",
            is_agent=False, system_prompt=None, target_config={}, order_index=0,
        )
        assert target_id > 0

    def test_register_plugin_and_activate(self) -> None:
        conn, store = self._make_store()
        rev_id = store.create_run(
            "run-1", score_schema="v1", storage_profile="compact",
            runner_mode="http", config={}
        )
        store.register_plugin("p1", "1.0", name="P1", max_score=20,
                              supports_streaming=True, metadata={})
        store.activate_plugin(rev_id, "p1", "1.0")

    def test_ensure_cell(self) -> None:
        conn, store = self._make_store()
        rev_id = store.create_run(
            "run-1", score_schema="v1", storage_profile="compact",
            runner_mode="http", config={}
        )
        target_id = store.register_target(
            rev_id, run_id="run-1", logical_name="m1", runner="http",
            source="local", api_model="m1", target_signature="s1",
            is_agent=False, system_prompt=None, target_config={}, order_index=0,
        )
        store.register_plugin("p1", "1.0", name="P1", max_score=20,
                              supports_streaming=True, metadata={})
        cell_id = store.ensure_cell(rev_id, target_id, "p1", "1.0")
        assert cell_id > 0
        # ensure_cell should return same id
        cell_id2 = store.ensure_cell(rev_id, target_id, "p1", "1.0")
        assert cell_id == cell_id2

    def test_record_attempt_and_select(self) -> None:
        conn, store = self._make_store()
        rev_id = store.create_run(
            "run-1", score_schema="v1", storage_profile="compact",
            runner_mode="http", config={}
        )
        target_id = store.register_target(
            rev_id, run_id="run-1", logical_name="m1", runner="http",
            source="local", api_model="m1", target_signature="s1",
            is_agent=False, system_prompt=None, target_config={}, order_index=0,
        )
        store.register_plugin("p1", "1.0", name="P1", max_score=20,
                              supports_streaming=True, metadata={})
        store.activate_plugin(rev_id, "p1", "1.0")
        cell_id = store.ensure_cell(rev_id, target_id, "p1", "1.0")
        attempt_id = store.record_attempt(rev_id, cell_id, {
            "attempt_number": 1, "prompt": "p", "content": "c", "thinking": "",
            "max_tokens": 4096, "output_tokens": 50, "thinking_tokens": 0,
            "total_tokens": 50, "tps": 10.0, "finish_reason": "stop",
            "response_nature": "ok", "retry_reason": None, "prompt_altered": None,
            "truncated": False, "truncated_due_to_time": False,
            "failure_cause": None, "stream_ok": True, "repeating": False,
            "empty_reason": None, "error": None, "score": 10.0,
            "rubric": [], "diagnostics": {}, "status": "ok",
        }, selected=True)
        assert attempt_id > 0


# ---------------------------------------------------------------------------
# benchmark.sqlite_judges edge cases
# ---------------------------------------------------------------------------


class TestSQLiteJudges:
    def _make_store(self) -> tuple[sqlite3.Connection, Any]:
        from benchmark.sqlite_judges import SQLiteJudgeStore
        from benchmark.sqlite_schema import configure_connection

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        configure_connection(conn)
        from benchmark.sqlite_schema import initialize_schema
        initialize_schema(conn)
        return conn, SQLiteJudgeStore(conn)

    def _create_rev(self, conn: sqlite3.Connection) -> int:
        from benchmark.sqlite_benchmarks import SQLiteBenchmarkStore
        bench = SQLiteBenchmarkStore(conn)
        return bench.create_run(
            "run-1", score_schema="v1", storage_profile="compact",
            runner_mode="http", config={}
        )

    def test_register_judge(self) -> None:
        conn, store = self._make_store()
        rev_id = self._create_rev(conn)
        store.register_judge(rev_id, "judge-model", source="http")

    def test_register_contract(self) -> None:
        conn, store = self._make_store()
        from benchmark.sqlite_benchmarks import SQLiteBenchmarkStore
        bench = SQLiteBenchmarkStore(conn)
        rev_id = bench.create_run(
            "run-1", score_schema="v1", storage_profile="compact",
            runner_mode="http", config={}
        )
        bench.register_plugin("p1", "1.0", name="P1", max_score=20,
                              supports_streaming=True, metadata={})
        bench.activate_plugin(rev_id, "p1", "1.0")
        store.register_contract(
            "contract-1", plugin_id="p1", plugin_version="1.0",
            prompt_version="v1", instructions_version="v1",
            response_schema_hash="abc", contract={"schema": {}},
            contract_hash="abc",
        )

    def test_activate_contract(self) -> None:
        conn, store = self._make_store()
        rev_id = self._create_rev(conn)
        from benchmark.sqlite_benchmarks import SQLiteBenchmarkStore
        bench = SQLiteBenchmarkStore(conn)
        bench.register_plugin("p1", "1.0", name="P1", max_score=20,
                              supports_streaming=True, metadata={})
        bench.activate_plugin(rev_id, "p1", "1.0")
        store.register_contract(
            "contract-1", plugin_id="p1", plugin_version="1.0",
            prompt_version="v1", instructions_version="v1",
            response_schema_hash="abc", contract={},
            contract_hash="abc",
        )
        store.activate_contract(rev_id, "p1", "contract-1")


# ---------------------------------------------------------------------------
# benchmark.storage – SQLiteRunStore lifecycle
# ---------------------------------------------------------------------------


class TestSQLiteRunStoreLifecycle:
    def test_start_and_prepare(self, tmp_path: Path) -> None:
        from benchmark.runtime_records import PluginRecord, TargetRecord
        from benchmark.storage import RunIdentity, SQLiteRunStore

        db = str(tmp_path / "test.db")
        store = SQLiteRunStore(db)
        store.start_run(RunIdentity("run-1", 1))
        assert store.revision_id is not None
        target = TargetRecord(
            logical_name="model-a", runner="http", source="local",
            api_model="model-a", target_signature="sig1",
        )
        plugin = PluginRecord(
            plugin_id="test-plugin", plugin_version="1.0",
            name="Test", max_score=20, supports_streaming=True,
        )
        store.prepare_run([target], [plugin])
        cell_id = store.get_cell_id("model-a", "http", "test-plugin")
        assert cell_id is not None

    def test_record_judge_result_caches(self) -> None:
        from benchmark.storage import JsonRunStore

        mock_state = MagicMock()
        store = JsonRunStore(mock_state)
        store.record_judge_result("m1", "http", "p1", score=15)
        mock_state.update_judge_result.assert_called_once()

    def test_save_snapshot_json(self) -> None:
        from benchmark.storage import JsonRunStore

        mock_state = MagicMock()
        mock_state.compact_journal.return_value = True
        store = JsonRunStore(mock_state)
        result = store.save_snapshot("/tmp/path.json")
        assert result is True

    def test_latest_results_json(self) -> None:
        from benchmark.storage import JsonRunStore

        mock_state = MagicMock()
        mock_state.latest_results.return_value = [{"model": "m1"}]
        store = JsonRunStore(mock_state)
        results = store.latest_results()
        assert results == [{"model": "m1"}]

    def test_update_model_json_noop(self) -> None:
        from benchmark.storage import JsonRunStore

        mock_state = MagicMock()
        store = JsonRunStore(mock_state)
        store.update_model("m1", status="ok")
        # BenchmarkState owns JSON mutations - this is a no-op on JsonRunStore
