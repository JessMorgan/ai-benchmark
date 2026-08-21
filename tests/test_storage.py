"""Tests for the Stage 1 storage contracts and JSON compatibility adapters."""
import json
import os
import tempfile
import unittest

from benchmark.state import BenchmarkState
from benchmark.storage import (
    DebugLogStore,
    JsonDebugLogStore,
    JsonPayloadStore,
    JsonReportSource,
    JsonRunStore,
    PayloadStore,
    ReportSource,
    RunStore,
    latest_result_rows,
)


class TestStorageProtocols(unittest.TestCase):
    def test_json_run_store_implements_run_store(self):
        state = BenchmarkState({"model": "Local"}, ["plugin"])
        self.assertIsInstance(JsonRunStore(state), RunStore)

    def test_protocols_are_runtime_checkable(self):
        self.assertTrue(isinstance(JsonPayloadStore(), PayloadStore))
        self.assertTrue(isinstance(JsonDebugLogStore(), DebugLogStore))
        self.assertIsInstance(JsonReportSource(), ReportSource)


class TestJsonRunStore(unittest.TestCase):
    def test_result_and_judge_updates_delegate_to_state(self):
        state = BenchmarkState({"model": "Local"}, ["plugin"])
        store = JsonRunStore(state)
        result = {"model": "model", "status": "ok", "plugin_score": 10}
        store.record_result(result)
        store.update_model("model", status="completed")
        store.record_judge_result(
            "model", "http", "plugin", score=9, confidence="high",
        )
        self.assertEqual(store.latest_results()[0]["plugin_judge_score"], 9)
        self.assertEqual(state.snapshot()["model"]["status"], "completed")

    def test_snapshot_delegates_to_json_state(self):
        state = BenchmarkState({"model": "Local"}, ["plugin"])
        store = JsonRunStore(state)
        state.update("model", status="completed")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "benchmark_state.json")
            self.assertTrue(store.save_snapshot(path))
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        self.assertEqual(data["model_info"]["model"]["status"], "completed")


class TestJsonReportSource(unittest.TestCase):
    def test_loads_directory_and_file_paths(self):
        data = {
            "active_plugins": ["plugin"],
            "session_seed": 12,
            "results": [
                {"model": "model", "status": "ok"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "benchmark_state.json")
            with open(state_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
            source = JsonReportSource()
            self.assertEqual(source.load_results(tmpdir)[1:], (["plugin"], 12))
            self.assertEqual(source.load_results(state_path)[0], data["results"])


class TestStorageHelpers(unittest.TestCase):
    def test_latest_result_rows_deduplicates_by_runner(self):
        rows = [
            {"model": "m", "runner": "http", "value": 1},
            {"model": "m", "runner": "http", "value": 2},
            {"model": "m", "runner": "pi", "value": 3},
        ]
        self.assertEqual(
            latest_result_rows(rows),
            [rows[1], rows[2]],
        )

    def test_json_payload_store_explicitly_has_no_payload_table(self):
        with self.assertRaises(NotImplementedError):
            JsonPayloadStore().put("prompt", b"hello")

    def test_json_debug_log_store_appends_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "logs", "model.log")
            store = JsonDebugLogStore()
            store.append(path, "one")
            store.append(path, b"two")
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), b"onetwo")
            store.close()


if __name__ == "__main__":
    unittest.main()
