"""Coverage-focused tests for the orchestration and preload coordinators.

``benchmark/orchestration._handle_early_command_exits`` routes one-shot CLI
sub-commands (storage measurement/comparison, JSON→SQLite import,
ChatPlayground config, Pi/schema probes) that exit the process before the
benchmark loop.  ``benchmark/preload_coordinator.PreloadCoordinator`` warms
models once per (source, api_model) with thread-safe dedup.
"""
import json
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from benchmark import orchestration
from benchmark.cli_parser import build_parser
from benchmark.preload_coordinator import PreloadCoordinator


def _base_args(**overrides):
    args = build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class OrchestrationMeasureStorageTest(unittest.TestCase):
    def test_measure_storage_success_exits_zero(self):
        args = _base_args(measure_storage=True)
        with mock.patch("benchmark.storage_measure.measure_storage",
                        return_value={"json": {"bytes": 10}}) as ms:
            with mock.patch("sys.stdout"):
                with self.assertRaises(SystemExit) as ctx:
                    orchestration._handle_early_command_exits(args, None)
        self.assertEqual(ctx.exception.code, 0)
        ms.assert_called_once()

    def test_measure_storage_error_exits_one(self):
        args = _base_args(measure_storage=True)
        with mock.patch("benchmark.storage_measure.measure_storage",
                        side_effect=OSError("boom")):
            with self.assertRaises(SystemExit) as ctx:
                orchestration._handle_early_command_exits(args, None)
        self.assertNotEqual(ctx.exception.code, 0)


class OrchestrationCompareStorageTest(unittest.TestCase):
    def test_compare_storage_equivalent_exits_zero(self):
        args = _base_args(compare_storage=("state.json", "run.sqlite3"))
        report = SimpleNamespace(as_dict=lambda: {"equivalent": True},
                                 equivalent=True)
        with mock.patch("benchmark.orchestration.JsonReportSource") as jsrc, \
             mock.patch("benchmark.orchestration.SQLiteReportSource") as ssrc, \
             mock.patch("benchmark.storage_validation.compare_read_models",
                        return_value=report) as cmp, \
             mock.patch("benchmark.orchestration.latest_result_rows",
                        side_effect=lambda r: r):
            jsrc.return_value.load_results.return_value = ([], [], 1)
            ssrc.open.return_value.load_results.return_value = ([], [], 1, None)
            with mock.patch("sys.stdout"):
                with self.assertRaises(SystemExit) as ctx:
                    orchestration._handle_early_command_exits(args, None)
        self.assertEqual(ctx.exception.code, 0)
        cmp.assert_called_once()

    def test_compare_storage_mismatch_exits_one(self):
        args = _base_args(compare_storage=("state.json", "run.sqlite3"))
        report = SimpleNamespace(as_dict=lambda: {"equivalent": False},
                                 equivalent=False)
        with mock.patch("benchmark.orchestration.JsonReportSource") as jsrc, \
             mock.patch("benchmark.orchestration.SQLiteReportSource") as ssrc, \
             mock.patch("benchmark.storage_validation.compare_read_models",
                        return_value=report):
            jsrc.return_value.load_results.return_value = ([], [], 1)
            ssrc.open.return_value.load_results.return_value = ([], [], 1, None)
            with mock.patch("sys.stdout"):
                with self.assertRaises(SystemExit) as ctx:
                    orchestration._handle_early_command_exits(args, None)
        self.assertNotEqual(ctx.exception.code, 0)

    def test_compare_storage_error_exits_one(self):
        args = _base_args(compare_storage=("state.json", "run.sqlite3"))
        with mock.patch("benchmark.orchestration.JsonReportSource") as jsrc, \
             mock.patch("benchmark.orchestration.SQLiteReportSource"):
            jsrc.return_value.load_results.side_effect = OSError("boom")
            with self.assertRaises(SystemExit) as ctx:
                orchestration._handle_early_command_exits(args, None)
        self.assertNotEqual(ctx.exception.code, 0)


class OrchestrationImportToSqliteTest(unittest.TestCase):
    def test_import_source_not_found_exits_one(self):
        args = _base_args(import_to_sqlite="/no/such/state.json")
        with self.assertRaises(SystemExit) as ctx:
            orchestration._handle_early_command_exits(args, None)
        self.assertEqual(ctx.exception.code, 1)

    def test_import_output_exists_without_overwrite_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = os.path.join(tmp, "benchmark_state.json")
            db = os.path.join(tmp, "run.sqlite3")
            with open(state, "w", encoding="utf-8") as handle:
                json.dump({"active_plugins": []}, handle)
            with open(db, "w", encoding="utf-8") as handle:
                handle.write("existing")
            args = _base_args(import_to_sqlite=state)
            with self.assertRaises(SystemExit) as ctx:
                orchestration._handle_early_command_exits(args, None)
            self.assertEqual(ctx.exception.code, 2)

    def test_import_success_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = os.path.join(tmp, "benchmark_state.json")
            with open(state, "w", encoding="utf-8") as handle:
                json.dump({"active_plugins": []}, handle)
            args = _base_args(import_to_sqlite=state,
                              sqlite_output=os.path.join(tmp, "out.sqlite3"))
            summary = SimpleNamespace(run_id="r", revision_id="1",
                                      __dict__={"run_id": "r"})
            with mock.patch("benchmark.orchestration.LegacySQLiteImporter") as imp:
                imp.import_path.return_value = summary
                with mock.patch("sys.stdout"):
                    with self.assertRaises(SystemExit) as ctx:
                        orchestration._handle_early_command_exits(args, None)
            self.assertEqual(ctx.exception.code, 0)

    def test_import_error_exits_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = os.path.join(tmp, "benchmark_state.json")
            with open(state, "w", encoding="utf-8") as handle:
                json.dump({"active_plugins": []}, handle)
            args = _base_args(import_to_sqlite=state,
                              sqlite_output=os.path.join(tmp, "out.sqlite3"))
            with mock.patch("benchmark.orchestration.LegacySQLiteImporter") as imp:
                imp.import_path.side_effect = ValueError("bad")
                with self.assertRaises(SystemExit) as ctx:
                    orchestration._handle_early_command_exits(args, None)
            self.assertNotEqual(ctx.exception.code, 0)

    def test_import_overwrite_removes_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = os.path.join(tmp, "benchmark_state.json")
            db = os.path.join(tmp, "run.sqlite3")
            with open(state, "w", encoding="utf-8") as handle:
                json.dump({"active_plugins": []}, handle)
            with open(db, "w", encoding="utf-8") as handle:
                handle.write("old")
            args = _base_args(import_to_sqlite=state, overwrite_sqlite=True)
            summary = SimpleNamespace(run_id="r", revision_id="1",
                                      __dict__={"run_id": "r"})
            with mock.patch("benchmark.orchestration.LegacySQLiteImporter") as imp:
                imp.import_path.return_value = summary
                with mock.patch("sys.stdout"):
                    with self.assertRaises(SystemExit):
                        orchestration._handle_early_command_exits(args, None)
            self.assertFalse(os.path.exists(db))


class OrchestrationOtherCommandsTest(unittest.TestCase):
    def test_chatplayground_config_success_exits_zero(self):
        args = _base_args(chatplayground_config=True)
        with mock.patch("benchmark.chatplayground.generate_config",
                        return_value={"sources": {}}):
            with mock.patch("sys.stdout"):
                with self.assertRaises(SystemExit) as ctx:
                    orchestration._handle_early_command_exits(args, None)
        self.assertEqual(ctx.exception.code, 0)

    def test_chatplayground_config_error_exits_one(self):
        args = _base_args(chatplayground_config=True)
        with mock.patch("benchmark.chatplayground.generate_config",
                        side_effect=RuntimeError("boom")):
            with self.assertRaises(SystemExit) as ctx:
                orchestration._handle_early_command_exits(args, None)
        self.assertNotEqual(ctx.exception.code, 0)

    def test_cfg_none_returns_when_no_early_command(self):
        args = _base_args()
        # No early-exit flags set; cfg=None makes the function return False path.
        orchestration._handle_early_command_exits(args, None)

    def test_pi_probe_exits_zero(self):
        args = _base_args(pi_probe=True, timeout=30)
        cfg = {"sources": {"S": {}}, "timeout": 300}
        with mock.patch("benchmark.configuration.resolve_targets",
                        return_value={"m": {"source": "S",
                                            "api_model": "x", "pi": {}}}), \
             mock.patch("benchmark.pi.resolve_pi_worker",
                        return_value=(None, None)), \
             mock.patch("benchmark.pi.run_pi_probe",
                        return_value={"passed": True}), \
             mock.patch("sys.stdout"):
            with self.assertRaises(SystemExit) as ctx:
                orchestration._handle_early_command_exits(args, cfg)
        self.assertEqual(ctx.exception.code, 0)

    def test_schema_sentinel_exits_zero(self):
        args = _base_args(schema_sentinel=True, timeout=30)
        cfg = {"sources": {"S": {}}, "timeout": 300}
        with mock.patch("benchmark.configuration.resolve_targets",
                        return_value={"m": {"source": "S",
                                            "api_model": "x"}}), \
             mock.patch("benchmark.core.run_schema_sentinel",
                        return_value={"ok": True}), \
             mock.patch("sys.stdout"):
            with self.assertRaises(SystemExit) as ctx:
                orchestration._handle_early_command_exits(args, cfg)
        self.assertEqual(ctx.exception.code, 0)


class PreloadCoordinatorTest(unittest.TestCase):
    def _coordinator(self, **overrides):
        state = mock.Mock()
        state.snapshot.return_value = {"m": {"status": "pending"}}
        preload = {"attempted": 0, "succeeded": 0, "failed": 0,
                   "total_preload_time": 0, "per_model": {}}
        run_info = {"preload": preload}
        source_config = {"S": {"preload": True}}
        defaults = dict(
            state=state, source_config=source_config,
            raw_targets={}, args=_base_args(), output_dir="/tmp",
            session_seed=0, stop_event=threading.Event(),
            run_info=run_info, runner_mode="http",
        )
        defaults.update(overrides)
        return PreloadCoordinator(**defaults), state, run_info

    def test_pi_runner_skips_preload(self):
        coordinator, _state, _info = self._coordinator()
        self.assertTrue(coordinator.ensure_preloaded(
            "m", {"source": "S", "api_model": "x"}, "pi"))

    def test_disabled_preload_returns_true(self):
        coordinator, _state, _info = self._coordinator(
            source_config={"S": {"preload": False}})
        self.assertTrue(coordinator.ensure_preloaded(
            "m", {"source": "S", "api_model": "x"}, "http"))

    def test_no_preload_flag_disables(self):
        coordinator, _state, _info = self._coordinator(
            args=_base_args(no_preload=True))
        self.assertTrue(coordinator.ensure_preloaded(
            "m", {"source": "S", "api_model": "x"}, "http"))

    def test_successful_preload_returns_true(self):
        coordinator, state, info = self._coordinator()
        result = mock.Mock(success=True, elapsed=1.0)
        with mock.patch("benchmark.core.preload_model",
                        return_value=result):
            with mock.patch("benchmark.core.resolve_preload_timeout",
                            return_value=10):
                ok = coordinator.ensure_preloaded(
                    "m", {"source": "S", "api_model": "x"}, "http")
        self.assertTrue(ok)
        self.assertEqual(info["preload"]["succeeded"], 1)
        state.update.assert_called()

    def test_failed_preload_returns_false(self):
        coordinator, state, info = self._coordinator()
        result = mock.Mock(success=False, elapsed=2.0, error="boom")
        with mock.patch("benchmark.core.preload_model",
                        return_value=result):
            with mock.patch("benchmark.core.resolve_preload_timeout",
                            return_value=10):
                ok = coordinator.ensure_preloaded(
                    "m", {"source": "S", "api_model": "x"}, "http")
        self.assertFalse(ok)
        self.assertEqual(info["preload"]["failed"], 1)

    def test_cached_success_returns_true_immediately(self):
        coordinator, _state, _info = self._coordinator()
        coordinator._ok.add(("S", "x"))
        self.assertTrue(coordinator.ensure_preloaded(
            "m", {"source": "S", "api_model": "x"}, "http"))

    def test_cached_failure_returns_false_immediately(self):
        coordinator, _state, _info = self._coordinator()
        coordinator._failed.add(("S", "x"))
        self.assertFalse(coordinator.ensure_preloaded(
            "m", {"source": "S", "api_model": "x"}, "http"))

    def test_preload_model_exception_is_captured(self):
        coordinator, state, info = self._coordinator()
        with mock.patch("benchmark.core.preload_model",
                        side_effect=RuntimeError("network")):
            with mock.patch("benchmark.core.resolve_preload_timeout",
                            return_value=10):
                ok = coordinator.ensure_preloaded(
                    "m", {"source": "S", "api_model": "x"}, "http")
        self.assertFalse(ok)
        self.assertEqual(info["preload"]["failed"], 1)

    def test_inflight_non_owner_waits(self):
        coordinator, _state, _info = self._coordinator()
        inflight_event = threading.Event()
        coordinator._inflight[("S", "x")] = inflight_event
        with mock.patch("benchmark.core.preload_model",
                        side_effect=AssertionError("owner must run")):
            # Non-owner path: no preload_model call; waits then returns False.
            thread = threading.Thread(
                target=lambda: inflight_event.set())
            thread.start()
            result = coordinator.ensure_preloaded(
                "m", {"source": "S", "api_model": "x"}, "http")
            thread.join()
        self.assertIs(result, False)

    def test_debug_logs_creates_log_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            coordinator, state, info = self._coordinator(
                args=_base_args(debug_logs=True), output_dir=tmp)
            result = mock.Mock(success=True, elapsed=1.0)
            with mock.patch("benchmark.core.preload_model",
                            return_value=result) as pm:
                with mock.patch("benchmark.core.resolve_preload_timeout",
                                return_value=10):
                    coordinator.ensure_preloaded(
                        "m", {"source": "S", "api_model": "x"}, "http")
            log_path = pm.call_args.kwargs["log_path"]
            self.assertTrue(log_path.endswith("preload.log"))

    def test_runners_both_mode_marks_opencode(self):
        state = mock.Mock()
        state.snapshot.return_value = {
            "m": {"status": "pending"},
            "m [opencode]": {"status": "pending"},
        }
        preload = {"attempted": 0, "succeeded": 0, "failed": 0,
                   "total_preload_time": 0, "per_model": {}}
        run_info = {"preload": preload}
        coordinator = PreloadCoordinator(
            state=state, source_config={"S": {"preload": True}},
            raw_targets={}, args=_base_args(), output_dir="/tmp",
            session_seed=0, stop_event=threading.Event(),
            run_info=run_info, runner_mode="both")
        result = mock.Mock(success=True, elapsed=1.0)
        with mock.patch("benchmark.core.preload_model",
                        return_value=result):
            with mock.patch("benchmark.core.resolve_preload_timeout",
                            return_value=10):
                coordinator.ensure_preloaded(
                    "m", {"source": "S", "api_model": "x"}, "http")
        self.assertEqual(preload["succeeded"], 1)


if __name__ == "__main__":
    unittest.main()
