"""Coverage-focused tests for the non-benchmark CLI dispatcher and shutdown.

``benchmark/command_dispatch.py`` routes one-off CLI commands (check-sqlite,
report generation, judge-queue building, plugin listing, shell completion,
config dumping/conversion) and raises ``SystemExit`` to halt the process after
each.  ``benchmark/shutdown_coordinator.py`` drains persistence during
shutdown.  These tests drive every branch without starting a benchmark run.
"""
import json
import os
import sqlite3
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from benchmark import command_dispatch
from benchmark.commands import check_sqlite, generate_reports, list_plugins
from benchmark.shutdown_coordinator import ShutdownCoordinator


class DispatchEarlyCommandCheckSqliteTest(unittest.TestCase):
    def test_check_sqlite_healthy_exits_zero(self):
        args = SimpleNamespace(
            check_sqlite="run.sqlite3", generate_reports=None,
            output_format=None, revision=None, build_judge_queue=None,
            judge_queue_output=None, no_judge_spread=False,
            judge_spread_threshold=0.1, no_judge_deviation=False,
            judge_deviation_threshold=0.1, list_plugins=False,
            generate_shell_completion=None, dump_default_config=False,
            base_url=None, api_key=None, convert_config=None,
        )
        with mock.patch("benchmark.command_dispatch.check_sqlite",
                        return_value={"ok": True, "sqlite_integrity": "ok", "issues": []}) as check:
            with self.assertRaises(SystemExit) as ctx:
                command_dispatch.dispatch_early_command(args)
        self.assertEqual(ctx.exception.code, 0)
        check.assert_called_once_with("run.sqlite3")

    def test_check_sqlite_unhealthy_exits_nonzero(self):
        args = SimpleNamespace(
            check_sqlite="run.sqlite3", generate_reports=None,
            output_format=None, revision=None, build_judge_queue=None,
            judge_queue_output=None, no_judge_spread=False,
            judge_spread_threshold=0.1, no_judge_deviation=False,
            judge_deviation_threshold=0.1, list_plugins=False,
            generate_shell_completion=None, dump_default_config=False,
            base_url=None, api_key=None, convert_config=None,
        )
        with mock.patch("benchmark.command_dispatch.check_sqlite",
                        return_value={"ok": False, "sqlite_integrity": "bad", "issues": []}):
            with self.assertRaises(SystemExit) as ctx:
                command_dispatch.dispatch_early_command(args)
        self.assertEqual(ctx.exception.code, 1)

    def test_check_sqlite_error_prints_and_exits(self):
        args = SimpleNamespace(
            check_sqlite="missing", generate_reports=None,
            output_format=None, revision=None, build_judge_queue=None,
            judge_queue_output=None, no_judge_spread=False,
            judge_spread_threshold=0.1, no_judge_deviation=False,
            judge_deviation_threshold=0.1, list_plugins=False,
            generate_shell_completion=None, dump_default_config=False,
            base_url=None, api_key=None, convert_config=None,
        )
        with mock.patch("benchmark.command_dispatch.check_sqlite",
                        side_effect=sqlite3.Error("boom")):
            with mock.patch("sys.stderr") as stderr:
                with self.assertRaises(SystemExit) as ctx:
                    command_dispatch.dispatch_early_command(args)
        self.assertNotEqual(ctx.exception.code, 0)
        stderr.write.assert_called()


class DispatchEarlyCommandGenerateReportsTest(unittest.TestCase):
    def _args(self, **overrides):
        defaults = dict(
            check_sqlite=None, output_format=["csv"], revision=None,
            build_judge_queue=None, judge_queue_output=None,
            no_judge_spread=False, judge_spread_threshold=0.1,
            no_judge_deviation=False, judge_deviation_threshold=0.1,
            list_plugins=False, generate_shell_completion=None,
            dump_default_config=False, base_url=None, api_key=None,
            convert_config=None,
        )
        defaults.update(overrides)
        defaults["generate_reports"] = defaults.get("generate_reports", "run-dir")
        return SimpleNamespace(**defaults)

    def test_generate_reports_success_prints_and_exits_zero(self):
        with mock.patch("benchmark.command_dispatch.generate_reports",
                        return_value=["wrote results.csv"]):
            with mock.patch("sys.stdout") as stdout:
                with self.assertRaises(SystemExit) as ctx:
                    command_dispatch.dispatch_early_command(self._args())
        self.assertEqual(ctx.exception.code, 0)
        stdout.write.assert_called()

    def test_generate_reports_missing_format_exits_two(self):
        with self.assertRaises(SystemExit) as ctx:
            command_dispatch.dispatch_early_command(
                self._args(output_format=None))
        self.assertEqual(ctx.exception.code, 2)

    def test_generate_reports_error_exits_nonzero(self):
        with mock.patch("benchmark.command_dispatch.generate_reports",
                        side_effect=OSError("boom")):
            with self.assertRaises(SystemExit) as ctx:
                command_dispatch.dispatch_early_command(self._args())
        self.assertNotEqual(ctx.exception.code, 0)


class DispatchEarlyCommandMiscTest(unittest.TestCase):
    def _base(self):
        return SimpleNamespace(
            check_sqlite=None, generate_reports=None, output_format=None,
            revision=None, build_judge_queue=None, judge_queue_output=None,
            no_judge_spread=False, judge_spread_threshold=0.1,
            no_judge_deviation=False, judge_deviation_threshold=0.1,
            list_plugins=False, generate_shell_completion=None,
            dump_default_config=False, base_url=None, api_key=None,
            convert_config=None,
        )

    def test_build_judge_queue_success_print_path(self):
        args = self._base()
        args.build_judge_queue = "state.json"
        args.judge_queue_output = "queue.md"
        with mock.patch("benchmark.command_dispatch.write_disagreement_queue",
                        return_value="/tmp/queue.md") as writer:
            with mock.patch("sys.stdout"):
                with self.assertRaises(SystemExit) as ctx:
                    command_dispatch.dispatch_early_command(args)
        self.assertEqual(ctx.exception.code, 0)
        writer.assert_called_once_with(
            "state.json", "queue.md", spread_threshold=0.1,
            deviation_threshold=0.1)

    def test_build_judge_queue_disables_thresholds(self):
        args = self._base()
        args.build_judge_queue = "state.json"
        args.no_judge_spread = True
        args.no_judge_deviation = True
        with mock.patch("benchmark.command_dispatch.write_disagreement_queue",
                        return_value="q") as writer:
            with self.assertRaises(SystemExit):
                command_dispatch.dispatch_early_command(args)
        writer.assert_called_once_with(
            "state.json", None, spread_threshold=None, deviation_threshold=None)

    def test_build_judge_queue_error_exits_nonzero(self):
        args = self._base()
        args.build_judge_queue = "state.json"
        with mock.patch("benchmark.command_dispatch.write_disagreement_queue",
                        side_effect=ValueError("bad")):
            with self.assertRaises(SystemExit) as ctx:
                command_dispatch.dispatch_early_command(args)
        self.assertNotEqual(ctx.exception.code, 0)

    def test_list_plugins_exits_zero_and_prints(self):
        args = self._base()
        args.list_plugins = True
        with mock.patch("benchmark.command_dispatch.list_plugins",
                        return_value="plugin list"):
            with mock.patch("sys.stdout") as stdout:
                with self.assertRaises(SystemExit) as ctx:
                    command_dispatch.dispatch_early_command(args)
        self.assertEqual(ctx.exception.code, 0)
        stdout.write.assert_any_call("plugin list")

    def test_generate_shell_completion_exits_zero(self):
        args = self._base()
        args.generate_shell_completion = "bash"
        with mock.patch("benchmark.command_dispatch.generate_shell_completion",
                        return_value="# completion"):
            with self.assertRaises(SystemExit) as ctx:
                command_dispatch.dispatch_early_command(args)
        self.assertEqual(ctx.exception.code, 0)

    def test_dump_default_config_without_base_url(self):
        args = self._base()
        args.dump_default_config = True
        with mock.patch("benchmark.command_dispatch.dump_default_config") as dump:
            with self.assertRaises(SystemExit) as ctx:
                command_dispatch.dispatch_early_command(args)
        self.assertEqual(ctx.exception.code, 0)
        dump.assert_called_once()

    def test_dump_default_config_with_base_url(self):
        args = self._base()
        args.dump_default_config = True
        args.base_url = "http://ollama"
        args.api_key = "key"
        with mock.patch("benchmark.command_dispatch.generate_config_from_api",
                        return_value={"sources": {}}):
            with self.assertRaises(SystemExit):
                command_dispatch.dispatch_early_command(args)

    def test_convert_config_yaml_to_json(self):
        args = self._base()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "cfg.yaml")
            with open(cfg, "w", encoding="utf-8") as handle:
                handle.write("sources: {}\n")
            args.convert_config = cfg
            with mock.patch("benchmark.command_dispatch.load_config",
                            return_value={"sources": {}}):
                with mock.patch("sys.stdout") as stdout:
                    with self.assertRaises(SystemExit) as ctx:
                        command_dispatch.dispatch_early_command(args)
        self.assertEqual(ctx.exception.code, 0)
        stdout.write.assert_called()

    def test_convert_config_json_to_yaml(self):
        args = self._base()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "cfg.json")
            with open(cfg, "w", encoding="utf-8") as handle:
                json.dump({"sources": {}}, handle)
            args.convert_config = cfg
            with mock.patch("benchmark.command_dispatch.load_config",
                            return_value={"sources": {}}):
                with mock.patch("sys.stdout") as stdout:
                    with self.assertRaises(SystemExit) as ctx:
                        command_dispatch.dispatch_early_command(args)
        self.assertEqual(ctx.exception.code, 0)
        stdout.write.assert_any_call("sources: {}\n")

    def test_convert_config_missing_file_exits_one(self):
        args = self._base()
        args.convert_config = "/no/such/config.yaml"
        with self.assertRaises(SystemExit) as ctx:
            command_dispatch.dispatch_early_command(args)
        self.assertEqual(ctx.exception.code, 1)

    def test_convert_config_unsupported_format_exits_one(self):
        args = self._base()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "cfg.ini")
            with open(cfg, "w", encoding="utf-8") as handle:
                handle.write("x=1\n")
            args.convert_config = cfg
            with self.assertRaises(SystemExit) as ctx:
                command_dispatch.dispatch_early_command(args)
        self.assertEqual(ctx.exception.code, 1)

    def test_no_command_returns_false(self):
        self.assertFalse(command_dispatch.dispatch_early_command(self._base()))


class CommandsModuleTest(unittest.TestCase):
    def test_check_sqlite_happy_path_returns_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "run.sqlite3")
            from benchmark.sqlite_schema import connect_database
            connection = connect_database(db)
            connection.close()
            report = check_sqlite(db)
        self.assertTrue(report["ok"])

    def test_check_sqlite_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            check_sqlite("/no/such/file.sqlite3")

    def test_generate_reports_from_json_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = os.path.join(tmp, "benchmark_state.json")
            with open(state, "w", encoding="utf-8") as handle:
                json.dump({
                    "active_plugins": ["rate-limiter"],
                    "session_seed": 7,
                    "results": [{
                        "model": "demo", "status": "ok",
                        "total_time": 1.0, "rate-limiter_score": 10,
                    }],
                }, handle)
            lines = generate_reports(tmp, ["csv"])
        self.assertTrue(any("results.csv" in line for line in lines))

    def test_generate_reports_missing_plugin_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = os.path.join(tmp, "benchmark_state.json")
            with open(state, "w", encoding="utf-8") as handle:
                json.dump({
                    "active_plugins": ["not-a-real-plugin"],
                    "session_seed": 1,
                    "results": [],
                }, handle)
            with self.assertRaises(ValueError):
                generate_reports(tmp, ["csv"])

    def test_list_plugins_returns_string(self):
        self.assertIsInstance(list_plugins(), str)
        self.assertTrue(list_plugins().strip())


class ShutdownCoordinatorTest(unittest.TestCase):
    def _coordinator(self, **overrides):
        state = SimpleNamespace(
            run_store=mock.Mock(),
            close_run_store=mock.Mock(return_value=True),
            consume_journal_failures=mock.Mock(return_value=[]),
        )
        state._run_store = mock.Mock(backend_name="sqlite")
        state._run_store.writer.failures = []
        flusher = mock.Mock()
        flusher.stop.return_value = True
        failures = []
        coordinator = ShutdownCoordinator(
            flusher=flusher or overrides.pop("flusher", flusher),
            shutdown_timeout=overrides.pop("shutdown_timeout", 3.0),
            report_persistence_failure=lambda stage, exc: failures.append(
                (stage, exc)),
            state=overrides.pop("state", state),
            state_file="/tmp/state.json",
            plugin_versions={"rate-limiter": "1.0.0"},
            persistence_lock=threading.Lock(),
        )
        return coordinator, state, flusher, failures

    def test_stop_flusher_success(self):
        coordinator, _state, flusher, _failures = self._coordinator()
        self.assertTrue(coordinator.stop_flusher())
        flusher.stop.assert_called_once_with(timeout=3.0)

    def test_stop_flusher_timeout_reports_failure(self):
        flusher = mock.Mock()
        flusher.stop.return_value = False
        failures = []
        coordinator = ShutdownCoordinator(
            flusher=flusher, shutdown_timeout=2.0,
            report_persistence_failure=lambda stage, exc: failures.append(
                (stage, exc)),
            state=mock.Mock(), state_file="/tmp/s.json",
            plugin_versions={}, persistence_lock=threading.Lock(),
        )
        self.assertFalse(coordinator.stop_flusher())
        self.assertTrue(failures)

    def test_save_final_state_success(self):
        coordinator, state, _flusher, _failures = self._coordinator()
        self.assertTrue(coordinator.save_final_state())
        state.run_store.save_snapshot.assert_called_once_with(
            "/tmp/state.json", plugin_versions={"rate-limiter": "1.0.0"},
            raise_on_error=True)

    def test_save_final_state_raises_on_journal_failures(self):
        state = SimpleNamespace(
            run_store=mock.Mock(),
            close_run_store=mock.Mock(return_value=True),
            consume_journal_failures=mock.Mock(
                return_value=["flush failed"]),
        )
        coordinator = ShutdownCoordinator(
            flusher=mock.Mock(), shutdown_timeout=3.0,
            report_persistence_failure=lambda *a: None,
            state=state, state_file="/tmp/s.json",
            plugin_versions={}, persistence_lock=threading.Lock(),
        )
        with self.assertRaises(RuntimeError):
            coordinator.save_final_state()

    def test_close_backend_success(self):
        coordinator, state, _flusher, _failures = self._coordinator()
        self.assertTrue(coordinator.close_backend())
        state.close_run_store.assert_called_once_with(timeout=3.0)

    def test_close_backend_timeout_reports_failure(self):
        state = SimpleNamespace(close_run_store=mock.Mock(return_value=False))
        failures = []
        coordinator = ShutdownCoordinator(
            flusher=mock.Mock(), shutdown_timeout=1.0,
            report_persistence_failure=lambda stage, exc: failures.append(
                (stage, exc)),
            state=state, state_file="/tmp/s.json",
            plugin_versions={}, persistence_lock=threading.Lock(),
        )
        self.assertFalse(coordinator.close_backend())
        self.assertTrue(failures)

    def test_check_sqlite_writer_failures_reports(self):
        failures = []
        state = SimpleNamespace()
        state._run_store = mock.Mock(backend_name="sqlite")
        state._run_store.writer.failures = ["write failed"]
        coordinator = ShutdownCoordinator(
            flusher=mock.Mock(), shutdown_timeout=1.0,
            report_persistence_failure=lambda stage, exc: failures.append(
                (stage, exc)),
            state=state, state_file="/tmp/s.json",
            plugin_versions={}, persistence_lock=threading.Lock(),
        )
        coordinator.check_sqlite_writer_failures()
        self.assertEqual(failures, [("sqlite writer", "write failed")])

    def test_check_sqlite_writer_no_backend(self):
        failures = []
        state = SimpleNamespace()
        coordinator = ShutdownCoordinator(
            flusher=mock.Mock(), shutdown_timeout=1.0,
            report_persistence_failure=failures.append,
            state=state, state_file="/tmp/s.json",
            plugin_versions={}, persistence_lock=threading.Lock(),
        )
        coordinator.check_sqlite_writer_failures()
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
