"""Coverage-focused tests for the benchmark.cli helpers.

test_tui_cells.py exercises the merged-cell and live-indicator helpers; this
file covers the width/abbreviation helpers, the 429-sleep lookup, the runner
pipeline error paths, the fallback TUI loop, and the plain-text TUI entry.
"""
import threading
import time
import unittest
from unittest import mock

from benchmark import cli


class TestCharDisplayWidthEdges(unittest.TestCase):
    def test_combining_and_controls_are_zero_width(self):
        self.assertEqual(cli._char_display_width("\u0301"), 0)  # combining acute
        self.assertEqual(cli._char_display_width("\x1b"), 0)    # ESC control
        self.assertEqual(cli._char_display_width("\r"), 0)
        self.assertEqual(cli._char_display_width("\n"), 0)

    def test_wide_unicode_is_two_columns(self):
        self.assertEqual(cli._char_display_width("界"), 2)
        self.assertEqual(cli._char_display_width("🙂"), 2)

    def test_ascii_and_arrows_are_one_column(self):
        self.assertEqual(cli._char_display_width("a"), 1)
        self.assertEqual(cli._char_display_width("→"), 1)


class TestGraphemeClusters(unittest.TestCase):
    def test_newline_terminates_row(self):
        clusters = list(cli._grapheme_clusters("ab\ncd"))
        self.assertEqual(clusters, ["a", "b"])

    def test_control_chars_skipped(self):
        clusters = list(cli._grapheme_clusters("a\x1bb"))
        self.assertEqual(clusters, ["a", "b"])

    def test_zwj_keeps_emoji_together(self):
        clusters = list(cli._grapheme_clusters("A\u200dB"))
        self.assertEqual(clusters, ["A\u200dB"])

    def test_regional_indicator_pair_forms_flag(self):
        clusters = list(cli._grapheme_clusters("\U0001f1fa\U0001f1f8"))
        self.assertEqual(clusters, ["\U0001f1fa\U0001f1f8"])

    def test_combining_mark_stays_with_base(self):
        clusters = list(cli._grapheme_clusters("e\u0301"))
        self.assertEqual(clusters, ["e\u0301"])


class TestTruncateAndSliceEdges(unittest.TestCase):
    def test_truncate_nonpositive_width_returns_empty(self):
        self.assertEqual(cli._truncate_display_width("abc", 0), "")
        self.assertEqual(cli._truncate_display_width("abc", -1), "")

    def test_slice_negative_start_clamped(self):
        self.assertEqual(cli._slice_display_width("abc", -5, 3), "abc")

    def test_slice_nonpositive_width_returns_empty(self):
        self.assertEqual(cli._slice_display_width("abc", 0, 0), "")

    def test_slice_skips_wide_characters_before_viewport(self):
        # "界" is 2 columns; viewport starting at column 2 begins after it.
        self.assertEqual(cli._slice_display_width("界X", 2, 3), "X")


class TestFormatModelRow(unittest.TestCase):
    def _snap(self, status="running", running_pids=None, source="Local", preloading=False):
        s = {"status": status, "source": source}
        if running_pids is not None:
            s["running_pids"] = running_pids
        if preloading:
            s["preloading"] = True
        return s

    def test_pending_queued_completed_failed_glyphs(self):
        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        plugin.supports_streaming = True
        abbrevs = {"Local": "LC"}
        for status in ("pending", "queued", "completed", "failed", "weird"):
            frozen, plugin_str = cli._format_model_row(
                "model-a", self._snap(status=status), 1, [plugin], abbrevs)
            self.assertIsInstance(frozen, str)
            self.assertIsInstance(plugin_str, str)

    def test_preloading_and_running_glyphs(self):
        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        plugin.supports_streaming = True
        abbrevs = {"Local": "LC"}
        frozen, _ = cli._format_model_row(
            "model-a", self._snap(preloading=True), 1, [plugin], abbrevs)
        self.assertIn("🔄", frozen)
        frozen2, _ = cli._format_model_row(
            "model-a", self._snap(running_pids=["rate-limiter"]), 1, [plugin], abbrevs)
        self.assertIn("🔷", frozen2)

    def test_zero_vote_row_has_no_judge_marker(self):
        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        plugin.supports_streaming = True
        state = self._snap(status="running")
        state.update({
            "rate-limiter_score": 80,
            "judge_models": ["judge-a"],
            "rate-limiter_judge_queued": True,
            "rate-limiter_judge_votes": [],
            "rate-limiter_judge_complete": False,
        })
        frozen, _ = cli._format_model_row(
            "model-a", state, 7, [plugin], {"Local": "LC"})
        self.assertNotIn("⚖️", frozen)

    def test_completed_judge_uses_row_checkmark(self):
        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        plugin.supports_streaming = True
        state = self._snap(status="completed")
        state.update({
            "rate-limiter_score": 80,
            "judge_models": ["judge-a"],
            "rate-limiter_judge_votes": [{
                "model": "judge-a", "score": 80, "confidence": "high",
                "rationale": "valid",
            }],
            "rate-limiter_judge_complete": True,
        })
        frozen, _ = cli._format_model_row(
            "model-a", state, 7, [plugin], {"Local": "LC"},
            active_judge_targets=set())
        self.assertIn("✅", frozen)

    def test_model_number_column_keeps_following_columns_aligned(self):
        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        plugin.supports_streaming = True
        base = self._snap(status="completed")
        base.update({"rate-limiter_score": 80})

        rows = []
        for extra in (
            {},
            {
                "judge_models": ["judge-a", "judge-b"],
                "rate-limiter_judge_queued": True,
                "rate-limiter_judge_votes": [{
                    "model": "judge-a", "score": 80, "confidence": "high",
                    "rationale": "valid",
                }],
            },
            {
                "judge_models": ["judge-a"],
                "rate-limiter_judge_votes": [{
                    "model": "judge-a", "score": 80, "confidence": "high",
                    "rationale": "valid",
                }],
            },
        ):
            state = {**base, **extra}
            frozen, _ = cli._format_model_row(
                "model-a", state, 7, [plugin], {"Local": "LC"})
            rows.append(frozen)

        self.assertEqual(cli.FROZEN_VIEW_WIDTH, 35)
        frozen_header = " ".join(
            f"{header:>{width}}"
            for header, width in [
                ("#", cli.MODEL_NUMBER_COLUMN_WIDTH),
                ("S", 4),
                ("Model", 18),
                ("St", 4),
            ]
        )
        self.assertEqual(cli._display_width(frozen_header) + 1, cli.FROZEN_VIEW_WIDTH)
        for row in rows:
            self.assertEqual(cli._display_width(row), cli.FROZEN_VIEW_WIDTH - 1)
            self.assertEqual(cli._display_width(row[:row.index("LC")]), 6)
            self.assertEqual(cli._display_width(row[:row.index("model-a")]), 10)
            self.assertEqual(cli._display_width(row[:row.rindex("✅")]), 30)

    def test_judge_markers_require_all_current_judges(self):
        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        plugin.supports_streaming = True
        state = self._snap(status="completed")
        state.update({
            "rate-limiter_score": 80,
            "judge_models": ["judge-a", "judge-b"],
            "rate-limiter_judge_queued": True,
            "rate-limiter_judge_votes": [{
                "model": "judge-a", "score": 80, "confidence": "high",
                "rationale": "valid",
            }],
            "rate-limiter_judge_complete": True,
        })
        frozen, plugin_str = cli._format_model_row(
            "model-a", state, 7, [plugin], {"Local": "LC"},
            active_judge_targets={"model-a"})
        self.assertIn("7⚖️", frozen)
        self.assertIn("80 ⚖️1", plugin_str)
        self.assertEqual(cli._display_width(plugin_str), cli.PLUGIN_BLOCK_WIDTH)

    def test_historical_votes_without_active_judging_have_no_row_marker(self):
        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        plugin.supports_streaming = True
        state = self._snap(status="completed")
        state.update({
            "rate-limiter_score": 80,
            "judge_models": ["judge-a", "judge-b"],
            "rate-limiter_judge_queued": False,
            "rate-limiter_judge_votes": [{"model": "judge-a", "score": 80}],
        })
        frozen, _ = cli._format_model_row(
            "model-a", state, 7, [plugin], {"Local": "LC"})
        self.assertNotIn("⚖️", frozen)

    def test_zero_vote_completed_row_has_no_judge_marker(self):
        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        plugin.supports_streaming = True
        state = self._snap(status="completed")
        state.update({
            "rate-limiter_score": 80,
            "judge_models": ["judge-a", "judge-b", "judge-c"],
            "rate-limiter_judge_queued": True,
            "rate-limiter_judge_votes": [],
        })
        frozen, plugin_str = cli._format_model_row(
            "model-a", state, 7, [plugin], {"Local": "LC"})
        self.assertNotIn("⚖️", frozen)
        self.assertNotIn("⚖️", plugin_str)

    def test_unknown_source_abbr_fallback(self):
        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        plugin.supports_streaming = True
        frozen, _ = cli._format_model_row(
            "model-a", self._snap(source="Unknown Source"), 1, [plugin], {})
        self.assertIn("Unk", frozen)


class TestPadAndSourceAbbr(unittest.TestCase):
    def test_pad_display_width_pads_to_target(self):
        self.assertEqual(cli._pad_display_width("ab", 4), "ab  ")
        self.assertEqual(cli._pad_display_width("abcd", 2), "abcd")

    def test_source_abbr_missing_and_none(self):
        self.assertEqual(cli._source_abbr({}, "SomeSource"), "Som")
        self.assertEqual(cli._source_abbr({}, None), "???")
        self.assertEqual(cli._source_abbr({"Local": "LC"}, "Local"), "LC")


class TestBuildSleepingLookup(unittest.TestCase):
    def test_splits_pipe_keys(self):
        backoff = {
            "sleeping": {
                "Local|model-a|rate-limiter": {"wake_ts": 100.0},
            }
        }
        lookup = cli._build_sleeping_lookup(backoff)
        self.assertIn(("Local", "model-a", "rate-limiter"), lookup)

    def test_empty_sleeping_returns_empty(self):
        self.assertEqual(cli._build_sleeping_lookup({}), {})


class TestFmtValueEdges(unittest.TestCase):
    def test_none_renders_dash(self):
        self.assertEqual(cli._fmt_value(None), "-")

    def test_format_failure_falls_back_to_str(self):
        self.assertEqual(cli._fmt_value(object(), fmt=".1f"), str(object()))


class TestSliceBoundaryInsideWideCluster(unittest.TestCase):
    def test_viewport_start_straddles_wide_cluster(self):
        # "界" is 2 columns; a viewport starting at column 1 must skip past
        # the whole cluster rather than emitting a dangling half-width glyph.
        self.assertEqual(cli._slice_display_width("界X", 1, 3), "X")


class TestGraphemeTrailingYield(unittest.TestCase):
    def test_trailing_control_char_leaves_empty_cluster(self):
        # A control char terminates the row, so the final ``if cluster:``
        # yield is skipped when the input ends on one.
        self.assertEqual(list(cli._grapheme_clusters("ab\x1b")), ["a", "b"])


class TestBuildLiveIndicatorsUnknownPid(unittest.TestCase):
    def test_unknown_pid_is_skipped(self):
        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        s = {"running_pids": ["ghost"], "status": "running"}
        self.assertEqual(cli._build_live_indicators(s, [plugin]), "")


class TestStartRunnerPipelineErrors(unittest.TestCase):
    def test_opencode_failure_calls_on_error_and_still_runs_http(self):
        calls = []
        errors = []
        targets_by_source = {"Source": ["model-a"]}
        opencode_pending = {"Source": ["model-a"]}
        http_pending = {"Source": {"model-a"}}
        stop_event = threading.Event()

        def run_target(target_name, runner):
            if runner == "opencode":
                raise RuntimeError("opencode died")
            calls.append((target_name, runner))

        threads = cli._start_runner_pipeline(
            targets_by_source, opencode_pending, http_pending,
            run_target, stop_event, lambda n, r, e: errors.append((n, r, e)),
        )
        for thread in threads:
            thread.join(timeout=2)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0][:2], ("model-a", "opencode"))
        self.assertEqual(calls, [("model-a", "http")])

    def test_http_failure_calls_on_error(self):
        errors = []
        targets_by_source = {"Source": ["model-a"]}
        opencode_pending = {"Source": []}
        http_pending = {"Source": {"model-a"}}
        stop_event = threading.Event()

        def run_target(target_name, runner):
            raise RuntimeError("http died")

        threads = cli._start_runner_pipeline(
            targets_by_source, opencode_pending, http_pending,
            run_target, stop_event, lambda n, r, e: errors.append((n, r, e)),
        )
        for thread in threads:
            thread.join(timeout=2)
        self.assertEqual(errors[0][:2], ("model-a", "http"))

    def test_source_with_no_targets_is_skipped(self):
        threads = cli._start_runner_pipeline(
            {"Source": []}, {"Source": []}, {"Source": set()},
            lambda *a: None, threading.Event(), lambda *a: None,
        )
        self.assertEqual(threads, [])


class TestPromptRestartOrContinueEdges(unittest.TestCase):
    def test_eof_choice_quits(self):
        with mock.patch("builtins.input", side_effect=EOFError()):
            self.assertEqual(cli._prompt_restart_or_continue(scripted=False), "quit")

    def test_restart_choice(self):
        with mock.patch("builtins.input", return_value="r"):
            self.assertEqual(cli._prompt_restart_or_continue(scripted=False), "restart")

    def test_quit_choice(self):
        with mock.patch("builtins.input", return_value="q"):
            self.assertEqual(cli._prompt_restart_or_continue(scripted=False), "quit")

    def test_invalid_then_valid_choice(self):
        with mock.patch("builtins.input", side_effect=["x", "c"]):
            self.assertEqual(cli._prompt_restart_or_continue(scripted=False), "continue")


class TestWriteRunInfoFailure(unittest.TestCase):
    def test_write_failure_prints_warning(self):
        with mock.patch("builtins.open", side_effect=OSError("readonly fs")), \
                mock.patch("sys.stderr") as stderr:
            cli._write_run_info("/tmp/nonexistent-dir", {"status": "ok"})
        written = "".join(call.args[0] for call in stderr.write.call_args_list)
        self.assertIn("Could not write run-info.json", written)


class TestInject429Stats(unittest.TestCase):
    def test_injects_stats_and_returns_run_info(self):
        stats = {
            "total_retries": 3,
            "plugin_stats": {
                "rate-limiter": {"retries": 2, "total_sleep_time": 45.5},
            },
        }
        with mock.patch("benchmark.cli.get_429_stats", return_value=stats):
            run_info = cli._inject_429_stats({})
        self.assertEqual(run_info["backoff_429"]["total_retries"], 3)
        self.assertEqual(
            run_info["backoff_429"]["per_plugin"]["rate-limiter"]["retries"], 2)


class TestFallbackTuiLoop(unittest.TestCase):
    def test_loop_renders_snapshot_and_exits_on_stop(self):
        import threading

        class _State:
            def __init__(self):
                self._completed = 1
                self._total = 2

            def snapshot(self):
                return {
                    "model-a": {"preloading": True,
                                "preload_start_ts": time.monotonic() - 5},
                    "model-b": {"status": "queued",
                                "running_pids": ["rate-limiter"],
                                "attempt_start": time.monotonic() - 3,
                                "last_error": "boom"},
                }

            @property
            def completed(self):
                return self._completed

            @property
            def total(self):
                return self._total

        stop = threading.Event()
        # Let the loop body run once, then trip the event on the first wait.
        def _trip_after_first_wait(timeout):
            stop.set()

        stop.wait = _trip_after_first_wait
        with mock.patch("benchmark.cli.get_active_request_count", return_value=2), \
                mock.patch("benchmark.cli.get_429_stats",
                           return_value={"sleeping": {"Local|m|p": {}}}), \
                mock.patch("sys.stdout") as stdout:
            cli._fallback_tui_loop(_State(), stop, session_seed=42)
        written = "".join(call.args[0] for call in stdout.write.call_args_list)
        self.assertIn("Seed: 42", written)
        self.assertIn("1/2 completed", written)
        self.assertIn("Preloading model-a", written)
        self.assertIn("model-b", written)
        self.assertIn("boom", written)


class TestEnableFaulthandler(unittest.TestCase):
    def test_enable_and_register_sigusr1(self):
        with mock.patch("benchmark.cli.faulthandler.enable") as enable, \
                mock.patch("benchmark.cli.faulthandler.register") as register, \
                mock.patch("benchmark.cli.signal.SIGUSR1", 30, create=True):
            cli._enable_faulthandler()
        enable.assert_called_once()
        register.assert_called_once_with(30)

    def test_skips_sigusr1_when_absent(self):
        # A plain object with no SIGUSR1 attribute (e.g. Windows) must skip
        # the register call without error.
        class _NoSIGUSR1:
            pass

        with mock.patch("benchmark.cli.faulthandler.enable") as enable, \
                mock.patch("benchmark.cli.faulthandler.register") as register, \
                mock.patch("benchmark.cli.signal", new=_NoSIGUSR1()):
            cli._enable_faulthandler()
        enable.assert_called_once()
        register.assert_not_called()


if __name__ == "__main__":
    unittest.main()
