"""Coverage-focused tests for the benchmark.cli TUI helpers.

test_tui_cells.py exercises the merged-cell and live-indicator helpers; this
file covers the remaining rendering helpers (model rows, live activity,
recent errors, footer variants), the width/abbreviation helpers, the
429-sleep lookup, the runner pipeline error paths, and tui_main itself.
"""
import threading
import time
import unittest
from unittest import mock

from benchmark import cli


class _FakeWindow:
    """Minimal curses stand-in recording calls."""

    def __init__(self, height=40, width=80):
        self.height = height
        self.width = width
        self.calls = []
        self.lines = {}

    def getmaxyx(self):
        return (self.height, self.width)

    def move(self, y, x):
        self.calls.append(("move", y, x))

    def clrtoeol(self):
        self.calls.append(("clrtoeol",))

    def erase(self):
        self.calls.append(("erase",))

    def addstr(self, y, x, text, attr=0):
        self.calls.append(("addstr", y, x, text, attr))
        self.lines[y] = text

    def nodelay(self, flag):
        self.calls.append(("nodelay", flag))

    def keypad(self, flag):
        self.calls.append(("keypad", flag))

    def refresh(self):
        self.calls.append(("refresh",))

    def instr(self, y, x, n):
        return self.lines.get(y, "")[:n].encode()


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


class TestWrBounds(unittest.TestCase):
    def test_out_of_bounds_write_is_ignored(self):
        window = _FakeWindow()
        cli._wr(window, 80, 40, 40, 0, "text")   # y == max_y
        cli._wr(window, 80, 40, 0, 80, "text")   # x == max_x
        cli._wr(window, 80, 40, -1, 0, "text")
        self.assertFalse(any(c[0] == "addstr" for c in window.calls))

    def test_curses_error_is_swallowed(self):
        class _Boom(_FakeWindow):
            def move(self, y, x):
                raise cli.curses.error("resized")

        cli._wr(_Boom(), 80, 40, 1, 0, "text")  # must not raise


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
            "rate-limiter_judge_votes": [{"model": "judge-a", "score": 80}],
            "rate-limiter_judge_complete": True,
        })
        frozen, _ = cli._format_model_row(
            "model-a", state, 7, [plugin], {"Local": "LC"})
        self.assertIn("7✅", frozen)

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
                "rate-limiter_judge_votes": [{"model": "judge-a", "score": 80}],
            },
            {
                "judge_models": ["judge-a"],
                "rate-limiter_judge_votes": [{"model": "judge-a", "score": 80}],
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
            "rate-limiter_judge_votes": [{"model": "judge-a", "score": 80}],
            "rate-limiter_judge_complete": True,
        })
        frozen, plugin_str = cli._format_model_row(
            "model-a", state, 7, [plugin], {"Local": "LC"})
        self.assertIn("7⚖️", frozen)
        self.assertIn("80 ⚖️ 1", plugin_str)
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


class TestRenderModelRows(unittest.TestCase):
    def test_renders_rows_and_clears_tail(self):
        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        plugin.supports_streaming = True
        snap_items = [
            ("m1", {"status": "completed", "source": "Local",
                    "rate-limiter_score": 10.0}),
            ("m2", {"status": "failed", "source": "Local"}),
            ("m3", {"status": "running", "source": "Local",
                    "running_pids": ["rate-limiter"]}),
        ]
        window = _FakeWindow(40, 80)
        with mock.patch("benchmark.cli.curses.color_pair", side_effect=[1, 3, 2]):
            cli._render_model_rows(
                window, 80, 40, snap_items, [plugin], {"Local": "LC"},
                scroll_y=0, scroll_x=0, visible_rows=5, frozen_width=34,
                model_top=4, sleeping_lookup={},
            )
        self.assertIn("m1", window.lines.get(4, ""))
        self.assertIn("m2", window.lines.get(5, ""))

    def test_scroll_offset_shifts_rows(self):
        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        plugin.supports_streaming = True
        snap_items = [
            ("m1", {"status": "pending", "source": "Local"}),
            ("m2", {"status": "pending", "source": "Local"}),
        ]
        window = _FakeWindow(40, 80)
        cli._render_model_rows(
            window, 80, 40, snap_items, [plugin], {"Local": "LC"},
            scroll_y=1, scroll_x=0, visible_rows=5, frozen_width=34,
            model_top=4, sleeping_lookup={},
        )
        self.assertIn("m2", window.lines.get(4, ""))
        self.assertNotIn("m1", window.lines.get(4, ""))


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


class TestRenderTableHeadings(unittest.TestCase):
    def test_renders_headings_when_tall_enough(self):
        window = _FakeWindow(40, 80)
        with mock.patch("benchmark.cli.curses.A_UNDERLINE", 4):
            plugin_hdr = cli._render_table_headings(
                window, 80, 40, scroll_x=0,
                frozen_cols=[("#", 4), ("S", 4)],
                plugin_cols=[("RatSc", 5), ("RatTok", 6)],
                frozen_width=12,
            )
        self.assertIn("RatSc", plugin_hdr)
        self.assertTrue(any(c[0] == "addstr" for c in window.calls))

    def test_short_terminal_skips_heading(self):
        window = _FakeWindow(3, 80)
        plugin_hdr = cli._render_table_headings(
            window, 80, 3, scroll_x=0,
            frozen_cols=[("#", 4)], plugin_cols=[("RatSc", 5)],
            frozen_width=8,
        )
        self.assertIn("RatSc", plugin_hdr)
        self.assertFalse(any(c[0] == "addstr" for c in window.calls))


class TestRenderLiveActivity(unittest.TestCase):
    def test_renders_live_models_preloading_and_sleeping(self):
        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        plugin.supports_streaming = True
        snap = {
            "model-a": {"source": "Local", "running_pids": ["rate-limiter"],
                        "rate-limiter_start_ts": time.monotonic() - 3,
                        "rate-limiter_bytes_received": 16},
            "model-b": {"source": "Local", "preloading": True,
                        "preload_start_ts": time.monotonic() - 2},
        }
        sleeping_lookup = {
            ("Local", "model-a", "rate-limiter"): {
                "wake_ts": time.time() + 10,
                "attempts": 1,
                "max_attempts": 3,
            }
        }
        window = _FakeWindow(40, 100)
        cli._render_live_activity(
            window, 100, 40, snap, {"Local": "LC"},
            live_models=["model-a"], live_top=10, live_height=3, log_top=36,
            active_plugins=[plugin], sleeping_lookup=sleeping_lookup,
            preloading_models=["model-b"],
        )
        self.assertIn("Live:", window.lines.get(10, ""))
        self.assertIn("Preloading model model-b", window.lines.get(12, ""))
        self.assertIn("429 Sleeping:", window.lines.get(13, ""))

    def test_renders_active_judge_activity_without_tokens(self):
        window = _FakeWindow(40, 120)
        cli._render_live_activity(
            window, 120, 40, {}, {"Local": "LC"}, [],
            live_top=10, live_height=3, log_top=20,
            active_plugins=[], sleeping_lookup={},
            judge_activities=[{
                "judge": "judge-model",
                "target": "target-model",
                "plugin": "rate-limiter",
                "tokens": 42,
                "elapsed": 5,
            }],
        )
        rendered = " ".join(window.lines.get(y, "") for y in range(10, 20))
        self.assertIn("⚖️ Judge judge-model [target-model rate-limiter (5s)]", rendered)
        self.assertNotIn("42", rendered)
        self.assertNotIn("tok", rendered)

    def test_groups_multiple_active_cells_for_one_judge(self):
        window = _FakeWindow(40, 120)
        cli._render_live_activity(
            window, 120, 40, {}, {"Local": "LC"}, [],
            live_top=10, live_height=3, log_top=20,
            active_plugins=[], sleeping_lookup={},
            judge_activities=[
                {
                    "judge": "judge-model",
                    "target": "target-a",
                    "plugin": "rate-limiter",
                    "elapsed": 2,
                },
                {
                    "judge": "judge-model",
                    "target": "target-b",
                    "plugin": "wireframes",
                    "elapsed": 7,
                },
                {
                    "judge": "other-judge",
                    "target": "target-c",
                    "plugin": "tool-calling",
                    "elapsed": 3,
                },
            ],
        )
        rendered = " ".join(window.lines.get(y, "") for y in range(10, 20))
        self.assertIn(
            "⚖️ Judge judge-model [target-a rate-limiter (2s)] "
            "[target-b wireframes (7s)]",
            rendered,
        )
        self.assertIn(
            "⚖️ Judge other-judge [target-c tool-calling (3s)]",
            rendered,
        )
        self.assertEqual(rendered.count("⚖️ Judge judge-model"), 1)
        self.assertNotIn("tok", rendered)


class TestRenderRecentErrors(unittest.TestCase):
    def test_renders_errors_when_present(self):
        state = mock.MagicMock()
        state.recent_log.return_value = [(time.time(), "model-a", "boom")]
        window = _FakeWindow(40, 80)
        with mock.patch("benchmark.cli.curses.color_pair", return_value=3):
            cli._render_recent_errors(window, 80, 40, state, log_top=30, footer_line=36)
        text = " ".join(window.lines.get(y, "") for y in range(30, 36))
        self.assertIn("Errors:", text)
        self.assertIn("boom", text)


class TestRenderFooterVariants(unittest.TestCase):
    def test_all_done_message(self):
        window = _FakeWindow()
        cli._render_footer(window, 80, 40, [], [], 39)
        self.assertIn("All models complete", window.lines.get(39, ""))

    def test_active_and_queued_parts(self):
        window = _FakeWindow()
        cli._render_footer(window, 80, 40, ["m1"], ["m2"], 39)
        rendered = window.lines.get(39, "")
        self.assertIn("1 active", rendered)
        self.assertIn("1 queued", rendered)

    def test_preloading_details_with_seconds(self):
        window = _FakeWindow()
        cli._render_footer(window, 80, 40, [], [], 39,
                           preloading_models=["model-a"],
                           preloading_details=[("model-a", 8.4)])
        rendered = window.lines.get(39, "")
        self.assertIn("Preloading model-a 8s", rendered)

    def test_preloading_count_only(self):
        window = _FakeWindow()
        cli._render_footer(window, 80, 40, [], [], 39,
                           preloading_models=["model-a"])
        rendered = window.lines.get(39, "")
        self.assertIn("1 preloading", rendered)

    def test_judge_progress_is_rendered_per_model(self):
        window = _FakeWindow()
        cli._render_footer(
            window, 120, 40, [], [], 39,
            judge_progress={
                "Big Pickle": {"completed": 4, "expected": 17},
                "gemini/gemini-2.5-flash-lite": {"completed": 5, "expected": 17},
            },
        )
        rendered = window.lines.get(39, "")
        self.assertIn("Judging", rendered)
        self.assertIn("[Big Pickle: 4/17]", rendered)
        self.assertIn("[gemini/gemini-2.5-flash-lite: 5/17]", rendered)


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


class TestWrEmptySafeText(unittest.TestCase):
    def test_truncated_to_empty_skips_addstr(self):
        window = _FakeWindow()
        cli._wr(window, 80, 40, 1, 79, "ab")  # max_x - x - 1 == 0
        self.assertFalse(any(c[0] == "addstr" for c in window.calls))


class TestGraphemeTrailingYield(unittest.TestCase):
    def test_trailing_control_char_leaves_empty_cluster(self):
        # A control char terminates the row, so the final ``if cluster:``
        # yield is skipped when the input ends on one.
        self.assertEqual(list(cli._grapheme_clusters("ab\x1b")), ["a", "b"])


class TestRenderHeaderSmallTerminal(unittest.TestCase):
    def test_one_row_terminal_skips_summary_and_separator(self):
        window = _FakeWindow(1, 80)
        snap = {"m1": {"status": "completed"}}
        cli._render_header_and_summary(window, 80, 1, snap, 1, 1, [], [], [],
                                       0, 1, 1, None, 0, 0)
        self.assertTrue(any(c[0] == "addstr" for c in window.calls))

    def test_two_row_terminal_skips_separator(self):
        window = _FakeWindow(2, 80)
        snap = {"m1": {"status": "completed"}}
        cli._render_header_and_summary(window, 80, 2, snap, 1, 1, [], [], [],
                                       0, 1, 1, None, 0, 0)
        self.assertTrue(any(c[0] == "addstr" for c in window.calls))


class TestRenderModelRowsExceptions(unittest.TestCase):
    def test_color_pair_raising_is_swallowed(self):
        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        plugin.supports_streaming = True
        snap_items = [
            ("m1", {"status": "completed", "source": "Local"}),
            ("m2", {"status": "failed", "source": "Local"}),
            ("m3", {"status": "running", "source": "Local",
                    "running_pids": ["rate-limiter"]}),
        ]
        window = _FakeWindow(40, 80)
        with mock.patch("benchmark.cli.curses.color_pair",
                        side_effect=Exception("no colors")):
            cli._render_model_rows(
                window, 80, 40, snap_items, [plugin], {"Local": "LC"},
                scroll_y=0, scroll_x=0, visible_rows=5, frozen_width=34,
                model_top=4, sleeping_lookup={},
            )
        self.assertIn("m1", window.lines.get(4, ""))

    def test_tail_clear_move_raising_is_swallowed(self):
        class _BoomMove(_FakeWindow):
            def move(self, y, x):
                if y >= 7:
                    raise cli.curses.error("resized")
                return super().move(y, x)

        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        plugin.supports_streaming = True
        snap_items = [("m1", {"status": "completed", "source": "Local"})]
        window = _BoomMove(40, 80)
        cli._render_model_rows(
            window, 80, 40, snap_items, [plugin], {"Local": "LC"},
            scroll_y=0, scroll_x=0, visible_rows=5, frozen_width=34,
            model_top=4, sleeping_lookup={},
        )  # must not raise


class TestBuildLiveIndicatorsUnknownPid(unittest.TestCase):
    def test_unknown_pid_is_skipped(self):
        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        s = {"running_pids": ["ghost"], "status": "running"}
        self.assertEqual(cli._build_live_indicators(s, [plugin]), "")


class TestRenderLiveActivityEdges(unittest.TestCase):
    def test_live_model_with_error_and_log_overflow(self):
        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        plugin.supports_streaming = True
        snap = {
            "model-a": {"source": "Local", "running_pids": ["rate-limiter"],
                        "rate-limiter_start_ts": time.monotonic() - 3,
                        "rate-limiter_bytes_received": 16, "last_error": "boom"},
        }
        window = _FakeWindow(40, 100)
        cli._render_live_activity(
            window, 100, 40, snap, {"Local": "LC"},
            live_models=["model-a"], live_top=10, live_height=3, log_top=12,
            active_plugins=[plugin], sleeping_lookup={},
        )
        rendered = " ".join(window.lines.get(y, "") for y in range(10, 13))
        self.assertIn("boom", rendered)

    def test_sleeping_overflow_breaks(self):
        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        plugin.supports_streaming = True
        snap = {}
        sleeping_lookup = {
            ("Local", "model-a", "rate-limiter"): {
                "wake_ts": time.time() + 10, "attempts": 1, "max_attempts": 3},
        }
        window = _FakeWindow(40, 100)
        cli._render_live_activity(
            window, 100, 40, snap, {"Local": "LC"},
            live_models=[], live_top=10, live_height=3, log_top=11,
            active_plugins=[plugin], sleeping_lookup=sleeping_lookup,
        )  # sleeping header writes, per-item row exceeds log_top -> break


class TestRenderRecentErrorsEdges(unittest.TestCase):
    def test_no_errors_skips_section(self):
        state = mock.MagicMock()
        state.recent_log.return_value = []
        window = _FakeWindow(40, 80)
        cli._render_recent_errors(window, 80, 40, state, log_top=30, footer_line=36)
        text = " ".join(window.lines.get(y, "") for y in range(30, 36))
        self.assertNotIn("Errors:", text)

    def test_errors_overflow_to_footer_line_break(self):
        state = mock.MagicMock()
        state.recent_log.return_value = [
            (time.time(), "m1", "e1"),
            (time.time(), "m2", "e2"),
            (time.time(), "m3", "e3"),
        ]
        window = _FakeWindow(40, 80)
        with mock.patch("benchmark.cli.curses.color_pair", return_value=3):
            cli._render_recent_errors(window, 80, 40, state, log_top=34, footer_line=36)
        text = " ".join(window.lines.get(y, "") for y in range(34, 36))
        self.assertIn("e1", text)
        self.assertNotIn("e3", text)  # footer-line break truncated the list


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


class TestTuiMain(unittest.TestCase):
    def _state(self):
        state = mock.MagicMock()
        state.snapshot.return_value = {
            "model-a": {"status": "completed", "source": "Local",
                        "rate-limiter_score": 10.0},
            "model-b": {"status": "running", "source": "Local",
                        "running_pids": ["rate-limiter"],
                        "rate-limiter_start_ts": time.monotonic() - 2,
                        "rate-limiter_bytes_received": 8},
        }
        state.completed = 1
        state.total = 2
        state.recent_log.return_value = []
        return state

    def test_runs_one_frame_and_exits_on_stop(self):
        import threading
        window = _FakeWindow(40, 100)
        stop_event = threading.Event()
        plugin = mock.MagicMock()
        plugin.id = "rate-limiter"
        plugin.supports_streaming = True

        curses = mock.MagicMock()
        curses.initscr.return_value = window
        curses.has_colors.return_value = False
        curses.error = OSError

        def _stop_after_sleep(delay):
            stop_event.set()

        with mock.patch("benchmark.cli.curses", curses), \
                mock.patch("benchmark.cli.get_active_request_count", return_value=0), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}), \
                mock.patch("benchmark.cli.time.sleep", side_effect=_stop_after_sleep):
            cli.tui_main(self._state(), stop_event, 2, [plugin], session_seed=42)
        self.assertTrue(any(c[0] == "refresh" for c in window.calls))

    def test_curses_init_failure_falls_back(self):
        import threading
        stop_event = threading.Event()

        curses = mock.MagicMock()
        curses.initscr.side_effect = OSError("no tty")

        with mock.patch("benchmark.cli.curses", curses), \
                mock.patch("benchmark.cli._fallback_tui_loop") as fallback:
            cli.tui_main(self._state(), stop_event, 2, [])
        fallback.assert_called_once()


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


class TestHandleTuiInput(unittest.TestCase):
    def _window_with_key(self, key):
        window = mock.Mock()
        window.getch.return_value = key
        return window

    def test_getch_error_returns_same_position(self):
        window = mock.Mock()
        window.getch.side_effect = OSError("resize")
        self.assertEqual(
            cli._handle_tui_input(window, 2, 3, 10, 5, 80, 34, 100), (2, 3))

    def test_arrow_keys_navigate(self):
        with mock.patch("benchmark.cli.curses.KEY_UP", 1), \
                mock.patch("benchmark.cli.curses.KEY_DOWN", 2):
            self.assertEqual(
                cli._handle_tui_input(self._window_with_key(1), 5, 0, 10, 5, 80, 34, 100),
                (4, 0))
            self.assertEqual(
                cli._handle_tui_input(self._window_with_key(2), 5, 0, 10, 5, 80, 34, 100),
                (6, 0))

    def test_page_space_home_end_keys(self):
        with mock.patch("benchmark.cli.curses.KEY_PPAGE", 3), \
                mock.patch("benchmark.cli.curses.KEY_NPAGE", 4), \
                mock.patch("benchmark.cli.curses.KEY_HOME", 5), \
                mock.patch("benchmark.cli.curses.KEY_END", 6):
            self.assertEqual(
                cli._handle_tui_input(self._window_with_key(3), 8, 0, 10, 5, 80, 34, 100),
                (3, 0))
            self.assertEqual(
                cli._handle_tui_input(self._window_with_key(4), 8, 0, 10, 5, 80, 34, 100),
                (10, 0))
            self.assertEqual(
                cli._handle_tui_input(self._window_with_key(5), 8, 0, 10, 5, 80, 34, 100),
                (0, 0))
            self.assertEqual(
                cli._handle_tui_input(self._window_with_key(6), 8, 0, 10, 5, 80, 34, 100),
                (10, 0))

    def test_left_right_scroll(self):
        with mock.patch("benchmark.cli.curses.KEY_LEFT", 7), \
                mock.patch("benchmark.cli.curses.KEY_RIGHT", 8):
            self.assertEqual(
                cli._handle_tui_input(self._window_with_key(7), 5, 20, 10, 5, 80, 34, 100),
                (5, 12))
            self.assertEqual(
                cli._handle_tui_input(self._window_with_key(8), 5, 0, 10, 5, 80, 34, 100),
                (5, 8))
            # Right-scroll clamps to the plugin header width minus viewport.
            self.assertEqual(
                cli._handle_tui_input(self._window_with_key(8), 5, 100, 10, 5, 80, 34, 100),
                (5, 55))

    def test_scroll_clamped_to_range(self):
        window = self._window_with_key(-1)
        self.assertEqual(
            cli._handle_tui_input(window, -5, 0, 10, 5, 80, 34, 100), (0, 0))
        self.assertEqual(
            cli._handle_tui_input(window, 500, 0, 10, 5, 80, 34, 100), (10, 0))


if __name__ == "__main__":
    unittest.main()
