"""Tests for the per-plugin merged-cell visualization in ``ai-benchmark.py``.

Python identifiers disallow hyphens so the file ``ai-benchmark.py`` cannot
be imported by ``import ai_benchmark`` directly. We load the module via
``importlib.util.spec_from_file_location`` so unittest can still exercise
the helper functions.
"""
import importlib.util
import pathlib
import sys
import time
import unittest
from unittest import mock


_THIS_DIR = pathlib.Path(__file__).resolve().parent
_AI_BENCHMARK_PATH = _THIS_DIR.parent / "ai-benchmark.py"
_spec = importlib.util.spec_from_file_location("_ai_benchmark_module", _AI_BENCHMARK_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load spec for {_AI_BENCHMARK_PATH}")
ai_benchmark = importlib.util.module_from_spec(_spec)
sys.modules["_ai_benchmark_module"] = ai_benchmark
_spec.loader.exec_module(ai_benchmark)


class _FakeWindow:
    """Small single-row curses stand-in for testing ``_wr``."""

    def __init__(self, width):
        self.width = width
        self.line = [" "] * width
        self.calls = []

    def move(self, y, x):
        self.calls.append(("move", y, x))

    def clrtoeol(self):
        self.calls.append(("clrtoeol",))
        self.line = [" "] * self.width

    def addstr(self, y, x, text, attr=0):
        self.calls.append(("addstr", y, x, text, attr))
        for char in text:
            if x >= self.width:
                break
            self.line[x] = char
            x += 2 if ai_benchmark._char_display_width(char) == 2 else 1


class TestTuiWriteHelper(unittest.TestCase):
    """Regression tests for stale characters and terminal-edge writes."""

    def test_shorter_next_frame_cannot_leave_leading_or_trailing_characters(self):
        window = _FakeWindow(12)

        ai_benchmark._wr(window, 12, 1, 0, 0, "038s")
        ai_benchmark._wr(window, 12, 1, 0, 0, "38s")

        self.assertEqual("".join(window.line[:3]), "38s")
        self.assertEqual("".join(window.line[3:]), " " * 9)
        self.assertEqual(
            [call[0] for call in window.calls],
            ["move", "clrtoeol", "addstr", "move", "clrtoeol", "addstr"],
        )

    def test_wide_unicode_is_clipped_by_terminal_columns(self):
        window = _FakeWindow(8)

        ai_benchmark._wr(window, 8, 1, 0, 0, "🔄123456")

        addstr = next(call for call in window.calls if call[0] == "addstr")
        self.assertEqual(addstr[3], "🔄12345")
        self.assertLessEqual(ai_benchmark._display_width(addstr[3]), 7)

    def test_zwj_emoji_is_kept_as_one_cluster_when_clipped(self):
        """Clipping must not leave a dangling ZWJ or variation selector."""
        family = "👨‍👩‍👧‍👦"
        rendered = ai_benchmark._truncate_display_width("A" + family + "B", 2)

        self.assertEqual(rendered, "A")
        self.assertNotIn("\\u200d", rendered)
        self.assertLessEqual(ai_benchmark._display_width(rendered), 3)

    def test_combining_mark_and_skin_tone_modifier_stay_with_base(self):
        """Grapheme extensions are included only with their base cluster."""
        text = "é👍🏽X"

        self.assertEqual(ai_benchmark._truncate_display_width(text, 1), "é")
        self.assertEqual(ai_benchmark._truncate_display_width(text, 2), "é")
        self.assertEqual(ai_benchmark._truncate_display_width(text, 3), "é👍🏽")

    def test_control_characters_are_removed_before_curses_write(self):
        """Model text cannot inject a cursor-moving terminal control."""
        window = _FakeWindow(40)

        ai_benchmark._wr(window, 40, 1, 0, 0, "ok\u001b[2J\u001b[H done")

        addstr = next(call for call in window.calls if call[0] == "addstr")
        self.assertEqual(addstr[3], "ok[2J[H done")
        self.assertNotIn("\u001b", addstr[3])

    def test_boundary_write_is_not_retried_after_curses_error(self):
        class ErrorWindow(_FakeWindow):
            def addstr(self, *args):
                self.calls.append(("addstr", args))
                raise ai_benchmark.curses.error("edge")

        window = ErrorWindow(8)
        ai_benchmark._wr(window, 8, 1, 0, 0, "1234567")

        self.assertEqual(len([call for call in window.calls if call[0] == "addstr"]), 1)

    def test_footer_redraw_clears_a_longer_previous_message(self):
        window = _FakeWindow(60)

        ai_benchmark._render_footer(window, 60, 1, [], [], 0)
        ai_benchmark._render_footer(window, 60, 1, ["model"], [], 0)

        rendered = "".join(window.line)
        self.assertTrue(rendered.startswith(" 1 active"))
        self.assertNotIn("All models complete", rendered)
        self.assertEqual(rendered.rstrip(), " 1 active")


class TestPluginCellBlock(unittest.TestCase):
    """The ``_plugin_cell_block`` helper produces a single 26-char cell per
    plugin, collapsing the existing 4-cell results layout to a bracket-
    delimited status message when the plugin is in flight or 429-sleeping.
    """

    def setUp(self):
        self.p_streaming = mock.MagicMock()
        self.p_streaming.id = "rate-limiter"
        self.p_streaming.supports_streaming = True
        self.p_nonstream = mock.MagicMock()
        self.p_nonstream.id = "counter"
        self.p_nonstream.supports_streaming = False

    def test_block_is_always_exactly_26_chars(self):
        """All branches produce a fixed-width cell so vertical alignment
        holds against the existing ``plugin_cols`` table."""
        cases = [
            ({"running_pids": []}, None),                              # queued
            ({"running_pids": ["rate-limiter"]}, None),                # in-flight
            ({"running_pids": ["rate-limiter"]}, 24),                  # 429 sleep
            ({
                "running_pids": [],
                "rate-limiter_score": 95.0,
                "rate-limiter_output_tokens": 123,
                "rate-limiter_response_time": 45.6,
                "rate-limiter_tps": 2.5,
            }, None),                                                   # completed
        ]
        for s, sleeping_remaining in cases:
            with self.subTest(s=s, sleeping_remaining=sleeping_remaining):
                kwargs = {}
                if sleeping_remaining is not None:
                    s = {**s, "source": "Local", "api_model": "model-a"}
                    kwargs["sleeping_lookup"] = {
                        ("Local", "model-a", "rate-limiter"): {
                            "wake_ts": time.time() + sleeping_remaining,
                            "attempts": 1,
                            "max_attempts": 3,
                        }
                    }
                block = ai_benchmark._plugin_cell_block(
                    "rate-limiter", s, self.p_streaming, **kwargs)
                self.assertEqual(len(block), ai_benchmark.PLUGIN_BLOCK_WIDTH)

    def test_queued_no_results_shows_dash_placeholders(self):
        """A model not yet running shows the standard 4-cell layout
        with ``-`` placeholders for missing values (no bracket status
        text)."""
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", {"running_pids": []}, self.p_streaming, None)
        self.assertNotIn("[", block)
        # All numeric cells render as ``-`` placeholders.
        self.assertIn("    -", block,
                      "missing values render as '-' placeholders")
        # The legacy per-plugin streaming-glyph column (``st``) was
        # deleted -- the block no longer carries a trailing
        # streamed-state marker. Verify by checking the block ends
        # with the tps column's ``-`` placeholder, not a separate
        # ``st`` placeholder.
        self.assertEqual(block.strip().split()[-1], "-",
                         "last token is the '-' tps placeholder (no st column)")

    def test_in_flight_streaming_plugin_shows_streaming_after_first_token(self):
        """If a streaming-capable plugin has received its first token,
        the merged cell says ``[streaming]`` (the bare bracket form,
        which is the same label as the no-first-tok state and is
        distinguished by the elapsed suffix once the wait crosses
        2s or by ``[streaming - N tok]`` once bytes accumulate)."""
        s = {
            "running_pids": ["rate-limiter"],
            "rate-limiter_first_tok_ts": 1234.5,
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, None)
        self.assertIn("[streaming]", block)
        # Must not look like the non-streaming branch.
        self.assertNotIn("[requested]", block)
        # Must not look like a tok-count indicator (no bytes here).
        self.assertNotIn("tok", block)

    def test_in_flight_streaming_plugin_shows_streaming_without_first_token(self):
        """If a streaming-capable plugin is in flight but has NOT yet
        received its first token, the merged cell says ``[streaming]``
        (the renamed label that subsumes the old ``[waiting]``). The
        no-first-tok state and the just-received-first-tok transient
        share the bare ``[streaming]`` label and are distinguished by
        the elapsed suffix once the wait crosses 2s."""
        s = {"running_pids": ["rate-limiter"]}
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, None)
        self.assertIn("[streaming]", block)
        # Must not look like the non-streaming branch.
        self.assertNotIn("[requested]", block)
        # Must not look like a tok-count indicator (no bytes yet).
        self.assertNotIn("tok", block)

    def test_in_flight_non_streaming_plugin_shows_requested_label(self):
        """A non-streaming plugin in flight has no first-token concept;
        use ``[requested]`` (the renamed transport-only label that
        subsumes the old ``[in flight]``) instead of ``[running]``
        (which would collide with status="running")."""
        s = {"running_pids": ["counter"]}
        block = ai_benchmark._plugin_cell_block(
            "counter", s, self.p_nonstream, None)
        self.assertIn("[requested]", block)
        self.assertNotIn("[running]", block,
                         "label must not collide with status=running glyph")
        self.assertNotIn("[streaming]", block)

    def test_429_sleep_overrides_in_flight_status(self):
        """If this plugin is in a 429 backoff, the bracket text shows the
        countdown regardless of per-plugin transport state -- the
        operator cares more about wall-clock backoff than the per-
        plugin status."""
        s = {
            "source": "Local",
            "api_model": "model-a",
            "running_pids": ["rate-limiter"],
            "rate-limiter_first_tok_ts": 1234.5,  # would otherwise be [streaming]
        }
        sleeping_lookup = {
            ("Local", "model-a", "rate-limiter"): {
                "wake_ts": time.time() + 24,
                "attempts": 1,
                "max_attempts": 3,
            }
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, sleeping_lookup=sleeping_lookup)
        self.assertIn("[429 sleeping 24s]", block)
        self.assertNotIn("[streaming]", block,
                         "429 must override the per-plugin streaming label")
        self.assertNotIn("[requested]", block,
                         "429 must override the per-plugin requested label")

    def test_429_sleep_not_shown_for_completed_plugins(self):
        """A plugin that already has results is not in flight, so its
        cell should keep its numeric results even when another plugin
        for the same model is 429-sleeping."""
        s = {
            "source": "Local",
            "api_model": "model-a",
            "running_pids": ["wireframes"],
            "rate-limiter_score": 95.0,
            "rate-limiter_output_tokens": 123,
            "rate-limiter_response_time": 45.6,
            "rate-limiter_tps": 2.5,
        }
        sleeping_lookup = {
            ("Local", "model-a", "wireframes"): {
                "wake_ts": time.time() + 7,
                "attempts": 1,
                "max_attempts": 3,
            }
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, sleeping_lookup=sleeping_lookup)
        self.assertNotIn("[429 sleeping", block,
                         "completed plugin cell must not inherit another plugin's 429 status")
        self.assertIn("95.0", block)

    def test_429_sleep_clamps_at_zero_seconds(self):
        """A wake_ts in the past clamps to ``0`` so the bracket stays
        well-formed rather than rendering a negative duration."""
        s = {
            "source": "Local",
            "api_model": "model-a",
            "running_pids": ["rate-limiter"],
        }
        sleeping_lookup = {
            ("Local", "model-a", "rate-limiter"): {
                "wake_ts": time.time() - 1.0,
                "attempts": 2,
                "max_attempts": 3,
            }
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, sleeping_lookup=sleeping_lookup)
        self.assertIn("[429 sleeping 0s]", block)

    def test_429_sleep_only_for_matching_pid(self):
        """Only the plugin that triggered the 429 shows the sleep
        bracket; another in-flight plugin for the same model keeps its
        normal streaming/requested indicator."""
        s = {
            "source": "Local",
            "api_model": "model-a",
            "running_pids": ["rate-limiter", "wireframes"],
            "wireframes_first_chunk_seen": True,
            "wireframes_bytes_received": 64,
        }
        sleeping_lookup = {
            ("Local", "model-a", "rate-limiter"): {
                "wake_ts": time.time() + 12,
                "attempts": 1,
                "max_attempts": 3,
            }
        }
        rl_block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, sleeping_lookup=sleeping_lookup)
        wf_block = ai_benchmark._plugin_cell_block(
            "wireframes", s, self.p_streaming, sleeping_lookup=sleeping_lookup)
        self.assertIn("[429 sleeping 12s]", rl_block)
        self.assertIn("[streaming - 16 tok]", wf_block)
        self.assertNotIn("429 sleeping", wf_block)

    def test_429_sleep_per_plugin_has_different_remaining(self):
        """Each plugin has its own wake_ts, so the countdown shown in the
        table differs per-plugin."""
        s = {
            "source": "Local",
            "api_model": "model-a",
            "running_pids": ["rate-limiter", "moe-dense"],
        }
        sleeping_lookup = {
            ("Local", "model-a", "rate-limiter"): {
                "wake_ts": time.time() + 5,
                "attempts": 1,
                "max_attempts": 3,
            },
            ("Local", "model-a", "moe-dense"): {
                "wake_ts": time.time() + 55,
                "attempts": 2,
                "max_attempts": 3,
            },
        }
        rl_block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, sleeping_lookup=sleeping_lookup)
        md_block = ai_benchmark._plugin_cell_block(
            "moe-dense", s, self.p_streaming, sleeping_lookup=sleeping_lookup)
        self.assertIn("[429 sleeping 5s]", rl_block)
        self.assertIn("[429 sleeping 55s]", md_block)

    def test_completed_plugin_shows_numeric_results(self):
        """A plugin whose task has finished shows the standard 4-cell
        layout with the recorded score / tokens / time / tps. The
        legacy per-plugin streaming-glyph column (``st``) was deleted
        as redundant -- the merged status block already conveys
        in-flight state, and post-flight the plugin isn't streaming
        anymore."""
        s = {
            "running_pids": [],
            "rate-limiter_score": 95.0,
            "rate-limiter_output_tokens": 123,
            "rate-limiter_response_time": 45.6,
            "rate-limiter_tps": 2.5,
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, None)
        # No bracket text in completed state.
        self.assertNotIn("[", block)
        self.assertNotIn("]", block)
        # Numeric results present in their respective columns.
        self.assertIn("95.0", block)
        self.assertIn("123", block)
        self.assertIn("45.6", block)
        self.assertIn("2.5", block)
        # Block width matches PLUGIN_BLOCK_WIDTH (no separate st column).
        self.assertEqual(len(block), ai_benchmark.PLUGIN_BLOCK_WIDTH)
        # Last token is the tps value (no trailing '-' st glyph).
        self.assertEqual(block.strip().split()[-1], "2.5",
                         "block ends with the tps value (st column deleted)")

    def test_streaming_plugin_shows_tok_count_after_bytes_accumulate(self):
        """Once the streaming callback has accumulated chars (mocked
        here as a stored counter), the cell shows
        ``[streaming - N tok]`` where N is chars // 4 (matching the
        ``count_tokens`` estimator + ``benchmark_core.add_bytes_received``
        which counts chars, not bytes, so live and completion numbers
        align for ASCII=CJK=emoji alike). The indicator gives the
        operator a wall-clock ticker on the in-flight progress.
        """
        s = {
            "running_pids": ["rate-limiter"],
            "rate-limiter_first_chunk_seen": True,
            "rate-limiter_bytes_received": 64,  # 64 // 4 = 16 tok
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, None)
        self.assertIn("[streaming - 16 tok]", block)
        # Bare "[streaming]" must NOT appear after substituting out
        # the enriched indicator -- the cell shows the tok-counter
        # form, not the bare bracket. The cells read
        # The cells here read ``first_chunk_seen`` (bool). The
        # ``first_tok_ts`` (timestamp) field is NOT legacy -- it
        # is still actively read by the live-footer consumer
        # (``_build_live_indicators``), with distinct semantics
        # (timestamp vs bool). ``mark_first_chunk_seen`` is the
        # canonical hook the SSE parse layer fires when the first
        # chunk lands; setting it here triggers the real
        # ``[streaming - N tok]`` cellular branch.
        leftover = block.replace("[streaming - 16 tok]", "")
        self.assertNotIn("[streaming]", leftover,
                         "cell should enrich to [streaming - N tok] when bytes accumulate")

    def test_streaming_plugin_keeps_bare_streaming_before_first_byte(self):
        """If first_tok_ts is set but bytes_received is 0 (just
        landed first-tok event but no delta has accumulated yet),
        keep the legacy ``[streaming]`` form rather than showing
        ``[streaming - 0 tok]`` (which would be visually noisy and
        duplicate the rare ``ft > 0 but no bytes`` edge case).
        """
        s = {
            "running_pids": ["rate-limiter"],
            "rate-limiter_first_tok_ts": 1234.5,
            "rate-limiter_bytes_received": 0,
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, None)
        self.assertIn("[streaming]", block)
        self.assertNotIn("tok", block,
                         "no bytes yet -> don't show 0-tok indicator")

    def test_in_flight_streaming_plugin_pre_first_chunk_shows_elapsed_seconds_after_threshold(self):
        """Streaming-capable plugin in flight BEFORE any first chunk
        has landed: once the wait crosses the 2s threshold,
        the cell switches from ``[streaming]`` to ``[streaming - Ns]``
        showing the seconds elapsed since dispatch.

        Pre-chunk display is elapsed seconds, NOT an estimated
        token count, because predicting tokens from seconds is
        misleading at this stage (real throughput varies wildly
        between providers / temperatures / prompt sizes). The
        ``_elapsed_suffix`` helper is shared with the
        ``[requested - Ns]`` non-streaming branch so the two
        pre-chunk indicators stay in sync.

        Below the threshold the bare ``[streaming]`` form is kept
        (no visual noise on quick plugins).
        """
        now = time.monotonic()
        # Set attempt_start 6s ago so elapsed clears the 2s threshold.
        s = {
            "running_pids": ["rate-limiter"],
            "rate-limiter_first_chunk_seen": False,
            "rate-limiter_bytes_received": 0,
            "attempt_start": now - 6.0,
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, None)
        elapsed = int(time.monotonic() - s["attempt_start"])
        self.assertIn(f"[streaming - {elapsed}s]", block,
                      "pre-chunk cell past threshold should show "
                      "[streaming - Ns] with elapsed seconds since dispatch")
        # The pre-chunk-with-bytes form is gone -- we use wall-clock
        # seconds, not an estimated tok count. The ``~`` tilde
        # glyph was the transparent estimate cue; it must never
        # appear in the streaming-pre-chunk branch.
        self.assertNotIn("~", block,
                         "tilde (~) is reserved for the estimate-guess "
                         "form which was removed in favour of seconds")
        self.assertNotIn("est.", block,
                         "'est.' prefix was removed from the streaming "
                         "pre-chunk branch in favour of wall-clock seconds")
        self.assertNotIn("tok", block,
                         "pre-chunk state (no bytes yet) must not show any "
                         "tok indicator -- the seconds-suffix form is the "
                         "operator's only signal until the real counter starts")

    def test_in_flight_streaming_plugin_post_first_chunk_does_not_use_estimate_marker(self):
        """Streaming-capable plugin in flight WITH first chunk seen
        AND positive byte count: the cell shows the real counter
        ``[streaming - N tok]`` -- NO ``~`` glyph, NO ``est.`` prefix.
        Even when bytes_received is positive after the first chunk,
        the operator reads the value as actually-received data.

        This test pins the contract that the operator NEVER sees a
        ``~`` decoration on the real counter -- after the tilde /
        est-tok ticker was removed in favour of wall-clock seconds,
        the post-chunk branch is bare by design (no estimate form
        to fall back to).
        """
        # No ``attempt_start`` needed: the post-chunk branch is
        # pure-attribute (depends on ``first_chunk_seen`` +
        # ``bytes_received``) and doesn't read``attempt_start``
        # once we know the first chunk has landed. Including the
        # past-threshold value would just confuse future readers
        # about which field drives this test.
        s = {
            "running_pids": ["rate-limiter"],
            "rate-limiter_first_chunk_seen": True,
            "rate-limiter_bytes_received": 64,    # 64 // 4 = 16 tok
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, None)
        self.assertIn("[streaming - 16 tok]", block)
        # Estimate decorations are now forbidden EVERYWHERE -- a
        # ``~`` would falsely imply "guess", and we no longer have
        # any estimate path to cue.
        self.assertNotIn("~", block,
                         "tilde (~) is reserved for the [removed] "
                         "estimate form; the seconds-suffix path uses "
                         "no decoration")
        self.assertNotIn("est.", block,
                         "'est.' prefix is reserved for the [removed] "
                         "estimate form; the seconds-suffix path uses "
                         "no decoration")

    def test_in_flight_streaming_plugin_shows_elapsed_seconds_form_above_threshold(self):
        """Streaming-capable plugin in flight with no first chunk
        yet: once the wait exceeds the 2s threshold
        (``_ELAPSED_THRESHOLD_S``), the bracket transitions from
        the bare ``[streaming]`` form to ``[streaming - Ns]``
        (re-using the same ``_elapsed_suffix`` helper that drives
        ``[requested - Ns]`` for non-streaming plugins). This is
        elapsed seconds since *this plugin's* dispatch -- NOT an
        estimated token count, since predicting tokens from seconds is
        misleading (actual throughput varies wildly between providers /
        temperatures / prompt sizes).

        Below the 2s threshold the bare ``[streaming]`` form is kept
        (no visual noise on quick plugins). A missing
        ``{pid}_start_ts`` falls back to the legacy model-level
        ``attempt_start``; a missing/zero start keeps the bare form
        -- we don't fabricate a meaningless elapsed value from epoch 0.
        """
        now = time.monotonic()
        # Above threshold (5s ago): expect the seconds-suffix form.
        s_above = {
            "running_pids": ["rate-limiter"],
            "rate-limiter_start_ts": now - 5.0,
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s_above, self.p_streaming, None)
        elapsed = int(time.monotonic() - s_above["rate-limiter_start_ts"])
        self.assertIn(f"[streaming - {elapsed}s]", block,
                      "above-threshold pre-chunk cell should show "
                      "[streaming - Ns] with elapsed seconds since dispatch")
        # The estimate-decoration forms are gone entirely -- the
        # operator sees wall-clock seconds, not a tok guess.
        self.assertNotIn("~", block)
        self.assertNotIn("est.", block)
        self.assertNotIn("tok", block,
                         "above-threshold pre-chunk state must show seconds, "
                         "not tokens -- tok would require a real chunk to have arrived")
        # Below threshold (1s ago): bare bracket, no suffix.
        s_below = {
            "running_pids": ["rate-limiter"],
            "attempt_start": now - 1.0,
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s_below, self.p_streaming, None)
        self.assertIn("[streaming]", block)
        self.assertNotIn("[streaming -", block,
                         "below-threshold streaming should not carry any suffix")
        # Missing attempt_start: bare bracket (no fabricated elapsed).
        s_none = {
            "running_pids": ["rate-limiter"],
            # no attempt_start key -- simulates a model that hasn't been dispatched yet
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s_none, self.p_streaming, None)
        self.assertIn("[streaming]", block)
        self.assertNotIn("[streaming -", block,
                         "missing attempt_start should keep bare bracket (no fabricated elapsed)")

    def test_in_flight_non_streaming_plugin_shows_elapsed_suffix_above_threshold(self):
        """Non-streaming-capable plugin in flight: once elapsed
        exceeds 2s, the bracket becomes ``[requested - Ns]`` so the
        operator can spot hung non-streaming requests. Below the
        threshold the bare ``[requested]`` form is kept.
        """
        now = time.monotonic()
        # Above threshold (5s ago): expect elapsed suffix.
        s_above = {
            "running_pids": ["counter"],
            "attempt_start": now - 5.0,
        }
        block = ai_benchmark._plugin_cell_block(
            "counter", s_above, self.p_nonstream, None)
        self.assertIn("[requested - 5s]", block)
        # Below threshold (1s ago): bare bracket.
        s_below = {
            "running_pids": ["counter"],
            "attempt_start": now - 1.0,
        }
        block = ai_benchmark._plugin_cell_block(
            "counter", s_below, self.p_nonstream, None)
        self.assertIn("[requested]", block)
        self.assertNotIn("[requested -", block,
                         "below-threshold non-streaming should not carry an elapsed suffix")


    def test_in_flight_streaming_plugin_thinking_only_shows_think_tok(self):
        """Thinking-capable plugin in flight with first chunk seen
        AND reasoning_content accumulated but ZERO content bytes:
        cells show the compact ``[thinking - N tok]`` form so the
        operator can distinguish thinking-phase data from "no first
        chunk yet" / pure content-streaming. The ``thinking`` keyword
        (rather than the verbose ``[streaming - N think-tok]``
        suffix) was chosen so the operator sees the keyword as the
        disambiguator, allowing a seamless hand-off to the
        content-counter ``[streaming - N tok]`` once primary content
        starts flowing -- see
        ``test_thinking_then_content_handoff_to_content_counter``.
        The previous ``assertNotIn("tok]", ...)`` guard no longer
        applies (the new form's suffix is literally ``tok]``); we
        now strip the new bracket and assert that no
        ``[streaming -`` content-counter bracket remains.
        """
        s = {
            "running_pids": ["rate-limiter"],
            "rate-limiter_first_chunk_seen": True,
            "rate-limiter_bytes_received": 0,                       # no content yet
            "rate-limiter_thinking_bytes_received": 200,            # 50 tok
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, None)
        self.assertIn("[thinking - 50 tok]", block,
                      "thinking-only cell should show "
                      "[thinking - N tok] with reasoning chars // 4")
        # Strip the expected bracket; what remains MUST NOT show a
        # *content* counter (``[streaming -``) because
        # ``bytes_received = 0`` by definition in this branch (the
        # 200 reasoning chars do not get counted as content).
        leftover = block.replace("[thinking - 50 tok]", "")
        self.assertNotIn("[streaming -", leftover,
                         "thinking-only cell MUST NOT include a "
                         "content counter (no content bytes yet)")
        # Bare ``[streaming]`` should also be absent so the
        # operator sees the ticking widget rather than the
        # ``no-data`` placeholder.
        self.assertNotIn("[streaming]", leftover,
                         "thinking-only cell should NOT show bare "
                         "[streaming] (reasoning bytes have accumulated)")

    def test_thinking_then_content_handoff_to_content_counter(self):
        """Once both thinking and content are accumulating, the
        content counter takes over (``[streaming - N tok]``) so
        the operator's eye tracks the final-answer token count
        rather than the chain-of-thought length -- exactly as the
        post-completion ``count_tokens(text)`` estimator reports
        for the final answer. The thinking-phase bracket flips
        from ``[thinking - N tok]`` to ``[streaming - N tok]``
        purely on the keyword change so there is no prefix churn
        in the cell.
        """
        s = {
            "running_pids": ["rate-limiter"],
            "rate-limiter_first_chunk_seen": True,
            "rate-limiter_bytes_received": 64,                       # 16 tok
            "rate-limiter_thinking_bytes_received": 200,             # 50 tok
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, None)
        self.assertIn("[streaming - 16 tok]", block,
                      "content present -> cell hands off to [streaming - N tok]")
        # The thinking-only counter must NOT appear when content has
        # also started (otherwise the operator would see TWO ticking
        # widgets and not know which one to read).
        self.assertNotIn("think-tok]", block,
                         "once content arrives the cell shows ONLY "
                         "the content counter")

    def test_in_flight_streaming_plugin_no_first_chunk_no_thinking_keeps_bare(self):
        """If ``mark_first_chunk_seen`` is False AND ``thinking_bytes``
        is 0 (early in the request -- no reasoning delta yet either),
        keep the bare ``[streaming]`` form. The thinking-only
        branch fires only AFTER at least one delta has landed.
        """
        s = {
            "running_pids": ["rate-limiter"],
            "rate-limiter_first_chunk_seen": False,
            "rate-limiter_bytes_received": 0,
            "rate-limiter_thinking_bytes_received": 0,
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, None)
        self.assertIn("[streaming]", block)
        # No ``think-tok`` count when no delta has landed yet
        # (the thinking counter is structurally meaningless at
        # ``first_chunk_seen=False``).
        self.assertNotIn("think-tok", block)


class TestBuildLiveIndicators(unittest.TestCase):
    """Tests for ``_build_live_indicators`` in ai-benchmark.py.

    The helper feeds the per-model row of the live TUI's ``Live:``
    section. With parallel plugin threads (``max_workers > 1``), a
    model can be running several plugins simultaneously; the
    operator needs to see every in-flight plugin's elapsed seconds,
    not a single model-level timer.

    Format requirements:
    * Space-separated ``"[<pid>: <N> tok (<e>s)]"`` entries for
      streaming plugins with bytes accumulated.
    * ``"[<pid>: waiting <e>s]"`` for streaming plugins with no
      first chunk yet.
    * ``"[<pid>: requested <e>s]"`` for non-streaming plugins.
    * Empty running_pids -> empty string.
    * Elapsed seconds are derived from ``{pid}_start_ts`` against
      the monotonic ``now`` parameter so tests are deterministic.

    Example expected output for two streaming + one waiting + one non-streaming:
        ``"[rate-limiter: 16 tok (4s)] [moe-dense: requested 4s] [wireframes: waiting 4s]"``
    """

    @staticmethod
    def _plugin(pid, *, streaming=True):
        return type("P", (), {"id": pid, "supports_streaming": streaming})()

    def test_two_streaming_plugins_and_one_waiting_in_running_pids_order(self):
        """Three streaming-capable plugins in flight: two with
        bytes, one waiting. Each bracket carries its own elapsed
        time, computed from ``{pid}_start_ts`` vs the supplied
        ``now``.
        """
        base_ts = 1000.0
        now = base_ts + 4.0
        s = {
            "running_pids": ["rate-limiter", "moe-dense", "wireframes"],
            "rate-limiter_first_tok_ts": base_ts,
            "rate-limiter_bytes_received": 64,     # 16 tok
            "rate-limiter_start_ts": base_ts,
            "moe-dense_start_ts": base_ts,
            "wireframes_first_tok_ts": base_ts,
            "wireframes_bytes_received": 128,    # 32 tok
            "wireframes_start_ts": base_ts,
        }
        plugins = [
            self._plugin("rate-limiter", streaming=True),
            self._plugin("moe-dense", streaming=True),
            self._plugin("wireframes", streaming=True),
        ]
        out = ai_benchmark._build_live_indicators(s, plugins, now=now)
        self.assertEqual(
            out,
            "[rate-limiter: 16 tok (4s)] [moe-dense: waiting 4s] [wireframes: 32 tok (4s)]",
        )

    def test_user_example_output_two_streaming_six_waiting(self):
        """Reproduces the user's conceptual example with per-plugin
        elapsed timing: two streaming plugins showing tok counts,
        followed by six other plugins each showing ``waiting`` with
        their own elapsed time.
        """
        base_ts = 1000.0
        now = base_ts + 6.0
        waiting_pids = [f"plugin-waiting-{i}" for i in range(6)]
        s = {
            "running_pids": [
                "rate-limiter",
                "software-architecture",
                *waiting_pids,
            ],
            "rate-limiter_first_tok_ts": base_ts,
            "rate-limiter_bytes_received": 64,            # 16 tok
            "rate-limiter_start_ts": base_ts,
            "software-architecture_first_tok_ts": base_ts,
            "software-architecture_bytes_received": 608,  # 152 tok
            "software-architecture_start_ts": base_ts,
        }
        for pid in waiting_pids:
            s[f"{pid}_start_ts"] = base_ts
        plugins = [self._plugin(p, streaming=True) for p in s["running_pids"]]
        out = ai_benchmark._build_live_indicators(s, plugins, now=now)
        expected = (
            "[rate-limiter: 16 tok (6s)] "
            "[software-architecture: 152 tok (6s)] "
            + " ".join(f"[plugin-waiting-{i}: waiting 6s]" for i in range(6))
        )
        self.assertEqual(out, expected)

    def test_non_streaming_plugin_in_flight_is_included(self):
        """Non-streaming-capable plugins now appear in the live
        footer with ``requested <e>s`` because their elapsed wait
        time is observable and useful, even though the transport
        does not stream.
        """
        base_ts = 1000.0
        now = base_ts + 4.0
        s = {
            "running_pids": ["rate-limiter", "structured-output"],
            "rate-limiter_first_tok_ts": base_ts,
            "rate-limiter_bytes_received": 64,  # 16 tok
            "rate-limiter_start_ts": base_ts,
            "structured-output_start_ts": base_ts,
        }
        plugins = [
            self._plugin("rate-limiter", streaming=True),
            self._plugin("structured-output", streaming=False),
        ]
        out = ai_benchmark._build_live_indicators(s, plugins, now=now)
        self.assertEqual(
            out,
            "[rate-limiter: 16 tok (4s)] [structured-output: requested 4s]",
        )

    def test_plugin_not_in_running_pids_is_excluded(self):
        """A plugin that completed (not in running_pids) and a
        plugin that never ran (also not in running_pids) are both
        excluded. The output is determined solely by running_pids.
        """
        base_ts = 1000.0
        now = base_ts + 2.0
        s = {
            "running_pids": ["rate-limiter"],
            "rate-limiter_start_ts": base_ts,
            # These are defined but plugins NOT in running_pids:
            "completed_plugin_first_tok_ts": 5.0,
            "completed_plugin_bytes_received": 200,
            "wireframes_first_tok_ts": 3.0,
            "wireframes_bytes_received": 100,
        }
        plugins = [
            self._plugin("rate-limiter", streaming=True),
            self._plugin("wireframes", streaming=True),
            self._plugin("completed_plugin", streaming=True),
        ]
        out = ai_benchmark._build_live_indicators(s, plugins, now=now)
        self.assertEqual(out, "[rate-limiter: waiting 2s]")

    def test_empty_running_pids_returns_empty_string(self):
        """No in-flight plugins -> empty string. The caller
        (``_render_live_activity``) skips the prefix when the result
        is empty.
        """
        s = {"running_pids": []}
        plugins = [self._plugin("any"), self._plugin("other")]
        self.assertEqual(ai_benchmark._build_live_indicators(s, plugins, now=1000.0), "")

    def test_only_waiting_returns_per_plugin_waiting(self):
        """All in-flight streaming-capable plugins have no first
        token + no bytes yet -> output is one ``waiting <e>s`` entry
        per plugin (no aggregate bucket).
        """
        base_ts = 1000.0
        now = base_ts + 3.0
        s = {
            "running_pids": ["rate-limiter", "wireframes", "moe-dense"],
        }
        for pid in s["running_pids"]:
            s[f"{pid}_start_ts"] = base_ts
        plugins = [self._plugin(p, streaming=True) for p in s["running_pids"]]
        out = ai_benchmark._build_live_indicators(s, plugins, now=now)
        self.assertEqual(
            out,
            "[rate-limiter: waiting 3s] [wireframes: waiting 3s] [moe-dense: waiting 3s]",
        )

    def test_streaming_first_chunk_but_no_bytes_shows_waiting(self):
        """Rare transient: first_tok_ts is set but bytes_received
        is still 0 (just landed first-tok, no delta has accumulated
        yet). The footer shows ``waiting <e>s`` -- emitting
        ``[name: 0 tok]`` would be visually noisy and the per-cell
        [streaming] already conveys "just started streaming".
        """
        base_ts = 1000.0
        now = base_ts + 2.0
        s = {
            "running_pids": ["rate-limiter", "wireframes"],
            "rate-limiter_first_tok_ts": base_ts,
            "rate-limiter_bytes_received": 64,  # 16 tok
            "rate-limiter_start_ts": base_ts,
            "wireframes_first_tok_ts": base_ts,
            "wireframes_start_ts": base_ts,
        }
        plugins = [
            self._plugin("rate-limiter", streaming=True),
            self._plugin("wireframes", streaming=True),
        ]
        out = ai_benchmark._build_live_indicators(s, plugins, now=now)
        self.assertEqual(out, "[rate-limiter: 16 tok (2s)] [wireframes: waiting 2s]")

    def test_thinking_only_live_indicator_with_elapsed(self):
        """Thinking-capable plugin in flight with first chunk seen
        AND ``reasoning_content`` accumulated but ZERO content bytes:
        the live footer shows ``[<pid>: thinking N tok (e s)]`` so
        the operator can tell data IS arriving on a deepseek-r1 /
        Qwen3 / o1-style run BEFORE primary content starts flowing.
        The compact ``thinking`` keyword (vs. the verbose
        ``[<pid>: N think-tok (e s)]``) keeps the bracket short
        AND uses the keyword as the disambiguator so a seamless
        hand-off to ``[<pid>: N tok (e s)]`` happens without any
        prefix churn once primary content starts. Falls through to
        ``[<pid>: waiting <e s>]`` when neither counter is
        positive and to ``[<pid>: requested <e s>]`` for
        non-streaming plugins.
        """
        base_ts = 1000.0
        now = base_ts + 7.0
        s = {
            "running_pids": ["rate-limiter"],
            "rate-limiter_first_tok_ts": base_ts,
            "rate-limiter_bytes_received": 0,                  # no content yet
            "rate-limiter_thinking_bytes_received": 196,       # 49 tok
            "rate-limiter_start_ts": base_ts,
        }
        plugins = [self._plugin("rate-limiter", streaming=True)]
        out = ai_benchmark._build_live_indicators(s, plugins, now=now)
        self.assertEqual(out, "[rate-limiter: thinking 49 tok (7s)]")

    def test_thinking_then_content_live_handoff_to_tok_counter(self):
        """Once both thinking AND content are accumulating, the live
        footer hands off to the content counter form (mirrors the
        cell-renderer handoff in
        ``test_thinking_then_content_handoff_to_content_counter``).
        """
        base_ts = 1000.0
        now = base_ts + 5.0
        s = {
            "running_pids": ["rate-limiter"],
            "rate-limiter_first_tok_ts": base_ts,
            "rate-limiter_bytes_received": 64,                # 16 tok
            "rate-limiter_thinking_bytes_received": 196,       # 49 tok
            "rate-limiter_start_ts": base_ts,
        }
        plugins = [self._plugin("rate-limiter", streaming=True)]
        out = ai_benchmark._build_live_indicators(s, plugins, now=now)
        self.assertEqual(out, "[rate-limiter: 16 tok (5s)]")

    def test_no_first_chunk_no_thinking_falls_back_to_waiting(self):
        """Sanity check: if no first chunk has landed AND thinking
        bytes is 0 (a deepseek-r1 run BEFORE the first reasoning
        delta has arrived), the live footer still shows the legacy
        ``[<pid>: waiting N s]`` form -- we do NOT fake a
        ``think-tok`` count from epoch 0.
        """
        base_ts = 1000.0
        now = base_ts + 4.0
        s = {
            "running_pids": ["rate-limiter"],
            "rate-limiter_start_ts": base_ts,
        }
        plugins = [self._plugin("rate-limiter", streaming=True)]
        out = ai_benchmark._build_live_indicators(s, plugins, now=now)
        self.assertEqual(out, "[rate-limiter: waiting 4s]")


if __name__ == "__main__":
    unittest.main()
