#!/usr/bin/env python3
"""
AI Benchmark — Plugin-based benchmark for code generation and reasoning.
Supports arbitrary task plugins, versioned results, and plugin selection.

Configuration: edit benchmark-config.json (or pass --config <path>).
API keys can use ${VAR} or ${VAR:default} syntax for env-var expansion.
"""
import argparse
import contextlib
import curses
import glob
import json
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
import unicodedata
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone

from benchmark.completions import generate_shell_completion
from benchmark.core import (
    BenchmarkState,
    PreloadResult,
    _apply_http_retry_default,
    _save_outputs,
    _unique_source_abbrevs,
    confidence_weighted_consensus,
    dump_default_config,
    generate_config_from_api,
    get_target_plugins_blacklist,
    judge_response,
    load_config,
    parse_plugin_temperatures,
    preload_model,
    resolve_model_thread_limit,
    resolve_preload_timeout,
    resolve_targets,
    run_model,
    save_judge_response,
)
from benchmark.http import (
    close_active_requests,
    get_429_stats,
    get_active_request_count,
    reset_429_stats,
)
from benchmark.opencode import (
    generate_config as generate_opencode_config,
)
from benchmark.opencode import (
    opencode_model_name,
    opencode_version,
    resolve_opencode_binary,
)
from benchmark.plugin import SCORE_SCHEMA
from benchmark.state import apply_state_recovery, prepare_state_recovery
from plugins import discover_plugins, format_plugin_list

_CORRUPTED_STATE_ABORT = "abort"

DEFAULT_CONFIG_PATH = "benchmark-config.json"


def _clear_restart_artifacts(state_file, output_dir):
    """Remove state/report/log artifacts for an explicit fresh restart."""
    if os.path.exists(state_file):
        os.remove(state_file)
    for path in glob.glob(os.path.join(output_dir, "results.*")):
        try:
            os.remove(path)
        except OSError:
            pass
    logs_dir = os.path.join(output_dir, "logs")
    if os.path.isdir(logs_dir):
        for path in glob.glob(os.path.join(logs_dir, "*.log")):
            try:
                os.remove(path)
            except OSError:
                pass


def _resolve_config_path(config_path):
    """Resolve a config path, falling back to .yaml/.yml when using the default.

    If ``config_path`` is the default and does not exist, try
    ``benchmark-config.yaml`` then ``benchmark-config.yml``. Returns the
    first existing path, or ``None`` if no fallback exists.
    """
    if os.path.exists(config_path):
        return config_path
    if os.path.basename(config_path) == "benchmark-config.json":
        base, _ = os.path.splitext(config_path)
        for ext in (".yaml", ".yml"):
            fallback = base + ext
            if os.path.exists(fallback):
                return fallback
    return None


def _merge_saved_targets(targets, state_models, saved_state, runner_mode):
    """Keep the current configuration authoritative for runnable targets.

    Saved ``model_info`` entries for models removed from the current config are
    deliberately not merged into ``targets`` or ``state_models``. This prevents
    resume from scheduling obsolete models. ``BenchmarkState.load_state`` still
    retains the saved ``results`` list, so historical rows remain available to
    reports without becoming executable work.

    The parameters are retained for a small amount of call-site/test
    compatibility; the saved state and runner mode are intentionally ignored.
    """
    del targets, state_models, saved_state, runner_mode
    return []


def _write_run_info(output_dir, run_info):
    """Persist run metadata to ``run-info.json`` in the output directory."""
    path = os.path.join(output_dir, "run-info.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2, default=str)
    except Exception as e:  # noqa: BLE001 - a report-write failure must not abort the run
        print(f"⚠️  Could not write run-info.json: {e}", file=sys.stderr)


def _scan_judge_sidecars(judge_input_dir):
    """Yield retained judge-input sidecars, ignoring incomplete files."""
    if not judge_input_dir or not os.path.isdir(judge_input_dir):
        return []
    jobs = []
    for root, _dirs, files in os.walk(judge_input_dir):
        for filename in files:
            if not filename.endswith(".json") or filename.endswith(".tmp"):
                continue
            path = os.path.join(root, filename)
            try:
                with open(path, encoding="utf-8") as handle:
                    item = json.load(handle)
                if all(key in item for key in ("target", "runner", "plugin", "prompt", "response")):
                    jobs.append((path, item))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
    return jobs


def _eligible_judge_sidecars(judge_input_dir, targets, state, active_plugin_ids,
                             judge_models):
    """Return retained sidecars for individually judgeable plugin results.

    Eligibility is deliberately per ``(target, runner, plugin)`` rather than
    per model. A model can have sibling plugins that failed, still be pending
    or running on resume, and still have a numeric score for this plugin that
    is valid judge input. Existing votes are checked per judge so a resumed
    run queues only the missing judge/model-plugin combinations.
    """
    latest = {
        (result.get("state_key", result.get("model")), result.get("runner", "http")): result
        for result in state.latest_results()
    }
    snapshot = state.snapshot()
    eligible = []
    for sidecar, item in _scan_judge_sidecars(judge_input_dir):
        target = item.get("target")
        runner = item.get("runner")
        plugin_id = item.get("plugin")
        state_key = item.get("state_key", target)
        if runner not in {"http", "opencode"} or plugin_id not in active_plugin_ids:
            continue
        if runner == "http":
            target_name = state_key
            expected_state_key = target_name
        else:
            suffix = " [opencode]"
            if not state_key.endswith(suffix):
                continue
            target_name = state_key[:-len(suffix)]
            expected_state_key = state_key
        if target_name not in targets or target != target_name:
            continue

        result = latest.get((expected_state_key, runner), {})
        info = snapshot.get(expected_state_key, {})
        score = result.get(f"{plugin_id}_score")
        if not (isinstance(score, (int, float)) and not isinstance(score, bool)):
            score = info.get(f"{plugin_id}_score")
        if not (isinstance(score, (int, float)) and not isinstance(score, bool)):
            continue

        result_votes = result.get(f"{plugin_id}_judge_votes", []) or []
        info_votes = info.get(f"{plugin_id}_judge_votes", []) or []
        votes = [*result_votes, *info_votes]
        judged_models = {
            vote.get("model") for vote in votes if isinstance(vote, dict)
        }
        # Do not trust a stale aggregate completion flag by itself: the
        # configured judge set is authoritative, and a missing judge remains
        # eligible even when an older result incorrectly said complete.
        if set(judge_models).issubset(judged_models):
            continue
        eligible.append((sidecar, item))
    return eligible


def _inject_429_stats(run_info):
    """Add the current 429 backoff statistics to a run-info dict."""
    backoff_429 = get_429_stats()
    run_info["backoff_429"] = {
        "total_retries": backoff_429.get("total_retries", 0),
        "per_plugin": {
            pid: {
                "retries": stats["retries"],
                "total_sleep_time": stats["total_sleep_time"],
            }
            for pid, stats in backoff_429.get("plugin_stats", {}).items()
        },
    }
    return run_info


def _char_display_width(char):
    """Return a conservative terminal-column width for one character.

    Terminals do not agree on the width of Unicode characters with East
    Asian Width ``A`` (ambiguous), and many emoji-capable terminals render
    symbols such as ``⚠`` as two columns even though Unicode classifies them
    as neutral. Under-counting one of those characters lets ``curses`` write
    past the right edge, where the terminal may wrap it onto the next row.
    That wrap is the source of the apparent prepended/stale characters.

    Over-counting ambiguous symbols is intentional: clipping a row one
    column early is harmless; allowing a row to wrap corrupts every row
    below it. ASCII remains one column wide.
    """
    if char in "\r\n" or unicodedata.combining(char):
        return 0
    if unicodedata.category(char) in {"Cc", "Cf"}:
        return 0
    east_asian_width = unicodedata.east_asian_width(char)
    if east_asian_width in ("W", "F"):
        return 2
    # Emoji/symbol code points in these ranges are commonly rendered in an
    # emoji presentation with two columns. Do not classify every ambiguous
    # character as wide: arrows and box-drawing glyphs are normally one
    # column in the Linux terminals this TUI targets, and over-counting them
    # would make narrow layouts needlessly lose useful text.
    codepoint = ord(char)
    if (0x2600 <= codepoint <= 0x27BF
            or 0x1F000 <= codepoint <= 0x1FAFF):
        return 2
    return 1


def _is_grapheme_extension(char):
    """Return whether ``char`` extends the preceding display cluster.

    This is a deliberately small, dependency-free approximation of Unicode
    grapheme breaking. It covers the sequences most likely to occur in status
    text: combining marks, variation selectors, emoji modifiers, and format
    characters such as the zero-width joiner.
    """
    codepoint = ord(char)
    category = unicodedata.category(char)
    return (
        unicodedata.combining(char) != 0
        or category in {"Mn", "Me", "Cf"}
        or 0x1F3FB <= codepoint <= 0x1F3FF  # emoji skin-tone modifiers
    )


def _grapheme_clusters(text):
    """Yield approximate terminal grapheme clusters from ``text``.

    Newlines terminate the row and C0/C1 control characters are omitted so
    model-provided text cannot move the cursor or inject terminal controls.
    A code point following a ZWJ remains in the same cluster, as do paired
    regional indicators used for flag emoji.
    """
    cluster = []
    regional_indicator = False
    for char in text:
        if char in "\r\n":
            break
        if unicodedata.category(char) == "Cc":
            continue
        if not cluster:
            cluster = [char]
            regional_indicator = 0x1F1E6 <= ord(char) <= 0x1F1FF
            continue
        previous = cluster[-1]
        is_regional = 0x1F1E6 <= ord(char) <= 0x1F1FF
        if (
            _is_grapheme_extension(char)
            or previous == "\u200d"
            or (regional_indicator and is_regional)
        ):
            cluster.append(char)
            if regional_indicator and is_regional:
                regional_indicator = False
        else:
            yield "".join(cluster)
            cluster = [char]
            regional_indicator = is_regional
    if cluster:
        yield "".join(cluster)


def _cluster_display_width(cluster):
    """Return a conservative terminal width for one grapheme cluster."""
    if "\u200d" in cluster or (
        sum(0x1F1E6 <= ord(char) <= 0x1F1FF for char in cluster) == 2
    ):
        # Joined emoji and flag pairs are rendered as one pictograph by
        # terminals even though they contain several Unicode code points.
        return 2
    return max((_char_display_width(char) for char in cluster), default=0)


def _display_width(text):
    """Return the terminal-column width of ``text`` without extra deps.

    Width is calculated per approximate grapheme cluster rather than per
    Python code point. This keeps joined emoji, flags, skin-tone modifiers,
    and combining marks together when a row is clipped.
    """
    return sum(_cluster_display_width(cluster) for cluster in _grapheme_clusters(text))


def _truncate_display_width(text, max_width):
    """Return a sanitized prefix no wider than ``max_width`` columns."""
    if max_width <= 0:
        return ""
    result = []
    width = 0
    for cluster in _grapheme_clusters(text):
        cluster_width = _cluster_display_width(cluster)
        if width + cluster_width > max_width:
            break
        result.append(cluster)
        width += cluster_width
    return "".join(result)


def _slice_display_width(text, start, max_width):
    """Return a grapheme-safe horizontal slice measured in display columns.

    ``scroll_x`` is a column offset, not a Python code-point offset. Slicing
    the raw string can split a ZWJ/combining cluster and can also disagree
    with the terminal when wide characters occur before the viewport.
    """
    start = max(start, 0)
    if max_width <= 0:
        return ""
    result = []
    skipped = 0
    width = 0
    for cluster in _grapheme_clusters(text):
        cluster_width = _cluster_display_width(cluster)
        if skipped + cluster_width <= start:
            skipped += cluster_width
            continue
        # A viewport boundary inside a wide cluster advances to the next
        # complete cluster rather than emitting a dangling grapheme.
        if skipped < start:
            skipped += cluster_width
            continue
        if width + cluster_width > max_width:
            break
        result.append(cluster)
        width += cluster_width
    return "".join(result)


def _tui_dimensions_changed(stdscr, dimensions, previous_dimensions):
    """Erase the virtual screen once after a terminal resize.

    A terminal can physically reflow old lines when it becomes narrower,
    while curses still has the old virtual contents. Clearing on the first
    frame at the new dimensions re-synchronizes both representations before
    the normal per-row redraw starts.
    """
    if dimensions == previous_dimensions:
        return previous_dimensions
    try:
        stdscr.erase()
    except curses.error:
        # The next frame will retry after curses finishes processing SIGWINCH.
        return previous_dimensions
    return dimensions


def _wr(stdscr, max_x, max_y, y, x, text, attr=0):
    """Clear and safely write one bounded terminal row.

    Keep one column unused at the right edge. Writing exactly through the
    lower/right boundary can trigger curses' automatic wrap (or a
    ``curses.error`` after a partial write), which leaves the virtual and
    physical screens out of sync and produces leading characters on the next
    frame. Do not retry a failed write at the current cursor position: a
    failed boundary write may already have advanced that cursor.
    """
    if not (0 <= y < max_y and 0 <= x < max_x):
        return
    safe_text = _truncate_display_width(text, max_x - x - 1)
    try:
        stdscr.move(y, x)
        stdscr.clrtoeol()
        if safe_text:
            stdscr.addstr(y, x, safe_text, attr)
    except curses.error:
        # Window too small or resized since getmaxyx(); skip this frame.
        pass


def _active_source_target_counts(snap):
    """Count active target pipelines once per source, across runner states."""
    active = {}
    seen = set()
    for name, info in snap.items():
        if not (info.get("preloading") or info.get("running_pids")):
            continue
        target_name = name.removesuffix(" [opencode]")
        key = (info.get("source", "?"), target_name)
        if key in seen:
            continue
        seen.add(key)
        source = key[0]
        active[source] = active.get(source, 0) + 1
    return active


def _fallback_tui_loop(state, stop_event, session_seed=None, model_thread_limits=None):
    """Fallback terminal UI when curses is unavailable."""
    while not stop_event.is_set():
        snap = state.snapshot()
        active = sum(
            1 for s in snap.values()
            if s.get("preloading") or s.get("running_pids") or s["status"] == "queued"
        )
        done = state.completed
        total = state.total
        seed_info = f"Seed: {session_seed}  |  " if session_seed is not None else ""
        http_threads = get_active_request_count()
        backoff_429 = get_429_stats()
        sleeping_model_count = len({
            tuple(key.rsplit("|", 1)[0].split("|", 1))
            for key in (backoff_429.get("sleeping") or {})
        })
        source_active = _active_source_target_counts(snap)
        slots = ""
        if model_thread_limits:
            slots = "  |  " + ", ".join(
                f"{source}: models {source_active.get(source, 0)}/{limit}"
                for source, limit in model_thread_limits.items()
            )
        parts = [
            (f"{seed_info}🔄 {active} active  |  ✅ {done}/{total} completed"
            f"  |  HTTP: {http_threads}  |  429⏸ {sleeping_model_count}"
            f"{slots}")
        ]
        for name, s in snap.items():
            if s.get("preloading"):
                elapsed = (time.monotonic() - s.get("preload_start_ts", 0)) if s.get("preload_start_ts") else 0
                parts.append(f"  🔄 Preloading {name[:30]} {elapsed:.0f}s")
            elif s.get("running_pids"):
                elapsed = (time.monotonic() - s.get("attempt_start", 0)) if s.get("attempt_start") else 0
                err = s.get("last_error", "")
                msg = f"  {name[:30]} {elapsed:.0f}s"
                if err:
                    msg += f"  {err}"
                parts.append(msg)
        sys.stdout.write(f"\r{' ' * 80}\r")
        sys.stdout.write(" | ".join(parts))
        sys.stdout.flush()
        # Sleep in short increments so Ctrl+C is handled promptly.
        stop_event.wait(0.2)
    print()


def _handle_tui_input(stdscr, scroll_y, scroll_x, max_row_offset, visible_rows, max_x, frozen_width, plugin_hdr_len):
    """Handle keyboard navigation and return updated scroll offsets."""
    try:
        key = stdscr.getch()
    except Exception:  # noqa: BLE001 - getch() can raise on resize or stdin interruption
        # getch() can raise on terminal resize or when stdin is interrupted.
        key = -1
    if key == curses.KEY_UP:
        scroll_y = max(0, scroll_y - 1)
    elif key == curses.KEY_DOWN:
        scroll_y = min(max_row_offset, scroll_y + 1)
    elif key == curses.KEY_PPAGE:
        scroll_y = max(0, scroll_y - visible_rows)
    elif key == ord(' ') or key == curses.KEY_NPAGE:
        scroll_y = min(max_row_offset, scroll_y + visible_rows)
    elif key == curses.KEY_HOME:
        scroll_y = 0
    elif key == curses.KEY_END:
        scroll_y = max_row_offset
    elif key == curses.KEY_LEFT:
        scroll_x = max(0, scroll_x - 8)
    elif key == curses.KEY_RIGHT:
        visible_width = max(0, max_x - frozen_width - 1)
        scroll_x = min(
            max(0, plugin_hdr_len - visible_width),
            scroll_x + 8,
        )
    scroll_y = max(0, min(max_row_offset, scroll_y))
    return scroll_y, scroll_x


def _render_header_and_summary(stdscr, max_x, max_y, snap, done, total, running, queued, pending,
                                scroll_y, visible_rows, total_models, session_seed,
                                http_threads, sleeping_model_count,
                                model_thread_limits=None):
    """Render the top header and summary statistics.

    ``http_threads`` is the count of in-flight HTTP responses (the wall-clock
    \"parallelism ceiling\" the network is asked to carry).    ``sleeping_model_count`` is the number of unique ``(source, model)`` pairs
    currently paused in a 429 backoff window \u2014 seeing this rise is how an
    operator notices that the benchmark is rate-limited rather than making
    progress.
    """
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).astimezone().strftime('%H:%M:%S')
    seed_info = f"Seed: {session_seed}  |  " if session_seed is not None else ""
    hdr = f"AI Benchmark \u2014 Parallel  |  {seed_info}{ts}"
    # Always draw through _wr, even when the terminal is narrower than the
    # complete message. Skipping the write leaves the previous wider frame
    # physically visible after a narrow/mobile resize.
    _wr(stdscr, max_x, max_y, 0, 0, hdr, curses.A_BOLD)

    failed_count = sum(1 for s in snap.values() if s["status"] == "failed")
    err_indicator = f"  |  \u26a0 {failed_count} failed" if failed_count else ""
    source_active = _active_source_target_counts(snap)
    slot_text = ""
    if model_thread_limits:
        slot_text = "  |  " + ", ".join(
            f"{source}: models {source_active.get(source, 0)}/{limit}"
            for source, limit in model_thread_limits.items()
        )
    summary = (f"Total: {total}  |  "
               f"Done: {done}  |  "
               f"Active: {len(running)}  |  "
               f"Queued: {len(queued + pending)}"
               f"  |  HTTP: {http_threads}"
               f"  |  429\u23f8 {sleeping_model_count}"
               f"{err_indicator}"
               f"  |  \u2191\u2193 rows {scroll_y + 1}-{min(total_models, scroll_y + visible_rows)}/{total_models}"
               f"  |  \u2190\u2192 cols"
               f"{slot_text}")
    if max_y > 1:
        # _wr performs display-column clipping and clears the remainder.
        _wr(stdscr, max_x, max_y, 1, 0, summary)

    if max_y > 2:
        _wr(stdscr, max_x, max_y, 2, 0, "\u2500" * min(max_x, 80))


def _render_table_headings(stdscr, max_x, max_y, scroll_x, frozen_cols, plugin_cols, frozen_width):
    """Render the frozen and plugin column headings."""
    frozen_hdr = " ".join(f"{h:>{w}}" for h, w in frozen_cols)
    plugin_hdr_parts = [f"{h:>{w}}" for h, w in plugin_cols]
    plugin_hdr = " ".join(plugin_hdr_parts)
    if max_y > 3:
        visible_plugin_hdr = _slice_display_width(
            plugin_hdr, scroll_x, max(0, max_x - frozen_width - 1)
        )
        _wr(stdscr, max_x, max_y, 3, 0, frozen_hdr + " " + visible_plugin_hdr, curses.A_UNDERLINE)
    return plugin_hdr


# Display width of the frozen table prefix, including the separator before
# the horizontally scrollable plugin columns. Keep this shared by the row
# formatter and the curses layout calculations.
FROZEN_VIEW_WIDTH = 34

# Width of the per-plugin cell block rendered by ``_plugin_cell_block``.
# The standard 4-cell results layout sums to 5+6+6+6=23 plus 3 single
# spaces between cells = 26 chars -- so a merged bracket status centred
# in this same 26-char span lines up under the existing sub-headers
# (``RateSc RateTok RateTm RateTPS``) without reshaping the
# ``plugin_cols`` table. The previous per-plugin streaming-glyph column
# (``<id>St`` width 5) was deleted as redundant: the merged status block
# already conveys in-flight state, and post-flight the plugin isn't
# streaming anymore, so the glyph was always ``-``.
PLUGIN_BLOCK_WIDTH = 26

# Wall-clock threshold (seconds) past which an in-flight plugin shows a
# secondary indicator so the operator can spot slow / hung requests vs
# normal quick ones. Used by ``_elapsed_suffix`` (the ``- Ns`` bracket
# suffix on both ``[streaming]`` and ``[requested]`` pre-chunk forms).
# Single source of truth so the two consumers don't drift out of sync.
_ELAPSED_THRESHOLD_S = 2


def _fmt_value(v, fmt=".1f"):
    """Format a single cell value; ``None`` renders as ``-``.

    Used by ``_plugin_cell_block`` so a missing result reads as ``-``
    rather than as the literal string ``"None"``.
    """
    if v is None:
        return "-"
    try:
        return f"{v:{fmt}}"
    except (ValueError, TypeError):
        return str(v)


def _elapsed_suffix(start_ts, threshold=_ELAPSED_THRESHOLD_S):
    """Return ``f" - {N}s"`` to append to in-flight brackets when a
    plugin has been waiting long enough to be worth flagging, else
    ``""``.

    Used by ``_plugin_cell_block`` to enrich BOTH the bare
    ``[requested]`` bracket (non-streaming-capable plugin in flight)
    AND the bare ``[streaming]`` bracket (streaming-capable plugin
    in flight but no first chunk yet) once the wait crosses
    ``threshold`` seconds. Default ``threshold`` is
    ``_ELAPSED_THRESHOLD_S`` -- the module-level constant so the
    two consumers (streaming-vs-non-streaming pre-chunk) stay in
    sync. Pre-chunk display is wall-clock seconds elapsed since
    *this plugin's* initial dispatch (NOT an estimated token count,
    which would be misleading at this stage because real throughput
    varies wildly between providers / temperatures / prompt sizes).

    Below the threshold we keep the bare bracket (no visual noise
    on quick plugins); above the threshold we surface ``- Ns`` so
    the operator can tell a stuck/hung plugin from a normal slow
    one. A missing/zero ``start_ts`` means the dispatch hasn't been
    recorded yet -- we return ``""`` in that case rather than
    fabricating a meaningless elapsed value.
    """
    if not start_ts:
        return ""
    elapsed = int(time.monotonic() - start_ts)
    if elapsed > threshold:
        return f" - {elapsed}s"
    return ""


def _judge_models(state):
    """Return the configured judge identities represented by a state row."""
    configured = state.get("judge_models")
    if isinstance(configured, (list, tuple, set)):
        return {model for model in configured if model}
    return set()


def _judge_votes(state, pid):
    """Return configured judge identities that have completed one plugin."""
    configured = _judge_models(state)
    if not configured:
        return set()
    return {
        vote.get("model")
        for vote in (state.get(f"{pid}_judge_votes") or [])
        if isinstance(vote, dict) and vote.get("model") in configured
    }


def _judge_score_marker(pid, state):
    """Return the compact judge status marker for one plugin score.

    A numeric benchmark score is judgeable even before its first judge has
    returned, so configured judging is shown as ``👩‍⚖️0`` rather than being
    mistaken for an unconfigured run. Completion is derived from the current
    configured judge set and recorded votes, not a stale aggregate flag.
    """
    configured = _judge_models(state)
    if not configured:
        return ""
    judged_models = _judge_votes(state, pid)
    if configured.issubset(judged_models):
        return "✅"
    if state.get(f"{pid}_judge_queued") or judged_models:
        return f"👩‍⚖️{len(judged_models)}"
    return ""


def _model_judge_marker(state, active_plugins=None):
    """Return the row-header marker for a model's aggregate judge state."""
    active_plugins = active_plugins or []
    configured = _judge_models(state)
    if not configured:
        return ""
    scored = [
        plugin.id for plugin in active_plugins
        if isinstance(state.get(f"{plugin.id}_score"), (int, float))
        and not isinstance(state.get(f"{plugin.id}_score"), bool)
    ]
    if not scored:
        return ""
    if all(configured.issubset(_judge_votes(state, pid)) for pid in scored):
        return "✅"
    if any(
        state.get(f"{pid}_judge_queued") or _judge_votes(state, pid)
        for pid in scored
    ):
        return "👩‍⚖️"
    return ""


def _plugin_cell_block(pid, s, p, sleeping_lookup=None):
    """Render a single per-model cell block for one plugin.

    The block is always ``PLUGIN_BLOCK_WIDTH`` (32) chars wide so it can
    be dropped into ``plugin_str`` in place of the existing 5-cell
    layout. The standard table keeps the existing 5 sub-headers per
    plugin (``RateSc RateTok RateTm RateTPS RateSt``), so a long plugin
    id like ``json-formatter`` truncates its sub-header prefix to the
    first 3 chars; the 32-char cell block deliberately matches the
    span of those 5 sub-headers so a centred bracket status reads as
    belonging to the plugin under whatever prefix the header shows.

    When the plugin is in flight (``pid in running_pids``) OR
    the model is currently in a 429 backoff sleep, the block collapses
    to a single bracket-delimited status centred in 26 chars:
        ``[streaming - N tok]``     -- streaming-capable in flight, real
                                        counter (first chunk seen; bytes > 0).
                                        ``N`` is chars // 4 so the live
                                        ticker matches the post-completion
                                        ``count_tokens(text)`` estimator.
        ``[streaming - Ns]``       -- streaming-capable in flight, no first
                                        chunk yet, wait crossed the elapsed
                                        threshold (default 2s). Operator
                                        sees wall-clock seconds elapsed
                                        since dispatch; an estimate of the
                                        eventual token count would be
                                        misleading at this stage.
        ``[streaming]``             -- streaming-capable in flight, no first
                                        chunk yet, fresh (<=2s). Bare
                                        bracket to avoid visual noise on
                                        quick plugins.
        ``[requested - Ns]``       -- non-streaming-capable in flight, wait
                                        crossed the elapsed threshold
        ``[requested]``             -- non-streaming-capable in flight, fresh
        ``[429 sleeping Xs]``      -- model is mid-backoff (pauses the
                                        plugin task)

    Wait-for-first-chunk is communicated as wall-clock seconds, NOT
    as an estimated token count. The operator can already see
    elapsed seconds elsewhere in the live footer, but pairing
    seconds with the in-flight bracket answers the most useful
    diagnostic question: "how long has this plugin been waiting
    for the server to respond?" Once a chunk arrives
    (``mark_first_chunk_seen`` called by the SSE parse layer +
    ``bytes_received`` becomes positive), the bracket transitions
    to ``[streaming - N tok]`` with a real counter incrementing on
    every ``on_chunk`` callback.

    The seconds counter is per-plugin, using the
    ``{pid}_start_ts`` timestamp recorded by
    ``BenchmarkState.start_plugin_run``. It therefore resets to
    zero for each plugin dispatch, so the second plugin for a
    model starts counting from its own request time rather than
    inheriting the elapsed time of the first plugin.

    When none of the above applies, the block falls back to the standard
    4-cell results layout (``score tok tm tps``). The previous per-plugin
    streaming-glyph column (``st``) was deleted as redundant (see the
    ``PLUGIN_BLOCK_WIDTH`` block comment).

    ``sleeping_lookup`` is a mapping from ``(source, api_model, pid)``
    to sleep info (``wake_ts``, ``attempts``, ``max_attempts``). It is
    used to show a per-plugin ``[429 sleeping Xs]`` bracket only for the
    plugin that is actually in backoff. The 429 message takes priority
    over the per-plugin transport status because the operator cares more
    about the wall-clock backoff than whether the plugin is streaming.
    """
    in_flight = pid in (s.get("running_pids") or [])
    # Per-plugin dispatch timestamp. Newer state records it via
    # ``start_plugin_run``; older state files may lack it, so fall
    # back to the legacy model-level ``attempt_start`` for graceful
    # backward compat.
    start_ts = s.get(f"{pid}_start_ts") or s.get("attempt_start") or 0
    source = s.get("source")
    api_model = s.get("api_model")
    if in_flight and sleeping_lookup:
        sleep_info = sleeping_lookup.get((source, api_model, pid))
        if sleep_info is not None:
            remaining = max(0, round(sleep_info["wake_ts"] - time.time()))
            text = f"[429 sleeping {remaining}s]"
            remaining = max(0, PLUGIN_BLOCK_WIDTH - _display_width(text))
            return " " * (remaining // 2) + text + " " * (remaining - remaining // 2)
    if in_flight:
        if p.supports_streaming:
            first_chunk_seen = bool(s.get(f"{pid}_first_chunk_seen", False))
            bytes_received = s.get(f"{pid}_bytes_received", 0) or 0
            thinking_bytes = s.get(f"{pid}_thinking_bytes_received", 0) or 0
            if first_chunk_seen and bytes_received:
                # Real counter: first chunk is in AND content bytes
                # have accumulated. chars // 4 matches the
                # ``count_tokens`` estimator in ``benchmark_core``;
                # this is NOT an estimate -- the operator sees the
                # exact tok count the post-completion calculation
                # will report (modulo streaming-vs-final parsing
                # edge cases). No ``~`` marker is needed because
                # this byte-derived counter is genuinely real.
                text = f"[streaming - {bytes_received // 4} tok]"
            elif first_chunk_seen and thinking_bytes:
                # Thinking-phase-only cell: the SSE parse layer has
                # recorded a first chunk (via ``mark_first_chunk_seen``)
                # and accumulated ``reasoning_content`` deltas, but
                # PRIMARY ``content`` is still empty. Render the
                # compact ``[thinking - N tok]`` form so the operator
                # can see data IS arriving on a deepseek-r1 / Qwen3 /
                # o1-style stream rather than confusing it with "no
                # first chunk yet". The ``thinking`` keyword
                # disambiguates from the post-content ``[streaming -
                # N tok]`` cell on the same row -- the operator's
                # eye reads the keyword, not the prefix, so once
                # primary content starts flowing the cell flips to
                # the content-counter form (``[streaming - N tok]``)
                # without any prefix churn. chars // 4 matches the
                # post-completion ``count_tokens`` estimator (and
                # matches the streaming content counter branch
                # above) so the live number is the number the
                # post-completion ``.think.txt`` file shows for
                # length / 4.
                text = f"[thinking - {thinking_bytes // 4} tok]"
            else:
                # Pre-chunk state (no first chunk yet OR first
                # chunk seen with bytes still 0, the rare
                # transient). Show wall-clock seconds elapsed
                # since this plugin's dispatch -- NOT an estimated
                # token count: predicting tokens from seconds is
                # misleading because actual throughput varies wildly
                # between providers / temperatures / prompt sizes.
                # The ``- Ns`` suffix makes per-plugin hangs
                # obvious in the table once the wait crosses the
                # module threshold (``_ELAPSED_THRESHOLD_S``);
                # below the threshold we keep the bare bracket to
                # avoid visual noise on quick responses.
                text = f"[streaming{_elapsed_suffix(start_ts)}]"
        else:
            # Non-streaming-capable plugin in flight. ``[requested]``
            # conveys "we sent the request, awaiting the buffered
            # response" (previously labelled ``[in flight]`` which
            # sounded like an upstream-status verb). The elapsed
            # suffix is added INSIDE the brackets (re-uses the
            # same ``_elapsed_suffix`` helper as the streaming
            # branch above). No token display for non-streaming --
            # the transport doesn't yield data until completion.
            text = f"[requested{_elapsed_suffix(start_ts)}]"
        remaining = max(0, PLUGIN_BLOCK_WIDTH - _display_width(text))
        return " " * (remaining // 2) + text + " " * (remaining - remaining // 2)
    # Standard 4-cell results layout -- widths sum to 5+6+6+6=23 with 3
    # single-space separators between cells = 26 chars, matching the
    # merged status width. The token cell shows the TOTAL (thinking +
    # content) count; the per-kind split is exposed in the CSV/MD/HTML/PDF
    # reports. Falls back to the legacy content-only count for state files
    # that predate the thinking/content split.
    score = s.get(f"{pid}_score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        marker = _judge_score_marker(pid, s)
        sc = f"{score:.0f}{marker}"
    else:
        sc = _fmt_value(score, ".0f")
    total_tokens = s.get(f"{pid}_total_tokens")
    tok = _fmt_value(
        total_tokens if total_tokens is not None else s.get(f"{pid}_output_tokens"),
        "d",
    )
    tm = _fmt_value(s.get(f"{pid}_response_time"))
    tps = _fmt_value(s.get(f"{pid}_tps"))
    score_field = " " * max(0, 5 - _display_width(sc)) + sc
    block = f"{score_field} {tok:>6} {tm:>6} {tps:>6}"
    return _pad_display_width(block, PLUGIN_BLOCK_WIDTH)


def _format_model_row(name, s, display_idx, active_plugins, source_abbrevs,
                      sleeping_lookup=None):
    """Format a single model row into frozen and plugin strings.

    ``sleeping_lookup`` maps ``(source, api_model, pid)`` to sleep info
    (with ``wake_ts``, ``attempts``, ``max_attempts``) so that a plugin
    cell can render its own per-plugin ``[429 sleeping Xs]`` bracket.
    Only the plugin that is actually in backoff is shown as sleeping;
    completed plugins keep their numeric results.
    """
    sv = s["status"]
    status_ch = {"pending": "\u23f3", "queued": "\u23f3",
                 "completed": "\u2705", "failed": "\u274c"}.get(sv, "?")
    if s.get("preloading"):
        status_ch = "\U0001f504"
    elif sv == "running" or s.get("running_pids"):
        status_ch = "\U0001f537"

    src_ab = _source_abbr(source_abbrevs, s.get("source"))
    model_disp = name[:16]
    judge_marker = _model_judge_marker(s, active_plugins)
    frozen = f"{display_idx:>3}{judge_marker}  {src_ab:<3} {model_disp:<18}  {status_ch:<3}"
    # Keep the plugin viewport anchored at the same terminal column as the
    # heading. Emoji status glyphs can occupy two columns, so Python's string
    # length is not sufficient to align this frozen prefix.
    frozen = _pad_display_width(
        _truncate_display_width(frozen, FROZEN_VIEW_WIDTH - 1),
        FROZEN_VIEW_WIDTH - 1,
    )

    # Each plugin contributes exactly one 32-char block (merged status
    # OR standard 5-cell results) so ``plugin_str`` has the same total
    # length and column geometry as before. Joins the per-plugin blocks
    # with single spaces, matching the existing column-join pattern.
    plugin_parts = [
        _plugin_cell_block(p.id, s, p, sleeping_lookup=sleeping_lookup)
        for p in active_plugins
    ]
    plugin_str = " ".join(plugin_parts)
    return frozen, plugin_str


def _render_model_rows(stdscr, max_x, max_y, snap_items, active_plugins, source_abbrevs,
                       scroll_y, scroll_x, visible_rows, frozen_width, model_top,
                       sleeping_lookup):
    """Render the scrollable model status table.

    ``sleeping_lookup`` maps ``(source_name, api_model, pid) -> sleep info``
    (with ``wake_ts``, ``attempts``, ``max_attempts``) so per-plugin 429
    backoff state can be folded into each plugin cell via the
    ``[429 sleeping Xs]`` bracket status. Plugins whose key is absent
    render normally without the indicator.
    """
    total_models = len(snap_items)
    for row_idx in range(visible_rows):
        abs_idx = scroll_y + row_idx
        if abs_idx >= total_models:
            break
        name, s = snap_items[abs_idx]
        display_idx = abs_idx + 1
        frozen, plugin_str = _format_model_row(
            name, s, display_idx, active_plugins, source_abbrevs,
            sleeping_lookup=sleeping_lookup,
        )
        visible_plugin = _slice_display_width(
            plugin_str, scroll_x, max(0, max_x - frozen_width - 1)
        )
        line = frozen + " " + visible_plugin

        attr = 0
        sv = s["status"]
        if sv == "completed":
            try:
                attr = curses.color_pair(1)
            except Exception:  # noqa: BLE001, S110 - color_pair() fails when colors are unavailable
                pass
        elif sv == "failed":
            try:
                attr = curses.color_pair(3)
            except Exception:  # noqa: BLE001, S110 - color_pair() fails when colors are unavailable
                pass
        elif sv == "running" or s.get("running_pids"):
            try:
                attr = curses.color_pair(2)
            except Exception:  # noqa: BLE001, S110 - color_pair() fails when colors are unavailable
                pass
        _wr(stdscr, max_x, max_y, model_top + row_idx, 0, line, attr)

    for r in range(model_top + min(visible_rows, max(0, total_models - scroll_y)), model_top + visible_rows):
        try:
            stdscr.move(r, 0)
            stdscr.clrtoeol()
        except Exception:  # noqa: BLE001, S110 - window may resize between getmaxyx() and paint
            pass


def _pad_display_width(text, target_width):
    """Right-pad ``text`` to a terminal-column width."""
    return text + " " * max(0, target_width - _display_width(text))


def _source_abbr(source_abbrevs, source):
    """Return a short abbreviation for a source, with a safe fallback."""
    if source in source_abbrevs:
        return source_abbrevs[source]
    if source is None:
        return "???"
    return str(source)[:3] or "???"


def _build_sleeping_lookup(backoff_429):
    """Convert the flat ``get_429_stats`` sleeping map into a per-plugin lookup.

    ``backoff_429`` is the dict returned by ``get_429_stats``. Its sleeping
    keys are strings of the form ``"source|model|pid"``. This helper splits
    them into ``(source, api_model, pid)`` tuples so the table renderer can
    look up the exact plugin's backoff state in O(1).
    """
    sleeping_lookup = {}
    for key, info in (backoff_429.get("sleeping") or {}).items():
        src, model, pid = key.split("|", 2)
        sleeping_lookup[(src, model, pid)] = info
    return sleeping_lookup


def _build_live_indicators(s, active_plugins, *, now=None):
    """Build the space-separated live-activity indicator string for a
    model's "Live:" footer row.

    Iterates ``s["running_pids"]`` in insertion order and emits one
    bracket per in-flight plugin. Each bracket includes the elapsed
    seconds since *that plugin's* dispatch (using the monotonic
    ``{pid}_start_ts`` timestamp), so the operator can see per-plugin
    wait time in the same place as the streaming token ticker.

    Format:
      * Streaming plugin with first chunk + bytes:
        ``"[<pid>: <N> tok (<e>s)]"``
      * Streaming plugin waiting for first chunk:
        ``"[<pid>: waiting <e>s]"``
      * Non-streaming plugin:
        ``"[<pid>: requested <e>s]"``

    Non-streaming plugins are now included because the elapsed seconds
    since their request is observable and useful, even though the
    transport itself does not yield streaming chunks.

    With parallel plugin threads (``max_workers > 1``), a model can
    carry several plugins in flight at once; this surfaces ALL of
    them rather than only the streaming-capable ones.

    Returns ``""`` when no plugin is in flight.

    Example output (two streaming + one waiting + one non-streaming):
        ``"[rate-limiter: 16 tok (4s)] [moe-dense: requested 6s] [wireframes: waiting 2s]"``
    """
    if now is None:
        now = time.monotonic()
    running_pids = s.get("running_pids") or []
    parts = []
    for pid in running_pids:
        plugin = next((p for p in active_plugins if p.id == pid), None)
        if plugin is None:
            continue
        start_ts = s.get(f"{pid}_start_ts") or 0
        elapsed = int(now - start_ts) if start_ts else 0
        ft = s.get(f"{pid}_first_tok_ts", 0) or 0
        bytes_received = s.get(f"{pid}_bytes_received", 0) or 0
        thinking_bytes = s.get(f"{pid}_thinking_bytes_received", 0) or 0
        if plugin.supports_streaming and ft and bytes_received:
            parts.append(f"[{pid}: {bytes_received // 4} tok ({elapsed}s)]")
        elif plugin.supports_streaming and ft and thinking_bytes:
            # Thinking-phase-only live indicator. Parallel to the
            # ``[thinking - N tok]`` cell form: a thinking-capable
            # model that has produced reasoning_content but not yet
            # primary content surfaces as ``[<pid>: thinking N tok (e s)]``
            # so the operator can tell data IS arriving on a
            # deepseek-r1 / Qwen3 / o1-style stream. The leading
            # ``thinking`` keyword disambiguates from the post-content
            # ``[<pid>: N tok (e s)]`` entry on the same line, so
            # operators can scan the keyword first and the bracket
            # flips cleanly to the content-counter form once primary
            # content starts flowing. Falls through to
            # ``[<pid>: waiting <e s>]`` once both byte counters and
            # the first-chunk flag are zero (the "actually waiting
            # for the first byte" transient) or to
            # ``[<pid>: requested <e s>]`` for non-streaming plugins.
            parts.append(f"[{pid}: thinking {thinking_bytes // 4} tok ({elapsed}s)]")
        elif plugin.supports_streaming:
            parts.append(f"[{pid}: waiting {elapsed}s]")
        else:
            parts.append(f"[{pid}: requested {elapsed}s]")
    return " ".join(parts)


def _render_live_activity(stdscr, max_x, max_y, snap, source_abbrevs, live_models,
                          live_top, live_height, log_top, active_plugins, sleeping_lookup,
                          preloading_models=None, judge_activities=None):
    """Render running models + 429-sleeping plugins in the live area.

    ``sleeping_lookup`` maps ``(source, api_model, pid)`` to sleep info so
    the live footer can show each 429-backoff plugin individually. Layout
    (rows counted from ``live_top`` upward):

      0      ``Live:``  (header)
      1..    one row per running model
      ?      ``429 Sleeping:``  (optional header)
      ?..    one row per ``(source, model, pid)`` currently in a 429 backoff sleep

    The streaming/waiting indicator (``(stream)`` / ``(wait)``) is shown for
    any running plugin that supports streaming \u2014 for non-streaming plugins,
    the indicator is omitted so we never lie about a transport detail we
    cannot observe.
    """
    live_row = live_top
    _wr(stdscr, max_x, max_y, live_row, 0, "Live:", curses.A_BOLD)
    live_row += 1

    # Live models are rendered with their per-plugin indicators; any plugin
    # currently in 429 backoff is also listed separately below so the operator
    # sees both the model's active thread state and the per-plugin backoff
    # countdown.
    for nm, s in ((nm, snap.get(nm) or {}) for nm in live_models):
        if live_row >= log_top:
            break
        src_ab = _source_abbr(source_abbrevs, s.get("source"))
        err = s.get("last_error", "")
        msg = f" \U0001f537 [{src_ab}] {nm[:36]}"
        # Per-plugin live indicators. Each bracket includes the elapsed
        # seconds since *that plugin's* dispatch, so the footer never
        # inherits the elapsed time of an earlier plugin. Monotonic
        # timestamps from ``BenchmarkState.start_plugin_run`` keep the
        # counters correct even if the system clock jumps.
        indicators = _build_live_indicators(s, active_plugins)
        if indicators:
            msg += "  " + indicators
        if err:
            msg += f"  {err}"
        _wr(stdscr, max_x, max_y, live_row, 0, msg)
        live_row += 1

    for nm in (preloading_models or []):
        if live_row >= log_top:
            break
        s = snap.get(nm) or {}
        src_ab = _source_abbr(source_abbrevs, s.get("source"))
        elapsed = int(max(0, time.monotonic() - (s.get("preload_start_ts") or time.monotonic())))
        _wr(stdscr, max_x, max_y, live_row, 0,
            f" 🔄 [{src_ab}] Preloading model {nm[:36]} {elapsed}s")
        live_row += 1

    for activity in (judge_activities or []):
        if live_row >= log_top:
            break
        _wr(
            stdscr, max_x, max_y, live_row, 0,
            f" 👩‍⚖️ [Judge {activity['judge']}] {activity['target']} "
            f"[{activity['plugin']} {activity['tokens']} tok {activity['elapsed']}s]",
        )
        live_row += 1

    if sleeping_lookup and live_row + 1 < log_top:
        _wr(stdscr, max_x, max_y, live_row, 0, "429 Sleeping:", curses.A_BOLD)
        live_row += 1
        for (src_name, api_model, pid), info in sleeping_lookup.items():
            if live_row >= log_top:
                break
            src_ab = _source_abbr(source_abbrevs, src_name)
            wake_ts = info["wake_ts"]
            remaining = max(0, round(wake_ts - time.time()))
            msg = (f" \U0001f4a4 [{src_ab}] {api_model[:36]} ({pid}) "
                   f"[429 {info['attempts']}/{info['max_attempts']} {remaining}s]")
            _wr(stdscr, max_x, max_y, live_row, 0, msg)
            live_row += 1

    for r in range(live_row, log_top):
        try:
            stdscr.move(r, 0)
            stdscr.clrtoeol()
        except Exception:  # noqa: BLE001, S110 - window may resize between getmaxyx() and paint
            pass


def _render_recent_errors(stdscr, max_x, max_y, state, log_top, footer_line):
    """Render the recent errors section."""
    from datetime import datetime, timezone
    log_row = log_top
    recent_errors = state.recent_log(2)
    if recent_errors:
        _wr(stdscr, max_x, max_y, log_row, 0, "Errors:", curses.A_BOLD)
        log_row += 1
        for ts_entry, model_entry, msg_entry in recent_errors:
            if log_row >= footer_line:
                break
            t_str = datetime.fromtimestamp(ts_entry, tz=timezone.utc).astimezone().strftime('%H:%M:%S')
            err_msg = f"  {t_str} [{model_entry[:20]}]: {msg_entry}"
            _wr(stdscr, max_x, max_y, log_row, 0, err_msg, curses.color_pair(3))
            log_row += 1
    for r in range(log_row, footer_line):
        try:
            stdscr.move(r, 0)
            stdscr.clrtoeol()
        except Exception:  # noqa: BLE001, S110 - window may resize between getmaxyx() and paint
            pass


def _render_footer(stdscr, max_x, max_y, live_models, queuing, footer_line,
                   preloading_models=None, preloading_details=None,
                   judge_progress=None):
    """Render the bottom status line, including active preload probes.

    ``preloading_details`` is an optional sequence of ``(name, seconds)``
    pairs. Keeping it separate from ``preloading_models`` preserves the
    small helper API used by older callers while allowing the curses footer
    to show the requested ``Preloading model Ns`` status instead of only a
    count.
    """
    preloading_models = preloading_models or []
    preloading_details = preloading_details or []
    judge_progress = judge_progress or {}
    judge_parts = []
    for model, values in judge_progress.items():
            completed = values.get("completed", 0)
            expected = values.get("expected", 0)
            judge_parts.append(f"[{model}: {completed}/{expected}]")

    judge_line = f"Judging {' '.join(judge_parts)}" if judge_parts else ""
    if not live_models and not queuing and not preloading_models and not judge_line:
        msg = " All models complete — generating outputs..."
    else:
        parts = []
        if live_models:
            parts.append(f"{len(live_models)} active")
        if preloading_details:
            parts.extend(
                f"Preloading {name[:30]} {seconds:.0f}s"
                for name, seconds in preloading_details
            )
        elif preloading_models:
            parts.append(f"{len(preloading_models)} preloading")
        if queuing:
            parts.append(f"{len(queuing)} queued")
        if judge_line:
            parts.append(judge_line)
        msg = " " + "  |  ".join(parts)
    if judge_line and (not live_models and not queuing and not preloading_models):
        msg = " " + judge_line
    _wr(stdscr, max_x, max_y, footer_line, 0, msg)


def tui_main(state, stop_event, num_sources, active_plugins, session_seed=None,
             model_thread_limits=None):
    """Run ncurses TUI in a daemon thread. Updates every 200ms."""
    try:
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        stdscr.nodelay(1)
        stdscr.keypad(True)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
    except Exception:  # noqa: BLE001 - any curses init failure falls back to a plain loop
        _fallback_tui_loop(state, stop_event, session_seed, model_thread_limits)
        return

    try:
        LIVE_HEIGHT = max(3, num_sources + 1)
        scroll_y = 0
        scroll_x = 0

        src_snap = {s["source"] for s in state.snapshot().values()}
        source_abbrevs = _unique_source_abbrevs(src_snap)

        frozen_cols = [("#", 4), ("S", 4), ("Model", 18), ("St", 4)]
        frozen_width = FROZEN_VIEW_WIDTH

        plugin_cols = []
        for p in active_plugins:
            # Each plugin gets 4 sub-headers (Sc/Tok/Tm/TPS). The
            # previous per-plugin streaming-glyph column (``St``) was
            # deleted as redundant: the merged bracket status block
            # already conveys in-flight state, and post-flight the
            # plugin isn't streaming anymore, so the glyph was always
            # ``-`` (see the ``PLUGIN_BLOCK_WIDTH`` block comment).
            plugin_cols.extend([
                (f"{p.id[:3]}Sc", 5),
                (f"{p.id[:3]}Tok", 6),
                (f"{p.id[:3]}Tm", 6),
                (f"{p.id[:3]}TPS", 6),
            ])

        _last_tui_error_ts = 0.0
        previous_dimensions = None
        while not stop_event.is_set():
            try:
                max_y, max_x = stdscr.getmaxyx()
                dimensions = (max_y, max_x)
                if dimensions != previous_dimensions:
                    previous_dimensions = _tui_dimensions_changed(
                        stdscr, dimensions, previous_dimensions
                    )
                snap = state.snapshot()
                snap_items = list(snap.items())
                done = state.completed
                total = state.total
                # In-flight models are identified by a non-empty ``running_pids`` list
                # (the canonical source-of-truth populated by
                # ``BenchmarkState.start_plugin_run``). The previous
                # ``s["status"].startswith("running_")`` check relied on a
                # legacy pid-suffix status string that the runtime no longer
                # writes (see commit message); reading the list directly is
                # robust to parallel plugin threads and to status mutations
                # from outer callers (e.g. ``status="completed"`` set after
                # the last plugin finishes).
                running = [n for n, s in snap.items() if s.get("running_pids")]
                preloading = [n for n, s in snap.items() if s.get("preloading")]
                queued = [n for n, s in snap.items() if s["status"] == "queued" and not s.get("preloading")]
                pending = [n for n, s in snap.items() if s["status"] == "pending" and not s.get("preloading")]

                FOOTER_LINE = max_y - 1
                MAX_LOG_ROWS = 3
                LOG_TOP = FOOTER_LINE - MAX_LOG_ROWS
                LIVE_TOP = LOG_TOP - LIVE_HEIGHT
                MODEL_BOTTOM = LIVE_TOP - 1
                MODEL_TOP = 4
                VISIBLE_ROWS = max(0, MODEL_BOTTOM - MODEL_TOP)

                http_threads = get_active_request_count()
                backoff_429 = get_429_stats()
                # Per-plugin 429 lookup keyed by ``(source, api_model, pid)``
                # so each plugin cell can fold its own backoff countdown
                # into the model row. Plugins whose key is absent render
                # normally without the indicator.
                sleeping_lookup = _build_sleeping_lookup(backoff_429)
                sleeping_model_count = len({(src, model) for (src, model, _pid) in sleeping_lookup})

                _render_header_and_summary(
                    stdscr, max_x, max_y, snap, done, total, running, queued, pending,
                    scroll_y, VISIBLE_ROWS, len(snap), session_seed,
                    http_threads, sleeping_model_count, model_thread_limits
                )

                plugin_hdr = _render_table_headings(
                    stdscr, max_x, max_y, scroll_x, frozen_cols, plugin_cols, frozen_width
                )

                max_row_offset = max(0, len(snap_items) - VISIBLE_ROWS)
                scroll_y, scroll_x = _handle_tui_input(
                    stdscr, scroll_y, scroll_x, max_row_offset, VISIBLE_ROWS, max_x,
                    frozen_width, _display_width(plugin_hdr)
                )

                _render_model_rows(
                    stdscr, max_x, max_y, snap_items, active_plugins, source_abbrevs,
                    scroll_y, scroll_x, VISIBLE_ROWS, frozen_width, MODEL_TOP,
                    sleeping_lookup,
                )

                if MODEL_BOTTOM >= 0:
                    _wr(stdscr, max_x, max_y, MODEL_BOTTOM, 0, "\u2500" * min(max_x, 60))

                _render_live_activity(
                    stdscr, max_x, max_y, snap, source_abbrevs, running,
                    LIVE_TOP, LIVE_HEIGHT, LOG_TOP, active_plugins, sleeping_lookup,
                    preloading_models=preloading,
                    judge_activities=state.judge_activity_snapshot(),
                )

                _render_recent_errors(stdscr, max_x, max_y, state, LOG_TOP, FOOTER_LINE)

                queuing = queued + pending
                preload_details = [
                    (
                        name,
                        max(0.0, time.monotonic() - (snap[name].get("preload_start_ts") or time.monotonic())),
                    )
                    for name in preloading
                    if name in snap
                ]
                _render_footer(
                    stdscr, max_x, max_y, running, queuing, FOOTER_LINE,
                    preloading_models=preloading,
                    preloading_details=preload_details,
                    judge_progress=state.judge_progress_snapshot(),
                )

                stdscr.refresh()
            except Exception:  # noqa: BLE001 - a render error must not kill the TUI thread
                # Log to a file so the screen isn't corrupted and the benchmark
                # workers can keep running. Throttle to avoid a runaway log.
                now = time.time()
                if now - _last_tui_error_ts > 5.0:
                    _last_tui_error_ts = now
                    try:
                        with open("tui_render_errors.log", "a", encoding="utf-8") as f:
                            traceback.print_exc(file=f)
                    except Exception:  # noqa: BLE001, S110 - logging a render error must not crash the TUI thread
                        pass
            time.sleep(0.2)

    finally:
        curses.echo()
        curses.nocbreak()
        try:
            curses.endwin()
        except Exception:  # noqa: BLE001, S110 - endwin() can fail on a broken terminal
            pass


def _targets_for_runner(targets, state_models, runner):
    """Return targets with a saved/configured identity for ``runner``."""
    suffix = " [opencode]" if runner == "opencode" else ""
    return {
        name: info
        for name, info in targets.items()
        if f"{name}{suffix}" in state_models
    }


def _build_runner_queues(targets, snapshot, runner_mode, source_config,
                         *, rerun_failed=True):
    """Build pending runner queues from the loaded state snapshot.

    ``rerun_failed`` mirrors the resume option. Keeping this decision in the
    queue builder is important: ``BenchmarkState.load_state`` can preserve a
    failed status, but a status-only ``!= completed`` check would immediately
    put that target back on the scheduler queue anyway.
    """
    def needs_run(state):
        return (
            state is not None
            and state.get("status") != "completed"
            and (rerun_failed or state.get("status") != "failed")
        )

    if runner_mode == "both":
        targets_by_source = {src: [] for src in source_config}
        opencode_pending = {src: [] for src in targets_by_source}
        http_pending = {src: set() for src in targets_by_source}
        for name, info in targets.items():
            opencode_state = snapshot.get(f"{name} [opencode]")
            opencode_needed = needs_run(opencode_state)
            http_state = snapshot.get(name)
            http_needed = needs_run(http_state)
            if opencode_needed:
                opencode_pending[info["source"]].append(name)
            if http_needed:
                http_pending[info["source"]].add(name)
            if opencode_needed or http_needed:
                targets_by_source[info["source"]].append(name)
        return targets_by_source, opencode_pending, http_pending

    phase_runner = runner_mode
    source_queues = {src: [] for src in {info["source"] for info in targets.values()}}
    for name, info in targets.items():
        state_key = name if phase_runner == "http" else f"{name} [opencode]"
        if needs_run(snapshot.get(state_key)):
            source_queues[info["source"]].append(name)
    return source_queues


class SourceModelScheduler:
    """Run a FIFO queue of target pipelines with a source-local bound."""

    def __init__(self, source, max_models, target_names, run_target,
                 stop_event, on_error, *, runner_label="model",
                 peak_callback=None, on_complete=None):
        self.source = source
        self.max_models = max(1, int(max_models))
        self.target_names = list(target_names)
        self.run_target = run_target
        self.stop_event = stop_event
        self.on_error = on_error
        self.runner_label = runner_label
        self.peak_callback = peak_callback
        self.on_complete = on_complete

    def run_until_drained(self):
        """Submit at most ``max_models`` targets and refill as they finish."""
        next_index = 0
        futures = {}
        active = 0
        executor = ThreadPoolExecutor(max_workers=self.max_models)
        try:
            def submit_next():
                nonlocal next_index, active
                if self.stop_event.is_set() or next_index >= len(self.target_names):
                    return False
                target_name = self.target_names[next_index]
                next_index += 1
                # The scheduler's FIFO queue and this one-shot submission
                # path are the claim guard: a target is inserted into exactly
                # one future before any refill can advance the queue.
                futures[executor.submit(self.run_target, target_name)] = target_name
                active += 1
                if self.peak_callback:
                    self.peak_callback(self.source, active)
                return True

            for _ in range(self.max_models):
                if not submit_next():
                    break
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    target_name = futures.pop(future)
                    active -= 1
                    if self.peak_callback:
                        self.peak_callback(self.source, active)
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001 - isolate one target failure
                        self.on_error(target_name, self.runner_label, exc)
                    submit_next()
                if self.stop_event.is_set():
                    for future in futures:
                        future.cancel()
                    break
        finally:
            # Do not let the executor context manager wait indefinitely after
            # cancellation: active HTTP/subprocess work is interrupted by the
            # caller before workers are joined. Normal completion still shuts
            # down synchronously so no executor thread leaks into output work.
            if self.stop_event.is_set():
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
            else:
                executor.shutdown(wait=True)
        if not self.stop_event.is_set() and self.on_complete:
            self.on_complete(self.source)



def _configure_judge_source(benchmark_limits, source, full_limit,
                            benchmark_active, pool):
    """Configure the judge reservation for one source.

    During benchmark overlap, reserve one judge worker only when another
    source slot remains available. Sources with no benchmark work start their
    full judge pool immediately; completion callbacks later call
    ``pool.expand_full()`` for active sources.
    """
    full_limit = max(1, int(full_limit))
    if not benchmark_active:
        pool.start(full_limit)
    elif full_limit > 1:
        benchmark_limits[source] = max(1, full_limit - 1)
        pool.start(1)


class SourceJudgeWorkerPool:
    """Run judge jobs with a source-local, dynamically expandable pool.

    The pool starts with the reserved judge capacity used while benchmark
    targets are active. Once the source benchmark scheduler completes, callers
    can expand it to ``limit`` so all source slots judge concurrently.
    """

    def __init__(self, source, limit, process_job, stop_event):
        self.source = source
        self.limit = max(1, int(limit))
        self.process_job = process_job
        self.stop_event = stop_event
        self.queue = queue.Queue()
        self._stop = object()
        self._lock = threading.Lock()
        self._threads = []

    @property
    def thread_count(self):
        """Return the number of workers started for this source."""
        with self._lock:
            return len(self._threads)

    def enqueue(self, job):
        """Queue one judge job for this source."""
        self.queue.put(job)

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                job = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is self._stop:
                break
            self.process_job(job)

    def start(self, count=1):
        """Start workers up to ``count`` without exceeding the source limit."""
        with self._lock:
            target = min(self.limit, max(0, int(count)))
            new_threads = []
            for index in range(len(self._threads), target):
                thread = threading.Thread(
                    target=self._worker,
                    name=f"judge-worker-{self.source}-{index + 1}",
                    daemon=True,
                )
                self._threads.append(thread)
                new_threads.append(thread)
        for thread in new_threads:
            thread.start()

    def expand_full(self):
        """Release the benchmark reservation and use the full source pool."""
        self.start(self.limit)

    def stop(self, timeout=None):
        """Stop and join all workers, preserving already-processed jobs."""
        with self._lock:
            threads = list(self._threads)
        for _thread in threads:
            self.queue.put(self._stop)
        for thread in threads:
            thread.join(timeout=timeout)


def _start_runner_pipeline(targets_by_source, opencode_pending, http_pending,
                           run_target, stop_event, on_error,
                           model_thread_limits=None, peak_callback=None,
                           source_complete_callback=None):

    """Start one OpenCode-to-HTTP worker per source.

    ``run_target(name, runner)`` executes one configured target through the
    requested runner. In ``both`` mode, each source has exactly one worker,
    which processes each target as ``OpenCode -> HTTP`` before moving to the
    next target. Sources run concurrently with one another, but OpenCode and
    HTTP can never run at the same time for the same source.

    Targets whose OpenCode identity is already complete are sent directly to
    their pending HTTP step on resume. The returned threads are intentionally
    not joined here; the caller owns the join/interrupt policy so Ctrl+C can
    set ``stop_event`` and close active HTTP requests before waiting for workers
    to wind down.
    """
    threads = []

    def source_worker(source, target_names):
        limit = (model_thread_limits or {}).get(source, 1)

        def run_pipeline(target_name):
            skip_target = False
            if target_name in opencode_pending.get(source, []):
                try:
                    skip_target = run_target(target_name, "opencode") is False
                except Exception as exc:  # noqa: BLE001 - a runner crash is reported, not fatal
                    on_error(target_name, "opencode", exc)
            if stop_event.is_set() or skip_target:
                return
            if target_name in http_pending.get(source, set()):
                try:
                    run_target(target_name, "http")
                except Exception as exc:  # noqa: BLE001 - a runner crash is reported, not fatal
                    on_error(target_name, "http", exc)

        scheduler = SourceModelScheduler(
            source, limit, target_names, run_pipeline, stop_event, on_error,
            runner_label="pipeline", peak_callback=peak_callback,
            on_complete=source_complete_callback,
        )
        try:
            scheduler.run_until_drained()
        except Exception as exc:  # noqa: BLE001 - isolate unexpected source scheduler failures
            on_error(source, "scheduler", exc)

    for source, target_names in targets_by_source.items():
        if not target_names:
            continue

        thread = threading.Thread(
            target=source_worker,
            args=(source, target_names),
            name=f"runner-pipeline-{source}",
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    return threads


def _prompt_restart_or_continue(scripted=False):
    """Ask the user whether to restart or continue a run with changed plugins.

    In scripted mode, no input is requested and the run continues automatically,
    running any new plugins on all models while preserving existing results.
    """
    print("\nPlugin set has changed since the last run.")
    if scripted:
        print("   Scripted mode: continuing run and running new plugins on all models.",
              file=sys.stderr)
        return "continue"
    print("[r] Restart run (discard old state)")
    print("[c] Continue run (keep old data, run missing plugins)")
    print("[q] Quit")
    while True:
        try:
            choice = input("Choice [r/c/q]: ").strip().lower()
        except EOFError:
            choice = "q"
        if choice in ("r", "restart"):
            return "restart"
        if choice in ("c", "continue"):
            return "continue"
        if choice in ("q", "quit"):
            return "quit"
        print("Please enter r, c, or q.")


def _prompt_corrupt_state(recovery, scripted=False):
    """Show recovery counts and obtain an explicit corruption decision.

    ``continue`` is available only when a valid state candidate and results
    container were recovered. Scripted mode continues automatically only for a
    known, zero-loss repair; uncertain or lossy recovery aborts conservatively.
    """
    total = recovery.get("total_results")
    recoverable = recovery.get("recoverable_results", 0)
    lost = recovery.get("lost_results")
    total_text = str(total) if total is not None else "unknown"
    lost_text = str(lost) if lost is not None else "unknown"
    print("\n⚠️  Corrupted benchmark state detected.", file=sys.stderr)
    print(f"   Results in file: {total_text}", file=sys.stderr)
    print(f"   Results recoverable: {recoverable}", file=sys.stderr)
    print(f"   Results that would be lost: {lost_text}", file=sys.stderr)

    can_continue = (
        recovery.get("data") is not None
        and recovery.get("results_found")
        and recovery.get("counts_certain", False)
    )
    zero_loss = can_continue and lost == 0
    if zero_loss:
        print("   No completed results will be lost; applying the validated repair.",
              file=sys.stderr)
        return "continue"
    if scripted:
        if zero_loss:
            print("   Scripted mode: continuing with the validated zero-loss repair.",
                  file=sys.stderr)
            return "continue"
        print("   Scripted mode: aborting because recovery is uncertain or lossy.",
              file=sys.stderr)
        return _CORRUPTED_STATE_ABORT

    if can_continue:
        print("[c] Continue with recoverable results")
    else:
        print("[c] Continue is unavailable: no complete state candidate was recovered")
    print("[r] Restart run (discard the corrupted state)")
    print("[a] Abort and leave the corrupted file untouched")
    while True:
        try:
            choice = input("Choice [c/r/a]: ").strip().lower()
        except (EOFError, OSError):
            choice = "a"
        if choice in ("c", "continue") and can_continue:
            if lost:
                print(f"   Continuing will discard {lost} unrecoverable result(s).",
                      file=sys.stderr)
            return "continue"
        if choice in ("r", "restart"):
            return "restart"
        if choice in ("a", "abort", "q", "quit"):
            return _CORRUPTED_STATE_ABORT
        print("Please enter c, r, or a.")


def main():  # pragma: no cover - live benchmark orchestrator (no unit tests)
    try:
        subprocess.run(['stty', 'sane'], stderr=subprocess.DEVNULL,
                       stdin=sys.stdin, timeout=1, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass
    sys.stderr.write('\033[2J\033[H')
    sys.stderr.flush()

    parser = argparse.ArgumentParser(
        description="AI Model Benchmark — Run plugin-based benchmarks across multiple API sources.",
        epilog="Challenge plugins are loaded from plugins/challenges/ and report plugins from plugins/outputs/.\n\n"
               "Examples:\n"
               "  python ai-benchmark.py --restart\n"
               "  python ai-benchmark.py --config my-config.json\n"
               "  python ai-benchmark.py --out /tmp/bench-run --timeout 300\n"
               "  python ai-benchmark.py --plugins-whitelist rate-limiter\n"
               "  python ai-benchmark.py --dump-default-config --base-url http://localhost:11434 > config.json\n"
               "  python ai-benchmark.py --dump-default-config > benchmark-config.json\n\n"
               "Shell completions:\n"
               "  eval \"$(python ai-benchmark.py --generate-shell-completion bash)\"\n"
               "  python ai-benchmark.py --generate-shell-completion zsh > ~/.zsh/completions/_ai-benchmark.py\n"
               "  python ai-benchmark.py --generate-shell-completion fish > ~/.config/fish/completions/ai-benchmark.py.fish",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--restart', action='store_true',
                        help='Restart the run from scratch, discarding prior results')
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH,
                        help=f'Config file path (default: {DEFAULT_CONFIG_PATH})')
    parser.add_argument('--out', default=None,
                        help='Override output directory from config')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Override request timeout in seconds from config')
    parser.add_argument('--token-levels', type=int, nargs='+', default=None,
                        help='Override token levels (e.g. --token-levels 4096 8192 16384)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Default temperature for all plugins (overrides config; individual --plugin-temperature takes priority)')
    parser.add_argument('--plugin-temperature', type=str, nargs='+', default=None,
                        help='Per-plugin temperatures as id=value (e.g. --plugin-temperature rate-limiter=0.2 moe-dense=0.7)')
    parser.add_argument('--plugin-thread-limit', type=int, default=None,
                        help='Max threads per model for plugin execution. 0 means one thread per plugin (default: 1)')
    parser.add_argument('--plugins-whitelist', type=str, nargs='+', default=None,
                        help='Run only these plugins (e.g. --plugins-whitelist rate-limiter moe-dense)')
    parser.add_argument('--plugins-blacklist', type=str, nargs='+', default=None,
                        help='Run all plugins except these (e.g. --plugins-blacklist moe-dense)')
    parser.add_argument('--list-plugins', action='store_true',
                        help='List discovered challenge plugins (from plugins/challenges/) with their IDs, names, and versions, then exit')
    parser.add_argument('--generate-shell-completion', type=str, default=None,
                        choices=['bash', 'zsh', 'fish'],
                        help='Generate shell completion script for the specified shell and exit')
    parser.add_argument('--dump-default-config', action='store_true',
                        help='Print a default config file to stdout and exit')
    parser.add_argument('--convert-config', type=str, default=None,
                        help='Convert a YAML config to JSON or a JSON config to YAML and print to stdout')
    parser.add_argument('--base-url', default=None,
                        help='Base URL for model discovery via /v1/models API (used with --dump-default-config)')
    parser.add_argument('--api-key', default=None,
                        help='API key for model discovery (used with --dump-default-config --base-url)')
    parser.add_argument('--save-responses', action='store_true',
                        help='Save each model\'s plugin response text to <output_dir>/responses/')
    parser.add_argument('--judge-models', nargs='+', default=None, metavar='MODEL',
                        help='Judge benchmark responses with one configured model (repeatable in a future consensus mode)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Fixed random seed for all API requests (default: random)')
    retry_group = parser.add_mutually_exclusive_group()
    retry_group.add_argument('--retry-on-429', action='store_true', default=True,
                             help='Retry HTTP 429 responses with exponential backoff for any source '
                                  'that did not set its own max_429_retries (default: enabled). Each '
                                  'rate-limited request can sleep up to (max_429_retries x max_backoff_seconds) '
                                  'before failing. Use --no-retry-on-429 if you want the legacy fail-fast '
                                  'behaviour back.')
    retry_group.add_argument('--no-retry-on-429', action='store_false',
                             help='Disable HTTP 429 retries globally. Overrides per-source max_429_retries '
                                  'only when the source did not set its own; explicit per-source values '
                                  'are preserved.')
    parser.add_argument('--no-rerun-failed', action='store_true',
                        help='Do not re-run models that failed in a previous session')
    parser.add_argument('--scripted', action='store_true',
                        help='Non-interactive mode: never prompt for input; default to continuing runs')
    parser.add_argument('--runner', choices=['http', 'opencode', 'both'], default='http',
                        help='Execution runner: http (default), opencode, or both (per-target OpenCode-to-HTTP pipeline)')
    parser.add_argument('--no-install-opencode', action='store_true',
                        help='Do not auto-download OpenCode into .tools/opencode/ when it is missing or too old; fail with an error instead')
    parser.add_argument('--no-preload', action='store_true',
                        help='Disable per-source model pre-loading for this run')
    args = parser.parse_args()

    if args.list_plugins:
        print(format_plugin_list(discover_plugins()))
        sys.exit(0)

    if args.generate_shell_completion:
        print(generate_shell_completion(args.generate_shell_completion, discover_plugins()))
        sys.exit(0)

    if args.dump_default_config:
        if args.base_url:
            cfg = generate_config_from_api(args.base_url, args.api_key)
            print(json.dumps(cfg, indent=2))
        else:
            dump_default_config()
        sys.exit(0)

    if args.convert_config:
        if not os.path.exists(args.convert_config):
            print(f"❌ Config file not found: {args.convert_config}", file=sys.stderr)
            sys.exit(1)
        ext = os.path.splitext(args.convert_config)[1].lower()
        if ext not in (".json", ".yaml", ".yml"):
            print(f"❌ Unsupported config format: {ext}. Use .json, .yaml, or .yml.", file=sys.stderr)
            sys.exit(1)
        cfg = load_config(args.convert_config)
        if ext in (".yaml", ".yml"):
            print(json.dumps(cfg, indent=2))
        else:
            import yaml
            print(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
        sys.exit(0)

    config_path = _resolve_config_path(args.config)
    if config_path is None:
        print(f"❌ Config file not found: {args.config}\n"
              f"   Tried: benchmark-config.json, benchmark-config.yaml, benchmark-config.yml\n"
              f"   Copy benchmark-config.json or create one with --dump-default-config.",
              file=sys.stderr)
        sys.exit(1)
    cfg = load_config(config_path)
    # Apply the global --retry-on-429 / --no-retry-on-429 toggle before any
    # plugin sees the config so per-source defaults are aligned with the flag.
    _apply_http_retry_default(cfg, args.retry_on_429)
    source_config = cfg.get("sources", {})
    models = cfg.get("models", {})
    agents = cfg.get("agents", {})
    collisions = set(models) & set(agents)
    if collisions:
        print(f"❌ Model/agent name collision: {', '.join(sorted(collisions))}", file=sys.stderr)
        sys.exit(1)
    targets = resolve_targets(cfg)
    runner_mode = args.runner
    judge_models = list(dict.fromkeys(args.judge_models or []))
    unknown_judges = [name for name in judge_models if name not in models]
    if unknown_judges:
        print(
            f"❌ Unknown judge model(s): {', '.join(unknown_judges)}. "
            f"Choose from configured models: {', '.join(models) or '(none)'}.",
            file=sys.stderr,
        )
        sys.exit(1)
    judge_model = judge_models[0] if judge_models else None
    opencode_binary = None
    if runner_mode in ("opencode", "both"):
        try:
            opencode_binary = resolve_opencode_binary(
                allow_install=not args.no_install_opencode,
            )
        except RuntimeError as exc:
            print(f"❌ OpenCode unavailable: {exc}", file=sys.stderr)
            sys.exit(1)
        version = opencode_version(opencode_binary)
        print(f"🤖 OpenCode binary: {opencode_binary}"
              + (f" (v{version})" if version else ""), file=sys.stderr)
    output_dir = cfg.get("output_dir", "benchmark-results")
    if args.out:
        output_dir = args.out
    state_file = os.path.join(output_dir, "benchmark_state.json")

    # Keep one state identity per (configured target, runner). HTTP retains
    # the historical target key; OpenCode gets a stable suffix so `both`
    # can resume and report the two executions independently.
    state_models = {}
    for target_name, target_info in targets.items():
        if runner_mode in ("http", "both"):
            state_models[target_name] = {**target_info, "runner": "http"}
        if runner_mode in ("opencode", "both"):
            state_models[f"{target_name} [opencode]"] = {**target_info, "runner": "opencode"}

    timeout = cfg.get("timeout", 600)
    if args.timeout is not None:
        timeout = args.timeout

    token_levels = cfg.get("token_levels", [16384])
    if args.token_levels is not None:
        token_levels = args.token_levels

    whitelist = args.plugins_whitelist or cfg.get("plugins_whitelist") or None
    blacklist = args.plugins_blacklist or cfg.get("plugins_blacklist") or None
    if whitelist and blacklist:
        print("❌ Cannot specify both --plugins-whitelist and --plugins-blacklist.", file=sys.stderr)
        sys.exit(1)
    try:
        active_plugins = discover_plugins(whitelist=whitelist, blacklist=blacklist)
    except Exception as e:  # noqa: BLE001 - a broken plugin must fail loudly, not silently
        print(f"❌ Failed to discover plugins: {e}", file=sys.stderr)
        sys.exit(1)

    if not active_plugins:
        print("❌ No plugins selected. Check your whitelist/blacklist.", file=sys.stderr)
        sys.exit(1)

    # Per-plugin temperatures: CLI overrides config. Config keys may use either
    # hyphen or underscore, e.g. "rate-limiter_temperature" or "rate_servererature".
    plugin_temperatures = parse_plugin_temperatures(cfg)
    if args.temperature is not None:
        for plugin in active_plugins:
            plugin_temperatures[plugin.id] = args.temperature
    if args.plugin_temperature:
        for item in args.plugin_temperature:
            if "=" not in item:
                print(f"❌ Invalid --plugin-temperature value: {item}. Expected id=value.", file=sys.stderr)
                sys.exit(1)
            pid, temp_str = item.split("=", 1)
            try:
                plugin_temperatures[pid] = float(temp_str)
            except ValueError:
                print(f"❌ Invalid temperature for {pid}: {temp_str}", file=sys.stderr)
                sys.exit(1)
    cfg["plugin_temperatures"] = plugin_temperatures

    # Apply per-source plugin_thread_limit defaults and validate the separate
    # model-level source slots. The latter has no unlimited/zero meaning.
    for source, src_cfg in source_config.items():
        if not isinstance(src_cfg, dict):
            print(f"❌ Source '{source}' must be an object.", file=sys.stderr)
            sys.exit(1)
        src_cfg["plugin_thread_limit"] = src_cfg.get(
            "plugin_thread_limit", cfg.get("plugin_thread_limit", 1)
        )
    if args.plugin_thread_limit is not None:
        for src_cfg in source_config.values():
            src_cfg["plugin_thread_limit"] = args.plugin_thread_limit
    try:
        model_thread_limits = {
            source: resolve_model_thread_limit(
                source_config, source, cfg.get("model_thread_limit", 1)
            )
            for source in source_config
        }
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(1)
    for source, limit in model_thread_limits.items():
        if limit > 1 and source.lower() in {"ai server", "gaming pc"}:
            print(
                f"⚠️  {source}: model_thread_limit={limit} may exhaust local hardware; honoring explicit configuration.",
                file=sys.stderr,
            )

    print(f"📋 Loaded {len(targets)} targets ({len(models)} models, {len(agents)} agents) "
          f"across {len(source_config)} sources from {config_path}", file=sys.stderr)
    print("🧵 Model slots: " + "; ".join(
        f"{source}: {limit}" for source, limit in model_thread_limits.items()
    ), file=sys.stderr)
    print(f"🔌 Active plugins: {', '.join(p.name for p in active_plugins)} "
          f"(v{', v'.join(p.version for p in active_plugins)})", file=sys.stderr)
    print(f"📂 Output directory: {output_dir}", file=sys.stderr)
    print(f"🏃 Runner: {runner_mode}", file=sys.stderr)

    os.makedirs(output_dir, exist_ok=True)
    http_output_dir = os.path.join(output_dir, "http")
    opencode_output_dir = os.path.join(output_dir, "opencode")
    if runner_mode in ("http", "both"):
        os.makedirs(http_output_dir, exist_ok=True)
    opencode_config_path = None
    opencode_agent_ids = {}
    opencode_projection = None
    if runner_mode in ("opencode", "both"):
        try:
            generated = generate_opencode_config(
                source_config,
                targets,
                os.path.join(opencode_output_dir, "opencode.generated.json"),
                timeout=timeout,
                token_levels=token_levels,
                benchmark_config=cfg,
                plugin_temperatures=cfg.get("plugin_temperatures"),
            )
            opencode_config_path = generated["path"]
            opencode_agent_ids = generated["agent_ids"]
            opencode_projection = generated["projection"]
        except (OSError, ValueError) as exc:
            print(f"❌ Could not prepare OpenCode configuration: {exc}", file=sys.stderr)
            sys.exit(1)

    try:
        shutil.copy2(config_path, os.path.join(output_dir, os.path.basename(config_path)))
    except (OSError, shutil.Error) as e:
        print(f"⚠️  Could not copy config file to output directory: {e}", file=sys.stderr)
    state = None
    worker_errors = 0
    interrupted = False
    reset_429_stats()

    run_info = {
        "config_file": config_path,
        "cli_args": vars(args),
        "output_dir": output_dir,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": None,
        "status": "running",
        "total_targets": len(targets) * (2 if runner_mode == "both" else 1),
        "completed_targets": 0,
        "worker_errors": 0,
        "session_seed": None,
        "active_plugins": [p.id for p in active_plugins],
        "runner": runner_mode,
        "score_schema": SCORE_SCHEMA,
        "model_thread_limit": model_thread_limits,
        "peak_active_models": {source: 0 for source in model_thread_limits},
        "opencode_config": opencode_config_path,
        "opencode_projection": opencode_projection,
        "opencode_binary": opencode_binary,
        "targets": list(targets.keys()),
        "judge_model": judge_model,
        "judge_models": judge_models,
        "judge_prompt_version": "judge-v1" if judge_models else None,
        "judge_status": "disabled" if not judge_models else "pending",
        "judge_counts": {"queued": 0, "completed": 0, "failed": 0, "votes": 0},
        "preload": {
            "enabled_sources": [
                name for name, src_cfg in source_config.items()
                if not args.no_preload and isinstance(src_cfg, dict) and src_cfg.get("preload", False)
            ],
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
            "total_preload_time": 0.0,
            "per_model": {},
        },
    }

    try:
        if args.restart:
            _clear_restart_artifacts(state_file, output_dir)

        plugin_ids = [p.id for p in active_plugins]
        plugin_versions = {p.id: p.version for p in active_plugins}

        resumed = False
        restored_targets = []
        fresh_state = False
        if not args.restart and os.path.exists(state_file):
            try:
                try:
                    with open(state_file, encoding="utf-8") as f:
                        saved_state = json.load(f)
                except json.JSONDecodeError:
                    recovery = prepare_state_recovery(state_file)
                    choice = _prompt_corrupt_state(recovery, scripted=args.scripted)
                    if choice == "restart":
                        _clear_restart_artifacts(state_file, output_dir)
                        state = BenchmarkState(state_models, plugin_ids, runner=runner_mode)
                        fresh_state = True
                    elif choice == "continue":
                        backup = apply_state_recovery(state_file, recovery)
                        saved_state = recovery["data"]
                        if recovery.get("kind") == "known" and recovery.get("lost_results") == 0:
                            print(
                                f"⚠️  Repaired corrupted state file; backup saved as {backup}.",
                                file=sys.stderr,
                            )
                        else:
                            print(
                                f"   Recovery applied; original state saved as {backup}.",
                                file=sys.stderr,
                            )
                    else:
                        print("❌ Could not resume run: corruption recovery was aborted.",
                              file=sys.stderr)
                        print("   Aborting instead of silently discarding prior results.",
                              file=sys.stderr)
                        print("   The corrupted state file was left untouched.",
                              file=sys.stderr)
                        print("   Use --restart only if you intentionally want to discard it and start fresh.",
                              file=sys.stderr)
                        sys.exit(1)

                if not fresh_state:
                    saved_plugins = saved_state.get("active_plugins", [])
                    if set(saved_plugins) != set(plugin_ids):
                        print("\n⚠️  Plugin set has changed.", file=sys.stderr)
                        print(f"   Saved:   {', '.join(saved_plugins) or '(none)'}", file=sys.stderr)
                        print(f"   Current: {', '.join(plugin_ids)}", file=sys.stderr)
                        choice = _prompt_restart_or_continue(scripted=args.scripted)
                        if choice == "restart":
                            _clear_restart_artifacts(state_file, output_dir)
                            state = BenchmarkState(state_models, plugin_ids, runner=runner_mode)
                            fresh_state = True
                        elif choice != "continue":
                            sys.exit(0)

                    if not fresh_state:
                        restored_targets = _merge_saved_targets(
                            targets, state_models, saved_state, runner_mode,
                        )
                        if restored_targets:
                            run_info["targets"] = list(targets)
                            run_info["total_targets"] = (
                                len(state_models) if runner_mode == "both" else len(targets)
                            )
                            print(
                                f"   Restored {len(restored_targets)} saved target(s) absent from current config.",
                                file=sys.stderr,
                            )
                        state = BenchmarkState.load_state(
                            state_file, state_models, plugin_ids,
                            rerun_failed=not args.no_rerun_failed)
                        # State may contain results from another runner; retain
                        # them because identity is carried per model/result.
                        resumed = True

                if resumed:
                    completed = state.completed
                    total = state.total
                    print(f"📂 Resuming — {completed}/{total} models already completed. "
                          f"Failed models/plugins will be re-run.\n"
                          f"   Remove {state_file} or use --restart to start fresh.",
                          file=sys.stderr)

                    if completed == total and total > 0 and not judge_models:
                        print(f"\n{'='*70}")
                        print(f"✅ PRIOR RUN COMPLETE — {completed}/{total} successful")
                        print(f"   Results: {output_dir}/")
                        print(f"{'='*70}")
                        sys.exit(0)
            except Exception as e:  # noqa: BLE001 - abort rather than silently restarting
                # A failed resume must not silently discard prior results by
                # starting a fresh run: the operator may have hours of
                # completed work in this state file. Abort with the underlying
                # error so the state can be inspected or repaired, or the run
                # can be explicitly discarded with --restart.
                print(f"❌ Could not resume run: failed to load or clear the state file ({e}).",
                      file=sys.stderr)
                print("   Aborting instead of silently discarding prior results.",
                      file=sys.stderr)
                print(f"   Inspect or fix {state_file}, or pass --restart to discard it and start fresh.",
                      file=sys.stderr)
                sys.exit(1)
        else:
            state = BenchmarkState(state_models, plugin_ids, runner=runner_mode)

        # The active CLI judge configuration is authoritative on resume; do
        # not let a prior run's judge set drive stale row markers.
        state.set_judge_models(judge_models)

        if restored_targets and runner_mode in ("opencode", "both"):
            # OpenCode's generated projection is created before the state file
            # is loaded. Rebuild it after restoring removed targets so their
            # pending legs have valid agent/config entries when workers start.
            try:
                generated = generate_opencode_config(
                    source_config,
                    _targets_for_runner(targets, state_models, "opencode"),
                    os.path.join(opencode_output_dir, "opencode.generated.json"),
                    timeout=timeout,
                    token_levels=token_levels,
                    benchmark_config=cfg,
                    plugin_temperatures=cfg.get("plugin_temperatures"),
                )
                opencode_config_path = generated["path"]
                opencode_agent_ids = generated["agent_ids"]
                opencode_projection = generated["projection"]
            except (OSError, ValueError) as exc:
                print(f"❌ Could not refresh OpenCode configuration: {exc}", file=sys.stderr)
                sys.exit(1)

        # Use the CLI --seed if provided; otherwise preserve the seed from a
        # resumed state so report exports remain consistent.
        if args.seed is not None:
            session_seed = args.seed
        elif getattr(state, "session_seed", None) is not None:
            session_seed = state.session_seed
        else:
            session_seed = random.randint(0, 2**31 - 1)
        state.session_seed = session_seed
        run_info["session_seed"] = session_seed

        stop_event = threading.Event()

        # Run the TUI as a daemon so the process can exit promptly on Ctrl+C.
        # Without this, a stuck curses/fallback UI thread would block interpreter
        # shutdown, forcing the user to press Ctrl+C a second time.
        tui_thread = threading.Thread(
            target=tui_main,
            args=(state, stop_event, len(source_config), active_plugins, session_seed,
                  model_thread_limits),
            daemon=True,
        )
        tui_thread.start()

        time.sleep(0.3)

        total = state.total

        def _join_workers(timeout=None):
            """Wait for the current phase's source workers."""
            if not source_threads:
                return True
            if timeout is None:
                while any(t.is_alive() for t in source_threads.values()):
                    for t in source_threads.values():
                        t.join(timeout=0.2)
                return True
            for t in source_threads.values():
                t.join(timeout=timeout / max(len(source_threads), 1))
            return not any(t.is_alive() for t in source_threads.values())

        errors_lock = threading.Lock()
        persistence_lock = threading.Lock()
        preload_lock = threading.Lock()
        preloaded_ok = set()
        preload_failed = set()
        preload_inflight = {}
        raw_targets = {}
        raw_targets.update(cfg.get("models", {}))
        raw_targets.update(cfg.get("agents", {}))

        def _preload_is_enabled(source):
            src_cfg = source_config.get(source) or {}
            return (not args.no_preload and isinstance(src_cfg, dict)
                    and bool(src_cfg.get("preload", False)))

        def _set_preloading(target_name, target_info, enabled):
            """Mark both runner rows for a target as warming, when present."""
            keys = [target_name]
            if runner_mode == "both":
                keys.append(f"{target_name} [opencode]")
            now = time.monotonic() if enabled else 0
            snapshot = state.snapshot()
            for key in keys:
                # A completed leg must remain completed on resume. The shared
                # probe can still warm the model for pending legs, but it
                # must not turn an already-finished runner back into queued
                # work or overwrite its report state.
                if key in snapshot and snapshot[key].get("status") != "completed":
                    state.update(
                        key,
                        status="queued" if enabled else snapshot[key].get("status", "pending"),
                        preloading=enabled,
                        preload_start_ts=now,
                    )

        def _record_preload_failure(model_name, target_info, result, phase_runner):
            """Record a failed warm-up for the pending runner leg(s)."""
            error = f"preload failed: {result.error or 'empty preload response'}"
            source = target_info["source"]
            if runner_mode == "both" and phase_runner == "opencode":
                keys = [model_name, f"{model_name} [opencode]"]
            elif phase_runner == "opencode":
                keys = [f"{model_name} [opencode]"]
            else:
                keys = [model_name]
            snapshot = state.snapshot()
            for key in keys:
                info = snapshot.get(key)
                if info is None or info.get("status") == "completed":
                    continue
                runner = "opencode" if key.endswith(" [opencode]") else "http"
                state.add_result({
                    "model": model_name,
                    "state_key": key,
                    "api_model": target_info["api_model"],
                    "source": source,
                    "runner": runner,
                    "opencode_model": None,
                    "is_agent": target_info["is_agent"],
                    "system_prompt": target_info["system_prompt"],
                    "status": "error",
                    "stream_ok": False,
                    "ttft": None,
                    "total_time": 0.0,
                    "error": error,
                    "preload_time": result.elapsed,
                    "preload_error": result.error or "empty preload response",
                    "plugin_versions": plugin_versions,
                })
                state.update(
                    key,
                    status="failed",
                    error=error,
                    last_error=error,
                    elapsed=0.0,
                    preloading=False,
                    preload_start_ts=0,
                    preload_status="failed",
                    preload_time=result.elapsed,
                    preload_error=result.error or "empty preload response",
                )
                state.log(key, error)

        def _ensure_preloaded(model_name, target_info, phase_runner):
            """Warm a target once per source/model for this process."""
            if not _preload_is_enabled(target_info["source"]):
                return True
            key = (target_info["source"], target_info["api_model"])
            with preload_lock:
                if key in preloaded_ok:
                    return True
                if key in preload_failed:
                    return False
                inflight = preload_inflight.get(key)
                if inflight is None:
                    inflight = threading.Event()
                    preload_inflight[key] = inflight
                    preload_owner = True
                    _set_preloading(model_name, target_info, True)
                    run_info["preload"]["attempted"] += 1
                    run_info["preload"]["per_model"][f"{key[0]}/{key[1]}"] = {
                        "status": "running",
                        "timeout": resolve_preload_timeout(source_config, target_info["source"]),
                    }
                else:
                    preload_owner = False
            if not preload_owner:
                inflight.wait()
                with preload_lock:
                    return key in preloaded_ok
            started = time.time()
            timeout_limit = resolve_preload_timeout(source_config, target_info["source"])
            raw_cfg = raw_targets.get(model_name)
            drop_params = raw_cfg.get("drop_params", []) if isinstance(raw_cfg, dict) else []
            log_path = None
            result = None
            try:
                if args.save_responses:
                    preload_logs = os.path.join(output_dir, "logs")
                    os.makedirs(preload_logs, exist_ok=True)
                    log_path = os.path.join(preload_logs, "preload.log")
                result = preload_model(
                    source_config,
                    target_info["source"],
                    target_info["api_model"],
                    timeout_limit,
                    session_seed=session_seed,
                    stop_event=stop_event,
                    drop_params=drop_params,
                    log_path=log_path,
                )
            except Exception as exc:  # noqa: BLE001 - any preload failure is recorded as such
                result = PreloadResult(
                    success=False,
                    elapsed=round(time.time() - started, 1),
                    error=f"{type(exc).__name__}: {exc}",
                )
            try:
                elapsed = result.elapsed if result.elapsed is not None else round(time.time() - started, 1)
                model_key = f"{key[0]}/{key[1]}"
                with preload_lock:
                    run_info["preload"]["total_preload_time"] += elapsed
                    run_info["preload"]["per_model"][model_key] = {
                        "status": "ok" if result.success else "failed",
                        "timeout": timeout_limit,
                        "time": elapsed,
                    }
                    if result.success:
                        preloaded_ok.add(key)
                        run_info["preload"]["succeeded"] += 1
                        for state_key in (model_name, f"{model_name} [opencode]") if runner_mode == "both" else (model_name,):
                            if state_key in state.snapshot():
                                state.update(
                                    state_key, preloading=False, preload_start_ts=0,
                                    preload_status="ok", preload_time=elapsed,
                                    preload_error=None,
                                )
                    else:
                        preload_failed.add(key)
                        run_info["preload"]["failed"] += 1
                        _record_preload_failure(model_name, target_info, result, phase_runner)
                    return result.success
            except Exception as exc:  # noqa: BLE001 - release waiters with a deterministic failed preload
                # Probe execution already returned, but bookkeeping can still
                # fail (for example, a state update or diagnostic write). Do
                # not leave a successful cache entry or an unclassified
                # in-flight key behind: all later targets must see a stable
                # failed result for this source/model pair.
                failure = PreloadResult(
                    success=False,
                    elapsed=round(time.time() - started, 1),
                    error=f"preload bookkeeping failed: {type(exc).__name__}: {exc}",
                )
                with preload_lock:
                    preloaded_ok.discard(key)
                    preload_failed.add(key)
                    try:
                        run_info["preload"]["failed"] += 1
                        run_info["preload"]["per_model"][f"{key[0]}/{key[1]}"] = {
                            "status": "failed",
                            "timeout": timeout_limit,
                            "time": failure.elapsed,
                            "error": failure.error,
                        }
                    except (KeyError, TypeError):
                        pass
                try:
                    _record_preload_failure(model_name, target_info, failure, phase_runner)
                except Exception as record_exc:  # noqa: BLE001 - preserve the original preload failure
                    print(
                        f"⚠️  Could not record preload failure for {model_name}: {record_exc}",
                        file=sys.stderr,
                    )
                return False
            finally:
                with preload_lock:
                    preload_inflight.pop(key, None)
                    inflight.set()

        judge_input_dir = os.path.join(output_dir, "judge-inputs") if judge_models else None
        judge_sources = {name: targets[name]["source"] for name in judge_models}
        judge_source_limits = {
            source: max(1, int(model_thread_limits.get(source, 1)))
            for source in set(judge_sources.values())
        }
        judge_pools = {}
        judge_seen = set()
        judge_seen_lock = threading.Lock()
        judge_counts_lock = threading.Lock()
        judge_votes = {}
        judge_votes_lock = threading.Lock()
        existing_judge_counts = {model: 0 for model in judge_models}
        for result in state.latest_results():
            for plugin in active_plugins:
                votes = result.get(f"{plugin.id}_judge_votes", [])
                judged_models = {
                    vote.get("model") for vote in votes if isinstance(vote, dict)
                }
                for model in existing_judge_counts:
                    if model in judged_models:
                        existing_judge_counts[model] += 1
        state.set_judge_progress({
            model: {"completed": existing_judge_counts[model], "expected": 0}
            for model in judge_models
        })
        judge_effective_timeout = (cfg.get("judge", {}).get("timeout", timeout)
                                   if isinstance(cfg.get("judge"), dict) else timeout)
        judge_token_levels = (cfg.get("judge", {}).get("token_levels", [1024])
                              if isinstance(cfg.get("judge"), dict) else [1024])
        judge_temperature = (cfg.get("judge", {}).get("temperature", 0.0)
                             if isinstance(cfg.get("judge"), dict) else 0.0)

        def enqueue_judge(sidecar, target_name, runner, plugin_id):
            latest = {
                (result.get("state_key", result.get("model")), result.get("runner", "http")): result
                for result in state.latest_results()
            }
            try:
                with open(sidecar, encoding="utf-8") as handle:
                    item = json.load(handle)
            except (OSError, json.JSONDecodeError, TypeError):
                # The sidecar is durable best-effort input. A producer may
                # have failed between creating the path and completing its
                # atomic replace; skip it here and let startup/final scans
                # retry rather than failing the benchmark worker.
                return
            state_key = item.get("state_key", target_name)
            result = latest.get((state_key, runner), {})
            info = state.snapshot().get(state_key, {})
            score = result.get(f"{plugin_id}_score")
            if not (isinstance(score, (int, float)) and not isinstance(score, bool)):
                score = info.get(f"{plugin_id}_score")
            if not (isinstance(score, (int, float)) and not isinstance(score, bool)):
                return
            result_votes = result.get(f"{plugin_id}_judge_votes", []) or []
            info_votes = info.get(f"{plugin_id}_judge_votes", []) or []
            votes_by_model = {
                vote.get("model"): vote
                for vote in [*result_votes, *info_votes]
                if isinstance(vote, dict) and vote.get("model")
            }
            judged_models = set(votes_by_model)
            for judge_name in judge_models:
                if judge_name in judged_models:
                    continue
                source = judge_sources[judge_name]
                key = (os.path.abspath(sidecar), target_name, runner, plugin_id, judge_name)
                with judge_seen_lock:
                    if key in judge_seen:
                        continue
                    judge_seen.add(key)
                judge_pools[source].enqueue(
                    (sidecar, target_name, runner, plugin_id, judge_name)
                )
                state.update(state_key, **{f"{plugin_id}_judge_queued": True})
                state.increment_judge_progress(judge_name, expected=1)
                with judge_counts_lock:
                    run_info["judge_counts"]["queued"] += 1

        def process_judge_job(job):
            """Judge one sidecar with every configured judge and persist consensus."""
            sidecar, target_name, runner, plugin_id, judge_name = job
            item = {}
            try:
                with open(sidecar, encoding="utf-8") as handle:
                    item = json.load(handle)
                latest = {
                    (result.get("state_key", result.get("model")), result.get("runner", "http")): result
                    for result in state.latest_results()
                }.get((item.get("state_key", target_name), runner), {})
                state_key = item.get("state_key", target_name)
                live_info = state.snapshot().get(state_key, {})
                vote_key = f"{plugin_id}_judge_votes"
                existing_by_model = {
                    vote.get("model"): vote
                    for vote in [
                        *(latest.get(vote_key, []) or []),
                        *(live_info.get(vote_key, []) or []),
                    ]
                    if isinstance(vote, dict) and vote.get("model")
                }
                existing_votes = list(existing_by_model.values())
                if any(
                    vote.get("model") == judge_name
                    for vote in existing_votes
                    if isinstance(vote, dict)
                ):
                    state.increment_judge_progress(judge_name, completed=1)
                    return
                activity_id = state.start_judge_activity(
                    judge_name, target_name, plugin_id,
                )
                outcome = None
                try:
                    outcome = judge_response(
                        source_config,
                        judge_sources[judge_name],
                        targets[judge_name]["api_model"],
                        sidecar,
                        timeout=judge_effective_timeout,
                        token_levels=judge_token_levels,
                        temperature=judge_temperature,
                        drop_params=(raw_targets.get(judge_name, {}).get("drop_params", [])
                                     if isinstance(raw_targets.get(judge_name), dict) else []),
                        stop_event=stop_event,
                        log_path=os.path.join(output_dir, f"judge-{judge_name}.log"),
                    )
                finally:
                    if outcome is not None and outcome.response_text is not None:
                        state.update_judge_activity(
                            activity_id,
                            tokens=len(outcome.response_text) // 4,
                        )
                    state.finish_judge_activity(activity_id)
                vote = {
                    "model": judge_name,
                    "score": outcome.score,
                    "confidence": outcome.confidence,
                    "rationale": outcome.rationale,
                    "error": outcome.error,
                }
                if outcome.response_text is not None:
                    try:
                        save_judge_response(
                            output_dir, target_name, runner, plugin_id,
                            judge_name, outcome.response_text,
                        )
                    except OSError:
                        vote["error"] = (
                            vote["error"] or "could not save judge response artifact"
                        )
                vote_identity = (state_key, runner, plugin_id)
                with judge_votes_lock:
                    prior_votes = list(judge_votes.get(vote_identity, existing_votes))
                    prior_votes = [v for v in prior_votes if v.get("model") != judge_name]
                    prior_votes.append(vote)
                    judge_votes[vote_identity] = prior_votes
                    votes = list(prior_votes)
                consensus = confidence_weighted_consensus(votes)
                expected_judges = set(judge_models)
                received_judges = {
                    vote.get("model") for vote in votes if isinstance(vote, dict)
                }
                all_judges_finished = expected_judges.issubset(received_judges)
                judge_status = (
                    "failed" if all_judges_finished and consensus["error"]
                    and not any(vote.get("score") is not None for vote in votes)
                    else "partial" if all_judges_finished and any(vote.get("error") for vote in votes)
                    else "complete" if all_judges_finished else "running"
                )
                state.update_judge_result(
                    state_key, runner, plugin_id,
                    score=consensus["score"],
                    confidence=consensus["confidence"],
                    rationale=consensus["rationale"],
                    error=consensus["error"],
                    input_sha256=item.get("response_sha256"),
                    votes=votes,
                    status=judge_status,
                    complete=all_judges_finished,
                )
                state.increment_judge_progress(judge_name, completed=1)
                with judge_counts_lock:
                    if consensus["error"]:
                        run_info["judge_counts"]["failed"] += 1
                    else:
                        run_info["judge_counts"]["completed"] += 1
                    run_info["judge_counts"]["votes"] += 1
                with persistence_lock:
                    state.save_state(state_file, plugin_versions=plugin_versions)
                    _save_outputs(state, output_dir, active_plugins)
            except Exception as exc:  # noqa: BLE001 - isolate one judge job failure
                state_key = item.get("state_key", target_name)
                failure_vote = {
                    "model": judge_name,
                    "score": None,
                    "confidence": None,
                    "rationale": None,
                    "error": f"judge input failed: {type(exc).__name__}: {exc}",
                }
                vote_identity = (state_key, runner, plugin_id)
                with judge_votes_lock:
                    prior_votes = list(judge_votes.get(vote_identity, []))
                    prior_votes = [v for v in prior_votes if v.get("model") != judge_name]
                    prior_votes.append(failure_vote)
                    judge_votes[vote_identity] = prior_votes
                received_judges = {
                    vote.get("model") for vote in prior_votes if isinstance(vote, dict)
                }
                all_judges_finished = set(judge_models).issubset(received_judges)
                state.update_judge_result(
                    state_key, runner, plugin_id,
                    error=failure_vote["error"],
                    votes=prior_votes,
                    status="failed" if all_judges_finished else "running",
                    complete=all_judges_finished,
                )
                with judge_counts_lock:
                    run_info["judge_counts"]["failed"] += 1
                state.increment_judge_progress(judge_name, completed=1)
                with persistence_lock:
                    state.save_state(state_file, plugin_versions=plugin_versions)
                    _save_outputs(state, output_dir, active_plugins)

        judge_pools = {
            source: SourceJudgeWorkerPool(
                source, judge_source_limits[source], process_judge_job, stop_event,
            )
            for source in set(judge_sources.values())
        }

        # Queue retained results before benchmark workers start. On resume,
        # completed targets are deliberately absent from the benchmark queues,
        # but their durable sidecars still need judging immediately.
        for sidecar, item in _eligible_judge_sidecars(
            judge_input_dir, targets, state, {plugin.id for plugin in active_plugins},
            judge_models,
        ):
            enqueue_judge(sidecar, item["target"], item["runner"], item["plugin"])

        def source_benchmark_complete(source):
            """Release the source's reserved benchmark slots to judging."""
            pool = judge_pools.get(source)
            if pool is not None:
                pool.expand_full()

        def start_judge_if_async(benchmark_limits, benchmark_sources=None):
            """Reserve one judge slot, then expand after source completion.

            While a source is benchmarking, one judge worker is allowed only
            when another slot remains available. Once that source drains, all
            of its configured model slots become judge workers, including for a
            single configured judge model.

            """
            if not judge_models:
                return
            benchmark_sources = benchmark_sources or set()
            for source, pool in judge_pools.items():
                _configure_judge_source(
                    benchmark_limits,
                    source,
                    judge_source_limits[source],
                    source in benchmark_sources,
                    pool,
                )

        def stop_judge_workers():
            """Stop and join all source-local judge workers."""
            for pool in judge_pools.values():
                pool.stop(timeout=1.0)

        def finish_judge():
            """Drain retained judge jobs and join source-local workers."""
            if not judge_models:
                return
            jobs = _eligible_judge_sidecars(
                judge_input_dir, targets, state, {plugin.id for plugin in active_plugins},
                judge_models,
            )
            for sidecar, item in jobs:
                enqueue_judge(sidecar, item["target"], item["runner"], item["plugin"])
            for source, pool in judge_pools.items():
                # A source with no active benchmark scheduler should still
                # receive its full judge pool before the final drain.
                pool.start(judge_source_limits[source])
            stop_judge_workers()
            if judge_models:
                run_info["judge_status"] = "complete"

        def run_target(model_name, phase_runner):
            """Run one target through one runner and persist its progress."""
            nonlocal worker_errors
            model_blacklist = get_target_plugins_blacklist(raw_targets, model_name)
            model_active_plugins = [p for p in active_plugins if p.id not in model_blacklist]
            target_info = targets[model_name]
            if not _ensure_preloaded(model_name, target_info, phase_runner):
                with persistence_lock:
                    state.save_state(state_file, plugin_versions=plugin_versions)
                    _save_outputs(state, output_dir, active_plugins)
                return False
            state_key = model_name if phase_runner == "http" else f"{model_name} [opencode]"
            phase_output_dir = http_output_dir if phase_runner == "http" else opencode_output_dir
            mapped = None
            agent_id = None
            if phase_runner == "opencode":
                mapped = opencode_model_name(target_info["source"], target_info["api_model"])
                agent_id = opencode_agent_ids.get(model_name)
            # Per-target ``token_levels`` (model/agent dict or the
            # ``model_token_levels`` map, resolved by ``resolve_targets``)
            # beat the global config/CLI value for this target's legs.
            effective_token_levels = target_info.get("token_levels") or token_levels
            run_model(state_key, target_info["source"], state, model_active_plugins,
                      source_config, timeout, effective_token_levels, phase_output_dir,
                      session_seed=session_seed, global_cfg=cfg, stop_event=stop_event,
                      save_responses=args.save_responses,
                      judge_input_dir=judge_input_dir,
                      judge_enqueue=enqueue_judge if judge_models else None,
                      judge_model=judge_model,
                      judge_models=judge_models,
                      judge_prompt_version="judge-v1" if judge_models else None,
                      api_model=target_info["api_model"],
                      system_prompt=target_info["system_prompt"],
                      is_agent=target_info["is_agent"], runner=phase_runner,
                      opencode_config_path=opencode_config_path,
                      opencode_model=mapped, opencode_agent=agent_id,
                      opencode_binary=opencode_binary,
                      display_name=model_name, config_target_name=model_name)
            # OpenCode and HTTP pipeline workers can finish different targets
            # concurrently. Serialize persistence because save_state uses a
            # shared .tmp path and report generation writes shared output
            # files; the in-memory BenchmarkState itself remains thread-safe.
            with persistence_lock:
                state.save_state(state_file, plugin_versions=plugin_versions)
                _save_outputs(state, output_dir, active_plugins)

        peak_lock = threading.Lock()

        def record_peak(source, active):
            with peak_lock:
                run_info["peak_active_models"][source] = max(
                    run_info["peak_active_models"].get(source, 0), active
                )

        def on_worker_error(model_name, phase_runner, exc):
            nonlocal worker_errors
            with errors_lock:
                worker_errors += 1
            print(f"\\n❌ Worker exception ({model_name}, {phase_runner}): "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)

        if runner_mode == "both":
            # Pipeline each source independently with one execution slot:
            # each target runs OpenCode then HTTP before the source advances
            # to its next target. Sources still run concurrently, but there
            # is never an OpenCode/HTTP overlap within one source.
            snapshot = state.snapshot()
            targets_by_source, opencode_pending, http_pending = _build_runner_queues(
                targets, snapshot, runner_mode, source_config,
                rerun_failed=not args.no_rerun_failed,
            )
            benchmark_limits = dict(model_thread_limits)
            benchmark_sources = {
                source for source, names in targets_by_source.items() if names
            }
            start_judge_if_async(benchmark_limits, benchmark_sources)

            pipeline_threads = _start_runner_pipeline(
                targets_by_source, opencode_pending, http_pending,
                run_target, stop_event, on_worker_error,
                model_thread_limits=benchmark_limits,
                peak_callback=record_peak,
                source_complete_callback=source_benchmark_complete,
            )
            if pipeline_threads:
                try:
                    for thread in pipeline_threads:
                        thread.join()
                except KeyboardInterrupt:
                    interrupted = True
                    run_info["status"] = "interrupted"
                    stop_event.set()
                    print("\\n\\n⚠️  Ctrl+C — saving state and shutting down...", file=sys.stderr)
                    close_active_requests()
                    stop_judge_workers()
                    for thread in pipeline_threads:
                        thread.join(timeout=1.0)
        else:
            # Preserve the original single-runner source workers. Only
            # --runner both needs cross-runner coordination.
            phase_runner = runner_mode
            snapshot = state.snapshot()
            source_queues = _build_runner_queues(
                targets, snapshot, runner_mode, source_config,
                rerun_failed=not args.no_rerun_failed,
            )
            benchmark_limits = dict(model_thread_limits)
            benchmark_sources = {
                source for source, names in source_queues.items() if names
            }
            start_judge_if_async(benchmark_limits, benchmark_sources)

            source_threads = {}
            for source, model_names in source_queues.items():
                if not model_names:
                    continue

                def worker(source=source, model_names=model_names):
                    def run_one(model_name):
                        try:
                            run_target(model_name, phase_runner)
                        except Exception as exc:  # noqa: BLE001 - a worker crash is recorded per model
                            on_worker_error(model_name, phase_runner, exc)
                    scheduler = SourceModelScheduler(
                        source, benchmark_limits.get(source, 1), model_names, run_one,
                        stop_event, on_worker_error, peak_callback=record_peak,
                        on_complete=source_benchmark_complete,
                    )
                    scheduler.run_until_drained()

                thread = threading.Thread(target=worker, args=(), daemon=True)
                thread.start()
                source_threads[source] = thread

            if source_threads:
                try:
                    _join_workers()
                except KeyboardInterrupt:
                    interrupted = True
                    run_info["status"] = "interrupted"
                    stop_event.set()
                    print("\\n\\n⚠️  Ctrl+C — saving state and shutting down...", file=sys.stderr)
                    close_active_requests()
                    stop_judge_workers()
                    _join_workers(timeout=1.0)

        if not interrupted:
            finish_judge()
        stop_event.set()
        # The TUI thread is a daemon, so we don't need to wait for it. A short
        # timeout keeps the terminal tidy if it happens to finish quickly.
        tui_thread.join(timeout=0.5)

        with contextlib.suppress(Exception):
            state.save_state(state_file, plugin_versions=plugin_versions)

        if interrupted:
            done = state.completed
            print(f"✅ Saved state ({done}/{total} done). Re-run without --restart to continue.\n",
                  file=sys.stderr)
        else:
            _save_outputs(state, output_dir, active_plugins)
        final_results = state.latest_results()
        ok_count = len([r for r in final_results if r["status"] == "ok"])
        print(f"\n{'='*70}")
        print(f"AI BENCHMARK COMPLETE — {ok_count}/{total} successful "
              f"({worker_errors} worker errors)")
        print(f"Outputs: {output_dir}/")
        for fname in sorted(os.listdir(output_dir)):
            print(f"  - {fname}")
        print(f"{'='*70}")
    except KeyboardInterrupt:
        run_info["status"] = "interrupted"
        raise
    except Exception as exc:
        run_info["status"] = "crashed"
        run_info["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        run_info["end_time"] = datetime.now(timezone.utc).isoformat()
        run_info["completed_targets"] = state.completed if state is not None else 0
        run_info["worker_errors"] = worker_errors
        run_info["model_thread_limit"] = dict(model_thread_limits)
        if run_info["status"] == "running":
            run_info["status"] = "completed"
        _inject_429_stats(run_info)
        _write_run_info(output_dir, run_info)


if __name__ == "__main__":
    main()
