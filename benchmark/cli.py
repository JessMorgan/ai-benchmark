#!/usr/bin/env python3
"""
AI Benchmark — Plugin-based benchmark for code generation and reasoning.
Supports arbitrary task plugins, versioned results, and plugin selection.

Configuration: edit benchmark-config.json (or pass --config <path>).
API keys can use ${VAR} or ${VAR:default} syntax for env-var expansion.
"""
import faulthandler
import glob
import json
import os
import queue
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import unicodedata
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from typing import ClassVar

from textual.app import App
from textual.geometry import Region
from textual.widgets import Static
from wcwidth import wcswidth, wcwidth

from benchmark.completions import build_parser, generate_shell_completion
from benchmark.core import (
    FLUSH_INTERVAL_SECONDS,
    FLUSH_MAX_VOTES,
    JUDGE_DEFAULT_MAX_TOKENS,
    JUDGE_PROMPT_VERSION,
    BenchmarkState,
    PreloadResult,
    _apply_http_retry_default,
    _save_outputs,
    _unique_source_abbrevs,
    confidence_weighted_consensus,
    confidence_weighted_consensus_by_contract,
    dump_default_config,
    generate_config_from_api,
    get_target_plugins_blacklist,
    is_successful_judge_vote,
    judge_contract_id,
    judge_response,
    judge_votes_for_contract,
    load_config,
    load_dotenv_file,
    merge_judge_vote,
    parse_plugin_temperatures,
    preload_model,
    resolve_judge_request_params,
    resolve_model_thread_limit,
    resolve_preload_timeout,
    resolve_targets,
    run_model,
    run_schema_sentinel,
    save_judge_response,
    save_judge_response_metadata,
    summarize_judge_criteria,
    summarize_schema_compatibility,
)
from benchmark.http import (
    close_active_requests,
    get_429_stats,
    get_active_request_count,
    reset_429_stats,
)
from benchmark.judge_analysis import write_disagreement_queue
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
                             judge_models, judge_contracts=None):
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
        expected_contract = (judge_contracts or {}).get(plugin_id)
        judged_models = {
            vote.get("model") for vote in votes
            if is_successful_judge_vote(vote)
            and vote.get("judge_contract_id") == expected_contract
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

    Width comes from the ``wcwidth`` package, which encodes the width table
    terminal emulators have converged on (East Asian Width plus emoji
    presentation). Terminals still disagree on a few symbol code points that
    ``wcwidth`` classifies as one column but that many emoji-capable
    terminals render as two (e.g. ``⚠``); those are over-counted below.

    Under-counting one of those characters lets the TUI write past the
    right edge, where the terminal may wrap it onto the next row. That wrap
    is the source of the apparent prepended/stale characters. Over-counting
    is intentional: clipping a row one column early is harmless; allowing a
    row to wrap corrupts every row below it. ASCII remains one column wide.
    """
    if char in "\r\n" or unicodedata.category(char) in {"Cc", "Cf"}:
        return 0
    width = wcwidth(char)
    if width >= 2:
        return width
    # Emoji/symbol code points in these ranges are commonly rendered in an
    # emoji presentation with two columns. Do not classify every ambiguous
    # character as wide: arrows and box-drawing glyphs are normally one
    # column in the Linux terminals this TUI targets, and over-counting them
    # would make narrow layouts needlessly lose useful text.
    codepoint = ord(char)
    if (0x2600 <= codepoint <= 0x27BF
            or 0x1F000 <= codepoint <= 0x1FAFF):
        return 2
    return max(width, 0)


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
    """Return a conservative terminal width for one grapheme cluster.

    ``wcswidth`` measures a whole cluster at once, so joined emoji (ZWJ),
    keycaps, flags, and variation-selector emoji collapse to their single
    rendered-glyph width instead of summing their code points. The
    conservative over-count for ``⚠``-style symbols is applied on top.
    """
    width = wcswidth(cluster)
    if width >= 2:
        return width
    if any(_char_display_width(char) >= 2 for char in cluster):
        return 2
    return max(width, 0)


def _display_width(text):
    """Return the terminal-column width of ``text``.

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
    """Fallback terminal UI for non-interactive terminals (no Textual TUI)."""
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


# Display width of the frozen table prefix, including the separator before
# the horizontally scrollable plugin columns. Keep this shared by the row
# formatter and the row layout calculations.
# The model-number column reserves the judge marker's display width so a
# scales/checkmark marker cannot shift source, model, or status columns.
MODEL_NUMBER_COLUMN_WIDTH = 5
FROZEN_VIEW_WIDTH = 35

# Width of the per-plugin cell block rendered by ``_plugin_cell_block``.
# The standard 4-cell results layout uses a 9-column score field plus
# 6+6+6 token/time/TPS fields and three separators = 30 display columns.
# The score field contains a bracketed judge marker when present, so the
# score, judge status, and token value remain visually distinct even when the
# emoji glyph occupies two terminal columns.
# (``RateSc RateTok RateTm RateTPS``) without reshaping the
# ``plugin_cols`` table. The previous per-plugin streaming-glyph column
# (``<id>St`` width 5) was deleted as redundant: the merged status block
# already conveys in-flight state, and post-flight the plugin isn't
# streaming anymore, so the glyph was always ``-``.
# The score column is nine display columns wide. Judge markers keep a
# leading space from the score and put the count directly after the emoji
# (for example ``63 ⚖️2``), so the marker cannot visually merge with either
# the score or the token column.
SCORE_COLUMN_WIDTH = 9
PLUGIN_BLOCK_WIDTH = 30

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


_JUDGE_SCALES = "⚖️"


def _judge_models(state):
    """Return the configured judge identities represented by a state row."""
    configured = state.get("judge_models")
    if isinstance(configured, (list, tuple, set)):
        return {model for model in configured if model}
    return set()


def _judge_votes(state, pid):
    """Return configured judge identities with successful usable votes."""
    configured = _judge_models(state)
    if not configured:
        return set()
    return {
        vote.get("model")
        for vote in (state.get(f"{pid}_judge_votes") or [])
        if is_successful_judge_vote(vote)
        and vote.get("model") in configured
        and (
            state.get(f"{pid}_judge_selected_contract") is None
            or vote.get("judge_contract_id") == state.get(f"{pid}_judge_selected_contract")
        )
    }


def _judge_failed_count(pid, state):
    """Return distinct configured judges with failed attempts."""
    configured = _judge_models(state)
    return len({
        vote.get("model")
        for vote in (state.get(f"{pid}_judge_votes") or [])
        if isinstance(vote, dict)
        and vote.get("model") in configured
        and (
            state.get(f"{pid}_judge_selected_contract") is None
            or vote.get("judge_contract_id") == state.get(f"{pid}_judge_selected_contract")
        )
        and not is_successful_judge_vote(vote)
    })


def _judge_marker_parts(pid, state):
    """Return the compact ``(scale, fail)`` judge-marker segments.

    No leading or inter-segment spaces are included: the segments are glued
    directly to the score digits (``95⚖️1❌2``) so the score column stays
    tight. ``scale`` is the scales emoji plus the successful-judge count
    (``⚖️1``) for partial judging, ``✅`` for full completion, or ``""``;
    ``fail`` is the red-x plus the failed-judge count (``❌2``) or ``""``.
    Completion is derived from the current configured judge set and recorded
    votes, not a stale aggregate flag.
    """
    configured = _judge_models(state)
    if not configured:
        return "", ""
    judged_models = _judge_votes(state, pid)
    failed_count = _judge_failed_count(pid, state)
    if configured.issubset(judged_models):
        return "✅", ""
    scale = f"{_JUDGE_SCALES}{len(judged_models)}" if judged_models else ""
    fail = f"❌{failed_count}" if failed_count else ""
    return scale, fail


def _judge_score_marker(pid, state):
    """Return the space-free compact judge status for one plugin score.

    Partial judging is rendered as ``⚖️1`` / ``⚖️2``, full completion as
    ``✅``, failures as ``❌2``, and combinations as ``⚖️1❌3`` -- all with
    no spaces, so the marker clings to the score digits and cannot merge
    with either the score or the token column. Callers that need to line
    up like emojis across rows should use :func:`_judge_marker_parts` and
    the per-column slot widths from :func:`_plugin_judge_alignment`.
    """
    scale, fail = _judge_marker_parts(pid, state)
    return scale + fail


def _plugin_judge_alignment(snap_rows, pid):
    """Return per-column marker slot widths for one plugin, or ``None``.

    Scans every model row in the snapshot to find the widest score digits
    (``score_w``), the widest scale segment (``scale_w``, includes ``✅``),
    and the widest fail segment (``fail_w``) for ``pid``. When any marker
    exists and ``score_w + scale_w + fail_w`` fits the 9-column score
    budget, the caller renders every row in that column with these fixed
    slot widths so like-emojis (the scales markers, then the red-x
    markers) line up evenly across rows; ``None`` means the column has no
    judging data (or the slots would overflow the budget, in which case
    the compact no-space layout is kept instead).
    """
    score_w = scale_w = fail_w = 0
    for s in snap_rows:
        score = s.get(f"{pid}_score")
        if not (isinstance(score, (int, float)) and not isinstance(score, bool)):
            continue
        score_w = max(score_w, len(f"{score:.0f}"))
        scale, fail = _judge_marker_parts(pid, s)
        scale_w = max(scale_w, _display_width(scale) if scale else 0)
        fail_w = max(fail_w, _display_width(fail) if fail else 0)
    if scale_w == 0 and fail_w == 0:
        return None
    if score_w + scale_w + fail_w > SCORE_COLUMN_WIDTH:
        return None
    return score_w, scale_w, fail_w


def _model_judge_marker(state, active_plugins=None, active_judge_targets=None,
                        target_name=None):
    """Return the row-header marker for a model's aggregate judge state.

    The scales marker is transient: it is shown only while a judge request
    currently targets this model. Queued work and historical/partial votes
    belong in the per-score-cell markers, not in the row header. A completed
    model retains the checkmark when all of its scored plugins have votes from
    every configured judge.
    """
    active_plugins = active_plugins or []
    active_judge_targets = active_judge_targets or set()
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
    if target_name in active_judge_targets:
        return _JUDGE_SCALES
    if all(configured.issubset(_judge_votes(state, pid)) for pid in scored):
        return "✅"
    return ""


def _plugin_cell_block(pid, s, p, sleeping_lookup=None, judge_slots=None):
    """Render a single per-model cell block for one plugin.

    The block is always ``PLUGIN_BLOCK_WIDTH`` display columns wide so it can
    be dropped into ``plugin_str`` in place of the existing results layout.
    The standard table has four sub-headers per plugin (``RateSc RateTok
    RateTm RateTPS``), and the 30-column cell block matches that geometry.
    Judge markers are rendered with no spaces at all (the score digits
    directly abut ``⚖️N``/``❌N``, e.g. ``95⚖️1❌2``) so the score column
    stays tight; when ``judge_slots`` is supplied (see
    :func:`_plugin_judge_alignment`), like-emojis line up across rows in
    the same plugin column because every cell uses the same fixed segment
    widths for the score, scales/checkmark, and red-x parts.


    When the plugin is in flight (``pid in running_pids``) OR
    the model is currently in a 429 backoff sleep, the block collapses
    to a single bracket-delimited status centred in the fixed-width cell:
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
    # Standard 4-cell results layout -- widths sum to 9+6+6+6=27 with 3
    # single-space separators between cells = 30 display columns, matching
    # merged status width. The token cell shows the TOTAL (thinking +
    # content) count; the per-kind split is exposed in the CSV/MD/HTML/PDF
    # reports. Falls back to the legacy content-only count for state files
    # that predate the thinking/content split.
    score = s.get(f"{pid}_score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        scale, fail = _judge_marker_parts(pid, s)
        if judge_slots is not None:
            # Per-column fixed slots so like-emojis line up across rows:
            # the score is right-justified, the scales/checkmark segment
            # occupies ``scale_w`` columns, and the red-x segment occupies
            # ``fail_w`` columns. Missing segments are padded with spaces
            # (e.g. ``95⚖️1❌3`` next to ``100   ❌3``), so every row in the
            # column puts the same emoji at the same column. No spaces are
            # emitted between the score digits and the markers.
            score_w, scale_w, fail_w = judge_slots
            sc = f"{score:>{score_w}.0f}"
            if scale_w:
                sc += _pad_display_width(scale if scale else "", scale_w)
            if fail_w:
                sc += _pad_display_width(fail if fail else "", fail_w)
        else:
            sc = f"{score:.0f}{scale}{fail}"
    else:
        sc = _fmt_value(score, ".0f")
    total_tokens = s.get(f"{pid}_total_tokens")
    tok = _fmt_value(
        total_tokens if total_tokens is not None else s.get(f"{pid}_output_tokens"),
        "d",
    )
    tm = _fmt_value(s.get(f"{pid}_response_time"))
    tps = _fmt_value(s.get(f"{pid}_tps"))
    score_field = " " * max(0, SCORE_COLUMN_WIDTH - _display_width(sc)) + sc
    block = f"{score_field} {tok:>6} {tm:>6} {tps:>6}"
    return _pad_display_width(block, PLUGIN_BLOCK_WIDTH)


def _format_model_row(name, s, display_idx, active_plugins, source_abbrevs,
                      sleeping_lookup=None, active_judge_targets=None,
                      judge_slots=None):
    """Format a single model row into frozen and plugin strings.

    ``sleeping_lookup`` maps ``(source, api_model, pid)`` to sleep info
    (with ``wake_ts``, ``attempts``, ``max_attempts``) so that a plugin
    cell can render its own per-plugin ``[429 sleeping Xs]`` bracket.
    Only the plugin that is actually in backoff is shown as sleeping;
    completed plugins keep their numeric results.

    ``judge_slots`` is a mapping ``{pid: (score_w, scale_w, fail_w)}``
    produced by :func:`_plugin_judge_alignment` over the full snapshot so
    every row in a plugin column renders its judge markers with the same
    fixed segment widths (like-emojis line up). Rows without judge data
    still occupy the same per-cell width, keeping the score digit column
    consistent.
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
    # Activity uses the plain target name for both runners; OpenCode state
    # rows carry a `` [opencode]`` suffix.
    judge_marker = _model_judge_marker(
        s,
        active_plugins,
        active_judge_targets,
        name.removesuffix(" [opencode]"),
    )
    model_number = _pad_display_width(
        f"{display_idx:>3}{judge_marker}", MODEL_NUMBER_COLUMN_WIDTH
    )
    frozen = f"{model_number} {src_ab:<3} {model_disp:<18}  {status_ch:<3}"
    # Keep the plugin viewport anchored at the same terminal column as the
    # heading. Emoji status glyphs can occupy two columns, so Python's string
    # length is not sufficient to align this frozen prefix.
    frozen = _pad_display_width(
        _truncate_display_width(frozen, FROZEN_VIEW_WIDTH - 1),
        FROZEN_VIEW_WIDTH - 1,
    )

    # Each plugin contributes exactly one fixed-width block (merged status
    # OR standard 5-cell results) so ``plugin_str`` has the same total
    # length and column geometry as before. Joins the per-plugin blocks
    # with single spaces, matching the existing column-join pattern.
    plugin_parts = [
        _plugin_cell_block(
            p.id, s, p,
            sleeping_lookup=sleeping_lookup,
            judge_slots=(judge_slots or {}).get(p.id),
        )
        for p in active_plugins
    ]
    plugin_str = " ".join(plugin_parts)
    return frozen, plugin_str


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


# ---------------------------------------------------------------------------
# Textual live TUI (primary renderer). A Textual App re-renders the frame
# adaptively via a timer capped at ``_TUI_REFRESH_SECONDS`` (2 fps): the
# frame is rebuilt only when the state revision changed, the terminal
# resized, the operator scrolled, or live elapsed/countdown content is
# ticking, delegating resize, keyboard input, and screen writes to Textual
# so stale or partially-drawn rows cannot accumulate. Non-interactive
# terminals fall back to the plain-text loop above.
# ---------------------------------------------------------------------------

# Maximum refresh rate for the live TUI (2 fps). The frame is only rebuilt
# when something it displays actually changed (state revision, resize,
# scroll, or a live elapsed/countdown tick), so an idle run costs no CPU.
_TUI_REFRESH_SECONDS = 0.5

# Style names emitted by ``_build_frame_lines`` and mapped to Rich styles by
# ``_frame_lines_to_text``. Plain strings keep the frame builder headlessly
# testable without importing Rich/Textual.
_FRAME_STYLE_MAP = {
    "bold": "bold",
    "bold underline": "bold underline",
    "green": "green",
    "red": "red",
    "yellow": "yellow",
    # Rich has no plain "grey"/"gray" color name; a truecolor value
    # renders literally as dimmed grey in every terminal.
    "grey": "#808080",
}


def _build_frame_lines(state, active_plugins, source_abbrevs, frozen_hdr,
                       plugin_hdr, num_sources, scroll_y, scroll_x, size,
                       model_thread_limits=None, session_seed=None):
    """Build one full TUI frame as ``(text, style)`` line pairs.

    Headlessly testable: ``text`` is already truncated to the terminal width
    and ``style`` is a ``_FRAME_STYLE_MAP`` key or ``None`` for the default
    style.
    """
    max_y, max_x = size
    snap = state.snapshot()
    snap_items = list(snap.items())
    done = state.completed
    total = state.total
    running = [n for n, s in snap.items() if s.get("running_pids")]
    preloading = [n for n, s in snap.items() if s.get("preloading")]
    queued = [n for n, s in snap.items() if s["status"] == "queued" and not s.get("preloading")]
    pending = [n for n, s in snap.items() if s["status"] == "pending" and not s.get("preloading")]
    http_threads = get_active_request_count()
    backoff_429 = get_429_stats()
    sleeping_lookup = _build_sleeping_lookup(backoff_429)
    sleeping_model_count = len({(src, model) for (src, model, _pid) in sleeping_lookup})
    judge_activities = state.judge_activity_snapshot()
    active_judge_targets = {activity["target"] for activity in judge_activities}
    selected_snapshot = getattr(state, "judge_selected_snapshot", None)
    selected_judges = (
        set(selected_snapshot() or ()) if callable(selected_snapshot) else set()
    )
    active_judge_names = selected_judges | {
        activity["judge"] for activity in judge_activities
    }
    judge_progress = state.judge_progress_snapshot()

    live_height = max(3, num_sources + 1)
    visible_rows = max(0, max_y - 9 - live_height)
    frozen_width = FROZEN_VIEW_WIDTH

    def line(text, style=None):
        return (_truncate_display_width(text, max_x), style)

    lines = []

    ts = datetime.now(timezone.utc).astimezone().strftime('%H:%M:%S')
    seed_info = f"Seed: {session_seed}  |  " if session_seed is not None else ""
    lines.append(line(f"AI Benchmark \u2014 Parallel  |  {seed_info}{ts}", "bold"))
    failed_count = sum(1 for s in snap.values() if s["status"] == "failed")
    err_indicator = f"  |  \u26a0 {failed_count} failed" if failed_count else ""
    source_active = _active_source_target_counts(snap)
    slot_text = ""
    if model_thread_limits:
        slot_text = "  |  " + ", ".join(
            f"{source}: models {source_active.get(source, 0)}/{limit}"
            for source, limit in model_thread_limits.items()
        )
    summary = (
        f"Total: {total}  |  Done: {done}  |  Active: {len(running)}  |  "
        f"Queued: {len(queued + pending)}  |  HTTP: {http_threads}  |  "
        f"429\u23f8 {sleeping_model_count}{err_indicator}  |  "
        f"\u2191\u2193 rows {scroll_y + 1}-{min(total, scroll_y + visible_rows)}/{total}"
        f"  |  \u2190\u2192 cols{slot_text}"
    )
    if max_y > 1:
        lines.append(line(summary))
    if max_y > 2:
        lines.append(line("\u2500" * min(max_x, 80)))

    if max_y > 3:
        visible_plugin_hdr = _slice_display_width(
            plugin_hdr, scroll_x, max(0, max_x - frozen_width - 1)
        )
        lines.append(line(frozen_hdr + " " + visible_plugin_hdr, "bold underline"))
    # Per-plugin-column judge marker slots, computed once over the FULL
    # snapshot (not just the visible slice) so like-emojis line up in the
    # same columns regardless of scroll position.
    judge_slots = {}
    for plugin in active_plugins:
        slots = _plugin_judge_alignment(snap.values(), plugin.id)
        if slots is not None:
            judge_slots[plugin.id] = slots
    for row_idx in range(visible_rows):
        abs_idx = scroll_y + row_idx
        if abs_idx >= len(snap_items):
            break
        name, s = snap_items[abs_idx]
        frozen, plugin_str = _format_model_row(
            name, s, abs_idx + 1, active_plugins, source_abbrevs,
            sleeping_lookup=sleeping_lookup,
            active_judge_targets=active_judge_targets,
            judge_slots=judge_slots,
        )
        visible_plugin = _slice_display_width(
            plugin_str, scroll_x, max(0, max_x - frozen_width - 1)
        )
        row_line = frozen + " " + visible_plugin
        sv = s["status"]
        if sv == "completed":
            style = "green"
        elif sv == "failed":
            style = "red"
        elif sv == "running" or s.get("running_pids"):
            style = "yellow"
        else:
            style = None
        lines.append(line(row_line, style))
    lines.append(line("\u2500" * min(max_x, 60)))

    live_lines = [("Live:", "bold")]
    for nm, s in ((nm, snap.get(nm) or {}) for nm in running):
        if len(live_lines) >= live_height:
            break
        src_ab = _source_abbr(source_abbrevs, s.get("source"))
        err = s.get("last_error", "")
        msg = f" \U0001f537 [{src_ab}] {nm[:36]}"
        indicators = _build_live_indicators(s, active_plugins)
        if indicators:
            msg += "  " + indicators
        if err:
            msg += f"  {err}"
        live_lines.append((msg, None))
    for nm in preloading:
        if len(live_lines) >= live_height:
            break
        s = snap.get(nm) or {}
        src_ab = _source_abbr(source_abbrevs, s.get("source"))
        elapsed = int(max(0, time.monotonic() - (s.get("preload_start_ts") or time.monotonic())))
        live_lines.append((f" \U0001f504 [{src_ab}] Preloading model {nm[:36]} {elapsed}s", None))
    judge_groups = {}
    for activity in judge_activities:
        judge_groups.setdefault(activity["judge"], []).append(activity)
    for judge, activities in judge_groups.items():
        if len(live_lines) >= live_height:
            break
        cells = " ".join(
            f"[{activity['target']} {activity['plugin']} {activity['elapsed']}s "
            f"thinking={activity.get('thinking_tokens', 0)} "
            f"content={activity.get('content_tokens', 0)}]"
            for activity in activities
        )
        prog = judge_progress.get(judge) or {}
        progress_str = ""
        if prog:
            progress_str = (
                f" {prog.get('completed', 0)}\u2705{prog.get('failed', 0)}\u274c"
                f"{prog.get('expected', 0)}\u03a3"
            )
        live_lines.append((f" {_JUDGE_SCALES} Judge {judge}{progress_str} {cells}", None))
    if sleeping_lookup:
        if len(live_lines) < live_height:
            live_lines.append(("429 Sleeping:", "bold"))
        for (src_name, api_model, pid), info in sleeping_lookup.items():
            if len(live_lines) >= live_height:
                break
            src_ab = _source_abbr(source_abbrevs, src_name)
            remaining = max(0, round(info["wake_ts"] - time.time()))
            live_lines.append((
                f" \U0001f4a4 [{src_ab}] {api_model[:36]} ({pid}) "
                + f"[429 {info['attempts']}/{info['max_attempts']} {remaining}s]",
                None,
            ))
    for text, style in live_lines:
        lines.append(line(text, style))

    recent_errors = state.recent_log(2)
    if recent_errors:
        lines.append(line("Errors:", "bold"))
        for ts_entry, model_entry, msg_entry in recent_errors[:3]:
            t_str = datetime.fromtimestamp(ts_entry, tz=timezone.utc).astimezone().strftime('%H:%M:%S')
            lines.append(line(f"  {t_str} [{model_entry[:20]}]: {msg_entry}", "red"))

    queuing = queued + pending
    preload_details = [
        (name, max(0.0, time.monotonic() - (snap[name].get("preload_start_ts") or time.monotonic())))
        for name in preloading
        if name in snap
    ]
    running_judges = active_judge_names
    running_parts, waiting_parts, complete_parts, stopped_parts = [], [], [], []
    for model, values in judge_progress.items():
        part = (
            f"[{model}: {values.get('completed', 0)}\u2705{values.get('failed', 0)}\u274c"
            f"{values.get('expected', 0)}\u03a3]"
        )
        expected = values.get("expected", 0)
        if values.get("stopped"):
            stopped_parts.append(part)
        elif model in running_judges:
            running_parts.append(part)
        elif (expected > 0
              and values.get("completed", 0) + values.get("failed", 0) >= expected):
            complete_parts.append(part)
        else:
            waiting_parts.append(part)
    judge_line = f"Judging {' '.join(waiting_parts)}" if waiting_parts else ""
    if (not running and not queuing and not preloading
            and not judge_line and not active_judge_names):
        lines.append(line(" All models complete \u2014 generating outputs..."))
    else:
        parts = []
        if running:
            parts.append(f"{len(running)} active")
        if preload_details:
            parts.extend(
                f"Preloading {name[:30]} {seconds:.0f}s"
                for name, seconds in preload_details
            )
        elif preloading:
            parts.append(f"{len(preloading)} preloading")
        if queuing:
            parts.append(f"{len(queuing)} queued")
        if judge_line:
            parts.append(judge_line)
        footer = " " + "  |  ".join(parts)
        if _display_width(footer) <= max_x:
            lines.append(line(footer))
        else:
            # A large judge roster overflows one footer line; split the
            # judge progress onto its own wrapped line(s) so the details
            # aren't truncated away.
            base_parts = [p for p in parts if p != judge_line]
            if base_parts:
                lines.append(line(" " + "  |  ".join(base_parts)))
            lines.extend(line(wrapped) for wrapped in _wrap_judge_parts(waiting_parts, max_x))
    if running_parts:
        # Judges actively judging a cell right now (selected to run) render
        # green so they stand out from the waiting roster.
        lines.extend(
            line(wrapped, "green")
            for wrapped in _wrap_judge_parts(running_parts, max_x)
        )
    if complete_parts:
        # Judges whose whole workload is done render dimmed grey so they
        # recede while the others are still working.
        lines.extend(
            line(wrapped, "grey")
            for wrapped in _wrap_judge_parts(complete_parts, max_x)
        )
    if stopped_parts:
        # Judges halted by an exhausted 429 (terminal_429) render on their
        # own red footer line(s) so they stand out from the active roster.
        lines.extend(
            line(wrapped, "red")
            for wrapped in _wrap_judge_parts(stopped_parts, max_x)
        )

    return lines


def _wrap_judge_parts(judge_parts, max_width):
    """Wrap ``[model: N\u2705M\u274cT\u03a3]`` judge parts onto footer lines.

    Each line is prefixed `` Judging `` and holds as many parts as fit within
    ``max_width`` display columns, so a large judge roster spills onto a
    second (or third) footer line instead of being truncated away.
    """
    prefix = " Judging "
    lines_out = []
    current = ""
    for part in judge_parts:
        candidate = f"{current} {part}" if current else part
        if _display_width(prefix + candidate) > max_width and current:
            lines_out.append(prefix + current)
            current = part
        else:
            current = candidate
    if current:
        lines_out.append(prefix + current)
    return lines_out


def _frame_lines_to_text(lines):
    """Convert ``_build_frame_lines`` output to a styled Rich ``Text``."""
    from rich.text import Text

    text = Text(overflow="crop")
    for index, (content, style) in enumerate(lines):
        if index:
            text.append("\n")
        text.append(content, style=_FRAME_STYLE_MAP.get(style, ""))
    return text


def _line_cells(text, width):
    """Expand ``text`` into per-display-column cells, padded to ``width``.

    A wide grapheme (emoji, CJK) occupies its own column plus an empty
    continuation column, matching how the terminal lays the glyph out.
    Returns a list of exactly ``width`` entries so two lines can be diffed
    column-by-column regardless of their rendered widths.
    """
    cells = []
    for cluster in _grapheme_clusters(text):
        cluster_width = _cluster_display_width(cluster)
        if cluster_width <= 0:
            continue
        cells.append(cluster)
        cells.extend([""] * (cluster_width - 1))
    if len(cells) > width:
        del cells[width:]
    cells.extend([" "] * (width - len(cells)))
    return cells


def _changed_cell_spans(old_cells, new_cells):
    """Return ``(start, width)`` spans of columns that differ between cell lists.

    Returns an empty list when the lines render identically, so callers can
    skip the repaint entirely. Spans are horizontal only; the caller applies
    them to a single row of the terminal.
    """
    common = min(len(old_cells), len(new_cells))
    spans = []
    run_start = None
    for i in range(common):
        if old_cells[i] != new_cells[i]:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            spans.append((run_start, i - run_start))
            run_start = None
    if run_start is not None:
        spans.append((run_start, common - run_start))
    if common < len(new_cells):
        spans.append((common, len(new_cells) - common))
    return spans


class _FrameRow(Static):  # pragma: no cover - live interactive loop
    """One TUI frame line that repaints only the cells that changed.

    ``Static.update`` marks the entire widget dirty, so a whole-screen frame
    repainted every tick rewrites every cell of the terminal every 0.2s -
    visible as screen-wide cursor flicker on terminals without synchronized
    output. Each row is therefore its own widget, and ``update_line`` marks
    only the changed column spans (via ``refresh(*regions)``) so the
    compositor emits just those cells.
    """

    def __init__(self):
        super().__init__(markup=False)
        self.styles.width = "1fr"
        self.styles.height = 1
        self._line_text = ""
        self._line_style = None
        self._line_cells = None

    def render(self):
        from rich.text import Text

        text = Text(self._line_text)
        style = _FRAME_STYLE_MAP.get(self._line_style, "")
        if style:
            # Apply the style as a SPAN (not the Text base style) so Textual's
            # ``Content.from_rich_text`` converts it through the app's ANSI
            # theme (``Style.from_rich_style(..., ansi_theme)``). A base-style
            # string stays raw and is parsed with Rich's default theme, which
            # shifts named colors - e.g. ``green`` renders #008000 instead of
            # the app's MONOKAI green #98e024 (this was the colour regression
            # when the per-row rewrite moved from ``append()`` spans to a
            # base style).
            text.stylize(style)
        return text

    def update_line(self, content, style, width):
        """Switch to ``content`` (styled ``style``), repainting changed cells only.

        Returns ``True`` when the row changed (and a repaint was queued). The
        frame is rebuilt every tick but identical rows are left untouched, so
        an idle frame costs no terminal output at all.
        """
        if width <= 0:
            return False
        cells = _line_cells(content, width)
        if (self._line_cells is not None and cells == self._line_cells
                and style == self._line_style):
            return False
        spans = []
        if self._line_cells is not None:
            spans = _changed_cell_spans(self._line_cells, cells)
            if not spans and style != self._line_style:
                spans = [(0, width)]
        else:
            spans = [(0, width)]
        # Join cell atoms WITHOUT inserting a space for the empty
        # continuation cells that follow wide graphemes (an emoji occupies
        # two display columns but only one atom here). Emitting a literal
        # space there would push the following character apart, e.g.
        # rendering ``95⚖️1`` as ``95⚖️ 1`` on every repaint.
        self._line_text = "".join(cells)
        self._line_style = style
        self._line_cells = cells
        if spans:
            self.refresh(*[Region(x, 0, w, 1) for x, w in spans if w > 0])
        return True


class _BenchmarkTUIApp(App):  # pragma: no cover - live interactive loop
    """Textual app that re-renders the benchmark status frame adaptively.

    The frame is rebuilt at most every ``_TUI_REFRESH_SECONDS`` (2 fps), and
    only when something it displays actually changed: the state revision
    bumped, the terminal resized, the operator scrolled, or live elapsed/
    countdown content is ticking. An idle run rebuilds nothing. Each rebuilt
    frame is delivered to one ``_FrameRow`` widget per line; each row repaints
    only the cells that changed, so a repeated frame produces no terminal
    output.
    """

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("q", "quit_tui", "Quit"),
        ("up", "scroll_up", "Up"),
        ("down", "scroll_down", "Down"),
        ("left", "scroll_left", "Left"),
        ("right", "scroll_right", "Right"),
        ("pageup", "scroll_page_up", "Page Up"),
        ("pagedown", "scroll_page_down", "Page Down"),
        ("home", "scroll_home", "Top"),
        ("end", "scroll_end", "Bottom"),
        ("shift+left", "scroll_page_left", "Page Left"),
        ("shift+right", "scroll_page_right", "Page Right"),
        ("ctrl+left", "scroll_home_x", "Far Left"),
        ("ctrl+right", "scroll_end_x", "Far Right"),
    ]

    def __init__(self, state, stop_event, source_abbrevs, frozen_hdr,
                 plugin_hdr, num_sources, active_plugins, session_seed,
                 model_thread_limits):
        super().__init__()
        self._state = state
        self._stop_event = stop_event
        self._source_abbrevs = source_abbrevs
        self._frozen_hdr = frozen_hdr
        self._plugin_hdr = plugin_hdr
        self._num_sources = num_sources
        self._active_plugins = active_plugins
        self._session_seed = session_seed
        self._model_thread_limits = model_thread_limits
        self._scroll_y = 0
        self._scroll_x = 0
        self._rows: list[_FrameRow] = []
        self._last_frame_key = None

    def on_mount(self) -> None:
        self.set_interval(_TUI_REFRESH_SECONDS, self._refresh)
        self._refresh()

    def _sync_rows(self, lines) -> None:
        """Mount/unmount row widgets to match the frame, updating changed rows."""
        width = max(0, self.size.width)
        if len(self._rows) < len(lines):
            new_rows = []
            for i in range(len(self._rows), len(lines)):
                row = _FrameRow()
                row.id = f"row-{i}"
                self._rows.append(row)
                new_rows.append(row)
            # Batch the mounts so the initial frame costs a single layout pass
            # instead of one full repaint per row.
            self.mount(*new_rows)
        elif len(self._rows) > len(lines):
            for row in self._rows[len(lines):]:
                row.remove()
            del self._rows[len(lines):]
        for i, (content, style) in enumerate(lines):
            self._rows[i].update_line(content, style, width)

    def _visible_rows(self) -> int:
        live_height = max(3, self._num_sources + 1)
        return max(0, self.size.height - 9 - live_height)

    def _visible_cols(self) -> int:
        return max(0, self.size.width - FROZEN_VIEW_WIDTH - 1)

    def _max_row_offset(self) -> int:
        return max(0, len(self._state.snapshot()) - self._visible_rows())

    def _max_col_offset(self) -> int:
        return max(0, _display_width(self._plugin_hdr) - self._visible_cols())

    def _frame_key(self):
        """Identity of the frame's static inputs (excludes ticking content)."""
        return (
            self._state.revision,
            self.size.height, self.size.width,
            self._scroll_y, self._scroll_x,
        )

    def _live_content(self) -> bool:
        """True when time-based frame elements are ticking.

        Elapsed/countdown fields (streaming seconds, judge-activity elapsed,
        preload seconds, 429 sleeps) change over time even when no state
        mutation occurs. While any are visible the frame is rebuilt every
        tick (capped by ``_TUI_REFRESH_SECONDS``) so they stay live; when the
        run is fully idle the frame is skipped entirely.
        """
        return (
            self._state.has_live_work()
            or bool(self._state.judge_activity_snapshot())
            or bool((get_429_stats() or {}).get("sleeping"))
        )

    def _refresh(self) -> None:
        if self._stop_event.is_set():
            self.exit()
            return
        key = self._frame_key()
        if key == self._last_frame_key and not self._live_content():
            # Nothing the frame displays changed since the last tick.
            return
        self._last_frame_key = key
        lines = _build_frame_lines(
            self._state, self._active_plugins, self._source_abbrevs,
            self._frozen_hdr, self._plugin_hdr, self._num_sources,
            self._scroll_y, self._scroll_x, (self.size.height, self.size.width),
            model_thread_limits=self._model_thread_limits,
            session_seed=self._session_seed,
        )
        self._sync_rows(lines)

    def _cancel_requests(self) -> None:
        """Cancel benchmark and judge HTTP requests before leaving the TUI."""
        self._stop_event.set()
        close_active_requests()

    def on_unmount(self) -> None:
        """Ensure app teardown also cancels requests on non-keyboard exits."""
        self._cancel_requests()

    def action_quit_tui(self) -> None:
        self._cancel_requests()
        self.exit()

    def action_scroll_up(self) -> None:
        self._scroll_y = max(0, self._scroll_y - 1)

    def action_scroll_down(self) -> None:
        self._scroll_y = min(self._max_row_offset(), self._scroll_y + 1)

    def action_scroll_left(self) -> None:
        self._scroll_x = max(0, self._scroll_x - 8)

    def action_scroll_right(self) -> None:
        self._scroll_x = min(self._max_col_offset(), self._scroll_x + 8)

    def action_scroll_page_up(self) -> None:
        self._scroll_y = max(0, self._scroll_y - self._visible_rows())

    def action_scroll_page_down(self) -> None:
        self._scroll_y = min(
            self._max_row_offset(), self._scroll_y + self._visible_rows())

    def action_scroll_page_left(self) -> None:
        self._scroll_x = max(0, self._scroll_x - self._visible_cols())

    def action_scroll_page_right(self) -> None:
        self._scroll_x = min(
            self._max_col_offset(), self._scroll_x + self._visible_cols())

    def action_scroll_home(self) -> None:
        self._scroll_y = 0

    def action_scroll_end(self) -> None:
        self._scroll_y = self._max_row_offset()

    def action_scroll_home_x(self) -> None:
        self._scroll_x = 0

    def action_scroll_end_x(self) -> None:
        self._scroll_x = self._max_col_offset()


def _tui_main_textual(state, stop_event, num_sources, active_plugins, session_seed=None,
                      model_thread_limits=None):  # pragma: no cover - live interactive loop
    """Run the live TUI with Textual, re-rendering the full frame each tick."""
    source_abbrevs = _unique_source_abbrevs({s["source"] for s in state.snapshot().values()})
    frozen_cols = [("#", MODEL_NUMBER_COLUMN_WIDTH), ("S", 4), ("Model", 18), ("St", 4)]
    frozen_hdr = " ".join(f"{h:>{w}}" for h, w in frozen_cols)
    plugin_cols = []
    for p in active_plugins:
        plugin_cols.extend([
            (f"{p.id[:3]}Sc", SCORE_COLUMN_WIDTH),
            (f"{p.id[:3]}Tok", 6),
            (f"{p.id[:3]}Tm", 6),
            (f"{p.id[:3]}TPS", 6),
        ])
    plugin_hdr = " ".join(f"{h:>{w}}" for h, w in plugin_cols)

    _BenchmarkTUIApp(
        state, stop_event, source_abbrevs, frozen_hdr, plugin_hdr,
        num_sources, active_plugins, session_seed, model_thread_limits,
    ).run()


def _textual_tui_enabled():
    """Return whether the Textual TUI should be used (interactive terminal)."""
    if os.environ.get("AI_BENCHMARK_NO_TEXTUAL"):
        return False
    if os.environ.get("AI_BENCHMARK_FORCE_TEXTUAL"):
        return True
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):  # pragma: no cover
        return False


def tui_main(state, stop_event, num_sources, active_plugins, session_seed=None,
             model_thread_limits=None):
    """Run the live TUI: Textual on interactive terminals, plain text otherwise."""
    if _textual_tui_enabled():
        try:
            _tui_main_textual(state, stop_event, num_sources, active_plugins,
                              session_seed, model_thread_limits)
            return
        except Exception:  # noqa: BLE001 - a Textual failure must fall back to plain text
            try:
                with open("tui_render_errors.log", "a", encoding="utf-8") as handle:
                    traceback.print_exc(file=handle)
            except Exception:  # noqa: BLE001, S110 - logging must not crash the TUI thread
                pass
    _fallback_tui_loop(state, stop_event, session_seed, model_thread_limits)


def _targets_for_runner(targets, state_models, runner):
    """Return targets with a saved/configured identity for ``runner``."""
    suffix = " [opencode]" if runner == "opencode" else ""
    return {
        name: info
        for name, info in targets.items()
        if f"{name}{suffix}" in state_models
    }


def _mark_preload_failed(state, model_name, result, phase_runner, runner_mode):
    """Record a failed warm-up in the model's live state only.

    A preload failure means the model produced no per-plugin results, so it
    must not append a row to ``state.results``. A scoreless ``error`` row would
    become the model's latest result and mask later progress, causing a resumed
    run to re-run already-successful plugins. The ``failed`` status in
    ``model_info`` is authoritative for the TUI, queue builder, and resume
    re-queue.
    """
    error = f"preload failed: {result.error or 'empty preload response'}"
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



class _FlushGate:
    """Decide when in-memory changes warrant a full-state flush.

    The judge path used to persist the entire ``benchmark_state.json`` plus
    regenerate every report after each completed vote, which burned ~6 s of
    GIL-bound CPU and ~210 MB of disk writes per vote. ``_FlushGate`` throttles
    that: ``changed()`` is called once per in-memory change (a completed or
    failed judge vote) and returns ``True`` when a flush is due -- either
    ``interval`` seconds have elapsed since the last flush or ``max_changes``
    changes have accumulated, whichever comes first. The caller schedules the
    actual save only when ``changed()`` reports due, then calls ``reset()``.

    ``changed()``/``reset()`` are called from judge cell workers without the
    ``persistence_lock`` (the save itself runs on the background flusher
    thread, which owns the lock). A lost or duplicated due-decision is
    harmless: ``_BackgroundFlusher.request_flush()`` coalesces duplicates, and
    a lost request only defers the flush to the next cadence boundary (at most
    ``interval`` seconds later), never losing votes.
    """

    def __init__(self, interval=60.0, max_changes=10):
        try:
            self.interval = float(interval)
        except (TypeError, ValueError):
            self.interval = 60.0
        try:
            self.max_changes = max(1, int(max_changes))
        except (TypeError, ValueError):
            self.max_changes = 10
        self._last_flush = time.monotonic()
        self._changes = 0

    def changed(self):
        """Record one in-memory change; return True when a flush is due."""
        self._changes += 1
        return self._due()

    def _due(self):
        return (self._changes >= self.max_changes
                or time.monotonic() - self._last_flush >= self.interval)

    def reset(self):
        """Mark the current flush as completed, starting a fresh cadence."""
        self._last_flush = time.monotonic()
        self._changes = 0


class _BackgroundFlusher:
    """Serialize full-state snapshots on a dedicated thread.

    The judge path used to run the full-state save synchronously in the worker
    thread that completed the vote: ~6 s of GIL-bound deepcopy + JSON dump +
    report regeneration stalled every other judge worker (and the TUI) on the
    GIL, and later finishers queued behind ``persistence_lock`` instead of
    starting their next request. ``_BackgroundFlusher`` moves that
    serialization off the hot path: ``request_flush()`` is non-blocking (it
    only sets a pending flag), and one dedicated daemon thread runs
    ``flush_fn`` -- which must take ``persistence_lock`` itself -- for each
    batch of requests. Requests that arrive while a flush is running are
    coalesced into one follow-up flush, so at most one save is in flight and
    at most one is queued behind it. The flush persists only the state
    snapshot; report files are regenerated once at the end of the run.

    ``stop()`` drains any pending request (one final flush) before exiting, so
    callers can rely on the flusher for the tail; the benchmark additionally
    performs a guaranteed final ``save_state(raise_on_error=True)`` on the
    main thread after ``stop()``.
    """

    def __init__(self, flush_fn, name="background-flusher"):
        self._flush_fn = flush_fn
        self._condition = threading.Condition()
        self._pending = False
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run, name=name, daemon=True,
        )

    def start(self):
        self._thread.start()

    def request_flush(self):
        """Request a flush; never blocks the caller."""
        with self._condition:
            self._pending = True
            self._condition.notify()

    def stop(self, timeout=None):
        """Drain pending work (one final flush) and join the thread."""
        with self._condition:
            self._stopped = True
            self._condition.notify()
        self._thread.join(timeout=timeout)

    def _run(self):
        while True:
            with self._condition:
                while not self._pending and not self._stopped:
                    self._condition.wait()
                if self._pending:
                    self._pending = False
                elif self._stopped:
                    return
            try:
                self._flush_fn()
            except Exception as exc:  # noqa: BLE001 - keep the flusher alive
                print(
                    f"⚠️  Background state flush failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )


def _resolve_judge_plugin_limit(source_config, source):
    """Return the per-judge cell concurrency for ``source``.

    Mirrors ``plugin_thread_limit``: how many cells one judge model scores at
    once. Unlike the benchmark's per-target semantics, zero is not an
    unlimited value here -- fanning out an unbounded number of concurrent
    judge requests is a resource hazard, so a non-positive value serializes
    to one cell per judge.
    """
    cfg = source_config.get(source)
    value = cfg.get("plugin_thread_limit", 1) if isinstance(cfg, dict) else 1
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 1
    return value if value > 0 else 1


def _configure_judge_source(benchmark_limits, source, full_limit,
                            benchmark_active, pool):
    """Configure the judge reservation for one source.

    During benchmark overlap, reserve one judge model only when another
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


class _CombinedStopEvent:
    """Expose several cancellation events through the Event interface."""

    def __init__(self, *events):
        self._events = tuple(events)

    def is_set(self):
        return any(event.is_set() for event in self._events)

    def wait(self, timeout=None):
        if timeout is None:
            while not self.is_set():
                time.sleep(0.1)
            return True
        deadline = time.monotonic() + timeout
        while not self.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self.is_set()
            time.sleep(min(0.1, remaining))
        return True


class _JudgeQueue:
    """A single judge model's fresh-then-retry FIFO cell queue.

    Each judge owns exactly one queue so a source can run one judge to
    completion before loading another judge (keeping a local model resident)
    instead of round-robin swapping between judges every cell. Within a judge,
    never-judged cells still precede retried cells, and both tiers are served
    in arrival order.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._fresh = deque()
        self._retry = deque()
        self._unfinished_tasks = 0
        self._stop_tokens = 0

    @property
    def unfinished_tasks(self):
        """Expose queue accounting used by tests and shutdown diagnostics."""
        with self._condition:
            return self._unfinished_tasks

    @property
    def pending(self):
        """True while the judge still has unstarted cells queued."""
        with self._condition:
            return bool(self._fresh or self._retry)

    @staticmethod
    def _job_is_fresh(job):
        # ``expected_added`` is true when this judge has no prior vote for the
        # cell; failed/invalid prior attempts are retry work.
        return not isinstance(job, tuple) or len(job) <= 5 or bool(job[5])

    def put(self, job):
        bucket = self._fresh if self._job_is_fresh(job) else self._retry
        with self._condition:
            bucket.append(job)
            self._unfinished_tasks += 1
            self._condition.notify()

    def get(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                if self._stop_tokens:
                    self._stop_tokens -= 1
                    return _JUDGE_QUEUE_STOP
                if self._fresh or self._retry:
                    return self._fresh.popleft() if self._fresh else self._retry.popleft()
                if timeout is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()

    def task_done(self):
        with self._condition:
            if self._unfinished_tasks <= 0:
                raise ValueError("task_done() called too many times")
            self._unfinished_tasks -= 1
            if self._unfinished_tasks == 0:
                self._condition.notify_all()

    def join(self):
        with self._condition:
            while self._unfinished_tasks:
                self._condition.wait()

    def cancel_pending(self):
        """Discard queued, not-yet-started jobs while preserving active jobs."""
        with self._condition:
            pending = len(self._fresh) + len(self._retry)
            self._fresh.clear()
            self._retry.clear()
            self._unfinished_tasks -= pending
            self._condition.notify_all()

    def request_stop(self, count):
        with self._condition:
            self._stop_tokens += count
            self._condition.notify_all()


_JUDGE_QUEUE_STOP = object()
_NO_JUDGE = object()


class SourceJudgeWorkerPool:
    """Run judge jobs with per-source model and plugin concurrency.

    ``model_limit`` bounds how many distinct judge models run concurrently for
    the source; each active judge occupies exactly one model slot, mirroring
    ``model_thread_limit``. ``plugin_limit`` bounds how many cells one judge
    scores at once, mirroring ``plugin_thread_limit``. Judges are run to
    completion in discovery order before another judge is activated, which
    keeps a single local model resident instead of round-robin swapping
    between judges.
    """

    def __init__(self, source, model_limit, process_job, stop_event,
                 plugin_limit=1, on_selection_change=None):
        self.source = source
        self.model_limit = max(1, int(model_limit))
        self.plugin_limit = max(1, int(plugin_limit))
        self.process_job = process_job
        self.stop_event = stop_event
        self.on_selection_change = on_selection_change
        self._condition = threading.Condition()
        self._queues = {}          # judge -> _JudgeQueue
        self._order = []           # judge discovery order
        self._active = {}          # judge -> judge-runner thread
        self._active_limit = 0     # currently allowed concurrent judge models
        self._stopped = False

    @property
    def thread_count(self):
        """Number of judge models currently running for this source."""
        with self._condition:
            return len(self._active)

    @property
    def model_slots(self):
        """Currently allowed number of concurrent judge models (reservation)."""
        with self._condition:
            return self._active_limit

    @staticmethod
    def _job_key(job):
        if isinstance(job, tuple) and len(job) > 4:
            return job[4]
        return None

    def _queue_for(self, judge):
        queue = self._queues.get(judge)
        if queue is None:
            queue = _JudgeQueue()
            self._queues[judge] = queue
            self._order.append(judge)
        return queue

    def enqueue(self, job):
        """Queue one judge job, keyed by its judge model."""
        judge = self._job_key(job)
        with self._condition:
            self._queue_for(judge).put(job)
            self._activate_locked()

    def _next_pending_judge_locked(self):
        for judge in self._order:
            if judge in self._active:
                continue
            if self._queues[judge].unfinished_tasks > 0:
                return judge
        return _NO_JUDGE

    def _notify_selection(self, judge, selected):
        """Publish a judge-runner selection change without affecting workers."""
        if self.on_selection_change is None:
            return
        try:
            self.on_selection_change(judge, selected)
        except Exception as exc:  # noqa: BLE001 - TUI bookkeeping must not kill a judge
            print(
                f"⚠️  Judge selection update ({self.source}/{judge}) failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    def _activate_locked(self):
        """Start judge runners for pending judges while model slots are free."""
        while (not self._stopped and not self.stop_event.is_set()
               and len(self._active) < self._active_limit):
            judge = self._next_pending_judge_locked()
            if judge is _NO_JUDGE:
                break
            thread = threading.Thread(
                target=self._judge_runner,
                args=(judge,),
                name=f"judge-runner-{self.source}-{judge}",
                daemon=True,
            )
            self._active[judge] = thread
            self._notify_selection(judge, True)
            thread.start()

    def _judge_runner(self, judge):
        """Run one judge over its queued cells until drained.

        The judge holds a model slot while it still has cells, then tears down
        its cell workers and frees the slot so the next judge (in discovery
        order) can be loaded. Judge runners are daemonized so Ctrl+C cannot
        leave a process permanently stuck behind a provider that ignores
        cancellation; normal completion still drains and joins before exit.
        """
        queue = self._queues[judge]
        workers = []
        for index in range(self.plugin_limit):
            thread = threading.Thread(
                target=self._cell_worker,
                args=(queue,),
                name=f"judge-cell-{self.source}-{judge}-{index + 1}",
                daemon=True,
            )
            thread.start()
            workers.append(thread)
        # Drain this judge's currently queued cells before yielding the slot.
        while not self.stop_event.is_set() and queue.unfinished_tasks > 0:
            time.sleep(0.05)
        queue.request_stop(len(workers))
        for thread in workers:
            thread.join()
        with self._condition:
            self._active.pop(judge, None)
            # Clear the old selection before activating its replacement so
            # the live footer never treats a finished runner as selected after
            # another judge has taken its slot.
            self._notify_selection(judge, False)
            self._activate_locked()
            self._condition.notify_all()

    def _cell_worker(self, judge_queue):
        while True:
            try:
                job = judge_queue.get(timeout=0.2)
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue
            if job is _JUDGE_QUEUE_STOP:
                return
            try:
                # On cancellation, discard queued work instead of starting
                # another judge request. The active request receives the same
                # stop_event and can terminate cooperatively; every discarded
                # item still receives task_done so queue accounting remains
                # balanced.
                if self.stop_event.is_set():
                    continue
                try:
                    self.process_job(job)
                except Exception as exc:  # noqa: BLE001 - keep one bad job from killing the pool
                    # ``process_judge_job`` normally records its own failure,
                    # but the pool must remain live even if an unexpected
                    # callback bug escapes. Queue accounting is completed in
                    # the finally block and later jobs continue to drain.
                    print(
                        f"⚠️  Judge worker ({self.source}) failed: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
            finally:
                judge_queue.task_done()

    def start(self, count=1):
        """Allow up to ``count`` judge models to run concurrently."""
        with self._condition:
            self._active_limit = min(self.model_limit, max(0, int(count)))
            self._activate_locked()
            self._condition.notify_all()

    def expand_full(self):
        """Release the benchmark reservation and allow the full judge pool."""
        self.start(self.model_limit)

    def drain(self):
        """Wait until every queued job has finished, unless cancellation starts."""
        with self._condition:
            judge_queues = list(self._queues.values())
        for judge_queue in judge_queues:
            while judge_queue.unfinished_tasks:
                if self.stop_event.is_set():
                    return False
                time.sleep(0.05)
        with self._condition:
            while self._active:
                if self.stop_event.is_set():
                    return False
                self._condition.wait(timeout=0.05)
        return True

    def stop(self, timeout=None, *, drain=False):
        """Stop judges, optionally draining all queued jobs first.

        Normal completion uses ``drain=True`` and an unbounded join. The
        cancellation path skips the drain so Ctrl+C can save resumable state
        without waiting on new work.
        """
        drained = self.drain() if drain else False
        if not drained:
            with self._condition:
                queues = list(self._queues.values())
            for queue in queues:
                queue.cancel_pending()
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
            active = list(self._active.items())
            if not drain:
                for judge, _thread in active:
                    self._notify_selection(judge, False)
        for _judge, thread in active:
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


def _enable_faulthandler() -> None:
    """Dump a Python stack trace on fatal signals for crash diagnosis.

    ``faulthandler.enable()`` installs handlers for SIGSEGV/SIGFPE/SIGABRT/
    SIGBUS/SIGILL so a native crash (e.g. a Playwright/Chromium segfault in a
    worker thread) prints the Python stack to stderr instead of a bare
    "Segmentation fault". ``register(SIGUSR1)`` additionally lets an operator
    force a live stack dump of a wedged run with ``kill -USR1 <pid>``.
    """
    faulthandler.enable()
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1)


def _run_benchmark(tui_handoff=None):  # pragma: no cover - live benchmark orchestrator (no unit tests)
    """Run the full benchmark (setup, orchestration, and final output).

    ``tui_handoff`` is None on the non-interactive path, where the plain-text
    TUI runs in a daemon thread and this function owns the current thread. On
    the interactive Textual path the orchestrator runs in a worker thread and
    ``tui_handoff`` carries the launch arguments and stop/interrupt events for
    the main thread to drive the TUI (see ``main``).
    """
    # Dump Python stacks on fatal signals so a native crash or a wedged run is
    # diagnosable instead of ending in a bare "Segmentation fault".
    _enable_faulthandler()
    # Load a local .env file (if present) so ${VAR} config expansion and
    # env-driven tools (e.g. --chatplayground-config) can read credentials
    # without prefixing every command. Real environment variables take
    # precedence over file values.
    load_dotenv_file()
    try:
        subprocess.run(['stty', 'sane'], stderr=subprocess.DEVNULL,
                       stdin=sys.stdin, timeout=1, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass
    sys.stderr.write('\033[2J\033[H')
    sys.stderr.flush()

    parser = build_parser()
    args = parser.parse_args()

    if args.build_judge_queue:
        try:
            path = write_disagreement_queue(
                args.build_judge_queue,
                args.judge_queue_output,
                spread_threshold=(
                    None if args.no_judge_spread else args.judge_spread_threshold
                ),
                deviation_threshold=(
                    None if args.no_judge_deviation else args.judge_deviation_threshold
                ),
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"❌ Could not build judge disagreement queue: {exc}", file=sys.stderr)
            sys.exit(1)
        print(path)
        sys.exit(0)

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

    if args.chatplayground_config:
        from benchmark.chatplayground import generate_config as generate_chatplayground_config
        try:
            cfg = generate_chatplayground_config()
        except Exception as exc:  # noqa: BLE001 - browser/network tool; report and exit
            print(f"❌ Could not enumerate ChatPlayground models: {exc}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(cfg, indent=2))
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

    if args.schema_sentinel:
        targets_for_probe = resolve_targets(cfg)
        timeout = args.timeout if args.timeout is not None else int(cfg.get("timeout", 600))
        probe_results = []
        for target_name, target in targets_for_probe.items():
            result = run_schema_sentinel(
                source_config,
                target["source"],
                target["api_model"],
                timeout=timeout,
                drop_params=target.get("drop_params", []),
            )
            result["target"] = target_name
            result["is_agent"] = target.get("is_agent", False)
            probe_results.append(result)
        print(json.dumps({
            "probe": "schema-sentinel-v1",
            "scores_affected": False,
            "results": probe_results,
        }, indent=2))
        sys.exit(0)
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
    # Append-only result journal: every completed result is appended here so a
    # crash that truncates benchmark_state.json can still replay the results.
    journal_path = os.path.join(output_dir, "results.journal.jsonl")

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

    max_tokens = cfg.get("max_tokens", 16384)
    if args.max_tokens is not None:
        max_tokens = args.max_tokens
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        print("❌ max_tokens must be a positive integer scalar.", file=sys.stderr)
        sys.exit(1)

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
                max_tokens=max_tokens,
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
        "judge_prompt_version": JUDGE_PROMPT_VERSION if judge_models else None,
        "judge_contracts": (
            {plugin.id: judge_contract_id(plugin) for plugin in active_plugins}
            if judge_models else {}
        ),
        "judge_projection": "active-contract" if judge_models else None,
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
                        state.set_journal_path(journal_path, truncate=True)
                        fresh_state = True
                    elif choice == "continue":
                        # The result journal is authoritative for completed
                        # results: replay it into the recovery candidate so a
                        # crash loses at most the in-flight result, even when
                        # the byte-scanner cannot recover every row.
                        journal_results = BenchmarkState.replay_journal(journal_path)
                        if journal_results:
                            recovery["data"]["results"] = journal_results
                            recovery["candidate_bytes"] = None
                        backup = apply_state_recovery(state_file, recovery)
                        saved_state = recovery["data"]
                        if journal_results:
                            print(
                                f"   Recovered {len(journal_results)} result(s) "
                                "from the result journal.",
                                file=sys.stderr,
                            )
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
                            state.set_journal_path(journal_path, truncate=True)
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
                        # Preserve the existing journal rather than truncating
                        # it: the append-only log may hold results newer than
                        # the last state-file save, and a later crash whose
                        # state file is corrupt must still be able to replay
                        # pre-resume results from the journal.
                        state.set_journal_path(journal_path)
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
            state.set_journal_path(journal_path, truncate=True)

        # The active CLI judge configuration is authoritative on resume; do
        # not let a prior run's judge set drive stale row markers.
        state.set_judge_models(judge_models)
        active_judge_contracts = (
            {plugin.id: judge_contract_id(plugin) for plugin in active_plugins}
            if judge_models else {}
        )
        state.set_active_judge_contracts(active_judge_contracts)

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
                    max_tokens=max_tokens,
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

        tui_args = (state, stop_event, len(source_config), active_plugins,
                    session_seed, model_thread_limits)
        tui_thread = None
        tui_interrupt_event = None
        if tui_handoff is None:
            # Non-interactive fallback: the plain-text TUI is thread-safe, so
            # run it as a daemon while the orchestration owns this thread.
            tui_thread = threading.Thread(target=tui_main, args=tui_args, daemon=True)
            tui_thread.start()
            time.sleep(0.3)
        else:
            # Textual's Linux driver installs SIGTSTP/SIGCONT handlers, which
            # ``signal.signal`` only permits from the main thread. Hand the TUI
            # launch to the main thread and keep orchestrating here.
            tui_interrupt_event = tui_handoff["interrupt"]
            tui_handoff["args"] = tui_args
            tui_handoff["stop_event"] = stop_event
            tui_handoff["ready"].set()

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
                        _mark_preload_failed(state, model_name, result, phase_runner, runner_mode)
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
                    _mark_preload_failed(state, model_name, failure, phase_runner, runner_mode)
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
        judge_model_limits = {
            source: max(1, int(model_thread_limits.get(source, 1)))
            for source in set(judge_sources.values())
        }
        judge_plugin_limits = {
            source: _resolve_judge_plugin_limit(source_config, source)
            for source in set(judge_sources.values())
        }
        judge_pools = {}
        judge_seen = set()
        judge_seen_lock = threading.Lock()
        judge_counts_lock = threading.Lock()
        judge_votes = {}
        judge_votes_lock = threading.Lock()
        halted_judges = set()
        halted_judges_lock = threading.Lock()
        judge_stop_events = {model: threading.Event() for model in judge_models}
        judge_request_stop_events = {
            model: _CombinedStopEvent(stop_event, judge_stop_events[model])
            for model in judge_models
        }
        judge_contracts = active_judge_contracts
        existing_judge_counts = {model: 0 for model in judge_models}
        existing_judge_failures = {model: 0 for model in judge_models}
        existing_judge_expected = {model: 0 for model in judge_models}
        for result in state.latest_results():
            for plugin in active_plugins:
                expected_contract = judge_contracts.get(plugin.id)
                votes_by_model = {
                    vote.get("model"): vote
                    for vote in (result.get(f"{plugin.id}_judge_votes", []) or [])
                    if isinstance(vote, dict)
                    and vote.get("model")
                    and vote.get("judge_contract_id") == expected_contract
                }
                for model, vote in votes_by_model.items():
                    if model not in existing_judge_counts:
                        continue
                    existing_judge_expected[model] += 1
                    if is_successful_judge_vote(vote):
                        existing_judge_counts[model] += 1
                    else:
                        existing_judge_failures[model] += 1
        state.set_judge_progress({
            model: {
                "completed": existing_judge_counts[model],
                "failed": existing_judge_failures[model],
                "expected": existing_judge_expected[model],
            }
            for model in judge_models
        })

        def replace_judge_progress(judge_name, previous_vote, current_vote):
            """Replace one cell's prior outcome in the live footer totals."""
            previous_completed = int(
                previous_vote is not None and is_successful_judge_vote(previous_vote)
            )
            previous_failed = int(
                previous_vote is not None and not is_successful_judge_vote(previous_vote)
            )
            completed = int(is_successful_judge_vote(current_vote))
            failed = int(not is_successful_judge_vote(current_vote))
            return state.replace_judge_progress(
                judge_name,
                previous_completed=previous_completed,
                previous_failed=previous_failed,
                completed=completed,
                failed=failed,
            )

        judge_effective_timeout = (cfg.get("judge", {}).get("timeout", timeout)
                                   if isinstance(cfg.get("judge"), dict) else timeout)
        judge_max_tokens = (cfg.get("judge", {}).get("max_tokens", JUDGE_DEFAULT_MAX_TOKENS)
                            if isinstance(cfg.get("judge"), dict) else JUDGE_DEFAULT_MAX_TOKENS)
        if (isinstance(judge_max_tokens, bool)
                or not isinstance(judge_max_tokens, int)
                or judge_max_tokens <= 0):
            print("❌ judge.max_tokens must be a positive integer scalar.", file=sys.stderr)
            sys.exit(1)
        judge_temperature = (cfg.get("judge", {}).get("temperature", 0.0)
                             if isinstance(cfg.get("judge"), dict) else 0.0)
        judge_request_params = resolve_judge_request_params(cfg)

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
            contract_id = judge_contracts.get(plugin_id)
            votes_by_model = {
                vote.get("model"): vote
                for vote in [*result_votes, *info_votes]
                if isinstance(vote, dict)
                and vote.get("model")
                and vote.get("judge_contract_id") == contract_id
            }
            judged_models = {
                judge_name for judge_name, vote in votes_by_model.items()
                if is_successful_judge_vote(vote)
            }
            for judge_name in judge_models:
                if judge_name in judged_models:
                    continue
                with halted_judges_lock:
                    if judge_name in halted_judges:
                        continue
                source = judge_sources[judge_name]
                expected_added = judge_name not in votes_by_model
                key = (
                    os.path.abspath(sidecar), target_name, runner, plugin_id,
                    judge_name, contract_id,
                )
                with judge_seen_lock:
                    if key in judge_seen:
                        continue
                    judge_seen.add(key)
                judge_pools[source].enqueue(
                    (sidecar, target_name, runner, plugin_id, judge_name, expected_added)
                )
                state.update(state_key, **{f"{plugin_id}_judge_queued": True})
                if expected_added:
                    state.increment_judge_progress(judge_name, expected=1)
                with judge_counts_lock:
                    run_info["judge_counts"]["queued"] += 1

        def process_judge_job(job):
            """Judge one sidecar with every configured judge and persist consensus."""
            sidecar, target_name, runner, plugin_id, judge_name, expected_added = job
            with halted_judges_lock:
                if judge_name in halted_judges:
                    if expected_added:
                        state.increment_judge_progress(judge_name, expected=-1)
                    return
            item = {}
            previous_vote = None
            existing_votes = []
            state_key = (
                target_name if runner == "http"
                else f"{target_name} [opencode]"
            )
            plugin_obj = next(
                (p for p in active_plugins if p.id == plugin_id), None
            )
            contract_id = judge_contract_id(plugin_obj) if plugin_obj is not None else None
            try:
                with open(sidecar, encoding="utf-8") as handle:
                    item = json.load(handle)
                latest = {
                    (result.get("state_key", result.get("model")), result.get("runner", "http")): result
                    for result in state.latest_results()
                }.get((item.get("state_key", target_name), runner), {})
                state_key = item.get("state_key", state_key)
                live_info = state.snapshot().get(state_key, {})
                vote_key = f"{plugin_id}_judge_votes"
                expected_contract = item.get("judge_contract_id") or contract_id
                all_existing_by_identity = {
                    (vote.get("model"), vote.get("judge_contract_id")): vote
                    for vote in [
                        *(latest.get(vote_key, []) or []),
                        *(live_info.get(vote_key, []) or []),
                    ]
                    if isinstance(vote, dict) and vote.get("model")
                }
                all_existing_votes = list(all_existing_by_identity.values())
                existing_votes = judge_votes_for_contract(all_existing_votes, expected_contract)
                existing_by_model = {
                    vote.get("model"): vote for vote in existing_votes
                }
                previous_vote = existing_by_model.get(judge_name)
                if any(
                    vote.get("model") == judge_name
                    and is_successful_judge_vote(vote)
                    for vote in existing_votes
                    if isinstance(vote, dict)
                ):
                    # The persisted successful vote was already included in
                    # the initialized current-state totals. Duplicate queue
                    # delivery must not count it a second time.
                    return
                activity_id = state.start_judge_activity(
                    judge_name, target_name, plugin_id,
                )
                progress_chars = [0, 0]

                def judge_progress(content_delta, thinking_delta):
                    progress_chars[0] += len(content_delta or "")
                    progress_chars[1] += len(thinking_delta or "")
                    state.update_judge_activity(
                        activity_id,
                        thinking_tokens=progress_chars[1] // 4,
                        content_tokens=progress_chars[0] // 4,
                    )

                outcome = None
                try:
                    # Pass the real plugin instance so its judge sanitizer
                    # (e.g. tool-calling masks its <tool_call> tags) is
                    # applied when the judge prompt is built.
                    outcome = judge_response(
                        source_config,
                        judge_sources[judge_name],
                        targets[judge_name]["api_model"],
                        sidecar,
                        timeout=judge_effective_timeout,
                        max_tokens=judge_max_tokens,
                        temperature=judge_temperature,
                        request_params=judge_request_params,
                        drop_params=(raw_targets.get(judge_name, {}).get("drop_params", [])
                                     if isinstance(raw_targets.get(judge_name), dict) else []),
                        stop_event=judge_request_stop_events[judge_name],
                        log_path=os.path.join(output_dir, f"judge-{judge_name}.log"),
                        plugin=plugin_obj,
                        progress_callback=judge_progress,
                    )
                finally:
                    if outcome is not None and outcome.response_text is not None:
                        state.update_judge_activity(
                            activity_id,
                            content_tokens=len(outcome.response_text) // 4,
                        )
                    state.finish_judge_activity(activity_id)
                if outcome.terminal_429:
                    with halted_judges_lock:
                        halted_judges.add(judge_name)
                    state.update_judge_progress(judge_name, stopped=True)
                    judge_stop_events[judge_name].set()
                vote = {
                    "model": judge_name,
                    "score": outcome.score,
                    "confidence": outcome.confidence,
                    "rationale": outcome.rationale,
                    "criteria": outcome.criteria or [],
                    "judge_prompt_version": JUDGE_PROMPT_VERSION,
                    "judge_contract_id": contract_id,
                    "error": outcome.error,
                }
                response_text = outcome.response_text or ""
                artifact_error = None
                try:
                    # Always publish one raw artifact per attempt. A transport
                    # failure produces an empty .txt plus metadata rather than
                    # making the scheduler failure indistinguishable from a
                    # missing/abandoned job.
                    save_judge_response(
                        output_dir, target_name, runner, plugin_id,
                        judge_name, response_text, contract_id,
                    )
                    save_judge_response_metadata(
                        output_dir, target_name, runner, plugin_id, judge_name,
                        {
                            "target": target_name,
                            "runner": runner,
                            "plugin": plugin_id,
                            "judge_model": judge_name,
                            "judge_prompt_version": JUDGE_PROMPT_VERSION,
                            "judge_contract_id": contract_id,
                            "status": "error" if outcome.error else "ok",
                            "response_present": outcome.response_text is not None,
                            "response_empty": not bool(response_text.strip()),
                            "score": outcome.score,
                            "confidence": outcome.confidence,
                            "error": outcome.error,
                            "terminal_429": outcome.terminal_429,
                            "rationale": outcome.rationale,
                            "criteria": outcome.criteria or [],
                            "diagnostics": outcome.diagnostics,
                            "saved_at": datetime.now(timezone.utc).isoformat(),
                        },
                        contract_id,
                    )
                except OSError as exc:
                    artifact_error = f"could not save judge response artifact: {exc}"
                if artifact_error:
                    vote["error"] = vote["error"] or artifact_error
                vote_identity = (state_key, runner, plugin_id)
                with judge_votes_lock:
                    prior_all_votes = list(
                        judge_votes.get(vote_identity, all_existing_votes)
                    )
                    prior_all_votes = merge_judge_vote(prior_all_votes, vote)
                    judge_votes[vote_identity] = prior_all_votes
                    votes = judge_votes_for_contract(prior_all_votes, contract_id)
                consensus_by_contract = confidence_weighted_consensus_by_contract(
                    prior_all_votes,
                )
                consensus = consensus_by_contract.get(
                    contract_id, confidence_weighted_consensus(votes),
                )
                expected_judges = set(judge_models)
                received_judges = {
                    vote.get("model") for vote in votes
                    if is_successful_judge_vote(vote)
                }
                failed_judges = {
                    vote.get("model") for vote in votes
                    if isinstance(vote, dict)
                    and vote.get("model") in expected_judges
                    and not is_successful_judge_vote(vote)
                }
                all_judges_finished = expected_judges.issubset(
                    received_judges | failed_judges
                )
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
                    criteria=consensus.get("criteria", []),
                    consensus_by_contract=consensus_by_contract,
                    selected_contract=contract_id,
                    error=consensus["error"],
                    input_sha256=item.get("response_sha256"),
                    votes=prior_all_votes,
                    status=judge_status,
                    complete=(
                        all_judges_finished
                        and expected_judges.issubset(received_judges)
                    ),
                )
                # Progress counts completed usable judgments only. A failed
                # attempt remains visible in votes/artifacts, but its judge is
                # still eligible for a future resume retry.
                with judge_counts_lock:
                    replace_judge_progress(judge_name, previous_vote, vote)
                    if is_successful_judge_vote(vote):
                        run_info["judge_counts"]["completed"] += 1
                    else:
                        run_info["judge_counts"]["failed"] += 1
                    run_info["judge_counts"]["votes"] += 1
                with persistence_lock:
                    state.save_state(state_file, plugin_versions=plugin_versions)
                    _save_outputs(state, output_dir, active_plugins)
            except Exception as exc:  # noqa: BLE001 - isolate one judge job failure
                # Preserve a per-attempt diagnostic even when the sidecar,
                # transport, parser, or unexpected processing path fails
                # before a JudgeResult exists. Artifact failures are surfaced
                # rather than silently making an attempted job look absent.
                artifact_error = None
                try:
                    save_judge_response(
                        output_dir, target_name, runner, plugin_id, judge_name, "",
                        contract_id,
                    )
                    save_judge_response_metadata(
                        output_dir, target_name, runner, plugin_id, judge_name,
                        {
                            "target": target_name,
                            "runner": runner,
                            "plugin": plugin_id,
                            "judge_model": judge_name,
                            "judge_prompt_version": JUDGE_PROMPT_VERSION,
                            "judge_contract_id": contract_id,
                            "status": "exception",
                            "response_present": False,
                            "response_empty": True,
                            "error": f"{type(exc).__name__}: {exc}",
                            "saved_at": datetime.now(timezone.utc).isoformat(),
                        },
                        contract_id,
                    )
                except OSError as artifact_exc:
                    artifact_error = f"could not save judge failure artifact: {artifact_exc}"
                    print(f"⚠️  {artifact_error}", file=sys.stderr)
                state_key = item.get("state_key", state_key)
                if previous_vote is None:
                    latest = {
                        (result.get("state_key", result.get("model")),
                         result.get("runner", "http")): result
                        for result in state.latest_results()
                    }.get((state_key, runner), {})
                    live_info = state.snapshot().get(state_key, {})
                    all_existing_by_identity = {
                        (vote.get("model"), vote.get("judge_contract_id")): vote
                        for vote in [
                            *(latest.get(f"{plugin_id}_judge_votes", []) or []),
                            *(live_info.get(f"{plugin_id}_judge_votes", []) or []),
                        ]
                        if isinstance(vote, dict) and vote.get("model")
                    }
                    all_existing_votes = list(all_existing_by_identity.values())
                    existing_votes = judge_votes_for_contract(
                        all_existing_votes, contract_id,
                    )
                    previous_vote = next(
                        (vote for vote in existing_votes if vote.get("model") == judge_name),
                        None,
                    )
                failure_vote = {
                    "model": judge_name,
                    "score": None,
                    "confidence": None,
                    "rationale": None,
                    "criteria": [],
                    "judge_prompt_version": JUDGE_PROMPT_VERSION,
                    "judge_contract_id": contract_id,
                    "error": (
                        f"judge input failed: {type(exc).__name__}: {exc}"
                        + (f"; {artifact_error}" if artifact_error else "")
                    ),
                }
                vote_identity = (state_key, runner, plugin_id)
                with judge_votes_lock:
                    prior_all_votes = list(
                        judge_votes.get(vote_identity, all_existing_votes)
                    )
                    prior_all_votes = merge_judge_vote(prior_all_votes, failure_vote)
                    judge_votes[vote_identity] = prior_all_votes
                expected_judges = set(judge_models)
                current_votes = judge_votes_for_contract(prior_all_votes, contract_id)
                received_judges = {
                    vote.get("model") for vote in current_votes
                    if is_successful_judge_vote(vote)
                }
                failed_judges = {
                    vote.get("model") for vote in current_votes
                    if isinstance(vote, dict)
                    and vote.get("model") in expected_judges
                    and not is_successful_judge_vote(vote)
                }
                all_judges_finished = expected_judges.issubset(
                    received_judges | failed_judges
                )
                state.update_judge_result(
                    state_key, runner, plugin_id,
                    error=failure_vote["error"],
                    selected_contract=contract_id,
                    votes=prior_all_votes,
                    status="failed" if all_judges_finished else "running",
                    complete=(
                        all_judges_finished
                        and expected_judges.issubset(received_judges)
                    ),
                )
                with judge_counts_lock:
                    replace_judge_progress(judge_name, previous_vote, failure_vote)
                    run_info["judge_counts"]["failed"] += 1
                # Failed attempts do not advance completed progress and remain
                # eligible for retry on resume.
                with persistence_lock:
                    state.save_state(state_file, plugin_versions=plugin_versions)
                    _save_outputs(state, output_dir, active_plugins)

        judge_pools = {
            source: SourceJudgeWorkerPool(
                source, judge_model_limits[source], process_judge_job, stop_event,
                plugin_limit=judge_plugin_limits[source],
                on_selection_change=lambda judge, selected: state.set_judge_selected(
                    judge, selected,
                ),
            )
            for source in set(judge_sources.values())
        }

        # Queue retained results before benchmark workers start. On resume,
        # completed targets are deliberately absent from the benchmark queues,
        # but their durable sidecars still need judging immediately.
        for sidecar, item in _eligible_judge_sidecars(
            judge_input_dir, targets, state, {plugin.id for plugin in active_plugins},
            judge_models, judge_contracts,
        ):
            enqueue_judge(sidecar, item["target"], item["runner"], item["plugin"])

        def source_benchmark_complete(source):
            """Release the source's reserved benchmark slots to judging."""
            pool = judge_pools.get(source)
            if pool is not None:
                pool.expand_full()

        def start_judge_if_async(benchmark_limits, benchmark_sources=None):
            """Reserve one judge model slot, then expand after source completion.

            While a source is benchmarking, one judge model is allowed only
            when another slot remains available. Once that source drains, all
            of its configured model slots become judge models; each judge
            model scores up to ``plugin_thread_limit`` cells concurrently.

            """
            if not judge_models:
                return
            benchmark_sources = benchmark_sources or set()
            for source, pool in judge_pools.items():
                _configure_judge_source(
                    benchmark_limits,
                    source,
                    judge_model_limits[source],
                    source in benchmark_sources,
                    pool,
                )

        def stop_judge_workers(*, drain=False):
            """Stop and join all source-local judge workers."""
            for pool in judge_pools.values():
                pool.stop(timeout=None if drain else 1.0, drain=drain)

        def finish_judge():
            """Drain retained judge jobs and join source-local workers."""
            if not judge_models:
                return
            jobs = _eligible_judge_sidecars(
                judge_input_dir, targets, state, {plugin.id for plugin in active_plugins},
                judge_models, judge_contracts,
            )
            for sidecar, item in jobs:
                enqueue_judge(sidecar, item["target"], item["runner"], item["plugin"])
            for source, pool in judge_pools.items():
                # A source with no active benchmark scheduler should still
                # receive its full judge model pool before the final drain.
                pool.start(judge_model_limits[source])
            # Normal completion must not exit until every queued judge job has
            # called task_done and every worker has terminated.
            stop_judge_workers(drain=True)
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
            # Per-target scalar ``max_tokens`` beats the global config/CLI value.
            effective_max_tokens = target_info.get("max_tokens") or max_tokens
            run_model(state_key, target_info["source"], state, model_active_plugins,
                      source_config, timeout, effective_max_tokens, phase_output_dir,
                      session_seed=session_seed, global_cfg=cfg, stop_event=stop_event,
                      save_responses=args.save_responses,
                      judge_input_dir=judge_input_dir,
                      judge_enqueue=enqueue_judge if judge_models else None,
                      judge_model=judge_model,
                      judge_models=judge_models,
                      judge_prompt_version=JUDGE_PROMPT_VERSION if judge_models else None,
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

        if tui_interrupt_event is not None and tui_interrupt_event.is_set() \
                and not interrupted:
            # The Textual TUI owns the main thread in this mode, so Ctrl+C is
            # delivered there (which sets ``stop_event`` and the interrupt
            # event) rather than raised here. Record the external interrupt so
            # the run is reported as interrupted instead of completed.
            interrupted = True
            run_info["status"] = "interrupted"
            close_active_requests()
            stop_judge_workers()
        elif stop_event.is_set() and not interrupted:
            # ``q``/app teardown sets the shared stop event directly. Treat it
            # like Ctrl+C so we do not enter the final judge drain with queued
            # jobs that no worker is allowed to start.
            interrupted = True
            run_info["status"] = "interrupted"
            close_active_requests()
            stop_judge_workers()

        if not interrupted:
            finish_judge()
        stop_event.set()
        # The TUI thread is a daemon, so we don't need to wait for it. A short
        # timeout keeps the terminal tidy if it happens to finish quickly.
        if tui_thread is not None:
            tui_thread.join(timeout=0.5)

        with persistence_lock:
            # This final snapshot is serialized with all worker saves. Keep
            # the historical benchmark_state.json.tmp path, but never let
            # a final save race a judge worker's save. Unlike incremental
            # saves, failure here must be visible: reporting a completed run
            # without a durable final state would make judging progress
            # appear lost on resume.
            state.save_state(
                state_file,
                plugin_versions=plugin_versions,
                raise_on_error=True,
            )


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
        if state is not None:
            latest_results = state.latest_results()
            run_info["schema_compatibility"] = summarize_schema_compatibility(
                latest_results, active_plugins,
            )
            run_info["judge_criteria"] = summarize_judge_criteria(
                latest_results, active_plugins,
            )
        _write_run_info(output_dir, run_info)


def main():
    """Dispatch to the benchmark orchestrator, keeping the Textual TUI on the
    main thread when it is enabled.

    Textual's Linux driver installs SIGTSTP/SIGCONT handlers in its
    constructor, and ``signal.signal`` is only legal from the main thread of
    the main interpreter. Launching the TUI from a worker thread therefore
    crashes with ``ValueError: signal only works in main thread``. When the
    interactive TUI is enabled we invert the arrangement: the orchestrator runs
    in a worker thread and the TUI owns the main thread.
    """
    if not _textual_tui_enabled():
        _run_benchmark()
        return

    handoff = {"ready": threading.Event(), "interrupt": threading.Event()}
    outcome = {}

    def _run_orchestrator():
        try:
            _run_benchmark(handoff)
        except SystemExit as exc:
            outcome["exit_code"] = exc.code
        except BaseException as exc:  # noqa: BLE001 - re-raise fatal crashes here
            outcome["error"] = exc

    orchestrator = threading.Thread(target=_run_orchestrator, daemon=True)
    orchestrator.start()

    # Wait for setup to complete and hand over the TUI launch, or for the
    # process to exit early (e.g. --list-plugins / --dump-default-config).
    while not handoff["ready"].is_set() and orchestrator.is_alive():
        handoff["ready"].wait(0.05)

    if "args" in handoff:
        try:
            tui_main(*handoff["args"])
        except KeyboardInterrupt:
            # Ctrl+C landed on the main thread while the TUI was active; ask
            # the orchestrator to wind down gracefully.
            handoff["interrupt"].set()
            handoff["stop_event"].set()

    orchestrator.join()

    if "error" in outcome:
        raise outcome["error"]
    if "exit_code" in outcome:
        sys.exit(outcome["exit_code"])


if __name__ == "__main__":
    main()
