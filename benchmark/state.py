"""Benchmark state management.

This module holds the ``BenchmarkState`` class used to track model progress and
persist results across runs.
"""
import copy
import json
import os
import shutil
import tempfile
import threading
import time

from .plugin import SCORE_SCHEMA

# State saves intentionally retain the historical ``<state>.tmp`` path for
# operational compatibility. Serialize all writers so two threads cannot
# truncate or splice the shared temporary file before ``os.replace``.
_STATE_SAVE_LOCK = threading.Lock()

_CORRUPTED_STATE_REPAIRS = {
    # One historical state artifact contains this malformed JSON key. Keep
    # the repair deliberately byte-exact: broad control-character stripping
    # could silently alter model responses and error messages stored in
    # results. The larger affected run is recovered from its complete CSV
    # report by an explicit one-time migration, not by guessing at arbitrary
    # malformed JSON fragments.
    b'"moe-dense_first_chunk_see: : false,':
        b'"moe-dense_first_chunk_seen": false,',
}


def _valid_state_data(data):
    """Return whether ``data`` has the containers required for a state file."""
    return (
        isinstance(data, dict)
        and isinstance(data.get("model_info"), dict)
        and isinstance(data.get("results"), list)
        and isinstance(data.get("active_plugins"), list)
    )


def _known_repaired_bytes(raw):
    """Return bytes for the one audited repair, or ``None`` if inapplicable."""
    model_info_start = raw.find(b'"model_info"')
    results_start = raw.find(b'\n  "results"', model_info_start + 1)
    if model_info_start < 0 or results_start < 0:
        return None
    region = raw[model_info_start:results_start]
    replacements = 0
    for broken, fixed in _CORRUPTED_STATE_REPAIRS.items():
        count = region.count(broken)
        if count:
            region = region.replace(broken, fixed)
            replacements += count
    if replacements != 1:
        return None
    repaired = raw[:model_info_start] + region + raw[results_start:]
    try:
        data = json.loads(repaired)
    except json.JSONDecodeError:
        return None
    return repaired if _valid_state_data(data) else None


def _top_level_value(raw, key, default=None):
    """Decode one top-level JSON value from possibly-corrupt raw bytes."""
    marker = b'"' + key.encode("utf-8") + b'"'
    marker_start = raw.find(marker)
    if marker_start < 0:
        return default
    colon = raw.find(b":", marker_start + len(marker))
    if colon < 0:
        return default
    try:
        text = raw[colon + 1 :].decode("utf-8")
        value, _ = json.JSONDecoder().raw_decode(text.lstrip())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return default
    return value


def _scan_result_objects(raw, array_start):
    """Extract valid result objects and count every top-level array item.

    This is not a JSON sanitizer. The scanner only uses commas and brackets at
    the results-array level to identify *candidate* rows, then asks the
    standard JSON decoder to validate each candidate. Invalid and truncated
    candidates are counted as lost instead of silently disappearing from the
    loss summary. Bytes are never rewritten here.
    """
    if array_start < 0 or array_start >= len(raw) or raw[array_start] != ord("["):
        return [], 0, False, True
    objects = []
    item_start = None
    object_depth = 0
    nested_array_depth = 0
    in_string = False
    escaped = False
    structurally_uncertain = False

    def finish_item(end):
        nonlocal item_start
        if item_start is None:
            return
        span = raw[item_start:end].strip()
        item_start = None
        if not span:
            return
        try:
            value = json.loads(span)
        except json.JSONDecodeError:
            value = None
        objects.append(value if isinstance(value, dict) else None)

    i = array_start + 1
    while i < len(raw):
        byte = raw[i]
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
            i += 1
            continue
        if byte == ord('"'):
            if item_start is None:
                item_start = i
            in_string = True
        elif byte == ord("{"):
            if item_start is None:
                item_start = i
            object_depth += 1
        elif byte == ord("}") and object_depth:
            object_depth -= 1
        elif byte == ord("}") and object_depth == 0:
            structurally_uncertain = True
        elif byte == ord("["):
            if item_start is None:
                item_start = i
            nested_array_depth += 1
        elif byte == ord("]"):
            if object_depth == 0 and nested_array_depth == 0:
                finish_item(i)
                break
            nested_array_depth = max(0, nested_array_depth - 1)
        elif byte == ord(",") and object_depth == 0 and nested_array_depth == 0:
            finish_item(i)
        elif item_start is None and not chr(byte).isspace():
            # Capture malformed scalar tokens such as ``BAD`` so they count
            # as a result item rather than vanishing between commas.
            item_start = i
        i += 1

    # A missing closing bracket or a truncated final item is still a candidate
    # row. Count it as lost if it did not decode above.
    if item_start is not None:
        finish_item(len(raw))
        structurally_uncertain = True
    if object_depth or nested_array_depth or in_string:
        structurally_uncertain = True
    total = len(objects)
    return [obj for obj in objects if isinstance(obj, dict)], total, True, structurally_uncertain


def _scan_model_info(raw, object_start):
    """Extract valid model-info value objects from the outer mapping."""
    if object_start < 0 or object_start >= len(raw) or raw[object_start] != ord("{"):
        return {}
    result = {}
    value_start = None
    object_depth = 0
    in_string = False
    escaped = False
    i = object_start
    while i < len(raw):
        byte = raw[i]
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
            i += 1
            continue
        if byte == ord('"'):
            in_string = True
        elif byte == ord("{"):
            if object_depth == 1:
                value_start = i
            object_depth += 1
        elif byte == ord("}") and object_depth:
            object_depth -= 1
            if object_depth == 0:
                break
            if object_depth == 1 and value_start is not None:
                span = raw[value_start : i + 1]
                try:
                    value = json.loads(span)
                    prefix = raw[max(object_start, value_start - 512) : value_start]
                    key_match = list(__import__("re").finditer(
                        rb'"((?:\\.|[^"\\])*)"\s*:\s*$', prefix
                    ))
                    if key_match:
                        encoded_key = b'"' + key_match[-1].group(1) + b'"'
                        name = json.loads(encoded_key)
                        if isinstance(value, dict):
                            result[name] = value
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
                value_start = None
        i += 1
    return result


def prepare_state_recovery(path):
    """Inspect a corrupt state without changing it.

    The returned summary includes ``total_results``, ``recoverable_results`` and
    ``lost_results``. ``lost_results`` is ``None`` when no results container can
    be located, because the amount of data outside that container is unknown.
    ``data`` is a valid candidate state suitable for an explicitly approved
    recovery; it is never written by this function.
    """
    with open(path, "rb") as handle:
        raw = handle.read()

    repaired = _known_repaired_bytes(raw)
    if repaired is not None:
        data = json.loads(repaired)
        count = len(data["results"])
        return {
            "kind": "known", "data": data, "candidate_bytes": repaired,
            "total_results": count, "recoverable_results": count,
            "lost_results": 0, "results_found": True,
            "counts_certain": True,
        }

    results_marker = raw.find(b'"results"')
    results_colon = raw.find(b":", results_marker + 1) if results_marker >= 0 else -1
    results_start = raw.find(b"[", results_colon + 1) if results_colon >= 0 else -1
    results, total, results_found, structurally_uncertain = _scan_result_objects(raw, results_start)
    model_marker = raw.find(b'"model_info"')
    model_colon = raw.find(b":", model_marker + 1) if model_marker >= 0 else -1
    model_start = raw.find(b"{", model_colon + 1) if model_colon >= 0 else -1
    data = {
        "model_info": _scan_model_info(raw, model_start),
        "results": results,
        "active_plugins": _top_level_value(raw, "active_plugins", []),
        "plugin_versions": _top_level_value(raw, "plugin_versions", {}),
        "session_seed": _top_level_value(raw, "session_seed"),
        "runner": _top_level_value(raw, "runner", "http"),
    }
    if not isinstance(data["active_plugins"], list):
        data["active_plugins"] = []
    if not isinstance(data["plugin_versions"], dict):
        data["plugin_versions"] = {}
    recoverable = len(results)
    return {
        "kind": "partial", "data": data, "candidate_bytes": None,
        "total_results": total if results_found else None,
        "recoverable_results": recoverable,
        "lost_results": (
            total - recoverable if results_found and not structurally_uncertain else None
        ),
        "results_found": results_found,
        "counts_certain": results_found and not structurally_uncertain,
    }


def apply_state_recovery(path, recovery):
    """Atomically install an approved recovery and return a byte backup path."""
    data = recovery.get("data") if isinstance(recovery, dict) else None
    candidate = recovery.get("candidate_bytes") if isinstance(recovery, dict) else None
    if candidate is None:
        candidate = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    parsed = json.loads(candidate)
    if not _valid_state_data(parsed):
        raise ValueError("recovery candidate is not a valid benchmark state")
    directory = os.path.dirname(os.path.abspath(path))
    basename = os.path.basename(path)
    backup_prefix = (
        f"{basename}.pre-repair-"
        if recovery.get("kind") == "known"
        else f"{basename}.pre-recovery-"
    )
    backup_fd, backup = tempfile.mkstemp(
        prefix=backup_prefix, suffix=".bak", dir=directory
    )
    os.close(backup_fd)
    try:
        shutil.copy2(path, backup)
        tmp_fd, tmp = tempfile.mkstemp(
            prefix=f".{basename}.recovery-", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(tmp_fd, "wb") as handle:
                handle.write(candidate)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            raise
    except Exception:
        try:
            if os.path.exists(backup):
                os.remove(backup)
        except OSError:
            pass
        raise
    return backup


def repair_state_file(path):
    """Apply only the audited zero-loss repair, returning its backup path."""
    recovery = prepare_state_recovery(path)
    if recovery["kind"] != "known" or recovery["lost_results"] != 0:
        return None
    return apply_state_recovery(path, recovery)


class BenchmarkState:
    """Thread-safe shared state for parallel benchmark execution."""

    def __init__(self, models, plugin_ids, session_seed=None, runner="http"):
        self._lock = threading.Lock()
        self.results = []
        self._model_info = {}
        self._log = []
        # Monotonic counter bumped by every state mutation. The live TUI
        # polls ``revision`` to skip rebuilding its frame when nothing it
        # displays has changed since the last tick; a constant frame costs
        # no CPU or terminal output.
        self._revision = 0
        self.plugin_ids = list(plugin_ids)
        self.session_seed = session_seed
        self.runner = runner
        self.score_schema = SCORE_SCHEMA
        self._judge_progress = {}
        self._judge_activity = {}
        # Judge runners can remain selected between individual cell requests;
        # this is distinct from ``_judge_activity``, which only represents
        # requests currently executing. The TUI uses this transient set to
        # keep selected judges green across queue/transport gaps.
        self._judge_selected = set()
        self._next_judge_activity_id = 0
        # Optional append-only result journal (see ``set_journal_path``).
        self._journal_path = None
        for name, info in models.items():
            if isinstance(info, dict):
                source = info.get("source", "Default")
                api_model = info.get("api_model", name)
                system_prompt = info.get("system_prompt")
                is_agent = info.get("is_agent", False)
                info_runner = info.get("runner", runner)
            else:
                info_runner = runner
                source = info
                api_model = name
                system_prompt = None
                is_agent = False
            self._model_info[name] = {
                "source": source,
                "api_model": api_model,
                "system_prompt": system_prompt,
                "is_agent": is_agent,
                "runner": info_runner,
                "status": "pending",
                "ttft": None,
                "error": None, "elapsed": 0,
                "attempt": 0,
                "max_tok": 0,
                "attempt_start": 0,
                "last_error": "",
                "phase_detail": "",
                # Model warm-up state. These fields are intentionally
                # session-scoped; preload outcomes are summarized in
                # run-info.json and failed legs are represented in results.
                "preloading": False,
                "preload_start_ts": 0,
                "preload_status": None,
                "preload_time": None,
                "preload_error": None,
                # In-flight plugin task ids; canonical source-of-truth for the
                # live TUI's "[streaming]"/"[requested]" bracket cells
                # (pre-first-tok and post-first-tok transient share the same
                # label; the elapsed suffix once wait > 2s is the cue for
                # "no first chunk yet") and the table's yellow highlight.
                # ``__init__`` sets this to ``[]``; the
                # runtime tracks it via ``start_plugin_run`` / ``finish_plugin_run``
                # and ``load_state`` clears it on resume (no plugin task is
                # actually running until a worker picks the task up).
                "running_pids": [],
            }
            for pid in plugin_ids:
                self._model_info[name][f"{pid}_score"] = None
                self._model_info[name][f"{pid}_judge_score"] = None
                self._model_info[name][f"{pid}_judge_confidence"] = None
                self._model_info[name][f"{pid}_judge_rationale"] = None
                self._model_info[name][f"{pid}_judge_criteria"] = []
                self._model_info[name][f"{pid}_judge_consensus_by_contract"] = {}
                self._model_info[name][f"{pid}_judge_selected_contract"] = None
                self._model_info[name][f"{pid}_judge_error"] = None
                self._model_info[name][f"{pid}_judge_input_sha256"] = None
                self._model_info[name][f"{pid}_judge_votes"] = []
                self._model_info[name][f"{pid}_judge_complete"] = False
                self._model_info[name][f"{pid}_judge_queued"] = False
                self._model_info[name][f"{pid}_tps"] = None
                self._model_info[name][f"{pid}_response_time"] = None
                self._model_info[name][f"{pid}_output_tokens"] = None
                # Thinking/reasoning token count (from ``reasoning_content``)
                # and the combined content+thinking total, so reports can
                # break token usage down. ``None`` until the plugin runs.
                self._model_info[name][f"{pid}_thinking_tokens"] = None
                self._model_info[name][f"{pid}_total_tokens"] = None
                # Empty-response classification (None when the response had
                # content; otherwise one of "error", "thinking-truncation",
                # "thinking-only", "max-tokens", "empty"). Mirrored from
                # the result dict by ``_run_plugins.run_one`` and surfaced
                # in meta.json/CSV.
                self._model_info[name][f"{pid}_empty_reason"] = None
                # Streaming byte counter for in-flight plugins; incremented
                # via ``add_bytes_received`` per SSE delta by the runtime.
                # Starts at 0; ``start_plugin_run`` resets it on each
                # dispatch so retry runs don't carry a stale count.
                # ``finish_plugin_run`` leaves it in place (post-flight
                # cells fall back to the standard 4-cell results layout
                # after the per-plugin St column was deleted as redundant,
                # so this transient is harmless).
                self._model_info[name][f"{pid}_bytes_received"] = 0
                # First-chunk-received flag, callable via
                # ``mark_first_chunk_seen`` from the SSE parse layer the
                # moment the first delta lands. The flag drives a separate
                # downstream code path -- the live TUI's
                # ``[streaming - Ns]`` bracket form (pre-chunk, with
                # the seconds suffix past the elapsed threshold) versus
                # ``[streaming - N tok]`` (real counter, post-chunk with
                # nonzero ``bytes_received``). Default False so the
                # renderer can read it without a ``KeyError``; reset in
                # ``start_plugin_run`` to drop carry-over from a
                # previous dispatch.
                self._model_info[name][f"{pid}_first_chunk_seen"] = False
                # Timestamp of the first SSE non-empty delta, set in the
                # same atomic write as the flag flip. The live-footer
                # renderer (``_build_live_indicators``) reads this field
                # to switch the per-plugin live indicator from the
                # aggregate ``[pre-stream: K]`` bucket to the per-plugin
                # ``[<pid>: N tok]`` form. Default 0 so the reader can
                # tell a "no chunk has landed yet" snapshot from a real
                # timestamp; reset in ``start_plugin_run`` to drop
                # carry-over from a previous dispatch.
                self._model_info[name][f"{pid}_first_tok_ts"] = 0
                # Parallel reasoning/thinking byte counter incremented
                # via ``add_thinking_bytes_received`` per SSE
                # ``reasoning_content`` delta. Distinct from
                # ``{pid}_bytes_received`` (which counts PRIMARY
                # ``content`` only) so the live TUI can show a real
                # ticker on the thinking-only phase of a deepseek-r1 /
                # Qwen3 / o1-style stream BEFORE the response's main
                # content begins flowing. The cell renderer reads this
                # counter to switch from ``[streaming - Ns]`` to
                # ``[streaming - N think-tok]`` once reasoning deltas
                # have accumulated but no content chunk has yet landed.
                # Default 0 so the renderer can read it without a
                # ``KeyError``; reset in ``start_plugin_run`` to drop
                # carry-over from a previous dispatch.
                self._model_info[name][f"{pid}_thinking_bytes_received"] = 0

    def _mark_changed(self):
        """Bump the revision counter; must be called under ``self._lock``."""
        self._revision += 1

    @property
    def revision(self):
        """Monotonic counter bumped on every state mutation.

        Read without the lock: a plain integer attribute is GIL-atomic, and
        the TUI polls it every refresh tick, so lock-free reads keep the
        polling path cheap. A stale read only delays one repaint.
        """
        return self._revision

    def has_live_work(self):
        """True when any model is mid-task or preloading.

        The live TUI uses this to keep time-based elements (streaming
        seconds, preload elapsed) ticking even between state mutations.
        """
        with self._lock:
            return any(
                info.get("running_pids") or info.get("preloading")
                for info in self._model_info.values()
            )

    def update(self, model_name, **kwargs):
        with self._lock:
            self._mark_changed()
            self._model_info[model_name].update(kwargs)

    def start_plugin_run(self, model_name, pid):
        """Atomically mark a plugin task as in-flight on this model.

        Sets ``running_pids`` to include ``pid`` and the model's
        ``status`` to canonical ``"running"``. With parallel plugin
        threads (max_workers > 1), multiple pids accumulate; the
        ``finish_plugin_run`` call when the task returns drops just
        that pid. Lock is per-instance so concurrent calls from any
        number of workers execute atomically. Also zeroes
        ``f"{pid}_bytes_received"`` so a retry run doesn't surface a
        stale streaming byte count from the previous dispatch.
        """
        with self._lock:
            self._mark_changed()
            info = self._model_info[model_name]
            cur = list(info.get("running_pids") or [])
            if pid not in cur:
                cur.append(pid)
            info["running_pids"] = cur
            info["status"] = "running"
            # Reset the per-plugin streaming byte counter AND the
            # first-chunk flag so a retry dispatch doesn't carry
            # forward either signal from the previous (now-finished)
            # attempt. ``mark_first_chunk_seen`` is a sticky flag
            # within a single dispatch; ``start_plugin_run`` owns the
            # cross-dispatch reset semantics. This mirrors the
            # dispatch-time reset (``attempt_start`` is rewritten on
            # each entry to ``_run_plugins``).
            info[f"{pid}_bytes_received"] = 0
            info[f"{pid}_first_chunk_seen"] = False
            info[f"{pid}_first_tok_ts"] = 0
            # Reset the parallel thinking/reasoning byte counter so a
            # retry dispatch doesn't carry forward the previous
            # attempt's accumulated ``reasoning_content`` length.
            # Without this reset, a 429 retry on a thinking model would
            # resume the live TUI's ``[streaming - N think-tok]``
            # ticker at the N from the failed dispatch when the retry
            # hadn't yet produced any thinking content.
            info[f"{pid}_thinking_bytes_received"] = 0
            # Per-plugin dispatch timestamp so the live TUI can show
            # wall-clock seconds since *this* plugin's request started,
            # rather than the model-level timer which grows for the
            # whole model. Reset on every dispatch so retries and
            # sequential plugins get a fresh zero point.
            info[f"{pid}_start_ts"] = time.monotonic()

    def add_bytes_received(self, model_name, pid, n_bytes):
        """Atomically accumulate streaming bytes for a plugin task.

        Called once per parsed SSE delta by ``stream_request`` via an
        ``on_chunk`` callback installed in ``_run_plugin_task``. The
        bytes counter drives the table cell's ``[streaming - N tok]``
        bracket status and the live footer's ``[name: N tok]`` entries
        -- the operator gets a live "ticker" feel instead of a static
        "[streaming]" until the full response arrives. Falsy ``n_bytes``
        (None, 0, "") are ignored to avoid spurious zero-add writes; an
        unknown model/pid raises ``KeyError`` (Python standard behaviour)
        so the caller can detect a programming error immediately rather
        than silently drop bytes.

        Self-check (``RuntimeError``): if bytes are about to be added
        while ``{pid}_first_chunk_seen`` is still ``False``, the
        SSE-layer caller forgot to fire ``mark_first_chunk_seen`` on
        the first delta. This is the wiring contract: every code path
        that adds bytes MUST have already flipped the marker -- the
        cell renderer cannot switch from the pre-chunk
        ``[streaming]`` / ``[streaming - Ns]`` form to the real-
        counter form ``[streaming - N tok]`` without the marker
        being flipped first. We raise loudly here so the drift fails
        the test suite (and the live run) the moment a future caller
        forgets the hook, rather than silently keeping the bracket
        stuck on the pre-chunk form indefinitely. The check is inside
        ``self._lock`` so a concurrent ``mark_first_chunk_seen``
        cannot interleave between the read and the increment.

        Note: this method does NOT auto-fire ``mark_first_chunk_seen``;
        the SSE parse loop calls the marker independently so transport
        events (handshake ack, first delta, structured-error ack etc.)
        can flip the flag without growing the byte counter, and so the
        bytes-counting path stays purely additive for tests.
        """
        if not n_bytes:
            return
        with self._lock:
            self._mark_changed()
            info = self._model_info[model_name]
            if not info.get(f"{pid}_first_chunk_seen"):
                raise RuntimeError(
                    f"add_bytes_received({model_name!r}, {pid!r}, ...) called "
                    f"before mark_first_chunk_seen -- SSE parse layer forgot "
                    f"to fire the first-chunk marker. This is a wiring bug; "
                    f"see benchmark_core._run_plugin_task's on_chunk closure."
                )
            key = f"{pid}_bytes_received"
            info[key] = (info.get(key) or 0) + n_bytes

    def add_thinking_bytes_received(self, model_name, pid, n_bytes):
        """Atomically accumulate streaming reasoning/thinking bytes for
        a plugin task.

        Parallel to ``add_bytes_received`` but specifically for the
        ``reasoning_content`` chunks emitted by thinking-capable
        models BEFORE primary ``content`` starts flowing. Called once
        per parsed SSE ``reasoning_content`` delta by ``stream_request``
        via the ``on_think_chunk`` closure installed in
        ``_run_plugin_task``. The parallel counter drives the live
        TUI's ``[streaming - N think-tok]`` cell form and the live
        footer's ``[<pid>: N think-tok (e s)]`` indicator so the
        operator gets a real ticking widget on the thinking-only
        phase (a deepseek-r1 / Qwen3 / o1-style stream that has
        produced 2 000 chars of ``reasoning_content`` but zero chars
        of ``content`` shows a tokenised readout, not a seconds-only
        placeholder).

        Shares the same ``mark_first_chunk_seen`` wiring check as
        ``add_bytes_received`` -- the words "the request has begun"
        apply equally to the first ``reasoning_content`` delta as to
        the first ``content`` delta, so we reuse the same flag (set
        in the same atomic write as the bytes increment). Operators
        do not need to distinguish "first thinking delta arrived" from
        "first content delta arrived" as separate gates; they only
        need to know the response has begun.
        Falsy ``n_bytes`` and unknown model/pid behave identically to
        ``add_bytes_received``.
        """
        if not n_bytes:
            return
        with self._lock:
            self._mark_changed()
            info = self._model_info[model_name]
            if not info.get(f"{pid}_first_chunk_seen"):
                raise RuntimeError(
                    f"add_thinking_bytes_received({model_name!r}, {pid!r}, ...) called "
                    f"before mark_first_chunk_seen -- SSE parse layer forgot "
                    f"to fire the first-chunk marker. This is a wiring bug; "
                    f"see benchmark_core._run_plugin_task's on_think_chunk closure."
                )
            key = f"{pid}_thinking_bytes_received"
            info[key] = (info.get(key) or 0) + n_bytes

    def mark_first_chunk_seen(self, model_name, pid, ts=None):
        """Atomically mark that a first streaming chunk has arrived for
        a plugin task.

        Idempotent: calling repeatedly is safe (the flag only flips
        False -> True; we never set it back to False once a chunk has
        landed -- retry/resume semantics are owned by
        ``start_plugin_run`` which resets per-dispatch). The change
        drives the live TUI's bracket form: before the flag flips,
        the cell shows ``[streaming]`` or ``[streaming - Ns]``
        (depending on wall-clock wait vs the elapsed threshold);
        after the flag flips and ``bytes_received`` becomes positive,
        the cell shows the real ``[streaming - N tok]`` counter.

        The optional ``ts`` parameter atomically writes
        ``{pid}_first_tok_ts`` ONLY on the False -> True transition
        (i.e. the very first call that flips the flag). Subsequent
        calls preserve the original timestamp -- the live footer
        ``_build_live_indicators`` consumer needs the FIRST chunk's
        timestamp to compute "time-to-first-token" correctly, not
        whichever delta happens to be the Nth. ``ts`` is ``None``
        for backward compatibility with the bool-only contract
        (callers that don't care about the timer can skip it). The
        ``first_tok_ts`` write happens INSIDE the same ``self._lock``
        read-then-write window so a concurrent state reader never sees
        flag=True and ``first_tok_ts``=0 (which would render the per-
        plugin live indicator with no timestamp anchor).

        Decoupled from ``add_bytes_received`` so SSE-layer callers can
        flip the flag on whatever transport event constitutes "the
        response has actually begun" (first parsed delta, first
        non-heartbeat byte, first acknowledged HTTP chunk, etc.)
        without coupling to the bytes-accumulation path.
        """
        with self._lock:
            self._mark_changed()
            info = self._model_info[model_name]
            flag_key = f"{pid}_first_chunk_seen"
            was_false = not info[flag_key]
            info[flag_key] = True
            if was_false and ts is not None:
                info[f"{pid}_first_tok_ts"] = ts

    def finish_plugin_run(self, model_name, pid):
        """Atomically clear a plugin task's in-flight marker on this model.

        Removes ``pid`` from ``running_pids`` but leaves ``status``
        alone; the outer task (``run_model``) commits the final
        ``completed`` / ``failed`` / ``pending`` status once all
        in-flight plugin tasks have resolved. The brief "finalising"
        window between the last plugin finishing and the outer status
        update is therefore rendered as ``status="running"`` with
        ``running_pids=[]`` -- an unambiguous, momentary snapshot.
        """
        with self._lock:
            self._mark_changed()
            info = self._model_info[model_name]
            cur = [p for p in (info.get("running_pids") or []) if p != pid]
            info["running_pids"] = cur
            # Status intentionally untouched; see docstring.

    def add_result(self, result):
        with self._lock:
            self._mark_changed()
            # Judge workers may finish a plugin after its score is published
            # but before the enclosing model result row is appended. Carry the
            # per-plugin judge fields from live model_info into this new row so
            # a concurrent judge update is not lost on resume.
            state_key = result.get("state_key", result.get("model"))
            info = self._model_info.get(state_key, {})
            for key, value in info.items():
                if "_judge_" not in key and key != "judge_status":
                    continue
                # Default-initialized fields do not replace meaningful values
                # already carried by an older result row. Conversely, a live
                # judge update must win over a stale row value before append.
                if value not in (None, False, [], ""):
                    result[key] = value
                else:
                    result.setdefault(key, value)
            self.results.append(result)
            self._journal_append(result)

    def set_journal_path(self, path, truncate=False):
        """Enable append-only result journaling to ``path``.

        Every completed result is appended as one JSON line, so a crash that
        truncates ``benchmark_state.json`` can still replay the completed
        results. ``truncate`` starts a fresh journal (safe once the main state
        file has been loaded successfully); it must be False when the state
        file is corrupt and the journal is needed for recovery.
        """
        self._journal_path = path
        if truncate and path:
            try:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("")
            except OSError:
                pass

    def _journal_append(self, result):
        """Best-effort append of one completed result to the journal."""
        if not self._journal_path:
            return
        try:
            with open(self._journal_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, default=str) + "\n")
                handle.flush()
        except OSError:
            pass

    @staticmethod
    def replay_journal(path):
        """Replay an append-only result journal into a list of results.

        Returns ``[]`` when the journal is missing or unreadable. A partial
        trailing line (crash mid-write) is tolerated and dropped.
        """
        if not path or not os.path.exists(path):
            return []
        results = []
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        return results

    def set_judge_progress(self, progress):
        """Replace live per-judge progress shown by the TUI footer."""
        with self._lock:
            self._mark_changed()
            self._judge_progress = {
                model: dict(values) for model, values in (progress or {}).items()
            }

    def update_judge_progress(self, model, **values):
        """Update one judge model's live progress counters."""
        with self._lock:
            self._mark_changed()
            current = dict(self._judge_progress.get(model, {}))
            current.update(values)
            self._judge_progress[model] = current

    def increment_judge_progress(self, model, *, expected=0, completed=0, failed=0):
        """Atomically increment a judge's expected, completed, and failed counts."""
        with self._lock:
            self._mark_changed()
            current = dict(self._judge_progress.get(model, {}))
            current["expected"] = current.get("expected", 0) + expected
            current["completed"] = current.get("completed", 0) + completed
            current["failed"] = current.get("failed", 0) + failed
            self._judge_progress[model] = current
            return dict(current)

    def replace_judge_progress(self, model, *, previous_completed=0,
                               previous_failed=0, completed=0, failed=0):
        """Replace one judge's prior outcome in the live progress totals.

        Judge retries replace the vote for a model/plugin cell rather than
        adding a second current outcome. In particular, a failed attempt that
        later succeeds must move from ``failed`` to ``completed`` instead of
        leaving both counters inflated. ``expected`` is intentionally
        unchanged because the cell was already part of the judge's workload.
        """
        with self._lock:
            self._mark_changed()
            current = dict(self._judge_progress.get(model, {}))
            current["completed"] = max(
                0,
                current.get("completed", 0) - previous_completed + completed,
            )
            current["failed"] = max(
                0,
                current.get("failed", 0) - previous_failed + failed,
            )
            self._judge_progress[model] = current
            return dict(current)

    def judge_progress_snapshot(self):
        """Return a copy of live per-judge progress for the TUI."""
        with self._lock:
            return {model: dict(values) for model, values in self._judge_progress.items()}

    def set_judge_selected(self, judge_model, selected):
        """Mark whether a judge runner is selected for work on its source."""
        with self._lock:
            self._mark_changed()
            if selected:
                self._judge_selected.add(judge_model)
            else:
                self._judge_selected.discard(judge_model)

    def judge_selected_snapshot(self):
        """Return judge runners selected between individual cell requests."""
        with self._lock:
            return set(self._judge_selected)

    def start_judge_activity(self, judge_model, target, plugin):
        """Register one currently executing judge request for the live TUI."""
        with self._lock:
            self._mark_changed()
            activity_id = self._next_judge_activity_id
            self._next_judge_activity_id += 1
            self._judge_activity[activity_id] = {
                "judge": judge_model,
                "target": target,
                "plugin": plugin,
                "tokens": 0,
                "thinking_tokens": 0,
                "content_tokens": 0,
                "started": time.monotonic(),
            }
            return activity_id

    def update_judge_activity(self, activity_id, *, tokens=None,
                               thinking_tokens=None, content_tokens=None):
        """Update live total, thinking, and content tokens for one judge."""
        with self._lock:
            self._mark_changed()
            activity = self._judge_activity.get(activity_id)
            if activity is None:
                return
            if thinking_tokens is not None:
                activity["thinking_tokens"] = max(0, int(thinking_tokens))
            if content_tokens is not None:
                activity["content_tokens"] = max(0, int(content_tokens))
            if tokens is not None:
                activity["tokens"] = max(0, int(tokens))
            elif thinking_tokens is not None or content_tokens is not None:
                activity["tokens"] = (
                    activity["thinking_tokens"] + activity["content_tokens"]
                )

    def clear_judge_queued(self, target, plugin):
        """Clear the transient queued marker for a judgeable plugin."""
        with self._lock:
            self._mark_changed()
            if target in self._model_info:
                self._model_info[target][f"{plugin}_judge_queued"] = False

    def finish_judge_activity(self, activity_id):
        """Remove one completed judge request from the live TUI."""
        with self._lock:
            self._mark_changed()
            self._judge_activity.pop(activity_id, None)

    def judge_activity_snapshot(self):
        """Return active judge requests with current elapsed seconds."""
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "judge": activity["judge"],
                    "target": activity["target"],
                    "plugin": activity["plugin"],
                    "tokens": activity.get("tokens", 0),
                    "thinking_tokens": activity.get("thinking_tokens", 0),
                    "content_tokens": activity.get("content_tokens", 0),
                    "elapsed": max(0, int(now - activity["started"])),
                }
                for activity in self._judge_activity.values()
            ]

    def update_judge_result(self, state_key, runner, plugin_id, *, score=None,
                            confidence=None, rationale=None, criteria=None,
                            consensus_by_contract=None, selected_contract=None,
                            error=None,
                            input_sha256=None, votes=None, status=None,
                            complete=None):
        """Persist one judge outcome in live model info and latest result.

        Judge updates are independent of benchmark completion and therefore
        must not append a second benchmark result row or change its status.
        """
        fields = {
            f"{plugin_id}_judge_score": score,
            f"{plugin_id}_judge_confidence": confidence,
            f"{plugin_id}_judge_rationale": rationale,
            f"{plugin_id}_judge_criteria": criteria if criteria is not None else [],
            f"{plugin_id}_judge_error": error,
        }
        if consensus_by_contract is not None:
            fields[f"{plugin_id}_judge_consensus_by_contract"] = consensus_by_contract
        if selected_contract is not None:
            fields[f"{plugin_id}_judge_selected_contract"] = selected_contract
        if input_sha256 is not None:
            fields[f"{plugin_id}_judge_input_sha256"] = input_sha256
        if votes is not None:
            fields[f"{plugin_id}_judge_votes"] = votes
        if status is not None:
            fields["judge_status"] = status
        if complete is not None:
            fields[f"{plugin_id}_judge_complete"] = complete
            if complete:
                fields[f"{plugin_id}_judge_queued"] = False
        with self._lock:
            self._mark_changed()
            if state_key in self._model_info:
                self._model_info[state_key].update(fields)
            for result in reversed(self.results):
                result_key = result.get("state_key", result.get("model"))
                if result_key == state_key and result.get("runner", "http") == runner:
                    result.update(fields)
                    break

    def set_judge_models(self, judge_models):
        """Refresh the active judge identities on live and persisted rows."""
        models = list(dict.fromkeys(judge_models or []))
        with self._lock:
            self._mark_changed()
            for info in self._model_info.values():
                info["judge_models"] = list(models)
            for result in self.results:
                result["judge_models"] = list(models)

    def set_active_judge_contracts(self, contracts):
        """Project the current judge contracts onto live/current result rows.

        Resume can load a completed model without calling ``run_model`` again.
        In that case its live ``model_info`` may still identify the previous
        judge contract, causing the TUI to render historical votes even though
        the current judge progress correctly starts at zero. Update the active
        contract before rendering and clear only the legacy flat projection
        when it changes. Versioned votes and consensus remain untouched.
        """
        if not contracts:
            return
        projection_fields = {
            "judge_score": None,
            "judge_confidence": None,
            "judge_rationale": None,
            "judge_criteria": [],
            "judge_error": None,
            "judge_input_sha256": None,
            "judge_complete": False,
            "judge_queued": False,
        }
        with self._lock:
            self._mark_changed()
            current_results = {}
            for result in reversed(self.results):
                identity = (
                    result.get("state_key", result.get("model")),
                    result.get("runner", "http"),
                )
                if identity not in current_results:
                    current_results[identity] = result

            containers = [*self._model_info.values(), *current_results.values()]
            for container in containers:
                changed = False
                for plugin_id, contract_id in contracts.items():
                    selected_key = f"{plugin_id}_judge_selected_contract"
                    if container.get(selected_key) == contract_id:
                        continue
                    container[selected_key] = contract_id
                    changed = True
                    for suffix, value in projection_fields.items():
                        container[f"{plugin_id}_{suffix}"] = (
                            list(value) if isinstance(value, list) else value
                        )
                if changed:
                    container["judge_status"] = "pending"

    def snapshot(self):
        with self._lock:
            return {k: dict(v) for k, v in self._model_info.items()}

    @property
    def completed(self):
        with self._lock:
            return sum(1 for s in self._model_info.values() if s["status"] == "completed")

    @property
    def total(self):
        return len(self._model_info)

    def log(self, model_name, msg):
        with self._lock:
            self._mark_changed()
            self._log.append((time.time(), model_name, msg))
            if len(self._log) > 100:
                self._log = self._log[-100:]

    def recent_log(self, n=5):
        with self._lock:
            return self._log[-n:]

    def save_state(self, path, plugin_versions=None, *, raise_on_error=False):
        with self._lock:
            # Preload indicators are session-only. Copy model info while
            # holding the lock, then omit the transient probe fields so a
            # resumed process cannot display stale activity or treat an old
            # warm-up as evidence that the backend is still warm.
            transient_preload_fields = {
                "preloading", "preload_start_ts", "preload_status",
                "preload_time", "preload_error",
            }
            persisted_model_info = {
                name: {
                    key: value for key, value in info.items()
                    if key not in transient_preload_fields
                }
                for name, info in self._model_info.items()
            }
            # Deep-copy all mutable containers while holding the state lock.
            # The JSON dump happens after releasing it, so live judge updates
            # cannot mutate nested votes/results halfway through serialization.
            data = copy.deepcopy({
                "model_info": persisted_model_info,
                "results": self.results,
                "active_plugins": self.plugin_ids,
                "plugin_versions": plugin_versions or {},
                "session_seed": self.session_seed,
                "runner": self.runner,
                "judge_progress": self._judge_progress,
                "score_schema": SCORE_SCHEMA,
            })
        tmp = path + ".tmp"
        try:
            # Keep the fixed temporary filename, but serialize the complete
            # write/flush/replace sequence. This prevents concurrent judge and
            # benchmark snapshots from interleaving through the same .tmp file.
            with _STATE_SAVE_LOCK:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
                # Persist the directory entry when the platform supports it;
                # the state file itself remains the stable atomic-replace path.
                try:
                    directory_fd = os.open(
                        os.path.dirname(os.path.abspath(path)) or ".", os.O_DIRECTORY
                    )
                except (AttributeError, OSError):
                    directory_fd = None
                if directory_fd is not None:
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            if raise_on_error:
                raise
            return False
        return True

    def latest_results(self):
        """Return only the most recent result per model (deduplicates across runs)."""
        with self._lock:
            seen = {}
            for r in self.results:
                key = (r.get("state_key", r["model"]), r.get("runner", "http"))
                seen[key] = r
            return list(seen.values())

    @classmethod
    def load_state(cls, path, models, plugin_ids, *, rerun_failed=True):
        with open(path) as f:
            data = json.load(f)
        session_seed = data.get("session_seed")
        runner = data.get("runner", "http")
        state = cls(models, plugin_ids, session_seed=session_seed, runner=runner)
        if data.get("score_schema") != SCORE_SCHEMA:
            raise ValueError(
                f"Unsupported score schema {data.get('score_schema')!r}; "
                f"start a new run with {SCORE_SCHEMA!r}"
            )
        saved_plugins = data.get("active_plugins", [])
        new_plugins = [pid for pid in plugin_ids if pid not in saved_plugins]
        saved_info = data.get("model_info", {})
        saved_results = data.get("results", [])
        latest_by_model = {}
        for r in saved_results:
            key = (r.get("state_key", r["model"]), r.get("runner", "http"))
            latest_by_model[key] = r
        for name, info in saved_info.items():
            if name in state._model_info:
                # Strip transient in-flight state on resume: no plugin task
                # is actually running until a worker picks the task up, so
                # carrying stale ``running_pids`` forward causes phantom
                # "[streaming]" bracket cells and a false-yellow highlight in the
                # live TUI. The migration used to *preserve* this state,
                # but that assumption broke once the visual-layer read
                # ``running_pids`` as the source-of-truth.
                info.pop("running_pids", None)
                for pid in plugin_ids:
                    info[f"{pid}_judge_queued"] = False
                saved_status = info.get("status")
                # Normalise both legacy "running_<pid>" pid-suffix strings
                # AND the canonical "running" status to "pending" so the
                # worker re-promotes them when the task is actually picked
                # up. This is a behaviour change from the prior
                # migration (which kept status="running" + running_pids
                # populated), justified by the phantom-state bug.
                if saved_status == "running" or (
                    isinstance(saved_status, str)
                    and saved_status.startswith("running_")
                ):
                    info["status"] = "pending"
                # Merge saved values into the fully-initialized default dict.
                # This keeps backward compatibility with older state files that
                # may be missing newer keys (e.g. phase_detail, attempt).
                state._model_info[name].update(info)
                # Preload indicators are deliberately session-only. A saved
                # state may have been written while a probe was in progress or
                # immediately after one finished, but a resumed process must
                # never display stale preload activity or reuse its outcome as
                # proof that the backend is warm.
                state._model_info[name].update({
                    "preloading": False,
                    "preload_start_ts": 0,
                    "preload_status": None,
                    "preload_time": None,
                    "preload_error": None,
                })
                if new_plugins:
                    for pid in new_plugins:
                        state._model_info[name].setdefault(f"{pid}_score", None)
                        state._model_info[name].setdefault(f"{pid}_judge_score", None)
                        state._model_info[name].setdefault(f"{pid}_judge_confidence", None)
                        state._model_info[name].setdefault(f"{pid}_judge_rationale", None)
                        state._model_info[name].setdefault(f"{pid}_judge_criteria", [])
                        state._model_info[name].setdefault(f"{pid}_judge_consensus_by_contract", {})
                        state._model_info[name].setdefault(f"{pid}_judge_selected_contract", None)
                        state._model_info[name].setdefault(f"{pid}_judge_error", None)
                        state._model_info[name].setdefault(f"{pid}_judge_input_sha256", None)
                        state._model_info[name].setdefault(f"{pid}_judge_votes", [])
                        state._model_info[name].setdefault(f"{pid}_judge_complete", False)
                        state._model_info[name].setdefault(f"{pid}_judge_queued", False)
                        state._model_info[name].setdefault(f"{pid}_tps", None)
                        state._model_info[name].setdefault(f"{pid}_response_time", None)
                        state._model_info[name].setdefault(f"{pid}_output_tokens", None)
                        state._model_info[name].setdefault(f"{pid}_thinking_tokens", None)
                        state._model_info[name].setdefault(f"{pid}_total_tokens", None)
                    if state._model_info[name].get("status") == "completed":
                        state._model_info[name]["status"] = "pending"
                elif info.get("status") == "completed":
                    expected_runner = state._model_info[name].get("runner", runner)
                    latest = latest_by_model.get((name, expected_runner))
                    if latest is not None:
                        result_plugins = set(latest.get("plugin_versions", {}))
                        needs_current_plugin_check = bool(result_plugins)
                        if (
                            needs_current_plugin_check
                            and (
                                not result_plugins.issubset(set(plugin_ids))
                                or any(f"{pid}_score" not in latest for pid in plugin_ids)
                            )
                        ):
                            state._model_info[name]["status"] = "pending"
        state.results = data.get("results", [])
        state._judge_progress = data.get("judge_progress", {})
        for result in state.results:
            for pid in plugin_ids:
                result.setdefault(f"{pid}_judge_criteria", [])
                result.setdefault(f"{pid}_judge_consensus_by_contract", {})
                result.setdefault(f"{pid}_judge_selected_contract", None)
                if f"{pid}_judge_complete" not in result:
                    result[f"{pid}_judge_complete"] = bool(
                        result.get(f"{pid}_judge_votes")
                        and result.get("judge_status") in {"complete", "partial", "failed"}
                    )
        for info in state._model_info.values():
            for pid in plugin_ids:
                info.setdefault(f"{pid}_judge_criteria", [])
                info.setdefault(f"{pid}_judge_consensus_by_contract", {})
                info.setdefault(f"{pid}_judge_selected_contract", None)
        for name, info in state._model_info.items():
            if info.get("status") == "completed":
                continue
            if not rerun_failed:
                continue
            # Reset failed (and any other non-completed) models so they are
            # re-run when the benchmark restarts.
            info["status"] = "pending"
            info["last_error"] = ""
            info["error"] = None
            info.setdefault("attempt_start", 0)
        return state


