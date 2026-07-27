"""Benchmark state management.

This module holds the ``BenchmarkState`` class used to track model progress and
persist results across runs.
"""
import json
import os
import threading
import time


class BenchmarkState:
    """Thread-safe shared state for parallel benchmark execution."""
    def __init__(self, models, plugin_ids, session_seed=None):
        self._lock = threading.Lock()
        self.results = []
        self._model_info = {}
        self._log = []
        self.plugin_ids = list(plugin_ids)
        self.session_seed = session_seed
        for name, info in models.items():
            if isinstance(info, dict):
                source = info.get("source", "Default")
                api_model = info.get("api_model", name)
                system_prompt = info.get("system_prompt")
                is_agent = info.get("is_agent", False)
            else:
                source = info
                api_model = name
                system_prompt = None
                is_agent = False
            self._model_info[name] = {
                "source": source,
                "api_model": api_model,
                "system_prompt": system_prompt,
                "is_agent": is_agent,
                "status": "pending",
                "ttft": None,
                "error": None, "elapsed": 0,
                "attempt": 0,
                "max_tok": 0,
                "attempt_start": 0,
                "last_error": "",
                "phase_detail": "",
                # In-flight plugin task ids; canonical source-of-truth for the
                # live TUI's "[waiting]"/"[streaming]" cells and the table's
                # yellow highlight. ``__init__`` sets this to ``[]``; the
                # runtime tracks it via ``start_plugin_run`` / ``finish_plugin_run``
                # and ``load_state`` clears it on resume (no plugin task is
                # actually running until a worker picks the task up).
                "running_pids": [],
            }
            for pid in plugin_ids:
                self._model_info[name][f"{pid}_score"] = None
                self._model_info[name][f"{pid}_tps"] = None
                self._model_info[name][f"{pid}_response_time"] = None
                self._model_info[name][f"{pid}_output_tokens"] = None
                # Streaming byte counter for in-flight plugins; incremented
                # via ``add_bytes_received`` per SSE delta by the runtime.
                # Starts at 0; ``start_plugin_run`` resets it on each
                # dispatch so retry runs don't carry a stale count.
                # ``finish_plugin_run`` leaves it in place (post-flight
                # cells fall back to the standard 5-cell results layout,
                # so this transient is harmless).
                self._model_info[name][f"{pid}_bytes_received"] = 0

    def update(self, model_name, **kwargs):
        with self._lock:
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
            info = self._model_info[model_name]
            cur = list(info.get("running_pids") or [])
            if pid not in cur:
                cur.append(pid)
            info["running_pids"] = cur
            info["status"] = "running"
            # Reset the per-plugin streaming byte counter so a retry
            # dispatch doesn't carry forward the count from the
            # previous (now-finished) attempt. This mirrors the
            # dispatch-time reset (``attempt_start`` is rewritten on
            # each entry to ``_run_plugins``).
            info[f"{pid}_bytes_received"] = 0

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
        """
        if not n_bytes:
            return
        with self._lock:
            key = f"{pid}_bytes_received"
            info = self._model_info[model_name]
            info[key] = (info.get(key) or 0) + n_bytes

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
            info = self._model_info[model_name]
            cur = [p for p in (info.get("running_pids") or []) if p != pid]
            info["running_pids"] = cur
            # Status intentionally untouched; see docstring.

    def add_result(self, result):
        with self._lock:
            self.results.append(result)

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
            self._log.append((time.time(), model_name, msg))
            if len(self._log) > 100:
                self._log = self._log[-100:]

    def recent_log(self, n=5):
        with self._lock:
            return self._log[-n:]

    def save_state(self, path, plugin_versions=None):
        with self._lock:
            data = {
                "model_info": self._model_info,
                "results": self.results,
                "active_plugins": self.plugin_ids,
                "plugin_versions": plugin_versions or {},
                "session_seed": self.session_seed,
            }
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    def latest_results(self):
        """Return only the most recent result per model (deduplicates across runs)."""
        with self._lock:
            seen = {}
            for r in self.results:
                seen[r["model"]] = r
            return list(seen.values())

    @classmethod
    def load_state(cls, path, models, plugin_ids, *, rerun_failed=True):
        with open(path) as f:
            data = json.load(f)
        session_seed = data.get("session_seed")
        state = cls(models, plugin_ids, session_seed=session_seed)
        saved_plugins = data.get("active_plugins", [])
        new_plugins = [pid for pid in plugin_ids if pid not in saved_plugins]
        saved_info = data.get("model_info", {})
        saved_results = data.get("results", [])
        latest_by_model = {}
        for r in saved_results:
            latest_by_model[r["model"]] = r
        for name, info in saved_info.items():
            if name in state._model_info:
                # Strip transient in-flight state on resume: no plugin task
                # is actually running until a worker picks the task up, so
                # carrying stale ``running_pids`` forward causes phantom
                # "[waiting]" cells and a false-yellow highlight in the
                # live TUI. The migration used to *preserve* this state,
                # but that assumption broke once the visual-layer read
                # ``running_pids`` as the source-of-truth.
                info.pop("running_pids", None)
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
                if new_plugins:
                    for pid in new_plugins:
                        state._model_info[name].setdefault(f"{pid}_score", None)
                        state._model_info[name].setdefault(f"{pid}_tps", None)
                        state._model_info[name].setdefault(f"{pid}_response_time", None)
                        state._model_info[name].setdefault(f"{pid}_output_tokens", None)
                    if state._model_info[name].get("status") == "completed":
                        state._model_info[name]["status"] = "pending"
                elif info.get("status") == "completed":
                    latest = latest_by_model.get(name)
                    if latest is not None:
                        result_plugins = set(latest.get("plugin_versions", {}).keys())
                        if result_plugins and result_plugins.issubset(set(plugin_ids)):
                            for pid in plugin_ids:
                                if f"{pid}_score" not in latest:
                                    state._model_info[name]["status"] = "pending"
                                    break
        state.results = data.get("results", [])
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


