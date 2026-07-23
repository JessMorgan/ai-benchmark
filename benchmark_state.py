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
            }
            for pid in plugin_ids:
                self._model_info[name][f"{pid}_score"] = None
                self._model_info[name][f"{pid}_tps"] = None
                self._model_info[name][f"{pid}_response_time"] = None
                self._model_info[name][f"{pid}_output_tokens"] = None

    def update(self, model_name, **kwargs):
        with self._lock:
            self._model_info[model_name].update(kwargs)

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
                state._model_info[name] = info
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


