"""Preload coordination extracted from ``_run_benchmark``.

``PreloadCoordinator`` warms models before benchmarking: one request per
(source, api_model) ensures the backend is ready.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any


class PreloadCoordinator:
    """Warms models and tracks preload outcomes across the run."""

    def __init__(
        self,
        *,
        state: Any,
        source_config: dict[str, Any],
        raw_targets: dict[str, Any],
        args: Any,
        output_dir: str,
        session_seed: int,
        stop_event: threading.Event,
        run_info: dict[str, Any],
        runner_mode: str,
    ) -> None:
        self._state = state
        self._source_config = source_config
        self._raw_targets = raw_targets
        self._args = args
        self._output_dir = output_dir
        self._session_seed = session_seed
        self._stop_event = stop_event
        self._run_info = run_info
        self._runner_mode = runner_mode

        self._lock = threading.Lock()
        self._ok: set[tuple[str, str]] = set()
        self._failed: set[tuple[str, str]] = set()
        self._inflight: dict[tuple[str, str], threading.Event] = {}

    # ── public API ──────────────────────────────────────────────────────

    def ensure_preloaded(
        self, model_name: str, target_info: dict[str, Any], phase_runner: str,
    ) -> bool:
        """Warm a target once per source/model.  Thread-safe."""
        from benchmark.core import PreloadResult, preload_model, resolve_preload_timeout

        if phase_runner == "pi" or not self._is_enabled(target_info["source"]):
            return True
        key = (target_info["source"], target_info["api_model"])
        with self._lock:
            if key in self._ok:
                return True
            if key in self._failed:
                return False
            inflight = self._inflight.get(key)
            if inflight is None:
                inflight = threading.Event()
                self._inflight[key] = inflight
                preload_owner = True
                self._set_preloading(model_name, target_info, phase_runner, True)
                self._run_info["preload"]["attempted"] += 1
                self._run_info["preload"]["per_model"][f"{key[0]}/{key[1]}"] = {
                    "status": "running",
                    "timeout": resolve_preload_timeout(
                        self._source_config, target_info["source"]),
                }
            else:
                preload_owner = False
        if not preload_owner:
            inflight.wait()
            with self._lock:
                return bool(key in self._ok)

        started = time.time()
        timeout_limit = resolve_preload_timeout(
            self._source_config, target_info["source"])
        raw_cfg = self._raw_targets.get(model_name)
        drop_params = (
            raw_cfg.get("drop_params", []) if isinstance(raw_cfg, dict) else [])
        log_path = None
        result = None
        try:
            if self._args.debug_logs or self._args.storage_profile == "debug":
                preload_logs = os.path.join(self._output_dir, "logs")
                os.makedirs(preload_logs, exist_ok=True)
                log_path = os.path.join(preload_logs, "preload.log")
            result = preload_model(
                self._source_config,
                target_info["source"], target_info["api_model"],
                timeout_limit,
                session_seed=self._session_seed,
                stop_event=self._stop_event,
                drop_params=drop_params, log_path=log_path)
        except Exception as exc:  # noqa: BLE001
            result = PreloadResult(success=False,
                elapsed=round(time.time() - started, 1),
                error=f"{type(exc).__name__}: {exc}")
        try:
            elapsed = (result.elapsed if result.elapsed is not None
                       else round(time.time() - started, 1))
            model_key = f"{key[0]}/{key[1]}"
            with self._lock:
                self._run_info["preload"]["total_preload_time"] += elapsed
                self._run_info["preload"]["per_model"][model_key] = {
                    "status": "ok" if result.success else "failed",
                    "timeout": timeout_limit, "time": elapsed}
                if result.success:
                    self._ok.add(key)
                    self._run_info["preload"]["succeeded"] += 1
                    self._mark_ok_in_state(
                        model_name, phase_runner, elapsed)
                else:
                    self._failed.add(key)
                    self._run_info["preload"]["failed"] += 1
                    self._mark_failed_in_state(
                        model_name, result, phase_runner)
                return bool(result.success)
        except Exception as exc:  # noqa: BLE001
            failure = PreloadResult(
                success=False,
                elapsed=round(time.time() - started, 1),
                error=f"preload bookkeeping failed: {type(exc).__name__}: {exc}")
            with self._lock:
                self._ok.discard(key)
                self._failed.add(key)
                try:
                    self._run_info["preload"]["failed"] += 1
                    self._run_info["preload"]["per_model"][model_key] = {
                        "status": "failed", "timeout": timeout_limit,
                        "time": failure.elapsed, "error": failure.error}
                except (KeyError, TypeError):
                    pass
            try:
                self._mark_failed_in_state(model_name, failure, phase_runner)
            except Exception as record_exc:  # noqa: BLE001
                print(
                    f"\u26a0\ufe0f  Could not record preload failure "
                    f"for {model_name}: {record_exc}", file=sys.stderr)
            return False
        finally:
            with self._lock:
                self._inflight.pop(key, None)
                inflight.set()

    # ── internal ────────────────────────────────────────────────────────

    def _is_enabled(self, source: str) -> bool:
        src_cfg = self._source_config.get(source) or {}
        return (
            not self._args.no_preload
            and isinstance(src_cfg, dict)
            and bool(src_cfg.get("preload", False))
        )

    def _set_preloading(
        self, target_name: str, target_info: dict[str, Any],
        phase_runner: str, enabled: bool,
    ) -> None:
        from benchmark.scheduling import _runner_state_key
        keys = [target_name]
        if self._runner_mode == "both":
            keys.append(_runner_state_key(target_name, "opencode"))
        elif self._runner_mode == "multi" or phase_runner != "http":
            keys.append(_runner_state_key(target_name, phase_runner))
        now = time.monotonic() if enabled else 0
        snapshot = self._state.snapshot()
        for key in keys:
            if key in snapshot and snapshot[key].get("status") != "completed":
                self._state.update(
                    key,
                    status="queued" if enabled
                    else snapshot[key].get("status", "pending"),
                    preloading=enabled, preload_start_ts=now)

    def _mark_ok_in_state(
        self, model_name: str, phase_runner: str, elapsed: float,
    ) -> None:
        from benchmark.scheduling import _runner_state_key
        preload_state_keys = (
            (model_name, _runner_state_key(model_name, "opencode"))
            if self._runner_mode == "both"
            else (model_name,) if phase_runner == "http"
            else (_runner_state_key(model_name, phase_runner),))
        for state_key in preload_state_keys:
            if state_key in self._state.snapshot():
                self._state.update(
                    state_key, preloading=False, preload_start_ts=0,
                    preload_status="ok", preload_time=elapsed,
                    preload_error=None)

    def _mark_failed_in_state(
        self, model_name: str, result: Any, phase_runner: str,
    ) -> None:
        from benchmark.cli import _mark_preload_failed
        _mark_preload_failed(
            self._state, model_name, result, phase_runner,
            self._runner_mode)
