"""Judge coordination extracted from ``_run_benchmark``.

``JudgeCoordinator`` owns the judge lifecycle: sidecar enqueue, per-judge-model
transport + parsing + consensus, progress tracking, and worker-pool management.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any

from benchmark.scheduling import _CombinedStopEvent, _FlushGate


class JudgeCoordinator:
    """Owns the complete judge lifecycle for one benchmark run."""

    def __init__(
        self,
        *,
        state: Any,
        source_config: dict[str, Any],
        targets: dict[str, Any],
        active_plugins: list[Any],
        judge_models: list[str],
        judge_contracts: dict[str, str],
        active_judge_contracts: dict[str, str],
        judge_sources: dict[str, str],
        judge_model_limits: dict[str, int],
        judge_plugin_limits: dict[str, int],
        judge_effective_timeout: float,
        judge_max_tokens: int,
        judge_temperature: float,
        judge_request_params: dict[str, Any] | None,
        output_dir: str,
        args: Any,
        model_thread_limits: dict[str, int],
        stop_event: threading.Event,
        raw_targets: dict[str, Any],
        run_info: dict[str, Any],
        flush_gate: _FlushGate,
        flusher: Any,
    ) -> None:
        self._state = state
        self._source_config = source_config
        self._targets = targets
        self._active_plugins = active_plugins
        self._judge_models = judge_models
        self._judge_contracts = judge_contracts
        self._judge_sources = judge_sources
        self._judge_model_limits = judge_model_limits
        self._judge_plugin_limits = judge_plugin_limits
        self._judge_effective_timeout = judge_effective_timeout
        self._judge_max_tokens = judge_max_tokens
        self._judge_temperature = judge_temperature
        self._judge_request_params = judge_request_params
        self._output_dir = output_dir
        self._args = args
        self._model_thread_limits = model_thread_limits
        self._stop_event = stop_event
        self._raw_targets = raw_targets
        self._run_info = run_info
        self._flush_gate = flush_gate
        self._flusher = flusher
        self._active_judge_contracts = active_judge_contracts

        self._judge_seen: set[tuple] = set()
        self._judge_seen_lock = threading.Lock()
        self._judge_counts_lock = threading.Lock()
        self._judge_votes: dict[tuple, list[dict[str, Any]]] = {}
        self._judge_votes_lock = threading.Lock()
        self._halted_judges: set[str] = set()
        self._halted_judges_lock = threading.Lock()
        self._judge_stop_events = {
            model: threading.Event() for model in judge_models
        }
        self._judge_request_stop_events = {
            model: _CombinedStopEvent(stop_event, self._judge_stop_events[model])
            for model in judge_models
        }
        self._pools: dict[str, Any] = {}

    @property
    def pools(self) -> dict[str, Any]:
        return self._pools

    def build_pools(self) -> None:
        from benchmark.scheduling import SourceJudgeWorkerPool
        self._pools = {
            source: SourceJudgeWorkerPool(
                source,
                self._judge_model_limits[source],
                self.process_judge_job,
                self._stop_event,
                plugin_limit=self._judge_plugin_limits[source],
                on_selection_change=lambda judge, selected: (
                    self._state.set_judge_selected(judge, selected)
                ),
            )
            for source in set(self._judge_sources.values())
        }

    def init_progress_from_state(self) -> None:
        from benchmark.judging import is_successful_judge_vote
        existing_judge_counts = {model: 0 for model in self._judge_models}
        existing_judge_failures = {model: 0 for model in self._judge_models}
        existing_judge_expected = {model: 0 for model in self._judge_models}
        for result in self._state.latest_results():
            for plugin in self._active_plugins:
                expected_contract = self._judge_contracts.get(plugin.id)
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
        self._state.set_judge_progress({
            model: {
                "completed": existing_judge_counts[model],
                "failed": existing_judge_failures[model],
                "expected": existing_judge_expected[model],
            }
            for model in self._judge_models
        })

    def enqueue_existing_sidecars(self) -> None:
        from benchmark.cli import _eligible_judge_sidecars
        judge_input_dir = (
            os.path.join(self._output_dir, "judge-inputs")
            if self._judge_models else None
        )
        if not judge_input_dir:
            return
        for sidecar, item in _eligible_judge_sidecars(
            judge_input_dir, self._targets, self._state,
            {plugin.id for plugin in self._active_plugins},
            self._judge_models, self._judge_contracts,
        ):
            self.enqueue_judge(sidecar, item["target"], item["runner"], item["plugin"])

    def source_benchmark_complete(self, source: str) -> None:
        pool = self._pools.get(source)
        if pool is not None:
            pool.expand_full()

    def start_judge_if_async(
        self, benchmark_limits: dict[str, int],
        benchmark_sources: set[str] | None = None,
    ) -> None:
        from benchmark.scheduling import _configure_judge_source
        if not self._judge_models:
            return
        for source, pool in self._pools.items():
            _configure_judge_source(
                benchmark_limits, source,
                self._judge_model_limits[source],
                source in (benchmark_sources or set()), pool,
            )

    def stop_judge_workers(self, *, drain: bool = False) -> None:
        for pool in self._pools.values():
            pool.stop(timeout=None if drain else 1.0, drain=drain)

    def finish_judge(self) -> None:
        from benchmark.cli import _eligible_judge_sidecars
        if not self._judge_models:
            return
        judge_input_dir = os.path.join(self._output_dir, "judge-inputs") if self._judge_models else None
        if judge_input_dir:
            jobs = _eligible_judge_sidecars(
                judge_input_dir, self._targets, self._state,
                {plugin.id for plugin in self._active_plugins},
                self._judge_models, self._judge_contracts,
            )
            for sidecar, item in jobs:
                self.enqueue_judge(sidecar, item["target"], item["runner"], item["plugin"])
        for source, pool in self._pools.items():
            pool.start(self._judge_model_limits[source])
        self.stop_judge_workers(drain=True)
        if self._judge_models:
            self._run_info["judge_status"] = "complete"

    def _replace_judge_progress(
        self, judge_name: str, previous_vote: dict[str, Any] | None,
        current_vote: dict[str, Any],
    ) -> None:
        from benchmark.judging import is_successful_judge_vote
        pc = int(previous_vote is not None and is_successful_judge_vote(previous_vote))  # noqa: SIM108
        pf = int(previous_vote is not None and not is_successful_judge_vote(previous_vote))  # noqa: SIM108
        self._state.replace_judge_progress(
            judge_name, previous_completed=pc, previous_failed=pf,
            completed=int(is_successful_judge_vote(current_vote)),
            failed=int(not is_successful_judge_vote(current_vote)),
        )

    def enqueue_judge(
        self, sidecar: str, target_name: str, runner: str, plugin_id: str,
    ) -> None:
        from benchmark.judging import is_successful_judge_vote
        latest = {
            (result.get("state_key", result.get("model")), result.get("runner", "http")): result
            for result in self._state.latest_results()
        }
        try:
            with open(sidecar, encoding="utf-8") as handle:
                item = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError, UnicodeDecodeError):
            return
        state_key = item.get("state_key", target_name)
        result = latest.get((state_key, runner), {})
        info = self._state.snapshot().get(state_key, {})
        score = result.get(f"{plugin_id}_score")
        if not (isinstance(score, (int, float)) and not isinstance(score, bool)):
            score = info.get(f"{plugin_id}_score")
        if not (isinstance(score, (int, float)) and not isinstance(score, bool)):
            return
        result_votes = result.get(f"{plugin_id}_judge_votes", []) or []
        info_votes = info.get(f"{plugin_id}_judge_votes", []) or []
        contract_id = self._judge_contracts.get(plugin_id)
        votes_by_model = {
            vote.get("model"): vote
            for vote in [*result_votes, *info_votes]
            if isinstance(vote, dict) and vote.get("model")
            and vote.get("judge_contract_id") == contract_id
        }
        judged_models = {
            judge_name for judge_name, vote in votes_by_model.items()
            if is_successful_judge_vote(vote)
        }
        for judge_name in self._judge_models:
            if judge_name in judged_models:
                continue
            with self._halted_judges_lock:
                if judge_name in self._halted_judges:
                    continue
            source = self._judge_sources[judge_name]
            expected_added = judge_name not in votes_by_model
            key = (
                os.path.abspath(sidecar), target_name, runner, plugin_id,
                judge_name, contract_id,
            )
            with self._judge_seen_lock:
                if key in self._judge_seen:
                    continue
                self._judge_seen.add(key)
            self._pools[source].enqueue(
                (sidecar, target_name, runner, plugin_id, judge_name, expected_added))
            self._state.update(state_key, **{f"{plugin_id}_judge_queued": True})
            if expected_added:
                self._state.increment_judge_progress(judge_name, expected=1)
            with self._judge_counts_lock:
                self._run_info["judge_counts"]["queued"] += 1

    def process_judge_job(self, job: tuple) -> None:
        from benchmark.core import JUDGE_PROMPT_VERSION, judge_response
        from benchmark.judging import (
            confidence_weighted_consensus,
            confidence_weighted_consensus_by_contract,
            is_successful_judge_vote,
            judge_contract_id,
            judge_votes_for_contract,
            merge_judge_vote,
            save_judge_response,
            save_judge_response_metadata,
        )
        from benchmark.results import save_judge_result
        from benchmark.runtime_records import JudgeAttemptRecord, JudgeVoteRecord

        sidecar, target_name, runner, plugin_id, judge_name, expected_added = job
        with self._halted_judges_lock:
            if judge_name in self._halted_judges:
                if expected_added:
                    self._state.increment_judge_progress(judge_name, expected=-1)
                return
        item: dict[str, Any] = {}
        previous_vote = None
        existing_votes: list[dict[str, Any]] = []
        state_key = target_name if runner == "http" else f"{target_name} [opencode]"
        plugin_obj = next((p for p in self._active_plugins if p.id == plugin_id), None)
        contract_id = judge_contract_id(plugin_obj) if plugin_obj is not None else None
        _last_attempt = [1]

        try:
            with open(sidecar, encoding="utf-8") as handle:
                item = json.load(handle)
            latest = {
                (result.get("state_key", result.get("model")), result.get("runner", "http")): result
                for result in self._state.latest_results()
            }.get((item.get("state_key", target_name), runner), {})
            state_key = item.get("state_key", state_key)
            live_info = self._state.snapshot().get(state_key, {})
            vote_key = f"{plugin_id}_judge_votes"
            expected_contract = item.get("judge_contract_id") or contract_id
            all_existing_by_identity = {
                (vote.get("model"), vote.get("judge_contract_id")): vote
                for vote in [
                    *(latest.get(vote_key, []) or []),
                    *(live_info.get(vote_key, []) or []),
                ] if isinstance(vote, dict) and vote.get("model")
            }
            all_existing_votes = list(all_existing_by_identity.values())
            existing_votes = judge_votes_for_contract(all_existing_votes, expected_contract)
            existing_by_model = {vote.get("model"): vote for vote in existing_votes}
            previous_vote = existing_by_model.get(judge_name)
            if any(
                vote.get("model") == judge_name and is_successful_judge_vote(vote)
                for vote in existing_votes if isinstance(vote, dict)
            ):
                return
            activity_id = self._state.start_judge_activity(judge_name, target_name, plugin_id)
            progress_chars = [0, 0]

            def _judge_attempt(attempt_number: int) -> None:
                progress_chars[:] = [0, 0]
                _last_attempt[0] = attempt_number
                self._state.set_judge_activity_attempt(activity_id, attempt_number)

            def _judge_progress(content_delta: str = "", thinking_delta: str = "") -> None:
                progress_chars[0] += len(content_delta or "")
                progress_chars[1] += len(thinking_delta or "")
                self._state.update_judge_activity(
                    activity_id, thinking_tokens=progress_chars[1] // 4,
                    content_tokens=progress_chars[0] // 4)

            outcome = None
            try:
                outcome = judge_response(
                    self._source_config, self._judge_sources[judge_name],
                    self._targets[judge_name]["api_model"], sidecar,
                    timeout=self._judge_effective_timeout,
                    max_tokens=self._judge_max_tokens,
                    temperature=self._judge_temperature,
                    request_params=self._judge_request_params,
                    drop_params=(
                        self._raw_targets.get(judge_name, {}).get("drop_params", [])
                        if isinstance(self._raw_targets.get(judge_name), dict) else []
                    ),
                    stop_event=self._judge_request_stop_events[judge_name],
                    log_path=(
                        os.path.join(self._output_dir, f"judge-{judge_name}.log.gz")
                        if self._args.debug_logs or self._args.storage_profile == "debug"
                        else None
                    ),
                    plugin=plugin_obj,
                    progress_callback=_judge_progress,
                    attempt_callback=_judge_attempt,
                )
            finally:
                if outcome is not None and outcome.response_text is not None:
                    self._state.update_judge_activity(
                        activity_id, content_tokens=len(outcome.response_text) // 4)
                self._state.finish_judge_activity(activity_id)
            if outcome.terminal_429:
                with self._halted_judges_lock:
                    self._halted_judges.add(judge_name)
                self._state.update_judge_progress(judge_name, stopped=True)
                self._judge_stop_events[judge_name].set()
            vote = save_judge_result(
                outcome, model_name=judge_name,
                judge_prompt_version=JUDGE_PROMPT_VERSION, judge_contract_id=contract_id)
            judge_cell_id = self._state.run_store.get_cell_id(state_key, runner, plugin_id)
            if judge_cell_id is not None:
                self._state.run_store.record_judge_attempt(judge_cell_id,
                    JudgeAttemptRecord(
                        judge_model=judge_name, contract_id=contract_id,
                        attempt_number=_last_attempt[0],
                        raw_response=outcome.response_text,
                        max_tokens=self._judge_max_tokens,
                        usage=outcome.diagnostics or {},
                        diagnostics=outcome.diagnostics or {},
                        finish_reason=(
                            outcome.diagnostics.get("finish_reason")
                            if isinstance(outcome.diagnostics, dict) else None),
                        error=outcome.error,
                        status="completed" if not outcome.error else "failed"),
                    vote=JudgeVoteRecord(
                        score=vote.get("score"), confidence=vote.get("confidence"),
                        rationale=vote.get("rationale"),
                        criteria=vote.get("criteria") or [],
                        error=vote.get("error"),
                        usable=is_successful_judge_vote(vote)))
            response_text = outcome.response_text or ""
            artifact_error = None
            try:
                save_judge_response(self._output_dir, target_name, runner, plugin_id,
                                    judge_name, response_text, contract_id)
                save_judge_response_metadata(self._output_dir, target_name, runner, plugin_id,
                    judge_name, {
                        "target": target_name, "runner": runner, "plugin": plugin_id,
                        "judge_model": judge_name,
                        "judge_prompt_version": JUDGE_PROMPT_VERSION,
                        "judge_contract_id": contract_id,
                        "status": "error" if outcome.error else "ok",
                        "response_present": outcome.response_text is not None,
                        "response_empty": not bool(response_text.strip()),
                        "score": outcome.score, "confidence": outcome.confidence,
                        "error": outcome.error, "terminal_429": outcome.terminal_429,
                        "rationale": outcome.rationale,
                        "criteria": outcome.criteria or [],
                        "diagnostics": outcome.diagnostics,
                        "saved_at": datetime.now(timezone.utc).isoformat(),
                    }, contract_id)
            except OSError as exc:
                artifact_error = f"could not save judge response artifact: {exc}"
            if artifact_error:
                vote["error"] = vote["error"] or artifact_error
            vote_identity = (state_key, runner, plugin_id)
            with self._judge_votes_lock:
                prior_all_votes = list(self._judge_votes.get(vote_identity, all_existing_votes))
                prior_all_votes = merge_judge_vote(prior_all_votes, vote)
                self._judge_votes[vote_identity] = prior_all_votes
                votes = judge_votes_for_contract(prior_all_votes, contract_id)
            consensus_by_contract = confidence_weighted_consensus_by_contract(prior_all_votes)
            consensus = consensus_by_contract.get(
                contract_id, confidence_weighted_consensus(votes))
            expected_judges = set(self._judge_models)
            received_judges = {v.get("model") for v in votes if is_successful_judge_vote(v)}
            failed_judges = {
                v.get("model") for v in votes
                if isinstance(v, dict) and v.get("model") in expected_judges
                and not is_successful_judge_vote(v)}
            all_judges_finished = expected_judges.issubset(received_judges | failed_judges)
            judge_status = (
                "failed" if all_judges_finished and consensus["error"]
                and not any(v.get("score") is not None for v in votes)
                else "partial" if all_judges_finished and any(v.get("error") for v in votes)
                else "complete" if all_judges_finished else "running")
            # ``update_judge_result`` is the canonical write path: it updates
            # the live in-memory state the TUI renders AND forwards to the
            # durable backend (SQLite) when one is attached. Calling the run
            # store directly here would bypass the live ``_model_info`` rows,
            # so SQLite runs would show no judge markers in the table.
            self._state.update_judge_result(
                state_key, runner, plugin_id,
                score=consensus["score"], confidence=consensus["confidence"],
                rationale=consensus["rationale"],
                criteria=consensus.get("criteria", []),
                consensus_by_contract=consensus_by_contract,
                selected_contract=contract_id, error=consensus["error"],
                input_sha256=item.get("response_sha256"),
                votes=prior_all_votes, status=judge_status,
                complete=(all_judges_finished and expected_judges.issubset(received_judges)))
            with self._judge_counts_lock:
                self._replace_judge_progress(judge_name, previous_vote, vote)
                if is_successful_judge_vote(vote):
                    self._run_info["judge_counts"]["completed"] += 1
                else:
                    self._run_info["judge_counts"]["failed"] += 1
                self._run_info["judge_counts"]["votes"] += 1
            if self._flush_gate.changed():
                self._flusher.request_flush()
                self._flush_gate.reset()
        except Exception as exc:  # noqa: BLE001
            self._record_judge_failure(
                exc, sidecar, item, target_name, runner, plugin_id,
                judge_name, state_key, previous_vote,
                all_existing_votes, contract_id, _last_attempt[0])

    def _record_judge_failure(
        self, exc: Exception, sidecar: str, item: dict[str, Any],
        target_name: str, runner: str, plugin_id: str,
        judge_name: str, state_key: str, previous_vote: dict[str, Any] | None,
        all_existing_votes: list[dict[str, Any]],
        contract_id: str | None, attempt_number: int,
    ) -> None:
        from benchmark.core import JUDGE_PROMPT_VERSION
        from benchmark.judging import (
            is_successful_judge_vote,
            judge_votes_for_contract,
            merge_judge_vote,
            save_judge_response,
            save_judge_response_metadata,
        )
        from benchmark.results import save_judge_result
        from benchmark.runtime_records import JudgeAttemptRecord, JudgeVoteRecord

        artifact_error = None
        try:
            save_judge_response(
                self._output_dir, target_name, runner, plugin_id, judge_name, "", contract_id)
            save_judge_response_metadata(
                self._output_dir, target_name, runner, plugin_id, judge_name, {
                    "target": target_name, "runner": runner, "plugin": plugin_id,
                    "judge_model": judge_name,
                    "judge_prompt_version": JUDGE_PROMPT_VERSION,
                    "judge_contract_id": contract_id,
                    "status": "exception", "response_present": False,
                    "response_empty": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                }, contract_id)
        except OSError as artifact_exc:
            artifact_error = f"could not save judge failure artifact: {artifact_exc}"
            print(f"\u26a0\ufe0f  {artifact_error}", file=sys.stderr)
        state_key = item.get("state_key", state_key)
        if previous_vote is None:
            latest = {
                (result.get("state_key", result.get("model")), result.get("runner", "http")): result
                for result in self._state.latest_results()
            }.get((state_key, runner), {})
            live_info = self._state.snapshot().get(state_key, {})
            all_existing_by_identity = {
                (vote.get("model"), vote.get("judge_contract_id")): vote
                for vote in [
                    *(latest.get(f"{plugin_id}_judge_votes", []) or []),
                    *(live_info.get(f"{plugin_id}_judge_votes", []) or []),
                ] if isinstance(vote, dict) and vote.get("model")
            }
            all_existing_votes = list(all_existing_by_identity.values())
            existing_votes = judge_votes_for_contract(all_existing_votes, contract_id)
            previous_vote = next(
                (v for v in existing_votes if v.get("model") == judge_name), None)
        failure_vote = save_judge_result(
            None, model_name=judge_name, judge_prompt_version=JUDGE_PROMPT_VERSION,
            judge_contract_id=contract_id,
            parsed_judge={"error": (
                f"judge input failed: {type(exc).__name__}: {exc}"
                + (f"; {artifact_error}" if artifact_error else ""))})
        judge_cell_id = self._state.run_store.get_cell_id(state_key, runner, plugin_id)
        if judge_cell_id is not None:
            self._state.run_store.record_judge_attempt(
                judge_cell_id,
                JudgeAttemptRecord(
                    judge_model=judge_name, contract_id=contract_id,
                    attempt_number=attempt_number, raw_response=None,
                    max_tokens=self._judge_max_tokens,
                    error=f"{type(exc).__name__}: {exc}", status="failed"),
                vote=JudgeVoteRecord(error=failure_vote["error"], usable=False))
        vote_identity = (state_key, runner, plugin_id)
        with self._judge_votes_lock:
            prior_all_votes = list(self._judge_votes.get(vote_identity, all_existing_votes))
            prior_all_votes = merge_judge_vote(prior_all_votes, failure_vote)
            self._judge_votes[vote_identity] = prior_all_votes
        expected_judges = set(self._judge_models)
        current_votes = judge_votes_for_contract(prior_all_votes, contract_id)
        received_judges = {v.get("model") for v in current_votes if is_successful_judge_vote(v)}
        failed_judges = {
            v.get("model") for v in current_votes
            if isinstance(v, dict) and v.get("model") in expected_judges
            and not is_successful_judge_vote(v)}
        all_judges_finished = expected_judges.issubset(received_judges | failed_judges)
        # Canonical write path — updates live state and forwards to the
        # durable backend (see success path above).
        self._state.update_judge_result(
            state_key, runner, plugin_id,
            error=failure_vote["error"], selected_contract=contract_id,
            votes=prior_all_votes,
            status="failed" if all_judges_finished else "running",
            complete=(all_judges_finished and expected_judges.issubset(received_judges)))
        with self._judge_counts_lock:
            self._replace_judge_progress(judge_name, previous_vote, failure_vote)
            self._run_info["judge_counts"]["failed"] += 1
        if self._flush_gate.changed():
            self._flusher.request_flush()
            self._flush_gate.reset()
