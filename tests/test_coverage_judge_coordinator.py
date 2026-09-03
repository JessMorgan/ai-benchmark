"""Coverage tests for :mod:`benchmark.judge_coordinator` (JudgeCoordinator).

These exercise the coordinator's lifecycle helpers (pool building, progress
initialisation, start/stop/benchmark hooks, judging enqueue) with heavyweight
dependencies (worker threads, network, state persistence) mocked away.
"""
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from benchmark.judge_coordinator import JudgeCoordinator


def _coordinator(**overrides):
    """Build a JudgeCoordinator wired to mocks for its heavyweight deps."""
    state = mock.Mock()
    state.latest_results.return_value = []
    state.snapshot.return_value = {}
    stop_event = threading.Event()
    defaults = dict(
        state=state,
        source_config={},
        targets={},
        active_plugins=[],
        judge_models=[],
        judge_contracts={},
        active_judge_contracts={},
        judge_sources={},
        judge_model_limits={},
        judge_plugin_limits={},
        judge_effective_timeout=60.0,
        judge_max_tokens=1024,
        judge_temperature=0.2,
        judge_request_params=None,
        output_dir=tempfile.mkdtemp(),
        args=mock.Mock(),
        model_thread_limits={},
        stop_event=stop_event,
        raw_targets={},
        run_info={"judge_counts": {"queued": 0, "completed": 0, "failed": 0, "votes": 0}},
        flush_gate=mock.Mock(),
        flusher=mock.Mock(),
    )
    defaults.update(overrides)
    return JudgeCoordinator(**defaults), state, stop_event


class JudgeCoordinatorPropsTest(unittest.TestCase):
    def test_pools_and_gates_properties(self):
        coord, _state, _stop = _coordinator()
        self.assertEqual(coord.pools, {})
        self.assertIsNone(coord.plugin_slot_gates)


class JudgeCoordinatorBuildPoolsTest(unittest.TestCase):
    def test_build_pools_simple(self):
        coord, _state, stop = _coordinator(
            judge_models=["judgeA"],
            judge_sources={"judgeA": "srcA"},
            judge_model_limits={"srcA": 1},
            judge_plugin_limits={"srcA": 1},
            targets={"judgeA": {"source": "srcA"}},
            source_config={"srcA": {"plugin_thread_limit": 2}},
        )
        with mock.patch("benchmark.scheduling.SourceJudgeWorkerPool") as pool_cls, \
             mock.patch("benchmark.scheduling.PluginSlotGateRegistry") as gates_cls:
            gates_cls.return_value.create = mock.Mock()
            coord.build_pools()
            self.assertEqual(list(coord.pools.keys()), ["srcA"])
            pool_cls.assert_called_once()
            # judge model sits on its own judge source -> plugin gate created.
            gates_cls.return_value.create.assert_called_once()
            self.assertIs(coord.plugin_slot_gates, gates_cls.return_value)

    def test_build_pools_skips_missing_source(self):
        coord, _state, stop = _coordinator(
            judge_models=["judgeA", "judgeB"],
            judge_sources={"judgeA": "srcA"},
            judge_model_limits={"srcA": 1},
            judge_plugin_limits={"srcA": 1},
        )
        with mock.patch("benchmark.scheduling.SourceJudgeWorkerPool"), \
             mock.patch("benchmark.scheduling.PluginSlotGateRegistry") as gates_cls:
            gates_cls.return_value.create = mock.Mock()
            coord.build_pools()
            # judgeA resolves to srcA; judgeB has no source -> its pool is skipped.
            self.assertEqual(list(coord.pools.keys()), ["srcA"])
            self.assertIs(coord.plugin_slot_gates, gates_cls.return_value)

    def test_build_pools_nonpositive_plugin_limit_skips_gate(self):
        coord, _state, stop = _coordinator(
            judge_models=["judgeA"],
            judge_sources={"judgeA": "srcA"},
            judge_model_limits={"srcA": 1},
            judge_plugin_limits={"srcA": 1},
            targets={"judgeA": {"source": "srcA"}},
            source_config={"srcA": {"plugin_thread_limit": 0}},
        )
        with mock.patch("benchmark.scheduling.SourceJudgeWorkerPool"), \
             mock.patch("benchmark.scheduling.PluginSlotGateRegistry") as gates_cls:
            gates_cls.return_value.create = mock.Mock()
            coord.build_pools()
            gates_cls.return_value.create.assert_not_called()


class JudgeCoordinatorInitProgressTest(unittest.TestCase):
    def test_init_progress_from_state_counts_votes(self):
        state = mock.Mock()
        state.latest_results.return_value = [{
            "model": "m1",
            "p1_judge_votes": [
                {"model": "judgeA", "judge_contract_id": "c1", "success": True},
                {"model": "judgeB", "judge_contract_id": "c1", "success": False},
                {"model": "judgeOther", "judge_contract_id": "c1", "success": True},
            ],
        }]
        plugin = mock.Mock()
        plugin.id = "p1"
        coord, _s, _stop = _coordinator(
            state=state,
            judge_models=["judgeA", "judgeB"],
            judge_contracts={"p1": "c1"},
            active_plugins=[plugin],
        )
        with mock.patch("benchmark.judging.is_successful_judge_vote",
                        side_effect=lambda v: bool(v.get("success"))):
            coord.init_progress_from_state()
        state.set_judge_progress.assert_called_once()
        progress = state.set_judge_progress.call_args[0][0]
        self.assertEqual(progress["judgeA"],
                         {"completed": 1, "failed": 0, "expected": 1})
        self.assertEqual(progress["judgeB"],
                         {"completed": 0, "failed": 1, "expected": 1})

    def test_init_progress_from_state_no_results(self):
        state = mock.Mock()
        state.latest_results.return_value = []
        coord, _s, _stop = _coordinator(
            state=state,
            judge_models=["judgeA"],
            judge_contracts={},
            active_plugins=[],
        )
        coord.init_progress_from_state()
        state.set_judge_progress.assert_called_once()
        self.assertEqual(state.set_judge_progress.call_args[0][0],
                         {"judgeA": {"completed": 0, "failed": 0, "expected": 0}})


class JudgeCoordinatorLifecycleHooksTest(unittest.TestCase):
    def test_source_benchmark_complete_expands_pool(self):
        coord, _state, _stop = _coordinator(
            judge_models=["judgeA"],
            judge_sources={"judgeA": "srcA"},
            judge_model_limits={"srcA": 1},
            judge_plugin_limits={"srcA": 1},
        )
        coord._pools["srcA"] = mock.Mock()
        coord.source_benchmark_complete("srcA")
        coord._pools["srcA"].expand_full.assert_called_once()

    def test_source_benchmark_complete_missing_pool(self):
        coord, _state, _stop = _coordinator()
        coord.source_benchmark_complete("nowhere")  # must not raise

    def test_start_judge_if_async_no_models(self):
        coord, _state, _stop = _coordinator()
        with mock.patch("benchmark.scheduling._configure_judge_source"):
            coord.start_judge_if_async({})  # returns early

    def test_start_judge_if_async_configures_sources(self):
        coord, _state, _stop = _coordinator(
            judge_models=["judgeA", "judgeB"],
            judge_sources={"judgeA": "srcA", "judgeB": "srcB"},
            judge_model_limits={"srcA": 2, "srcB": 3},
            judge_plugin_limits={"srcA": 1, "srcB": 1},
        )
        coord._pools["srcA"] = mock.Mock()
        coord._pools["srcB"] = mock.Mock()
        limits = {}
        with mock.patch("benchmark.scheduling._configure_judge_source") as cfg:
            coord.start_judge_if_async(limits, benchmark_queues={"srcA": ["m1"]})
        self.assertEqual(cfg.call_count, 2)

    def test_set_benchmark_active_updates_pool(self):
        coord, _state, _stop = _coordinator(
            judge_models=["judgeA"],
            judge_sources={"judgeA": "srcA"},
            judge_model_limits={"srcA": 1},
            judge_plugin_limits={"srcA": 1},
        )
        coord._pools["srcA"] = mock.Mock()
        coord.set_benchmark_active("srcA", ["m1"])
        coord._pools["srcA"].set_benchmark_models.assert_called_once_with(["m1"])

    def test_stop_judge_workers(self):
        coord, _state, _stop = _coordinator()
        pool = mock.Mock()
        coord._pools["srcA"] = pool
        coord.stop_judge_workers(drain=True)
        pool.stop.assert_called_once_with(timeout=None, drain=True)
        coord.stop_judge_workers(drain=False)
        pool.stop.assert_called_with(timeout=1.0, drain=False)


class JudgeCoordinatorReplaceProgressTest(unittest.TestCase):
    def test_replace_judge_progress(self):
        state = mock.Mock()
        coord, _s, _stop = _coordinator(state=state)
        prev = {"success": True}
        current = {"success": False}
        with mock.patch("benchmark.judging.is_successful_judge_vote",
                        side_effect=lambda v: bool(v.get("success"))):
            coord._replace_judge_progress("judgeA", prev, current)
        state.replace_judge_progress.assert_called_once_with(
            "judgeA", previous_completed=1, previous_failed=0,
            completed=0, failed=1,
        )

    def test_replace_judge_progress_no_previous(self):
        state = mock.Mock()
        coord, _s, _stop = _coordinator(state=state)
        with mock.patch("benchmark.judging.is_successful_judge_vote",
                        return_value=True):
            coord._replace_judge_progress("judgeA", None, {"success": True})
        state.replace_judge_progress.assert_called_once_with(
            "judgeA", previous_completed=0, previous_failed=0,
            completed=1, failed=0,
        )


class JudgeCoordinatorEnqueueJudgeTest(unittest.TestCase):
    def _setup_enqueue(self):
        state = mock.Mock()
        state.latest_results.return_value = []
        state.snapshot.return_value = {}
        coord, _s, _stop = _coordinator(
            state=state,
            judge_models=["judgeA"],
            judge_sources={"judgeA": "srcA"},
            judge_contracts={"p1": "c1"},
        )
        coord._pools["srcA"] = mock.Mock()
        return coord, state

    def test_enqueue_judge_good_sidecar(self):
        coord, state = self._setup_enqueue()
        with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump({"state_key": "m1", "plugin": "p1"}, handle)
            sidecar = handle.name
        try:
            state.latest_results.return_value = [
                {"state_key": "m1", "runner": "http", "p1_score": 0.9,
                 "p1_judge_votes": []},
            ]
            with mock.patch("benchmark.judging.is_successful_judge_vote",
                            return_value=False):
                coord.enqueue_judge(sidecar, "m1", "http", "p1")
            coord._pools["srcA"].enqueue.assert_called_once()
            job = coord._pools["srcA"].enqueue.call_args[0][0]
            self.assertEqual(job[1], "m1")
            self.assertEqual(job[3], "p1")
            self.assertEqual(job[4], "judgeA")
            self.assertTrue(job[5])  # expected_added
            state.increment_judge_progress.assert_called_once_with(
                "judgeA", expected=1)
            self.assertEqual(coord._run_info["judge_counts"]["queued"], 1)
        finally:
            os.unlink(sidecar)

    def test_enqueue_judge_missing_sidecar(self):
        coord, state = self._setup_enqueue()
        coord.enqueue_judge("/nonexistent/sidecar.json", "m1", "http", "p1")
        coord._pools["srcA"].enqueue.assert_not_called()

    def test_enqueue_judge_no_score_skips(self):
        coord, state = self._setup_enqueue()
        with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump({"state_key": "m1"}, handle)
            sidecar = handle.name
        try:
            coord.enqueue_judge(sidecar, "m1", "http", "p1")
            coord._pools["srcA"].enqueue.assert_not_called()
        finally:
            os.unlink(sidecar)

    def test_enqueue_judge_already_judged_skips(self):
        coord, state = self._setup_enqueue()
        with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump({"state_key": "m1"}, handle)
            sidecar = handle.name
        try:
            state.latest_results.return_value = [
                {"state_key": "m1", "runner": "http", "p1_score": 0.9,
                 "p1_judge_votes": [
                     {"model": "judgeA", "judge_contract_id": "c1",
                      "success": True},
                 ]},
            ]
            with mock.patch("benchmark.judging.is_successful_judge_vote",
                            return_value=True):
                coord.enqueue_judge(sidecar, "m1", "http", "p1")
            coord._pools["srcA"].enqueue.assert_not_called()
        finally:
            os.unlink(sidecar)


def _outcome(**overrides):
    """Build a fake judge_response outcome."""
    from types import SimpleNamespace

    defaults = dict(
        response_text="hello world",
        terminal_429=False,
        error=None,
        score=0.9,
        confidence=0.8,
        rationale="looks good",
        criteria=[{"name": "c", "score": 1}],
        diagnostics={},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _process_job(coord, job, flush_changed=False):
    """Run process_judge_job with controlled judging helpers."""
    vote = {"score": 0.9, "confidence": 0.8, "rationale": "r",
            "criteria": [{"name": "c", "score": 1}], "error": None}
    with mock.patch("benchmark.core.judge_response",
                    return_value=_outcome()) as jr, \
         mock.patch("benchmark.core.JUDGE_PROMPT_VERSION", "v1"), \
         mock.patch("benchmark.judging.judge_contract_id",
                    return_value="c1"), \
         mock.patch("benchmark.judging.is_successful_judge_vote",
                    return_value=True), \
         mock.patch("benchmark.judging.judge_votes_for_contract",
                    side_effect=lambda vs, c: vs), \
         mock.patch("benchmark.judging.merge_judge_vote",
                    side_effect=lambda vs, v: [*vs, v]), \
         mock.patch("benchmark.judging.confidence_weighted_consensus_by_contract",
                    return_value={"c1": {"score": 0.9, "confidence": 0.8,
                                         "rationale": "r", "error": None}}), \
         mock.patch("benchmark.judging.confidence_weighted_consensus",
                    return_value={"score": 0.9, "confidence": 0.8,
                                  "rationale": "r", "error": None}), \
         mock.patch("benchmark.judging.save_judge_response"), \
         mock.patch("benchmark.judging.save_judge_response_metadata"), \
         mock.patch("benchmark.results.save_judge_result", return_value=vote), \
         mock.patch("benchmark.runtime_records.JudgeAttemptRecord"), \
         mock.patch("benchmark.runtime_records.JudgeVoteRecord"):
        coord._flush_gate.changed.return_value = flush_changed
        coord.process_judge_job(job)
    return jr


class JudgeCoordinatorProcessJudgeJobTest(unittest.TestCase):
    def _coord(self):
        state = mock.Mock()
        state.latest_results.return_value = []
        state.snapshot.return_value = {}
        state.start_judge_activity.return_value = "act1"
        state.run_store.get_cell_id.return_value = None
        plugin = mock.Mock()
        plugin.id = "p1"
        coord, _s, _stop = _coordinator(
            state=state,
            judge_models=["judgeA"],
            judge_sources={"judgeA": "srcA"},
            targets={"judgeA": {"api_model": "modelX", "source": "srcA"}},
            active_plugins=[plugin],
            judge_contracts={"p1": "c1"},
        )
        return coord, state

    def test_process_judge_job_happy_path(self):
        coord, state = self._coord()
        with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump({"state_key": "m1", "runner": "http"}, handle)
            sidecar = handle.name
        try:
            job = (sidecar, "m1", "http", "p1", "judgeA", True)
            _process_job(coord, job)
            state.start_judge_activity.assert_called_once_with(
                "judgeA", "m1", "p1")
            state.finish_judge_activity.assert_called_once_with("act1")
            state.update_judge_result.assert_called_once()
            self.assertEqual(coord._run_info["judge_counts"]["completed"], 1)
        finally:
            os.unlink(sidecar)

    def test_process_judge_job_plugin_without_contract(self):
        coord, state = self._coord()
        coord._active_plugins = []
        job = ("/x/sidecar.json", "m1", "http", "p1", "judgeA", True)
        _process_job(coord, job)
        # contract_id is None -> returns before any judging.
        state.start_judge_activity.assert_not_called()

    def test_process_judge_job_halts_on_terminal_429(self):
        coord, state = self._coord()
        with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump({"state_key": "m1", "runner": "http"}, handle)
            sidecar = handle.name
        try:
            with mock.patch("benchmark.core.judge_response",
                            return_value=_outcome(terminal_429=True, error="429")), \
                 mock.patch("benchmark.core.JUDGE_PROMPT_VERSION", "v1"), \
                 mock.patch("benchmark.judging.judge_contract_id",
                            return_value="c1"), \
                 mock.patch("benchmark.judging.is_successful_judge_vote",
                            return_value=True), \
                 mock.patch("benchmark.judging.judge_votes_for_contract",
                            side_effect=lambda vs, c: vs), \
                 mock.patch("benchmark.judging.merge_judge_vote",
                            side_effect=lambda vs, v: [*vs, v]), \
                 mock.patch("benchmark.judging.confidence_weighted_consensus_by_contract",
                            return_value={"c1": {"score": None, "error": "429"}}), \
                 mock.patch("benchmark.judging.confidence_weighted_consensus",
                            return_value={"score": None, "error": "429"}), \
                 mock.patch("benchmark.judging.save_judge_response"), \
                 mock.patch("benchmark.judging.save_judge_response_metadata"), \
                 mock.patch("benchmark.results.save_judge_result",
                            return_value={"score": None, "error": "429",
                                          "confidence": None, "rationale": None,
                                          "criteria": []}), \
                 mock.patch("benchmark.runtime_records.JudgeAttemptRecord"), \
                 mock.patch("benchmark.runtime_records.JudgeVoteRecord"):
                coord.process_judge_job(
                    (sidecar, "m1", "http", "p1", "judgeA", True))
            self.assertIn("judgeA", coord._halted_judges)
            state.update_judge_progress.assert_called_once_with(
                "judgeA", stopped=True)
            self.assertTrue(coord._judge_stop_events["judgeA"].is_set())
        finally:
            os.unlink(sidecar)

    def test_process_judge_job_halted_judge_early_return(self):
        coord, state = self._coord()
        coord._halted_judges.add("judgeA")
        job = ("/x/sidecar.json", "m1", "http", "p1", "judgeA", True)
        with mock.patch("benchmark.judging.is_successful_judge_vote"):
            coord.process_judge_job(job)
        state.increment_judge_progress.assert_called_once_with(
            "judgeA", expected=-1)
        state.start_judge_activity.assert_not_called()


class JudgeCoordinatorRecordFailureTest(unittest.TestCase):
    def test_record_judge_failure(self):
        state = mock.Mock()
        state.latest_results.return_value = []
        state.snapshot.return_value = {}
        state.run_store.get_cell_id.return_value = None
        coord, _s, _stop = _coordinator(
            state=state,
            judge_models=["judgeA"],
            judge_sources={"judgeA": "srcA"},
        )
        exc = RuntimeError("boom")
        item = {"state_key": "m1"}
        with mock.patch("benchmark.core.JUDGE_PROMPT_VERSION", "v1"), \
             mock.patch("benchmark.judging.is_successful_judge_vote",
                        return_value=False), \
             mock.patch("benchmark.judging.judge_votes_for_contract",
                        side_effect=lambda vs, c: vs), \
             mock.patch("benchmark.judging.merge_judge_vote",
                        side_effect=lambda vs, v: [*vs, v]), \
             mock.patch("benchmark.judging.save_judge_response"), \
             mock.patch("benchmark.judging.save_judge_response_metadata"), \
             mock.patch("benchmark.results.save_judge_result",
                        return_value={"error": "judge input failed"}), \
             mock.patch("benchmark.core.judge_response"), \
             mock.patch("benchmark.runtime_records.JudgeAttemptRecord"), \
             mock.patch("benchmark.runtime_records.JudgeVoteRecord"):
            coord._record_judge_failure(
                exc, "/x/sidecar.json", item, "m1", "http", "p1",
                "judgeA", "m1", None, [], "c1", 1)
        state.update_judge_result.assert_called_once()
        self.assertEqual(coord._run_info["judge_counts"]["failed"], 1)

    def test_record_judge_failure_artifact_error(self):
        state = mock.Mock()
        state.latest_results.return_value = []
        state.snapshot.return_value = {}
        state.run_store.get_cell_id.return_value = None
        coord, _s, _stop = _coordinator(
            state=state,
            judge_models=["judgeA"],
            judge_sources={"judgeA": "srcA"},
        )
        exc = RuntimeError("boom")
        item = {"state_key": "m1"}
        with mock.patch("benchmark.core.JUDGE_PROMPT_VERSION", "v1"), \
             mock.patch("benchmark.judging.is_successful_judge_vote",
                        return_value=False), \
             mock.patch("benchmark.judging.judge_votes_for_contract",
                        side_effect=lambda vs, c: vs), \
             mock.patch("benchmark.judging.merge_judge_vote",
                        side_effect=lambda vs, v: [*vs, v]), \
             mock.patch("benchmark.judging.save_judge_response",
                        side_effect=OSError("disk full")), \
             mock.patch("benchmark.judging.save_judge_response_metadata",), \
             mock.patch("benchmark.results.save_judge_result",
                        return_value={"error": "judge input failed"}), \
             mock.patch("benchmark.core.judge_response"), \
             mock.patch("benchmark.runtime_records.JudgeAttemptRecord"), \
             mock.patch("benchmark.runtime_records.JudgeVoteRecord"), \
             mock.patch("sys.stderr"):
            coord._record_judge_failure(
                exc, "/x/sidecar.json", item, "m1", "http", "p1",
                "judgeA", "m1", None, [], "c1", 1)
        state.update_judge_result.assert_called_once()
        self.assertEqual(coord._run_info["judge_counts"]["failed"], 1)


if __name__ == "__main__":
    unittest.main()
