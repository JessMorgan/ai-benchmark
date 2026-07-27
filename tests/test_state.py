"""Tests for BenchmarkState."""
import json
import os
import tempfile
import threading
import unittest

from plugins import discover_plugins
from tests.utils import load_benchmark_module


class TestBenchmarkState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()
        cls.plugins = discover_plugins()
        cls.plugin_ids = [p.id for p in cls.plugins]

    def test_state_tracks_plugin_fields(self):
        models = {"model-a": "Source1", "model-b": "Source2"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        snap = state.snapshot()
        for name in models:
            for pid in self.plugin_ids:
                self.assertIn(f"{pid}_score", snap[name])
                self.assertIn(f"{pid}_tps", snap[name])

    def test_save_and_load_state(self):
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids, session_seed=12345)
        state.update("model-a", status="completed", **{"rate-limiter_score": 10.0})
        state.add_result({"model": "model-a", "status": "ok", "rate-limiter_score": 10.0})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path, plugin_versions={"rate-limiter": "1.0.0"})
            loaded = self.module.BenchmarkState.load_state(path, models, self.plugin_ids)
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["status"], "completed")
            self.assertEqual(snap["model-a"]["rate-limiter_score"], 10.0)
            self.assertEqual(loaded.session_seed, 12345)

    def test_latest_results_deduplicates(self):
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.add_result({"model": "model-a", "status": "ok", "rate-limiter_score": 5.0})
        state.add_result({"model": "model-a", "status": "ok", "rate-limiter_score": 10.0})
        results = state.latest_results()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["rate-limiter_score"], 10.0)

    def test_load_state_preserves_attempt_start(self):
        """Regression test: attempt_start must survive load_state for TUI."""
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.update("model-a", status="completed")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            loaded = self.module.BenchmarkState.load_state(path, models, self.plugin_ids)
            snap = loaded.snapshot()
            self.assertIn("attempt_start", snap["model-a"])

    def test_save_and_load_state_without_session_seed(self):
        """State files without session_seed load with None."""
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.update("model-a", status="completed")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            # Simulate an older state file without session_seed.
            with open(path) as f:
                data = json.load(f)
            data.pop("session_seed", None)
            with open(path, "w") as f:
                json.dump(data, f)
            loaded = self.module.BenchmarkState.load_state(path, models, self.plugin_ids)
            self.assertIsNone(loaded.session_seed)

    def test_load_state_with_missing_tui_keys(self):
        """Older state files missing newer TUI keys still load without KeyError."""
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.update("model-a", status="completed")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            with open(path) as f:
                data = json.load(f)
            # Strip newer keys that older state files may not have.
            for info in data.get("model_info", {}).values():
                info.pop("phase_detail", None)
                info.pop("attempt", None)
                info.pop("max_tok", None)
            with open(path, "w") as f:
                json.dump(data, f)
            loaded = self.module.BenchmarkState.load_state(path, models, self.plugin_ids)
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["status"], "completed")
            self.assertEqual(snap["model-a"]["phase_detail"], "")
            self.assertEqual(snap["model-a"]["attempt"], 0)
            self.assertEqual(snap["model-a"]["max_tok"], 0)

    def test_load_state_with_dict_model_config(self):
        """Regression test: dict-valued model entries resolve to source strings."""
        raw_models = {
            "model-a": "Source1",
            "model-b": {"source": "Source2", "drop_params": ["seed"]},
        }
        models_source_map = {
            name: (val.get("source", "Default") if isinstance(val, dict) else val)
            for name, val in raw_models.items()
        }
        state = self.module.BenchmarkState(models_source_map, self.plugin_ids)
        state.update("model-a", status="completed")
        state.update("model-b", status="completed")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            loaded = self.module.BenchmarkState.load_state(path, models_source_map, self.plugin_ids)
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["source"], "Source1")
            self.assertEqual(snap["model-b"]["source"], "Source2")
            # The TUI builds a set of source strings; ensure sources are hashable.
            self.assertEqual({s["source"] for s in snap.values()}, {"Source1", "Source2"})

    def test_completed_counts_only_successful_models(self):
        """completed should not treat failed models as finished."""
        models = {"model-a": "Source1", "model-b": "Source2", "model-c": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.update("model-a", status="completed")
        state.update("model-b", status="failed")
        state.update("model-c", status="pending")
        self.assertEqual(state.completed, 1)

    def test_load_state_resets_failed_models_to_pending(self):
        """Failed models are queued for rerun when a saved state is resumed."""
        models = {"model-a": "Source1", "model-b": "Source2"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.update("model-a", status="completed")
        state.update("model-b", status="failed", error="boom", last_error="boom")
        state.add_result({"model": "model-b", "status": "error", "error": "boom"})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            loaded = self.module.BenchmarkState.load_state(path, models, self.plugin_ids)
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["status"], "completed")
            self.assertEqual(snap["model-b"]["status"], "pending")
            self.assertEqual(snap["model-b"]["error"], None)
            self.assertEqual(snap["model-b"]["last_error"], "")

    def test_load_state_with_no_rerun_failed_preserves_failed_status(self):
        """With rerun_failed=False, failed models keep their failed status."""
        models = {"model-a": "Source1", "model-b": "Source2"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.update("model-a", status="completed")
        state.update("model-b", status="failed", error="boom", last_error="boom")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            loaded = self.module.BenchmarkState.load_state(
                path, models, self.plugin_ids, rerun_failed=False)
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["status"], "completed")
            self.assertEqual(snap["model-b"]["status"], "failed")
            self.assertEqual(snap["model-b"]["error"], "boom")

    def test_load_state_with_new_plugins_resets_completed_models(self):
        """When new plugins are added, completed models are re-queued to run them."""
        models = {"model-a": "Source1"}
        original_plugins = ["rate-limiter"]
        state = self.module.BenchmarkState(models, original_plugins)
        state.update("model-a", status="completed", **{"rate-limiter_score": 5.0})
        state.add_result({"model": "model-a", "status": "ok", "rate-limiter_score": 5.0})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            # Now load with an expanded plugin set.
            expanded_plugins = ["rate-limiter", "moe-dense"]
            loaded = self.module.BenchmarkState.load_state(path, models, expanded_plugins)
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["status"], "pending")
            self.assertIn("moe-dense_score", snap["model-a"])
            self.assertIsNone(snap["model-a"]["moe-dense_score"])
            # Old plugin results should be preserved in results.
            latest = loaded.latest_results()[0]
            self.assertEqual(latest["rate-limiter_score"], 5.0)

    def test_load_state_without_new_plugins_preserves_completed_status(self):
        """Without plugin changes, completed models stay completed."""
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.update("model-a", status="completed")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            loaded = self.module.BenchmarkState.load_state(path, models, self.plugin_ids)
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["status"], "completed")


    def test_start_plugin_run_sets_running_status_and_pid(self):
        """start_plugin_run promotes the model to in-flight with the canonical
        status="running" form AND records the pid in ``running_pids`` so the
        live TUI's "[waiting]"/"[streaming]" cells and yellow highlight both
        fire correctly. The previous pid-suffix status string only worked
        for the live-panel filter; ``running_pids`` is the canonical source
        of truth for everything else.
        """
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.start_plugin_run("model-a", "rate-limiter")
        snap = state.snapshot()
        self.assertEqual(snap["model-a"]["status"], "running")
        self.assertIn("rate-limiter", snap["model-a"]["running_pids"])

    def test_start_plugin_run_is_idempotent_on_same_pid(self):
        """Calling start_plugin_run twice with the same pid does not duplicate
        the entry in ``running_pids`` -- otherwise the table would render
        duplicate "[waiting]" cells and the per-model thread count would be
        wrong for plugins that the runtime touches multiple times.
        """
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.start_plugin_run("model-a", "rate-limiter")
        state.start_plugin_run("model-a", "rate-limiter")
        self.assertEqual(state.snapshot()["model-a"]["running_pids"], ["rate-limiter"])

    def test_start_plugin_run_accumulates_concurrent_pids(self):
        """With parallel plugin threads (max_workers > 1), ``running_pids``
        accumulates one entry per in-flight plugin. The previous pid-suffix
        status approach lost all but the most-recent plugin's marker because
        status was overwritten; the list form preserves them all so each
        plugin's cell can render its own "[waiting]" independently.
        """
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.start_plugin_run("model-a", "rate-limiter")
        state.start_plugin_run("model-a", "moe-dense")
        state.start_plugin_run("model-a", "wireframes")
        self.assertEqual(
            state.snapshot()["model-a"]["running_pids"],
            ["rate-limiter", "moe-dense", "wireframes"],
        )

    def test_finish_plugin_run_removes_pid_but_leaves_status(self):
        """finish_plugin_run removes the pid from ``running_pids`` but does
        NOT touch ``status`` -- the outer task commits the final
        "completed"/"failed"/"pending" once all in-flight plugins resolve.
        The brief finalising window with status="running" + running_pids=[]
        is unambiguous and matches the snap a TUI reader would expect.
        """
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.start_plugin_run("model-a", "rate-limiter")
        state.start_plugin_run("model-a", "moe-dense")
        state.finish_plugin_run("model-a", "rate-limiter")
        snap = state.snapshot()
        self.assertEqual(snap["model-a"]["running_pids"], ["moe-dense"])
        # Status stays "running" until the outer task commits a final value.
        self.assertEqual(snap["model-a"]["status"], "running")

    def test_finish_plugin_run_clears_empty_list(self):
        """When the last in-flight plugin finishes, ``running_pids`` is
        emptied. The TUI's live-panel filter (``s.get("running_pids")``)
        therefore drops the model out of "Live:" the moment its last plugin
        thread returns.
        """
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.start_plugin_run("model-a", "rate-limiter")
        state.finish_plugin_run("model-a", "rate-limiter")
        snap = state.snapshot()
        self.assertEqual(snap["model-a"]["running_pids"], [])
        self.assertEqual(snap["model-a"]["status"], "running")

    def test_finish_plugin_run_no_op_for_unknown_pid(self):
        """finish_plugin_run is a no-op when the pid wasn't tracked --
        protects against the rare case where finish is called without a
        matching start (e.g. worker thread cancelled before start_plugin_run
        completed). Idempotency keeps the live panel honest.
        """
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.start_plugin_run("model-a", "rate-limiter")
        state.finish_plugin_run("model-a", "does-not-exist")
        self.assertEqual(
            state.snapshot()["model-a"]["running_pids"], ["rate-limiter"])

    def test_load_state_clears_stale_running_pids(self):
        """Regression test: a saved state with stale ``running_pids`` (from
        a previously interrupted run) must NOT carry those pids forward --
        carrying them forward causes phantom "[waiting]" cells in the live
        TUI even though no plugin task is actually running on resume.
        """
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.update("model-a",
                     status="running",
                     running_pids=["rate-limiter", "moe-dense"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            loaded = self.module.BenchmarkState.load_state(
                path, models, self.plugin_ids)
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["running_pids"], [])
            # The fresh worker will re-promote this to "running" when the
            # task is actually picked up.

    def test_load_state_normalizes_legacy_running_pid_status_to_pending(self):
        """A saved state with the legacy pid-suffix ``"running_<pid>"``
        status (from the pre-``start_plugin_run`` writes) is normalised to
        ``"pending"`` on resume. Previously the migration preserved the
        status + populated ``running_pids`` -- which is what produced the
        phantom yellow highlight deepseek-r1-distill case. Stripping both
        fields on load gives the worker a clean slate to re-promote.
        """
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.update("model-a", status="running_wireframes")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            loaded = self.module.BenchmarkState.load_state(
                path, models, self.plugin_ids)
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["status"], "pending")
            self.assertEqual(snap["model-a"]["running_pids"], [])

    def test_load_state_normalizes_canonical_running_status_to_pending(self):
        """A saved state with ``status="running"`` (the post-fix canonical
        in-flight form) is also normalised to ``"pending"`` on resume -- a
        previously-interrupted worker thread is not magically running again
        after the process restarts, so the new run starts fresh.
        """
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.update("model-a", status="running")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            loaded = self.module.BenchmarkState.load_state(
                path, models, self.plugin_ids)
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["status"], "pending")
            self.assertEqual(snap["model-a"]["running_pids"], [])

    def test_load_state_preserves_non_in_flight_statuses(self):
        """Sanity test: the load_state migration only touches the in-flight
        cases ("running" / "running_<pid>"). Non-in-flight statuses
        ("completed", "failed", "pending") survive the migration unchanged
        so the completed-count / failed-reset behavior is unaffected.
        """
        models = {"model-a": "Source1", "model-b": "Source2", "model-c": "Source1"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        state.update("model-a", status="completed")
        state.update("model-b", status="failed")
        state.update("model-c", status="pending")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            loaded = self.module.BenchmarkState.load_state(
                path, models, self.plugin_ids)
            snap = loaded.snapshot()
            self.assertEqual(snap["model-a"]["status"], "completed")
            self.assertEqual(snap["model-b"]["status"], "pending")  # rerun default
            self.assertEqual(snap["model-c"]["status"], "pending")


    def test_add_bytes_received_accumulates_atomically(self):
        """Repeated calls to ``add_bytes_received`` from concurrent
        threads accumulate the byte counter monotonically without
        losing increments (lock correctness).

        Verifies the new plumbing for the live TUI's
        ``[streaming - N tok]`` cell + ``[name: N tok]`` live
        footer entry.
        """
        state = self.module.BenchmarkState({"m1": "Default"}, ["rate-limiter"])
        state.start_plugin_run("m1", "rate-limiter")
        N_THREADS = 8
        PER_THREAD = 50
        N_PER_CALL = 16  # bytes per call
        expected = N_THREADS * PER_THREAD * N_PER_CALL

        def worker():
            for _ in range(PER_THREAD):
                state.add_bytes_received("m1", "rate-limiter", N_PER_CALL)

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        snap = state.snapshot()["m1"]
        self.assertEqual(snap["rate-limiter_bytes_received"], expected,
                         f"expected {expected} after {N_THREADS}x{PER_THREAD}x{N_PER_CALL} adds, "
                         f"got {snap['rate-limiter_bytes_received']}")

    def test_add_bytes_received_ignores_falsy_n(self):
        """Falsy ``n_bytes`` (None, 0, "") is silently ignored so a
        stray zero-size delta from the SSE layer cannot overwrite the
        accumulated counter to 0.
        """
        state = self.module.BenchmarkState({"m1": "Default"}, ["rate-limiter"])
        state.start_plugin_run("m1", "rate-limiter")
        state.add_bytes_received("m1", "rate-limiter", 128)  # 128 bytes
        state.add_bytes_received("m1", "rate-limiter", 0)
        state.add_bytes_received("m1", "rate-limiter", None)
        state.add_bytes_received("m1", "rate-limiter", "")
        snap = state.snapshot()["m1"]
        self.assertEqual(snap["rate-limiter_bytes_received"], 128)

    def test_start_plugin_run_resets_bytes_received(self):
        """``start_plugin_run`` zeroes the bytes counter so a retry
        dispatch doesn't show stale bytes from the previous (now-
        finished) attempt. Mirrors the existing reset semantics for
        ``attempt_start``.
        """
        state = self.module.BenchmarkState({"m1": "Default"}, ["rate-limiter"])
        state.start_plugin_run("m1", "rate-limiter")
        state.add_bytes_received("m1", "rate-limiter", 500)
        self.assertEqual(state.snapshot()["m1"]["rate-limiter_bytes_received"], 500)
        # End the first attempt and re-dispatch.
        state.finish_plugin_run("m1", "rate-limiter")
        state.start_plugin_run("m1", "rate-limiter")
        self.assertEqual(state.snapshot()["m1"]["rate-limiter_bytes_received"], 0,
                         "start_plugin_run must zero bytes_received on each dispatch")

    def test_bytes_received_default_is_zero(self):
        """Every plugin gets a ``bytes_received`` field of 0 in the
        default model_info dict so the renderer can read it without a
        ``KeyError`` (matching the pattern of ``_score`` etc.).
        """
        state = self.module.BenchmarkState({"m1": "Default"}, ["rate-limiter", "wireframes"])
        snap = state.snapshot()["m1"]
        self.assertEqual(snap["rate-limiter_bytes_received"], 0)
        self.assertEqual(snap["wireframes_bytes_received"], 0)

if __name__ == "__main__":
    unittest.main()
