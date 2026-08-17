"""Tests for BenchmarkState."""
import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from benchmark.state import prepare_state_recovery, repair_state_file
from plugins import discover_plugins
from tests.utils import load_benchmark_module


class TestBenchmarkState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()
        cls.plugins = discover_plugins()
        cls.plugin_ids = [p.id for p in cls.plugins]

    def test_judge_progress_tracks_failed_attempts(self):
        state = self.module.BenchmarkState({"model-a": "Source1"}, ["fake"])
        state.set_judge_progress({"judge": {"completed": 2, "failed": 1, "expected": 3}})
        progress = state.increment_judge_progress("judge", failed=1)
        self.assertEqual(progress, {"completed": 2, "failed": 2, "expected": 3})
        self.assertEqual(state.judge_progress_snapshot()["judge"]["failed"], 2)

    def test_judge_progress_replaces_failed_vote_when_retry_succeeds(self):
        state = self.module.BenchmarkState({"model-a": "Source1"}, ["fake"])
        state.set_judge_progress({"judge": {"completed": 311, "failed": 31, "expected": 342}})
        progress = state.replace_judge_progress(
            "judge", previous_failed=1, completed=1,
        )
        self.assertEqual(progress, {"completed": 312, "failed": 30, "expected": 342})
        self.assertEqual(state.judge_progress_snapshot()["judge"], progress)

        unchanged = state.replace_judge_progress(
            "judge", previous_completed=1, completed=1,
        )
        self.assertEqual(unchanged, progress)

    def test_state_tracks_plugin_fields(self):
        models = {"model-a": "Source1", "model-b": "Source2"}
        state = self.module.BenchmarkState(models, self.plugin_ids)
        snap = state.snapshot()
        for name in models:
            for pid in self.plugin_ids:
                self.assertIn(f"{pid}_score", snap[name])
                self.assertIn(f"{pid}_tps", snap[name])

    def test_repair_known_corrupted_state_key(self):
        """The audited historical malformed key is repaired atomically."""
        models = {"model-a": "Source1"}
        state = self.module.BenchmarkState(models, ["moe-dense"])
        state.update("model-a", status="completed")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            with open(path, "rb") as handle:
                raw = handle.read()
            broken = b'"moe-dense_first_chunk_seen": false,'
            corrupted = raw.replace(broken, b'"moe-dense_first_chunk_see: : false,', 1)
            with open(path, "wb") as handle:
                handle.write(corrupted)
            with self.assertRaises(json.JSONDecodeError), open(path, encoding="utf-8") as handle:
                json.load(handle)

            backup = repair_state_file(path)
            self.assertIsNotNone(backup)
            with open(path, encoding="utf-8") as handle:
                repaired = json.load(handle)
            self.assertEqual(
                repaired["model_info"]["model-a"]["moe-dense_first_chunk_seen"],
                False,
            )
            self.assertTrue(os.path.exists(backup))
            with open(backup, "rb") as handle:
                self.assertIn(b'"moe-dense_first_chunk_see: : false,', handle.read())

    def test_repair_unknown_corruption_is_not_guessed(self):
        """Unknown state corruption must remain untouched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            with open(path, "wb") as handle:
                handle.write(b'{"unknown": : 1}')
            self.assertIsNone(repair_state_file(path))
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), b'{"unknown": : 1}')

    def test_prepare_recovery_counts_invalid_result_row_as_loss(self):
        """Partial inspection reports malformed rows instead of hiding them."""
        raw = (
            b'{"model_info": {}, "results": '
            b'[{"model":"a"}, BAD, {"model":"b"}], '
            b'"active_plugins": []}'
        )
        with tempfile.NamedTemporaryFile() as handle:
            handle.write(raw)
            handle.flush()
            recovery = prepare_state_recovery(handle.name)
        self.assertEqual(recovery["total_results"], 3)
        self.assertEqual(recovery["recoverable_results"], 2)
        self.assertEqual(recovery["lost_results"], 1)
        self.assertEqual([r["model"] for r in recovery["data"]["results"]], ["a", "b"])

    def test_prepare_recovery_counts_truncated_result_as_loss(self):
        """A complete prefix plus a truncated final row is counted as loss."""
        raw = (
            b'{"model_info": {}, "results": '
            b'[{"model":"a"}, {"model":"b"}'
        )
        with tempfile.NamedTemporaryFile() as handle:
            handle.write(raw)
            handle.flush()
            recovery = prepare_state_recovery(handle.name)
        self.assertEqual(recovery["total_results"], 2)
        self.assertEqual(recovery["recoverable_results"], 2)
        self.assertIsNone(recovery["lost_results"])
        self.assertFalse(recovery["counts_certain"])
        self.assertEqual(len(recovery["data"]["results"]), 2)

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
        live TUI's "[streaming]"/"[requested]" bracket cells and yellow
        highlight both fire correctly. The previous pid-suffix status string
        only worked for the live-panel filter; ``running_pids`` is the
        canonical source of truth for everything else.
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
        duplicate "[streaming]" cells and the per-model thread count would be
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
        plugin's cell can render its own "[streaming]" independently.
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
        carrying them forward causes phantom "[streaming]" cells in the live
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
        footer entry. Marks the first chunk first so the new
        self-check (``add_bytes_received`` raises
        ``RuntimeError`` if bytes arrive before the SSE layer
        fired ``mark_first_chunk_seen``) doesn't fire on the
        concurrent worker threads below.
        """
        state = self.module.BenchmarkState({"m1": "Default"}, ["rate-limiter"])
        state.start_plugin_run("m1", "rate-limiter")
        state.mark_first_chunk_seen("m1", "rate-limiter")
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

    def test_add_bytes_received_raises_if_first_chunk_seen_not_fired(self):
        """``add_bytes_received`` is wired to fail loudly if the SSE
        parse layer forgot to call        ``mark_first_chunk_seen`` on the
        first delta. The check is the runtime self-check that protects
        against the cell renderer getting stuck on the pre-chunk
        waiting form (``[streaming]`` or ``[streaming - Ns]``) when
        the real counter is actually arriving. Without this
        RuntimeError the
        drift would only surface visually -- the operator would see
        estimates that never converge to real byte counts.
        """
        state = self.module.BenchmarkState({"m1": "Default"}, ["rate-limiter"])
        state.start_plugin_run("m1", "rate-limiter")
        # Default state: first_chunk_seen is False. The mark has not
        # been fired, so any add_bytes_received call must raise.
        self.assertEqual(
            state.snapshot()["m1"]["rate-limiter_first_chunk_seen"], False)
        with self.assertRaises(RuntimeError) as cm:
            state.add_bytes_received("m1", "rate-limiter", 16)
        self.assertIn("mark_first_chunk_seen", str(cm.exception),
                      "RuntimeError message must cite the missing hook")
        self.assertIn("rate-limiter", str(cm.exception),
                      "RuntimeError message should reference the pid for debuggability")
        # Bytes counter must NOT have grown -- the assertion aborts
        # before the increment.
        snap = state.snapshot()["m1"]
        self.assertEqual(snap["rate-limiter_bytes_received"], 0,
                         "RuntimeError must fire before the bytes write commits")

    def test_add_bytes_received_succeeds_after_mark_first_chunk_seen(self):
        """Sanity pair to ``test_add_bytes_received_raises_if_first_chunk_seen_not_fired``:
        once ``mark_first_chunk_seen`` has fired (the SSE parse layer
        hooks the flag on the first non-empty delta), the bytes
        accumulator accepts writes normally. This pins both halves of
        the contract inside the same test class so a regression in
        either direction shows up next to the other.
        """
        state = self.module.BenchmarkState({"m1": "Default"}, ["rate-limiter"])
        state.start_plugin_run("m1", "rate-limiter")
        state.mark_first_chunk_seen("m1", "rate-limiter")
        # First write after marker: should succeed.
        state.add_bytes_received("m1", "rate-limiter", 64)  # 64 chars -> 16 tok
        self.assertEqual(state.snapshot()["m1"]["rate-limiter_bytes_received"], 64)

    def test_add_bytes_received_ignores_falsy_n(self):
        """Falsy ``n_bytes`` (None, 0, "") is silently ignored so a
        stray zero-size delta from the SSE layer cannot overwrite the
        accumulated counter to 0. Mark the first chunk first so the
        new self-check (``add_bytes_received`` raises when the marker
        is False) doesn't fire during the 128-byte seeding call.
        """
        state = self.module.BenchmarkState({"m1": "Default"}, ["rate-limiter"])
        state.start_plugin_run("m1", "rate-limiter")
        state.mark_first_chunk_seen("m1", "rate-limiter")
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
        ``attempt_start``. Mark the first chunk first so the bytes
        write below doesn't trip the new ``add_bytes_received``
        self-check.
        """
        state = self.module.BenchmarkState({"m1": "Default"}, ["rate-limiter"])
        state.start_plugin_run("m1", "rate-limiter")
        state.mark_first_chunk_seen("m1", "rate-limiter")
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

    def test_set_judge_models_refreshes_rows_and_results(self):
        state = self.module.BenchmarkState(
            {"m1": "Local"}, ["rate-limiter"]
        )
        state.add_result({"model": "m1", "rate-limiter_score": 80})
        state.set_judge_models(["judge-a", "judge-b", "judge-a"])
        self.assertEqual(
            state.snapshot()["m1"]["judge_models"], ["judge-a", "judge-b"]
        )
        self.assertEqual(
            state.latest_results()[0]["judge_models"], ["judge-a", "judge-b"]
        )

    def test_first_chunk_seen_default_is_false(self):
        """Every plugin gets a ``first_chunk_seen`` field of ``False``
        in the default model_info dict so the renderer can read it
        without a ``KeyError`` (matching the pattern of
        ``_bytes_received``). Bool default (not None) cleanly drives
        the ``first_chunk_seen and bytes_received`` branching in
        ``_plugin_cell_block``.
        """
        state = self.module.BenchmarkState(
            {"m1": "Default"}, ["rate-limiter", "wireframes"]
        )
        snap = state.snapshot()["m1"]
        self.assertEqual(snap["rate-limiter_first_chunk_seen"], False)
        self.assertEqual(snap["wireframes_first_chunk_seen"], False)

    def test_first_tok_ts_default_is_zero(self):
        """Every plugin gets a ``first_tok_ts`` field of ``0`` in the
        default model_info dict so the live-footer's
        ``_build_live_indicators`` reader can tell "no chunk has
        landed yet" (0) from a real timestamp (positive float) without
        a ``KeyError``. Default 0 (not None) cleanly drives the
        positive-timestamp gating the renderer uses to switch from
        the ``[pre-stream: K]`` aggregate to the per-plugin
        ``[<pid>: N tok]`` form.
        """
        state = self.module.BenchmarkState(
            {"m1": "Default"}, ["rate-limiter", "wireframes"]
        )
        snap = state.snapshot()["m1"]
        self.assertEqual(snap["rate-limiter_first_tok_ts"], 0)
        self.assertEqual(snap["wireframes_first_tok_ts"], 0)

    def test_mark_first_chunk_seen_writes_first_tok_ts_only_on_first_flip(self):
        """``mark_first_chunk_seen(model, pid, ts=T)`` writes
        ``{pid}_first_tok_ts = T`` ONLY on the False -> True
        transition -- subsequent calls preserve the original
        timestamp. This is the contract the live-footer's
        ``_build_live_indicators`` relies on: time-to-first-token
        is anchored at the actual first delta, not whichever delta
        happens to be the Nth that fires the closure. Without
        the gate, the timestamp could drift later on every chunk
        and the cell-vs-footer time delta would grow misleadingly.
        """
        state = self.module.BenchmarkState({"m1": "Default"}, ["rate-limiter"])
        # First call (False -> True): ts SHOULD be written.
        state.mark_first_chunk_seen("m1", "rate-limiter", ts=1000.5)
        self.assertEqual(state.snapshot()["m1"]["rate-limiter_first_tok_ts"], 1000.5,
                         "First mark_first_chunk_seen with ts= must write the timestamp")
        # Second call (already True): the NEW ts must NOT be written.
        state.mark_first_chunk_seen("m1", "rate-limiter", ts=2000.5)
        self.assertEqual(state.snapshot()["m1"]["rate-limiter_first_tok_ts"], 1000.5,
                         "Subsequent mark_first_chunk_seen calls must preserve the original ts")
        # Third call without ts (back-compat path): timestamp unchanged.
        state.mark_first_chunk_seen("m1", "rate-limiter")
        self.assertEqual(state.snapshot()["m1"]["rate-limiter_first_tok_ts"], 1000.5)
        # Back-compat: ts=None (no ts passed) does not overwrite.
        # This pins the bool-only contract from before the field existed.
        state = self.module.BenchmarkState({"m1": "Default"}, ["rate-limiter"])
        state.mark_first_chunk_seen("m1", "rate-limiter")  # no ts arg
        self.assertEqual(state.snapshot()["m1"]["rate-limiter_first_tok_ts"], 0,
                         "When ts is omitted the timestamp must stay at its default")

    def test_mark_first_chunk_seen_flips_flag_and_is_idempotent(self):
        """``mark_first_chunk_seen`` flips the per-plugin flag from
        ``False`` -> ``True`` (used by the live TUI's
        ``[streaming - N tok]`` cell path to switch from the
        estimate form to the real counter) and is idempotent under
        repeated calls -- the flag never regresses to ``False``
        within a single dispatch (cross-dispatch reset is owned
        by ``start_plugin_run``). Repeated calls also don't error
        -- the SSE parse loop may fire the marker on multiple
        events (first parsed delta, first non-heartbeat byte, etc.)
        and each call must be a no-op rather than a programming
        error.

        Distinct regression markers (single-set AND repeat-set) are
        folded into one test because the repeat path subsumes the
        single-flip path: if the flag regressed on the 2nd call,
        the 3rd-call check would catch it; if the 1st call didn't
        flip it, the post-1st-call check would catch it. Keeping
        them in one test keeps the regression surface compact.
        """
        state = self.module.BenchmarkState({"m1": "Default"}, ["rate-limiter"])
        # (1) Default-state: flag starts False.
        self.assertEqual(
            state.snapshot()["m1"]["rate-limiter_first_chunk_seen"], False
        )
        # (2) First call flips the flag.
        state.mark_first_chunk_seen("m1", "rate-limiter")
        self.assertEqual(
            state.snapshot()["m1"]["rate-limiter_first_chunk_seen"], True
        )
        # (3) Subsequent calls stay True (idempotent, no regression).
        state.mark_first_chunk_seen("m1", "rate-limiter")
        state.mark_first_chunk_seen("m1", "rate-limiter")
        self.assertEqual(
            state.snapshot()["m1"]["rate-limiter_first_chunk_seen"], True,
            "mark_first_chunk_seen must be idempotent: subsequent calls "
            "stay True once flipped"
        )

    def test_start_plugin_run_resets_first_chunk_seen(self):
        """``start_plugin_run`` zeros both ``bytes_received`` and
        ``first_chunk_seen`` so a retry dispatch doesn't carry a
        stale flag from the previous (now-finished) attempt. Without
        this reset, a retry would skip the pre-chunk waiting form
        (``[streaming]`` or ``[streaming - Ns]``) on its first 2s
        and jump straight to the real-counter form --
        visually lying to the operator about what's happening.
        """
        state = self.module.BenchmarkState({"m1": "Default"}, ["rate-limiter"])
        state.start_plugin_run("m1", "rate-limiter")
        state.mark_first_chunk_seen("m1", "rate-limiter")
        state.add_bytes_received("m1", "rate-limiter", 128)
        self.assertEqual(
            state.snapshot()["m1"]["rate-limiter_first_chunk_seen"], True
        )
        # End the first attempt and re-dispatch.
        state.finish_plugin_run("m1", "rate-limiter")
        state.start_plugin_run("m1", "rate-limiter")
        self.assertEqual(
            state.snapshot()["m1"]["rate-limiter_first_chunk_seen"], False,
            "start_plugin_run must reset first_chunk_seen on each dispatch"
        )
        self.assertEqual(
            state.snapshot()["m1"]["rate-limiter_bytes_received"], 0,
            "start_plugin_run must reset bytes_received on each dispatch"
        )

    def test_start_plugin_run_resets_start_ts(self):
        """``start_plugin_run`` records a fresh per-plugin dispatch
        timestamp in ``{pid}_start_ts`` and overwrites any stale value
        from a previous dispatch. This is what lets the live table's
        ``[streaming - Ns]`` / ``[requested - Ns]`` brackets reset to
        zero for each plugin instead of growing for the whole model.
        """
        state = self.module.BenchmarkState({"m1": "Default"}, ["rate-limiter"])
        state.start_plugin_run("m1", "rate-limiter")
        first_ts = state.snapshot()["m1"]["rate-limiter_start_ts"]
        self.assertGreater(first_ts, 0,
                           "start_plugin_run must set a positive start_ts")
        # Simulate a small delay, then re-dispatch the same plugin.
        time.sleep(0.01)
        state.finish_plugin_run("m1", "rate-limiter")
        state.start_plugin_run("m1", "rate-limiter")
        second_ts = state.snapshot()["m1"]["rate-limiter_start_ts"]
        self.assertGreater(second_ts, first_ts,
                           "start_plugin_run must reset start_ts on each dispatch")

    def test_start_plugin_run_resets_first_tok_ts(self):
        """Mirror of the bytes/first_chunk_seen reset: a retry
        dispatch must also zero ``first_tok_ts`` so the live
        footer's time-to-first-token calculation is anchored at
        the new attempt's first delta, not the previous attempt's
        timestamp. Without this reset, the live footer would show
        ``[<pid>: N tok]`` with an old timestamp, falsely implying
        a TTFT that didn't actually happen in this run.
        """
        state = self.module.BenchmarkState({"m1": "Default"}, ["rate-limiter"])
        state.start_plugin_run("m1", "rate-limiter")
        # Capture a real-feeling ts; the value itself doesn't matter.
        state.mark_first_chunk_seen("m1", "rate-limiter", ts=12345.67)
        self.assertEqual(
            state.snapshot()["m1"]["rate-limiter_first_tok_ts"], 12345.67
        )
        # End the first attempt and re-dispatch.
        state.finish_plugin_run("m1", "rate-limiter")
        state.start_plugin_run("m1", "rate-limiter")
        self.assertEqual(
            state.snapshot()["m1"]["rate-limiter_first_tok_ts"], 0,
            "start_plugin_run must zero first_tok_ts on each dispatch "
            "so the live footer doesn't anchor TTFT on the previous attempt"
        )

    def test_load_state_backfills_first_chunk_seen_default(self):
        """Older state files lacking ``{pid}_first_chunk_seen``
        resolve to ``False`` on ``load_state`` so the renderer can
        read the field without a ``KeyError`` after the field was
        added. Backward-compat guard.
        """
        state = self.module.BenchmarkState({"m1": "Default"}, ["rate-limiter"])
        state.update("m1", status="completed")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            with open(path) as f:
                data = json.load(f)
            # Strip the new key to simulate a state file written
            # before the field existed.
            for info in data.get("model_info", {}).values():
                info.pop("rate-limiter_first_chunk_seen", None)
            with open(path, "w") as f:
                json.dump(data, f)
            loaded = self.module.BenchmarkState.load_state(
                path, {"m1": "Default"}, ["rate-limiter"]
            )
            snap = loaded.snapshot()["m1"]
            self.assertEqual(
                snap["rate-limiter_first_chunk_seen"], False,
                "missing first_chunk_seen key in legacy state file must default to False"
            )

    def test_load_state_backfills_first_tok_ts_default(self):
        """Mirror of ``test_load_state_backfills_first_chunk_seen_default``:
        older state files lacking ``{pid}_first_tok_ts`` resolve to
        ``0`` on ``load_state`` so the live footer can read the field
        without a ``KeyError``. Backward-compat guard.
        """
        state = self.module.BenchmarkState({"m1": "Default"}, ["rate-limiter"])
        state.update("m1", status="completed")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            with open(path) as f:
                data = json.load(f)
            # Strip the new key to simulate a pre-field state file.
            for info in data.get("model_info", {}).values():
                info.pop("rate-limiter_first_tok_ts", None)
            with open(path, "w") as f:
                json.dump(data, f)
            loaded = self.module.BenchmarkState.load_state(
                path, {"m1": "Default"}, ["rate-limiter"]
            )
            snap = loaded.snapshot()["m1"]
            self.assertEqual(
                snap["rate-limiter_first_tok_ts"], 0,
                "missing first_tok_ts key in legacy state file must default to 0 "
                "so the live footer treats it as 'no chunk landed yet'"
            )


class TestStateCoverage(unittest.TestCase):
    """Coverage-gap tests for BenchmarkState edge paths."""

    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()

    def test_add_thinking_bytes_before_first_chunk_raises(self):
        """The wiring self-check fires when reasoning arrives before the
        first-chunk marker (mirrors add_bytes_received)."""
        state = self.module.BenchmarkState({"m1": "S"}, ["p"])
        with self.assertRaises(RuntimeError):
            state.add_thinking_bytes_received("m1", "p", 5)

    def test_add_thinking_bytes_requires_positive(self):
        state = self.module.BenchmarkState({"m1": "S"}, ["p"])
        state.mark_first_chunk_seen("m1", "p")
        state.add_thinking_bytes_received("m1", "p", 0)  # no-op
        self.assertEqual(
            state.snapshot()["m1"]["p_thinking_bytes_received"], 0)

    def test_add_thinking_bytes_accumulates(self):
        state = self.module.BenchmarkState({"m1": "S"}, ["p"])
        state.mark_first_chunk_seen("m1", "p", ts=100.0)
        state.add_thinking_bytes_received("m1", "p", 3)
        state.add_thinking_bytes_received("m1", "p", 4)
        snap = state.snapshot()["m1"]
        self.assertEqual(snap["p_thinking_bytes_received"], 7)
        # First-chunk timestamp only written on the False->True transition.
        state.mark_first_chunk_seen("m1", "p", ts=200.0)
        self.assertEqual(snap["p_first_tok_ts"], 100.0)

    def test_log_truncates_at_100_entries(self):
        state = self.module.BenchmarkState({"m1": "S"}, ["p"])
        for i in range(120):
            state.log("m1", f"msg-{i}")
        recent = state.recent_log(5)
        self.assertEqual(len(recent), 5)
        self.assertIn("msg-119", [entry[2] for entry in recent])
        self.assertNotIn("msg-0", [entry[2] for entry in recent])

    def test_total_counts_all_models(self):
        state = self.module.BenchmarkState({"m1": "S", "m2": "S"}, ["p"])
        self.assertEqual(state.total, 2)
        self.assertEqual(state.completed, 0)

    def test_concurrent_state_saves_leave_valid_json_and_no_tmp_file(self):
        """Concurrent writers serialize through the fixed state temp path."""
        state = self.module.BenchmarkState({"m1": "S"}, ["p"])
        state.update("m1", status="completed")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "benchmark_state.json")
            errors = []

            def save():
                try:
                    state.save_state(path)
                except Exception as exc:  # noqa: BLE001 - collect any unexpected save failure
                    errors.append(exc)

            threads = [threading.Thread(target=save) for _ in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2)

            self.assertFalse(errors)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual(saved["model_info"]["m1"]["status"], "completed")
            self.assertFalse(os.path.exists(path + ".tmp"))

    def test_save_state_raise_on_error_propagates_and_cleans_tmp(self):
        state = self.module.BenchmarkState({"m1": "S"}, ["p"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            with (
                mock.patch("json.dump", side_effect=OSError("disk full")),
                self.assertRaises(OSError),
            ):
                state.save_state(path, raise_on_error=True)
            self.assertFalse(os.path.exists(path + ".tmp"))

    def test_save_state_removes_tmp_on_write_failure(self):
        state = self.module.BenchmarkState({"m1": "S"}, ["p"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            # The tmp file is created first; json.dump then fails, so the
            # except branch must clean the tmp file up. A further OSError
            # during that cleanup is swallowed, not re-raised.
            with (
                mock.patch("json.dump", side_effect=OSError("disk full")),
                mock.patch("os.remove", side_effect=OSError("already gone")),
            ):
                state.save_state(path)  # must not raise
            self.assertFalse(os.path.exists(path))

    def test_load_state_ignores_unknown_models(self):
        """Saved info for models no longer in the config is skipped."""
        state = self.module.BenchmarkState({"old-model": "S"}, ["p"])
        state.update("old-model", status="completed")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save_state(path)
            loaded = self.module.BenchmarkState.load_state(
                path, {"new-model": "S"}, ["p"]
            )
            self.assertIn("new-model", loaded.snapshot())
            self.assertNotIn("old-model", loaded.snapshot())


class TestResultJournal(unittest.TestCase):
    """Append-only result journaling for crash-safe state recovery."""

    @classmethod
    def setUpClass(cls):
        cls.module = load_benchmark_module()

    def _state(self, journal_path=None):
        state = self.module.BenchmarkState({"model-a": "Source1"}, ["fake"])
        if journal_path:
            state.set_journal_path(journal_path)
        return state

    def test_add_result_appends_one_json_line_per_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = os.path.join(tmp, "results.journal.jsonl")
            state = self._state(journal)
            state.add_result({"model": "model-a", "status": "ok", "total_time": 1.0})
            state.add_result({"model": "model-b", "status": "error", "total_time": 2.0})
            with open(journal, encoding="utf-8") as handle:
                lines = handle.read().strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["model"], "model-a")
            self.assertEqual(json.loads(lines[1])["status"], "error")

    def test_add_result_noop_without_journal_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = os.path.join(tmp, "results.journal.jsonl")
            state = self._state()
            state.add_result({"model": "model-a", "status": "ok"})
            self.assertFalse(os.path.exists(journal))

    def test_replay_journal_returns_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = os.path.join(tmp, "results.journal.jsonl")
            state = self._state(journal)
            state.add_result({"model": "model-a", "status": "ok"})
            state.add_result({"model": "model-b", "status": "error"})
            self.assertEqual(
                [r["model"] for r in self.module.BenchmarkState.replay_journal(journal)],
                ["model-a", "model-b"],
            )

    def test_replay_journal_tolerates_partial_trailing_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = os.path.join(tmp, "results.journal.jsonl")
            with open(journal, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"model": "ok"}) + "\n")
                handle.write('{"model": "partial"')  # crash mid-write
            results = self.module.BenchmarkState.replay_journal(journal)
            self.assertEqual([r["model"] for r in results], ["ok"])

    def test_replay_journal_missing_file_returns_empty(self):
        self.assertEqual(
            self.module.BenchmarkState.replay_journal("/nonexistent/journal.jsonl"), []
        )

    def test_journal_write_failure_is_best_effort(self):
        state = self._state("/nonexistent-dir/journal.jsonl")
        # Must not raise: journaling is best-effort and never fails a result.
        state.add_result({"model": "model-a", "status": "ok"})
        self.assertEqual(len(state.results), 1)

    def test_set_journal_path_truncates_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = os.path.join(tmp, "results.journal.jsonl")
            state = self._state(journal)
            state.add_result({"model": "model-a", "status": "ok"})
            state.set_journal_path(journal, truncate=True)
            self.assertEqual(
                self.module.BenchmarkState.replay_journal(journal), []
            )

    def test_resume_does_not_truncate_existing_journal(self):
        # Regression: resuming (load_state + re-attach the journal without
        # truncating) must preserve pre-resume results so a later crash can
        # still replay them from the append-only journal.
        with tempfile.TemporaryDirectory() as tmp:
            journal = os.path.join(tmp, "results.journal.jsonl")
            state_file = os.path.join(tmp, "benchmark_state.json")

            # Session 1: complete a result (journaled) and save the state.
            state = self._state(journal)
            state.update("model-a", status="completed")
            state.add_result({"model": "model-a", "status": "ok", "total_time": 1.0})
            state.save_state(state_file)

            # Session 2: resume, re-attach the journal without truncating,
            # and complete another result.
            resumed = self.module.BenchmarkState.load_state(
                state_file, {"model-a": "Source1"}, ["fake"])
            resumed.set_journal_path(journal)
            resumed.add_result({"model": "model-a", "status": "ok", "total_time": 2.0})

            self.assertEqual(
                [r["model"] for r in self.module.BenchmarkState.replay_journal(journal)],
                ["model-a", "model-a"],
            )


if __name__ == "__main__":
    unittest.main()
