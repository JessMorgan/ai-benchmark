r"""Regression tests for the `purge-results` SKILL embedded script.

These tests pin two independent contracts that have already shipped as
distinct fixes:

1. **Extraction contract** — the embedded script's first line is
   ``#!/usr/bin/env python3`` and the script body lives inside a single
   ``\`\`\`python``` markdown fence, so invoking agents can recover the
   runnable script with a stable awk extraction (anchored on the
   shebang, exits just before the closing fence). The naive
   ``sed -n '/^```python$/,/^```$/p' | sed '1d;$d'`` extraction
   silently produces an empty script on SKILL.md re-renderings
   that strip the column-0 anchor; once that happens,
   ``identify_targets`` returns zero matches even when the state has
   them, so the user can never tell the difference between "no 1-tok
   entries to purge" and "broken pipeline".

2. **Bug-fix contract** — the prior `identify_targets` read metrics
   from `state.results` rather than `state.model_info`. That meant
   any prior partial purge that stripped `{pid}_score` from the
   latest res dict while `state.model_info` still carried a stale
   `_output_tokens == 1` would silently NOT be revisited. The fix:

   - iterates over `state_models` from `model_info.keys()` (so
     orphan model_info-only entries are still candidates), and
   - iterates over the canonical `plugin_ids` (from
     `active_plugins`), and
   - reads `score_val`/`tok_val` from `state.model_info`, and
   - guards `remove_pair` against `latest_idx=None` so orphan matches
     don't IndexError.

The 14-targets synthetic fixture pins all four guarantees at once:
the user's run-directory had 13 normal (in BOTH `results` AND
`model_info`) plus 1 orphan (in `model_info` only) one-token
entries, totalling exactly 14.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from typing import Any

_SKILL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    ".agents",
    "skills",
    "purge-results",
    "SKILL.md",
)


def _extract_purge_script(skill_path: str = _SKILL_PATH) -> str:
    r"""Recover the embedded ``purge-results`` script using the exact
    awk command the SKILL.md "How to invoke" section prescribes.

    The shebang pattern and the closing-fence exit pinpoint the
    script body without depending on markdown fence column-0
    anchoring; if SKILL.md somehow loses its starting ``\`\`\`python```
    line, this extraction fails fast (the ``startswith`` guard raises)
    rather than silently producing an empty script.
    """
    # The awk pattern is identical to what the SKILL.md docs:
    #   awk '/^#!\/usr\/bin\/env python3/{f=1} f && /^```/{exit} f' \
    #       .agents/skills/purge-results/SKILL.md > /tmp/purge-results.py
    proc = subprocess.run(
        [
            "awk",
            r"/^#!\/usr\/bin\/env python3/{f=1} f && /^```/{exit} f",
            skill_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    script = proc.stdout
    assert script.startswith("#!/usr/bin/env python3"), (
        "SKILL.md no longer carries the embedded purge-results "
        "script's shebang as the FIRST line; the awk extraction "
        "contract is broken (would have silently produced an "
        "empty script in the field)."
    )
    # Strip every line before checking the tail -- the script's
    # trailing call is indented under ``if __name__ == "__main__":``
    # so its raw text is four-space ``    main()``, not bare
    # ``main()``.
    non_blank = [line for line in script.splitlines() if line.strip()]
    assert non_blank[-1].strip() == "main()", (
        "SKILL.md extraction now drops the trailing main() call -- "
        "agents invoking `python3 /tmp/purge-results.py --apply` "
        "would no-op without ever calling main()."
    )
    return script


class _PurgeScriptNs:
    """Fresh exec-namespace wrapper so ``setUpClass`` can share the
    extracted module across tests without leaking stateful
    mutations between cases (a class-level setUp re-runs the
    awk extraction once per class)."""

    def __init__(self, src: str):
        ns: dict = {"__name__": "purge_results_test"}
        exec(compile(src, "<extracted>", "exec"), ns)  # noqa: S102 - executing the extracted test code is the point
        self.src = src
        self.ns = ns

    def __getattr__(self, name):
        return self.ns[name]


class TestPurgeResultsExtraction(unittest.TestCase):
    """Pin the awk extraction contract so future SKILL.md edits cannot
    silently regress the skill to a no-op."""

    @classmethod
    def setUpClass(cls):
        cls.script = _extract_purge_script()

    def test_first_line_is_shebang(self):
        self.assertEqual(self.script.splitlines()[0], "#!/usr/bin/env python3")

    def test_last_visible_line_is_main_call(self):
        non_blank = [line for line in self.script.splitlines() if line.strip()]
        self.assertEqual(non_blank[-1].strip(), "main()")

    def test_imports_as_python_module(self):
        ns = _PurgeScriptNs(self.script).ns
        for fname in ("identify_targets", "remove_pair", "main",
                      "discover_plugin_ids", "load_state"):
            self.assertIn(fname, ns, f"missing top-level def {fname}")

    def test_extracted_script_runs_help(self):
        """The extracted script must be runnable as a CLI: invoking
        ``python3 <script> --help`` should produce argparse usage.
        This pins the trailing ``main()`` call end-to-end, not just
        by grep — a SKILL.md edit that re-introduced the
        awk/missing-main() bug would now fail this test."""
        fd, tmp = tempfile.mkstemp(suffix="_purge_results_help.py")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(self.script)
            proc = subprocess.run(
                [sys.executable, tmp, "--help"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("purge-results", proc.stdout)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass


def _build_14_targets_state() -> dict:
    """Synthetic fixture mirroring the user's
    an earlier benchmark run: 13 models
    present in BOTH `state.results` AND `state.model_info` plus 1
    orphan model present ONLY in `state.model_info`, all 14 carrying
    ``pluginA_output_tokens == 1``. Plus a single noise model that has
    pluginA in BOTH dicts but NOT matching the filter, to confirm the
    filter doesn't over-match.
    """
    state: dict[str, Any] = {
        "active_plugins": ["pluginA", "pluginB"],
        "results": [],
        "model_info": {},
    }

    # 13 normal models: present in both dicts; pluginA has 1-tok.
    for i in range(13):
        m = f"model_{i:02d}"
        state["results"].append({
            "model": m,
            "pluginA_score": 5,
            "pluginA_output_tokens": 1,  # MATCH
            "pluginB_score": 80,
            "pluginB_output_tokens": 50,
            "pluginA_response_time": 12.0,
            "pluginA_stream_ok": True,
            "pluginA_judge_score": 90,
            "pluginA_judge_votes": 3,
            "pluginA_judge_complete": True,
            "pluginA_judge_queued": False,
            "pluginA_empty_reason": None,
            "pluginA_diagnostics": {"errors": []},
        })
        state["model_info"][m] = {
            "source": "local",
            "pluginA_score": 5,
            "pluginA_output_tokens": 1,  # canonical mirror MATCH
            "pluginA_bytes_received": 32,
            "pluginA_thinking_bytes_received": 8,
            "pluginA_first_chunk_seen": True,
            "pluginA_first_tok_ts": 1.234,
            "pluginA_start_ts": 1.0,
            "pluginA_judge_score": 90,
            "pluginA_judge_votes": 3,
            "pluginA_judge_complete": True,
            "pluginA_judge_queued": False,
            "pluginA_judge_rationale": "ok",
            "pluginA_empty_reason": None,
            "pluginA_diagnostics": {"errors": []},
            "pluginB_score": 80,
            "pluginB_output_tokens": 50,
        }

    # 1 orphan: present in model_info ONLY -- model never produced a
    # state.results row. pluginA still matches the --tok=1 filter
    # here. This is the case that the prior `latest_idx=None` fix
    # specifically protects against.
    state["model_info"]["model_orphan"] = {
        "source": "local",
        "pluginA_score": 1,
        "pluginA_output_tokens": 1,  # MATCH (orphan path)
        "pluginA_bytes_received": 4,
        "pluginA_judge_score": 70,
        "pluginA_empty_reason": "empty",
        "pluginA_diagnostics": {"errors": []},
    }
    # No state.results entry for model_orphan.

    # 1 noise: present in both; pluginA has 15 tokens, must NOT match.
    state["results"].append({
        "model": "model_noise",
        "pluginA_score": 50,
        "pluginA_output_tokens": 15,
        "pluginB_score": 90,
        "pluginB_output_tokens": 200,
    })
    state["model_info"]["model_noise"] = {
        "source": "local",
        "pluginA_score": 50,
        "pluginA_output_tokens": 15,  # NOT MATCH
        "pluginB_score": 90,
        "pluginB_output_tokens": 200,
    }

    return state


def _make_args(**overrides):
    """Build a SimpleNamespace mirroring argparse output for the
    fields ``identify_targets`` reads. ``args.apply`` is NOT set --
    ``identify_targets`` only consumes ``tok/score/fail/target/
    model/plugin``, so the apply flag has no observable effect
    at this layer (it's only consumed by ``main()``, which the
    CLI tests exercise end-to-end)."""
    ns = types.SimpleNamespace(
        tok=None,
        score=None,
        fail=False,
        target=None,
        model=None,
        plugin=None,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class TestPurgeResultsIdentifyTargets(unittest.TestCase):
    """Pin the post-fix behaviour that identified the user's bug:
    iterate over `state_models` and read metrics from `state.model_info`
    so a stale `_output_tokens == 1` in model_info is ALWAYS revisited
    even if the latest res dict has been stripped of its score keys.
    """

    @classmethod
    def setUpClass(cls):
        cls.script = _extract_purge_script()

    def setUp(self):
        self.purge = _PurgeScriptNs(self.script)

    def test_14_target_case_matches_exactly(self):
        """Phantom test: replicates the real 1-tok purge the user hit.
        Synthetic state has 13 normal + 1 orphan = 14 matches."""
        state = _build_14_targets_state()
        plugin_ids = self.purge.discover_plugin_ids(state)
        self.assertEqual(plugin_ids, ["pluginA", "pluginB"])
        args = _make_args(tok=1)
        targets = self.purge.identify_targets(state, args, plugin_ids)
        self.assertEqual(len(targets), 14)
        # Make sure every match is `pluginA` (only plugin with 1 tok).
        for _m, pid, _lidx, _reason in targets:
            self.assertEqual(pid, "pluginA", f"unexpected plugin: {pid}")

    def test_orphan_is_matched_with_latest_idx_none(self):
        """The orphan case from the user's run -- model_orphan exists
        in model_info only, has no state.results row, but its
        ``pluginA_output_tokens == 1`` still triggers a match. The
        returned ``latest_idx`` MUST be ``None`` (otherwise the
        `results` row would be phantom-attached)."""
        state = _build_14_targets_state()
        plugin_ids = self.purge.discover_plugin_ids(state)
        args = _make_args(tok=1)
        targets = self.purge.identify_targets(state, args, plugin_ids)
        orphans = [t for t in targets if t[0] == "model_orphan"]
        self.assertEqual(len(orphans), 1)
        self.assertIsNone(orphans[0][2],
                          "orphan must surface as latest_idx=None")

    def test_noise_model_is_not_matched(self):
        """The 15-tok noise model must NOT match the --tok=1 filter,
        proving the filter is exact (not <= or >=)."""
        state = _build_14_targets_state()
        plugin_ids = self.purge.discover_plugin_ids(state)
        args = _make_args(tok=1)
        targets = self.purge.identify_targets(state, args, plugin_ids)
        matched_models = {m for m, _, _, _ in targets}
        self.assertNotIn("model_noise", matched_models)

    def test_noise_model_info_keys_remain_intact(self):
        """Sanity pin: applying remove_pair to the 14 matched
        targets must NOT collateral-damage the noise model (it
        isn't in ``targets`` so remove_pair is never called on it).
        Verifies the under-the-hood invariant that surgical
        removal is per-pair, not per-model."""
        state = _build_14_targets_state()
        plugin_ids = self.purge.discover_plugin_ids(state)
        args = _make_args(tok=1)
        targets = self.purge.identify_targets(state, args, plugin_ids)
        copy_state = copy.deepcopy(state)
        for m, p, l_i, _ in targets:
            self.purge.remove_pair(copy_state, m, p, l_i)
        # The noise model should be untouched after the apply.
        # Pin by name (NOT by ``[-1]`` indexing) so a future
        # fixture reorder doesn't silently degrade this to
        # "two random entries compared".
        self.assertEqual(
            copy_state["model_info"]["model_noise"],
            state["model_info"]["model_noise"],
        )
        pre_noise = next(
            r for r in state["results"] if r["model"] == "model_noise"
        )
        post_noise = next(
            r for r in copy_state["results"] if r["model"] == "model_noise"
        )
        self.assertEqual(post_noise, pre_noise)

    def test_pluginb_is_not_matched_regardless_of_tok_filter(self):
        """pluginB has 50 tok for every model -- no pluginB match for
        `--tok 1`. Confirms filters hit only the targeted {pid} cells."""
        state = _build_14_targets_state()
        plugin_ids = self.purge.discover_plugin_ids(state)
        args = _make_args(tok=1)
        targets = self.purge.identify_targets(state, args, plugin_ids)
        for _m, pid, _, _ in targets:
            self.assertEqual(pid, "pluginA")

    def test_no_filters_returns_empty(self):
        """`--tok None --score None --fail False --target None
        --model None --plugin None` returns [] so an unfiltered run
        is a no-op (user must supply at least one criterion)."""
        state = _build_14_targets_state()
        plugin_ids = self.purge.discover_plugin_ids(state)
        args = _make_args()
        targets = self.purge.identify_targets(state, args, plugin_ids)
        self.assertEqual(targets, [])


class TestPurgeResultsRemovePair(unittest.TestCase):
    """Pin the surgical-strip contract AND the orphan no-IndexError
    fix in the same class so a future change to ``remove_pair`` cannot
    regress the orphan path."""

    @classmethod
    def setUpClass(cls):
        cls.script = _extract_purge_script()

    def setUp(self):
        self.purge = _PurgeScriptNs(self.script)

    def test_apply_strips_matched_keys_in_both_dicts(self):
        """After mutation on the 14-targets case, ``pluginA_*`` keys
        are STRIPPED from BOTH ``state.results[latest_idx]`` and
        ``state.model_info[model_key]``. ``pluginB_*`` keys and
        per-model core fields (e.g. ``source``) are PRESERVED so the
        other plugins and the model's metadata stay re-runnable."""
        state = _build_14_targets_state()
        plugin_ids = self.purge.discover_plugin_ids(state)
        args = _make_args(tok=1)
        targets = self.purge.identify_targets(state, args, plugin_ids)
        self.assertEqual(len(targets), 14)

        # Verify each pair INSIDE the loop. A post-loop check on the
        # loop variables would only inspect the LAST iteration, not
        # all 14.
        for m, pid, l_idx, _ in targets:
            self.purge.remove_pair(state, m, pid, l_idx)

            if l_idx is None:
                # Orphan path: model_info only, no state.results row.
                for suf in ("score", "output_tokens", "response_time",
                            "stream_ok", "judge_score", "empty_reason",
                            "diagnostics"):
                    self.assertNotIn(
                        f"{pid}_{suf}", state["model_info"][m],
                        f"orphan {m} left {pid}_{suf}",
                    )
                self.assertNotIn(
                    f"{pid}_bytes_received", state["model_info"][m],
                    f"orphan {m} left transient {pid}_bytes_received",
                )
            else:
                # Normal path: BOTH dicts stripped of pluginA_* (score,
                # timing, judge, residual, and transient keys).
                res = state["results"][l_idx]
                for suf in ("score", "output_tokens", "response_time",
                            "stream_ok", "judge_score", "judge_votes",
                            "judge_complete", "judge_queued",
                            "empty_reason", "diagnostics"):
                    self.assertNotIn(
                        f"{pid}_{suf}", res,
                        f"results[{l_idx}] left {pid}_{suf}",
                    )
                info = state["model_info"][m]
                for suf in ("score", "output_tokens", "response_time",
                            "stream_ok", "bytes_received",
                            "thinking_bytes_received", "first_chunk_seen",
                            "first_tok_ts", "start_ts", "judge_score",
                            "judge_votes", "judge_complete",
                            "judge_queued", "judge_rationale",
                            "empty_reason", "diagnostics"):
                    self.assertNotIn(
                        f"{pid}_{suf}", info,
                        f"model_info[{m}] left {pid}_{suf}",
                    )
                # pluginB intact in both.
                self.assertIn("pluginB_score", res)
                self.assertIn("pluginB_output_tokens", res)
                self.assertIn("pluginB_score", info)
                self.assertIn("pluginB_output_tokens", info)
                # core metadata intact (the model isn't nuked).
                self.assertIn("source", info)

    def test_remove_pair_orphan_does_not_indexerror(self):
        """The prior bug: ``remove_pair(state, m, pid, latest_idx=None)``
        used to call ``state['results'][latest_idx]`` and IndexError.
        Now it short-circuits the results-side strip and only mutates
        model_info. This case pins it directly without going through
        identify_targets."""
        state = _build_14_targets_state()
        before_info = copy.deepcopy(state["model_info"]["model_orphan"])
        # Direct call -- should not raise.
        self.purge.remove_pair(state, "model_orphan", "pluginA", None)
        after_info = state["model_info"]["model_orphan"]
        self.assertNotIn("pluginA_score", after_info)
        self.assertNotIn("pluginA_output_tokens", after_info)
        self.assertNotIn("pluginA_bytes_received", after_info)
        # Anything that wasn't a pluginA_* key is intact (e.g. source).
        self.assertEqual(after_info["source"], before_info["source"])

    def test_warn_unknown_plugin_keys_ignores_judge_subkeys(self):
        """`warn_unknown_plugin_keys` must NOT flag per-plugin judge
        sub-keys (e.g. ``pluginA_judge_score``) as unknown plugin ids --
        they live under a real plugin's prefix. Only genuinely unknown
        plugin ids (e.g. a stale ``ghost_score``) should warn. This keeps
        the post-purge stderr clean now that judge keys are stripped by
        default."""
        import contextlib
        import io

        state = _build_14_targets_state()
        # A genuinely unknown plugin id, plus the judge sub-keys that
        # used to be misreported as unknown plugin ids.
        state["model_info"]["model_00"]["ghost_score"] = 42

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.purge.warn_unknown_plugin_keys(
                state, ["pluginA", "pluginB"]
            )
        out = err.getvalue()
        self.assertIn("ghost", out)
        self.assertNotIn("pluginA_judge", out)
        self.assertNotIn("pluginB_judge", out)


class TestPurgeResultsDryRunVsApply(unittest.TestCase):
    """End-to-end pin via the script's own entry point: ``--tok 1``
    on the synthetic state prints the 14-target table in dry-run mode
    and DOES NOT write the file. Confirms the extracted script works
    as a CLI on synthetic state, not just via in-process exec."""

    @classmethod
    def setUpClass(cls):
        cls.script = _extract_purge_script()

    def setUp(self):
        # Materialise the script and synthetic state under
        # ``tempfile.mkdtemp`` so concurrent test runs don't trample
        # a hardcoded `/tmp/test_purge_results_*` path.
        self.run_dir = tempfile.mkdtemp(prefix="test_purge_results_run_")
        self.script_path = os.path.join(self.run_dir, "purge-cli.py")
        with open(self.script_path, "w") as f:
            f.write(self.script)
        with open(os.path.join(self.run_dir, "benchmark_state.json"), "w") as f:
            json.dump(_build_14_targets_state(), f)

    def tearDown(self):
        # Clean up everything we wrote (script + state + .bak).
        for root, _dirs, files in os.walk(self.run_dir):
            for f in files:
                try:
                    os.remove(os.path.join(root, f))
                except OSError:
                    pass
        try:
            os.rmdir(self.run_dir)
        except OSError:
            pass

    def _post_state(self) -> dict:
        with open(os.path.join(self.run_dir, "benchmark_state.json")) as f:
            return json.load(f)

    def _pre_state(self) -> dict:
        return _build_14_targets_state()

    def test_dry_run_prints_targets_no_mutation(self):
        proc = subprocess.run(
            [sys.executable, self.script_path,
             "--output-dir", self.run_dir,
             "--tok", "1"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("14 (model, plugin) pairs matched", proc.stdout)
        self.assertIn("DRY RUN", proc.stdout)
        # State is untouched.
        post = self._post_state()
        self.assertEqual(post, self._pre_state(),
                         "dry-run must NOT mutate benchmark_state.json")

    def test_apply_mutates_state_and_writes_backup(self):
        proc = subprocess.run(
            [sys.executable, self.script_path,
             "--output-dir", self.run_dir,
             "--tok", "1", "--apply"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("applied 14 purges", proc.stdout)

        # Post-mutation state has zero pluginA_output_tokens == 1.
        post = self._post_state()
        for m, info in post["model_info"].items():
            self.assertNotEqual(
                info.get("pluginA_output_tokens"), 1,
                f"model_info[{m}] still has pluginA_output_tokens=1",
            )

        # At least one backup exists.
        backups = [
            f for f in os.listdir(self.run_dir)
            if f.startswith("benchmark_state.json.pre-purge-")
            and f.endswith(".bak")
        ]
        self.assertTrue(backups,
                        "expected a pre-purge .bak backup file")
        for b in backups:
            with open(os.path.join(self.run_dir, b)) as f:
                bak_state = json.load(f)
            self.assertEqual(
                bak_state, self._pre_state(),
                "backup must equal the pre-mutation state verbatim",
            )


if __name__ == "__main__":
    unittest.main()
