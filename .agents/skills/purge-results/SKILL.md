# Skill: `purge-results`

## Description

Surgically removes specific `(model, plugin)` result entries from a benchmark
run's `benchmark_state.json` so the affected plugins are re-executed on the
next `ai-benchmark.py` resume. Other plugins' results stay intact — only the
removed ones re-run.

The script strips **both** the score/timing metrics **and** the transient
runtime state (bytes_received, first_chunk_seen, first_tok_ts, start_ts) so
the next run starts from a clean per-plugin slate.

## When to use

Load this skill when the user asks to:

- "re-run" / "rerun" certain plugins or models
- remove single-token / zero-token / N-token responses
- remove plugins where the model returned garbage (empty `.txt`)
- remove plugins where the score is `0` (legitimately low-quality output)
- remove plugins where the score is `"fail"` (HTTP/timeout failure)
- remove a specific `(model, plugin)` pair
- remove all results for a specific model
- remove all results for a specific plugin
- **inspect** what a previous purge did (`--diff <backup>`)

## Safety contract

1. **Always** run the script **WITHOUT** `--apply` first (dry-run mode).
2. Present the console summary table of `(Model, Plugin)` pairs to the user.
3. Once the user confirms the list is exactly what they want, re-run the
   **exact same command** with `--apply` appended.
4. The script always creates a timestamped `.bak` backup before any
   mutation: `benchmark_state.json.pre-purge-<ms-ts>.bak`. The millisecond
   suffix prevents collisions when multiple purges fire in the same
   second. The user can inspect or restore from any backup file.
5. The script never deletes `responses/<model>/<plugin>.txt` or
   `.meta.json` files — they will be overwritten on the next run. This
   keeps the backup fully sufficient for undo.

## Schema discovery (self-maintaining)

Plugin IDs are read from `state["active_plugins"]` (saved by
`BenchmarkState.save_state` at every checkpoint) rather than hard-coded.
This keeps the skill in step with `benchmark_state.py` automatically; if
a new plugin is added there, it becomes available to `--plugin` /
`--target` here without any edit.

Key suffixes stripped per pair are the schema constants
`SCORE_SUFFIXES` and `TRANSIENT_SUFFIXES` declared at the top of the
embedded script — they mirror what `BenchmarkState.__init__` writes
alongside the `{pid}_score` field. The script also performs a
self-check: if `state["model_info"]` contains `_score` keys whose plugin
id is NOT in `state["active_plugins"]`, the script prints a warning to
stderr so the operator knows the state file is out-of-sync.

## How to invoke

The agent executing this skill should:

1. **Extract the embedded script below** to a temporary file
   (e.g., `/tmp/purge-results.py`) so it can be run repeatedly
   with different filters. Use this `awk` extraction rather than
   strict `sed` markdown-fence matching — the range
   ``#!/usr/bin/env python3`` ... ``if __name__ == "__main__":``
   tolerates shell-quoting, heredoc, and fence-indent variations
   that the naive ``sed -n '/^```python$/,/^```$/p'`` regex misses
   in practice (the SKILL.md's fences can be re-rendered through
   intermediate formats that strip their column-0 anchor). The
   shebang is stable wherever the script is dropped, and the
   script always ends with the closing fence of the
   ``\`\`\`python``` block, so extracting from the shebang and
   exiting just before the closing fence (rather than matching
   a body anchor that drops the trailing ``main()`` call) is
   bullet-proof and preserves the runnable script verbatim:

```bash
awk '/^#!\/usr\/bin\/env python3/{f=1} f && /^```/{exit} f' \
    .agents/skills/purge-results/SKILL.md > /tmp/purge-results.py
```

2. Run it with the appropriate flags.

```bash
python3 /tmp/purge-results.py --output-dir <RUN_DIR> [filters] [--apply]
python3 /tmp/purge-results.py --output-dir <RUN_DIR> --diff <backup.bak>
```

### Flags

| Flag | Description |
|------|-------------|
| `--output-dir PATH`     | Run directory containing `benchmark_state.json` (default `.`) |
| `--tok N`               | Purge results where `{pid}_Output_Tokens == N` (e.g. `1` for single-token) |
| `--score N`             | Purge results where `{pid}_Score == N` (e.g. `0`) |
| `--fail`                | Purge results where `{pid}_Score == "fail"` |
| `--model M`             | Purge ALL plugins for a specific model (fuzzy name match) |
| `--plugin P`            | Purge ALL models for a specific plugin (must match a known plugin) |
| `--target M:P`          | Purge a SPECIFIC `(model, plugin)` pair (fuzzy match on both halves) |
| `--apply`               | Actually mutate; without it the run is a dry-run |
| `--diff <backup.bak>`   | Show what was removed in a previous purge (read-only) |
| `--quiet`               | Suppress per-pair tables, dry-run/apply banners, and per-pair diagnostic prints; only aggregate counts and the diff headline are emitted |

Filters combine with **AND** semantics. Examples:

```bash
# Single-token entries from the current run
python3 /tmp/purge-results.py --output-dir . --tok 1

# All fail-status results
python3 /tmp/purge-results.py --output-dir . --fail

# All rate-limiter results for one specific model
python3 /tmp/purge-results.py --output-dir . \
    --model "qwen3.5:9b-32k" --plugin rate-limiter --apply

# A single explicit pair
python3 /tmp/purge-results.py --output-dir . \
    --target "qwen3.5:9b-32k:multi-step" --apply

# Inspect what a previous purge did
python3 /tmp/purge-results.py --output-dir . \
    --diff benchmark_state.json.pre-purge-1700000000.123.bak
```

## The script

Save this entire block to `/tmp/purge-results.py` (or any path of your
choosing) and run it.

```python
#!/usr/bin/env python3
"""purge-results -- surgically remove (model, plugin) entries from
benchmark_state.json.

Default mode: dry-run (prints targets, mutates nothing).
With --apply:  backs up state, mutates score + transient keys for matched
              (model, plugin) pairs in BOTH state.results (latest dict per
              model) and state.model_info, then writes back.
With --diff:  reports what a previous backup-vs-current diff shows
              (counting ALL stripped score/timing keys per pair, plus the
              transient in-flight keys removed from model_info).
"""
import argparse
import json
import os
import re
import shutil
import sys
import time


# Per-plugin key suffixes stripped from BOTH state.results (latest dict per
# model) AND state.model_info. These mirror the keys BenchmarkState writes
# alongside the {pid}_score field in __init__.
SCORE_SUFFIXES = (
    "score", "tps", "response_time", "output_tokens",
    "thinking_tokens", "total_tokens",
    "stream_ok", "truncated", "repeating", "rubric",
)

# Transient in-flight fields stripped ONLY from state.model_info (they
# don't exist in latest results). Keeps the next dispatch from carrying
# stale streaming byte counts or first-chunk timestamps across the re-run.
TRANSIENT_SUFFIXES = (
    "bytes_received", "first_chunk_seen", "first_tok_ts", "start_ts",
)

# Per-model non-plugin fields preserved untouched during the surgery.
CORE_FIELDS = {
    "source", "api_model", "system_prompt", "is_agent",
    "status", "ttft", "error", "elapsed", "attempt",
    "max_tok", "attempt_start", "last_error", "phase_detail",
    "running_pids", "rubric",
}


def discover_plugin_ids(state):
    """Read plugin IDs from ``state['active_plugins']`` (saved every
    checkpoint by ``BenchmarkState.save_state``). Falls back to scanning
    ``state['model_info']`` for ``*_score`` keys so legacy state files
    that pre-date this convention still work.
    """
    pids = list(state.get("active_plugins") or [])
    if pids:
        return pids
    # Legacy fallback: derive from any _score key in model_info.
    seen = set()
    for info in state.get("model_info", {}).values():
        if isinstance(info, dict):
            seen.update(
                k[: -len("_score")]
                for k in info.keys()
                if k.endswith("_score")
            )
    return sorted(seen)


def warn_unknown_plugin_keys(state, plugin_ids):
    """Self-check: if ``state.model_info`` carries ``{pid}_score`` for
    plugin ids NOT in ``state.active_plugins`` (or the legacy fallback),
    warn to stderr so the operator knows those entries are descendants of
    a stale state file and won't be matched by any filter.
    """
    seen_pids = set()
    for info in state.get("model_info", {}).values():
        if isinstance(info, dict):
            for k in info.keys():
                if k.endswith("_score"):
                    seen_pids.add(k[: -len("_score")])
    extra = seen_pids - set(plugin_ids)
    if extra:
        print(
            f"warning: state.model_info has _score keys for plugin ids "
            f"NOT in active_plugins: {sorted(extra)}. The skill cannot "
            f"strip their per-plugin state; restore active_plugins or "
            f"use a newer purge-results version.",
            file=sys.stderr,
        )


def normalize(name):
    """Lowercase + drop all non-alphanumeric so CSV-style names fuzzy-match
    state-style names. e.g. ``qwen3-vl:9b-32k`` -> ``qwen3vl9b32k`` matches
    state key ``qwen3-vl_9b-32k``.
    """
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def match_model(query, keys):
    """Best-effort resolution of a user-supplied model name to a state key."""
    if query in keys:
        return query
    q_norm = normalize(query)
    for k in keys:
        if normalize(k) == q_norm:
            return k
    return None


def match_plugin(query, known_pids):
    """Resolve a user-supplied plugin name to a known plugin id (exact or
    fuzzy). ``known_pids`` is the discovered list so this never returns a
    value the runtime could not have written.
    """
    if query in known_pids:
        return query
    for pid in known_pids:
        if normalize(pid) == normalize(query):
            return pid
    return None


def load_state(path):
    if not os.path.exists(path):
        sys.exit(f"error: {path} not found")
    with open(path) as f:
        return json.load(f)


def find_latest_index(results, model_name):
    """Index of the LAST entry in ``state.results`` whose
    ``model == model_name``, mirroring ``BenchmarkState.latest_results()``.
    """
    idx = None
    for i, r in enumerate(results):
        if r.get("model") == model_name:
            idx = i
    return idx


def identify_targets(state, args, plugin_ids):
    """Walk ``state.model_info`` (canonical resume-mirror) keyed by the
    canonical ``plugin_ids`` list (from ``active_plugins``) and decide
    which `(model, plugin)` pairs match the filter set.

    The earlier version read metrics from ``state.results`` (latest dict
    per model) and discovered plugins via the latest res dict's
    ``{pid}_score`` keys. That version silently MISSED entries when a
    prior partial purge had stripped ``{pid}_score`` from the latest
    res dict while ``state.model_info`` STILL carried a stale
    ``_output_tokens == 1`` (or any other matching field) -- the
    ``{pid}_score``-keyed discovery loop never visited that plugin, so
    the record stayed in the state file even after ``--apply``. Reading
    metrics from ``model_info`` (which ``BenchmarkState.__init__`` and
    ``start_plugin_run`` keep authoritative) and iterating over the
    full ``plugin_ids`` set avoids that whole class of bug.

    Returns a list of ``(model_key, plugin_id, latest_idx, reason)``
    tuples. ``latest_idx`` may be ``None`` if the model was never
    observed in ``state.results`` (e.g. an orphan ``model_info`` entry
    from a partial run); ``remove_pair`` handles that case by stripping
    only the model_info mirror.
    """
    results = state.get("results", [])
    model_info = state.get("model_info", {})
    state_models = list(model_info.keys())

    latest_by_model = {}
    for i, r in enumerate(results):
        latest_by_model[r.get("model")] = i

    criterion_set = (
        args.tok is not None
        or args.score is not None
        or args.fail
        or args.target
        or args.model
        or args.plugin
    )
    if not criterion_set:
        return []

    targets = []
    for model_key in state_models:
        l_idx = latest_by_model.get(model_key)
        info = model_info.get(model_key, {})
        # Iterate over the canonical ``plugin_ids`` (from
        # ``active_plugins``), NOT ``plugin_ids_in_res``. A model whose
        # latest res dict has had its ``{pid}_score`` stripped is still
        # a candidate if ``state.model_info`` carries a matching field.
        for pid in plugin_ids:
            # Apply --target / --model / --plugin filters first.
            if args.target:
                tm, _, tp = args.target.partition(":")
                if match_model(tm, state_models) != model_key:
                    continue
                tp_match = match_plugin(tp, plugin_ids)
                if tp_match != pid:
                    continue
            if args.model:
                mm = match_model(args.model, state_models)
                if mm != model_key:
                    continue
            if args.plugin:
                pm = match_plugin(args.plugin, plugin_ids)
                if pm != pid:
                    continue

            # Read metrics from ``state.model_info`` (canonical
            # resume-mirror) rather than the latest res dict so a
            # post-purge stale value still gates a re-run.
            score_val = info.get(f"{pid}_score")
            tok_val = info.get(f"{pid}_output_tokens")

            reasons = []
            matched = False

            if args.target or args.model or args.plugin:
                matched = True
                reasons.append("explicit")
            if args.fail and score_val == "fail":
                matched = True
                reasons.append("score=fail")
            if args.score is not None and str(score_val) == str(args.score):
                matched = True
                reasons.append(f"score={score_val}")
            if args.tok is not None and str(tok_val) == str(args.tok):
                matched = True
                reasons.append(f"tok={tok_val}")

            if matched:
                targets.append((model_key, pid, l_idx, "+".join(reasons)))
    return targets


def remove_pair(state, model_key, plugin_id, latest_idx):
    """Surgically remove per-plugin keys for one `(model, plugin)` pair."""
    # 1. Latest result dict in state.results (score/timing keys only --
    #    transient fields don't live here). Guard against orphan
    #    model_info entries that never produced a ``state.results`` row
    #    -- ``identify_targets`` returns ``latest_idx=None`` for those
    #    so ``state[\"results\"][None]`` would otherwise IndexError.
    if latest_idx is not None:
        res = state["results"][latest_idx]
        for suffix in SCORE_SUFFIXES:
            res.pop(f"{plugin_id}_{suffix}", None)

    # 2. state.model_info[model_key] mirror (BOTH score/timing and
    #    transient keys so the next dispatch starts clean).
    info = state.get("model_info", {}).get(model_key)
    if info is not None:
        for suffix in SCORE_SUFFIXES:
            info.pop(f"{plugin_id}_{suffix}", None)
        for suffix in TRANSIENT_SUFFIXES:
            info.pop(f"{plugin_id}_{suffix}", None)
        # ``info["rubric"]`` is a per-plugin dict (vs ``res[*_rubric]``
        # which is a list). Strip the matching entry if present.
        rubric = info.get("rubric")
        if isinstance(rubric, dict) and plugin_id in rubric:
            del rubric[plugin_id]


def count_pair_diff(before_dict, after_dict, pid):
    """Number of per-plugin keys for ``pid`` present in ``before_dict``
    but absent from ``after_dict``. Used by ``--diff`` to report the
    full removal footprint, not just the ``_score`` key.

    Counts every key matching the ``{pid}_*`` prefix (excluding the
    rubric-list / rubric-dict combo which is a single field, not a
    sibling of the score/timing keys). The dict-entry removal of
    ``info["rubric"][pid]`` in ``remove_pair`` is INVISIBLE to this
    counter because the parent ``"rubric"`` key remains in the dict
    -- so it never double-counts.
    """
    removed = 0
    for k in list(before_dict.keys()):
        if not k.startswith(f"{pid}_"):
            continue
        if k not in after_dict:
            removed += 1
    return removed


def print_table(targets, quiet=False):
    if not targets:
        if not quiet:
            print("no matching (model, plugin) pairs found.")
        return
    if quiet:
        print(f"{len(targets)} (model, plugin) pairs matched")
        return
    print(f"{len(targets)} (model, plugin) pairs matched:")
    print()
    print(f"  {'Model':<42}  {'Plugin':<22}  Reason")
    print(f"  {'-'*42}  {'-'*22}  {'-'*30}")
    for m, p, _, reason in targets:
        print(f"  {m:<42}  {p:<22}  {reason}")


def run_diff(state_path, backup_path, quiet=False):
    """Show all per-plugin keys removed between a backup and the current
    state, not just ``_score``. Reports separately for the latest-dict
    in state.results (8 timing keys) and state.model_info (12 keys).
    """
    bak = load_state(backup_path)
    cur = load_state(state_path)
    bak_mi = bak.get("model_info", {})
    cur_mi = cur.get("model_info", {})
    bak_results = bak.get("results", [])
    cur_results = cur.get("results", [])

    cur_latest = {}
    for i, r in enumerate(cur_results):
        cur_latest[r.get("model")] = (i, r)
    bak_latest = {}
    for i, r in enumerate(bak_results):
        bak_latest[r.get("model")] = (i, r)

    pairs_lost_results = []  # (model, pid, count)
    pairs_lost_model_info = []  # (model, pid, count)
    for model, (b_idx, b_res) in bak_latest.items():
        c = cur_latest.get(model, (None, None))[1]
        if c is None:
            continue
        # All plugin ids that had scores in the backup.
        pids = {
            k[: -len("_score")]
            for k in b_res.keys()
            if k.endswith("_score")
        }
        for pid in pids:
            n = count_pair_diff(b_res, c, pid)
            if n:
                pairs_lost_results.append((model, pid, n))

    for model, b_info in bak_mi.items():
        c_info = cur_mi.get(model)
        if c_info is None:
            continue
        pids = {
            k[: -len("_score")]
            for k in b_info.keys()
            if k.endswith("_score")
        }
        # Also include pids that ONLY had transient keys in the backup.
        for k in b_info.keys():
            for suf in TRANSIENT_SUFFIXES:
                if k.endswith(f"_{suf}") and not k.endswith("_score"):
                    pids.add(k[: -len(f"_{suf}")])
                    break
        for pid in pids:
            n = count_pair_diff(b_info, c_info, pid)
            if n:
                pairs_lost_model_info.append((model, pid, n))

    print(
        f"diff {os.path.basename(backup_path)} -> "
        f"{os.path.basename(state_path)}"
    )
    print(
        f"  state.results (latest dict per model): "
        f"{sum(n for _, _, n in pairs_lost_results)} keys removed across "
        f"{len(pairs_lost_results)} pairs"
    )
    print(
        f"  state.model_info:                      "
        f"{sum(n for _, _, n in pairs_lost_model_info)} keys removed "
        f"across {len(pairs_lost_model_info)} pairs"
    )
    if not (pairs_lost_results or pairs_lost_model_info):
        print("  (no per-plugin keys differ)")
        return
    # Honour --quiet: the aggregate summary above stays (it's the headline
    # numbers the operator wants piped into logs), but the per-pair table
    # is suppressed when the operator explicitly asked for terse output.
    if quiet:
        return
    print()
    print(
        f"  {'Model':<42}  {'Plugin':<22}  "
        f"{'results':>8}  {'model_info':>10}"
    )
    print(f"  {'-'*42}  {'-'*22}  {'-'*8}  {'-'*10}")
    by_pair = {}
    for m, p, n in pairs_lost_results:
        by_pair.setdefault((m, p), [0, 0])[0] = n
    for m, p, n in pairs_lost_model_info:
        by_pair.setdefault((m, p), [0, 0])[1] = n
    for (m, p), (r_n, mi_n) in sorted(by_pair.items()):
        print(f"  {m:<42}  {p:<22}  {r_n:>8}  {mi_n:>10}")


def sanity_check_filters(args, plugin_ids, state_models):
    """Warn (not error) if a user-supplied model or plugin can't be
    resolved. Error-quiet here because the user may have meant a fuzzy
    match that we'll catch anyway via ``identify_targets`` returning
    empty, but a warning lets them know their typo is on the wrong side.
    """
    if args.model is not None:
        if match_model(args.model, state_models) is None:
            print(
                f"warning: --model '{args.model}' does not match any "
                f"state key (check spelling / CSV-style names).",
                file=sys.stderr,
            )
    if args.plugin is not None:
        if match_plugin(args.plugin, plugin_ids) is None:
            print(
                f"warning: --plugin '{args.plugin}' is not in "
                f"state.active_plugins ({plugin_ids!r}).",
                file=sys.stderr,
            )
    if args.target is not None:
        tm, _, tp = args.target.partition(":")
        if not tm or not tp:
            print(
                f"warning: --target '{args.target}' should be "
                f"'model:plugin' (got empty halves).",
                file=sys.stderr,
            )
        else:
            if match_model(tm, state_models) is None:
                print(
                    f"warning: --target model half '{tm}' does not "
                    f"match any state key.",
                    file=sys.stderr,
                )
            if match_plugin(tp, plugin_ids) is None:
                print(
                    f"warning: --target plugin half '{tp}' is not in "
                    f"state.active_plugins ({plugin_ids!r}).",
                    file=sys.stderr,
                )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "purge-results: remove (model, plugin) entries from "
            "benchmark_state.json so they re-run on resume."
        ),
    )
    parser.add_argument("--output-dir", default=".", help="run directory")
    parser.add_argument("--tok", type=int, help="filter by output_tokens == N")
    parser.add_argument(
        "--score",
        help="filter by score (string match, e.g. '0' or 'fail')",
    )
    parser.add_argument(
        "--fail", action="store_true", help="filter by score == 'fail'"
    )
    parser.add_argument("--model", help="filter to a single model (fuzzy)")
    parser.add_argument(
        "--plugin", help="filter to a single plugin (must match active_plugins)"
    )
    parser.add_argument(
        "--target", help="specific 'model:plugin' pair (fuzzy)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually mutate (default is dry-run)",
    )
    parser.add_argument(
        "--diff",
        metavar="BACKUP_PATH",
        help=(
            "show keys removed between BACKUP and current state "
            "(no mutation)"
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help=(
            "suppress per-pair table, dry-run/apply banners, and per-pair "
            "diagnostic prints; only aggregate counts and the diff "
            "headline are emitted"
        ),
    )
    args = parser.parse_args()

    state_path = os.path.join(args.output_dir, "benchmark_state.json")

    if args.diff:
        run_diff(state_path, args.diff, quiet=args.quiet)
        return

    state = load_state(state_path)
    plugin_ids = discover_plugin_ids(state)
    state_models = list(state.get("model_info", {}).keys())
    warn_unknown_plugin_keys(state, plugin_ids)
    sanity_check_filters(args, plugin_ids, state_models)

    targets = identify_targets(state, args, plugin_ids)
    print_table(targets, quiet=args.quiet)

    if not args.apply:
        if not args.quiet:
            print()
            print(
                "DRY RUN: state was NOT modified. "
                "Re-run with --apply to commit."
            )
        return

    if not targets:
        if not args.quiet:
            print("nothing to do.")
        return

    # Backup first. Millisecond suffix so two purges within the same
    # second produce distinct filenames.
    # Microsecond resolution -- two purges within the same microsecond
    # are vanishingly unlikely (Python's time.time() rarely has better
    # than µs granularity anyway) and would still be avoidable with
    # a counter if you ever script a burst.
    ts = f"{time.time():.6f}"
    backup_path = f"{state_path}.pre-purge-{ts}.bak"
    shutil.copy2(state_path, backup_path)
    if not args.quiet:
        print(f"\nbackup written: {backup_path}")

    for model_key, plugin_id, latest_idx, _ in targets:
        remove_pair(state, model_key, plugin_id, latest_idx)

    # Write back. Re-load to confirm parseable.
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, default=str)
    with open(state_path) as f:
        json.load(f)

    if not args.quiet:
        print(
            f"applied {len(targets)} purges to {state_path}.\n"
            f"  backup: {backup_path}\n"
            f"  next step: re-run `ai-benchmark.py` to "
            f"repopulate the freed slots."
        )


if __name__ == "__main__":
    main()
```

## Resume semantics — what changes after the purge

`benchmark_core.py`'s `run_model` resume path consults the latest result
dict per model (deduped by `BenchmarkState.latest_results()`) and re-uses a
plugin's score only when `_score != "fail"`. Removing the `{pid}_score`
key for a pair makes the resume path re-execute that plugin while keeping
every other plugin's score intact.

## Status verification after the purge

The script does NOT alter the per-model `status` field. After `apply`,
verify the projected re-run behaviour with a quick check:

```bash
python3 -c "
import json
s = json.load(open('benchmark_state.json'))
for m, info in s['model_info'].items():
    missing = [p for p in s['active_plugins']
               if f'{p}_score' not in info]
    if missing:
        print(f'{m:42s}  will re-run: {missing}')
"
```

Models that show no missing-plugin list are stable (no re-run). Models
with every plugin missing are functionally a fresh benchmark for that
target. Models with a partial missing list will re-run only the listed
plugins and re-use the rest.

## Verification checklist (after `--apply`)

1. `python3 -c "import json; json.load(open('benchmark_state.json'))"` —
   confirms the file still parses.
2. `python3 /tmp/purge-results.py --output-dir <RUN_DIR> --diff <backup>`
   — shows every key that was removed (counts + per-pair sample).
3. Re-run `python3 ai-benchmark.py --output-dir <RUN_DIR>` (no extra
   flags) — the same affected plugins will re-execute.

## Restoration (undo)

```bash
cp benchmark_state.json.pre-purge-<ms-ts>.bak benchmark_state.json
```

The backup contains the entire pre-mutation `state.results` and
`state.model_info`, so this is a full rollback. Response `.txt` and
`.meta.json` files are never touched by the skill and need no restoration.

## Edge cases handled

- **CSV vs state name normalisation**: `--model` and `--target` use
  `normalize()` so `qwen3-vl:9b-32k`, `qwen3-vl_9b-32k`, `qwen3vl9b32k`,
  etc. all resolve to the same state key.
- **Multiple results per model**: the script uses the LAST occurrence in
  `state.results` (matching `BenchmarkState.latest_results()`).
- **Pair absent from latest dict**: skipped silently (no score -> already
  queued for re-run; wouldn't have matched a `--score N` filter anyway).
- **Backups timestamp**: `<unix_ts>.<ms>` lets two purges within the same
  second accumulate distinct backups instead of clobbering one.
- **Filter AND semantics**: `--model X --plugin Y` purges only the
  intersection, not the union.
- **Schema drift**: a warning is printed to stderr if `_score` keys exist
  in `model_info` for plugin ids not in `active_plugins` (legacy or stale
  state). The migration is operator-driven; the script doesn't mutate
  `active_plugins`.
- **JSON re-formatting**: `json.dump(data, f, indent=2, default=str)` may
  re-order keys compared to a hand-pruned file. The downstream
  `BenchmarkState.load_state` only reads values so this is observationally
  equivalent, but a `git diff benchmark_state.json` after a purge will
  show formatting changes that are not semantically meaningful.
