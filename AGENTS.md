# AGENTS.md — project context for fresh agent sessions

Multi-source, plugin-based LLM benchmark over OpenAI-compatible APIs. Long-form
reference lives in `docs/` (links at the bottom). Read this file in under 60 s
to know what to grep first.

## Project map

- **`benchmark/cli.py`** — CLI, argparse, TUI, main orchestrator (the entry point).
  Importable as a module and exposed as the installed `ai-benchmark` console
  script via `[project.scripts]` in `pyproject.toml`.
- **`ai-benchmark.py`** — thin root launcher that delegates to `benchmark.cli.main`;
  kept so `python ai-benchmark.py` works for docs, hooks, and tests.
- **`benchmark/`** — the core library package:
  - **`benchmark/core.py`** — `_run_plugin_task`, `run_model`, `_run_plugins`, plugin dispatch + per-token bookkeeping.
  - **`benchmark/http.py`** — `stream_request`, `nonstream_request`, `_post_request_context`, 429 retry/backoff with jitter & `Retry-After`, `_set_429_sleep / _clear_429_sleep / get_429_stats / reset_429_stats`, `build_curl_cmd`, SSE parser.
  - **`benchmark/state.py`** — thread-safe `BenchmarkState`: `save_state`, `load_state`, `add_result`, `start_plugin_run`, `finish_plugin_run`, `mark_first_chunk_seen`, `add_bytes_received`, resume semantics.
  - **`benchmark/plugin.py`** — abstract base classes (`BenchmarkTaskPlugin`, `BenchmarkOutputPlugin`).
  - **`benchmark/outputs.py`** — `gen_markdown / gen_csv / gen_html / gen_pdf`.
  - **`benchmark/completions.py`** — `--generate-shell-completion`.
  - **`benchmark/opencode.py`** — the optional OpenCode subprocess runner.
  - **`benchmark/chatplayground.py`** — subprocess-proxy adapter for ChatPlayground.ai sources (`api_protocol: "chatplayground"`, username/password login, buffered answers, UI model enumeration). **`benchmark/chatplayground_worker.py`** — the Playwright browser worker it spawns; the runner never imports Playwright (its sync API is not thread-safe, and running it in a model worker thread segfaulted the run).
- **`plugins/challenges/`** — 18 task plugins; auto-discovered and metadata-validated. **`plugins/outputs/`** — md/csv/html/pdf output plugins.
- **`scripts/recover_state_from_csv.py`** — explicit, dry-run-by-default recovery of historical `benchmark_state.json` files from `results.csv`; accepts known historical plugin subsets and rejects unknown score columns.
- **`tests/`** — pytest/unittest suite. **`pyproject.toml`** — project metadata, runtime deps, `dev` dependency group, pytest/coverage/mypy/ruff config. **`uv.lock`** — pinned dependency tree (managed by `uv`).

## Smoke / test commands (no API key needed)

The project is managed with `uv` (`uv.lock` pins the dependency tree).
`uv sync` installs the package editable plus the `dev` dependency group; all
project commands run through `uv run`.

```sh
uv run ai-benchmark --list-plugins          # installed console script
python3 ai-benchmark.py --list-plugins      # every discovered plugin (repo launcher)
python3 ai-benchmark.py --dump-default-config      # JSON config template
python3 ai-benchmark.py --convert-config in.yml    # convert JSON<->YAML (stdout)
uv run pytest tests/ plugins/challenges/ plugins/outputs/ -q
uv run mypy benchmark/ ai-benchmark.py
uv run python -m py_compile ai-benchmark.py benchmark/*.py
```

## Key CLI flags (full reference in `docs/cli.md`)

- `--retry-on-429` / `--no-retry-on-429` — toggle HTTP 429 backoff (default
  ON; per-source `max_429_retries` overrides are preserved either way). After
  two consecutive plugin tests exhaust their 429 retries for one model in a
  run, the remaining tests for that model are cancelled. A successful or
  non-429 test resets the consecutive count. See
  `benchmark.http._post_request_context` for the retry math.
- `--no-rerun-failed` — keep `failed` models on resume; default re-runs them.
- `--scripted` — continue non-interactively instead of prompting on resume/plugin changes.
- `--restart` — discard prior state and start every model fresh.
- `--no-preload` — override per-source `preload: true` and skip model warm-up
  probes for the run; `preload_timeout` defaults to 300 seconds per source.
- `--save-responses` — write per-(model, plugin) response files plus
  `meta.json` with rubric breakdown + `error`/`traceback` on crash.
- `--judge-models MODEL [...]` — run confidence-weighted semantic judging;
  `--build-judge-queue STATE_FILE` ranks disagreements for manual review.
- `--plugin-temperature ID=VAL` — overrides the matching `_<id>_temperature`
  config key; per-plugin wins over the global `--temperature`.

## State schema (per model in `benchmark_state.json`)

`BenchmarkState._model_info[model]` has core fields (`source`, `api_model`,
`status`, `attempt`, `elapsed`, `last_error`, `running_pids`, `ttft`, …) and
per-plugin suffixed keys (the first three change on resume; the rest reset
per dispatch):

| Suffix | Source | Meaning |
|---|---|---|
| `_score` | resume | Float or `"fail"` — **re-run when absent or `"fail"`**, reuse otherwise (the gate `run_model` checks via `latest_results()`) |
| `_tps`, `_response_time`, `_output_tokens` | both | Last completed run metrics |
| `_stream_ok`, `_truncated`, `_repeating`, `_rubric` | both | Streaming + scoring flags |
| `_empty_reason` | both | Empty-response classification (`None`/`error`/`thinking-truncation`/`thinking-only`/`max-tokens`/`empty`); surfaced in meta.json + CSV |
| `_judge_votes`, `_judge_score`, `_judge_complete`, `_judge_error` | judge | Valid semantic-judge attempts, consensus, completion, and failure state |
| `_bytes_received`, `_first_chunk_seen`, `_first_tok_ts`, `_start_ts` | runtime | Live TUI; reset on `start_plugin_run` |

`state.results` is the list of result dicts; `latest_results()` keeps the
LAST entry per model.

## Output directory (per run)

- `results.md`, `results.csv`, `results.html`, `results.pdf` — final reports.
- `benchmark_state.json` — resume state.
- `run-info.json` — run metadata; **includes `backoff_429`
  (`{total_retries, per_plugin: {pid: {retries, total_sleep_time}}}`)**
  for post-run 429 analysis (see `_inject_429_stats` in `benchmark.cli`).
- `benchmark-config.{yaml,json}` — **copy of the input config persisted
  as a time capsule**; diff against the in-repo config to see what the
  operator actually ran.
- `logs/<model>.log` — curl + response bodies (when `--save-responses`).
- `responses/<model>/<plugin>.txt` — model output; `.prompt.txt` is the
  prompt; `.think.txt` and `<plugin>.meta.json` preserve reasoning and rubric
  diagnostics. Failed evaluations include `error`/`traceback` when
  `plugin.evaluate()` crashes.
- `judge-inputs/` — retained prompt/response sidecars used for resumable
  semantic judging; raw judge responses are stored beside benchmark artifacts.

## Built-in plugins (18)

The runtime inventory below matches `uv run ai-benchmark --list-plugins`:

| ID | Version | Max | Stream |
|---|---:|---:|---|
| `code-review` | 1.0.0 | 15 | No |
| `debug-consistency` | 0.1.0 | 20 | Yes |
| `debug-traversal` | 1.0.0 | 20 | Yes |
| `error-recovery` | 1.1.0 | 20 | Yes |
| `event-processor` | 0.1.0 | 20 | Yes |
| `instruction-following` | 1.0.0 | 20 | Yes |
| `long-context` | 0.1.0 | 20 | Yes |
| `moe-dense` | 1.0.1 | 17 | Yes |
| `multi-step` | 1.0.0 | 20 | Yes |
| `multi-turn-conversation` | 1.0.0 | 20 | Yes |
| `orchestration` | 1.0.0 | 16 | Yes |
| `prd-creation` | 1.0.0 | 22 | Yes |
| `rate-limiter` | 1.0.0 | 20 | Yes |
| `reasoning` | 1.0.0 | 20 | Yes |
| `software-architecture` | 1.0.0 | 20 | Yes |
| `structured-output` | 1.0.0 | 22 | No |
| `tool-calling` | 1.0.0 | 25 | Yes |
| `wireframes` | 1.0.0 | 20 | Yes |

`code-review` and `structured-output` set
`supports_streaming=False` and use the **non-streaming** request path. `moe-dense`
uses the streaming path. Code-shaped plugins
use a pytest-compatible assertion harness in the Podman sandbox, with a
resource-limited local fallback recorded as `local-restricted` when Podman is
unavailable.

## Git management
Before committing any complete change:
1. Run the full CI checks defined in `.github/workflows/tests.yml` (or their local equivalent): `uv sync --frozen`; `uv run pre-commit run --all-files --show-diff-on-failure`; `uv run coverage run -m pytest tests/ plugins/challenges/ plugins/outputs/ -q`; `uv run coverage report -m`; and `uv run coverage report --fail-under=90`. The CI matrix runs these checks on Python 3.10 through 3.14. Fix every issue reported by these checks before committing, then rerun the checks until they pass.
2. Update all relevant documentation to reflect the new reality, including `AGENTS.md`, `README.md`, `docs/`, plugin documentation, CLI/configuration references, and any other checked-in documentation affected by the change.
3. Confirm the documentation and runtime metadata agree, then commit the complete change to git.
4. When adding a footer or co-author attribution, include the agent name (e.g. OpenCode, FreeBuff, CodeBuff, Hermes Agent, etc.) and model name (e.g. GPT-5.6 Luna, Big Pickle, DeepSeek v4 Flash 0731, etc.)

## Plugin updates
1. **Update challenge-plugin versions when modified from what's in git.** This policy applies only to challenge plugins and their shared challenge-plugin code under `plugins/challenges/`; it does not apply to `plugins/__init__.py`, output plugins, documentation, or tests. Every internal or externally visible challenge-plugin code change requires a version bump. Non-scoring changes, such as behavior-preserving refactors or internal/API plumbing that cannot affect evaluation results, increment the revision (for example, `0.2.0` → `0.2.1`). A minor-version bump is reserved for changes that could affect scoring in any way, including prompt, rubric, scoring-code, validation, normalization, or execution changes (for example, `0.2.0` → `0.3.0`, resetting the revision). Complete rewrites or very major changes increment the major version (for example, `0.2.0` → `1.0.0`, resetting minor and revision). A bump is not required for every intermediate edit; record the largest applicable change from the version currently in git.

## Dependency policy

Prefer adding a well-maintained dependency over rolling your own, especially
for complex problems and for areas where a hand-rolled implementation would
only approximate the solution. The TUI's terminal-width math is the canonical
example: the hand-rolled East Asian Width table approximated `wcwidth` and
mis-counted keycap/VS16 emoji; swapping to the `wcwidth` package fixed the
gaps and removed a fragile, hand-maintained Unicode table.

When considering a dependency:

- Verify it is actively maintained, has a compatible license, and fits the
  problem (e.g. the small, pure-Python `wcwidth` over a heavier alternative).
- Prefer small, focused libraries over pulling in a large framework for one
  function.
- Keep roll-your-own code only where a dependency genuinely doesn't cover the
  need — e.g. the TUI's grapheme clustering for truncation/slicing, which
  `wcwidth` doesn't provide — and where that code is already correct and
  tested.

## Known gotchas (each is a recent fix; the regression test lives in `tests/`)

1. **`on_retry` closure-scope block.** `def on_retry():` was once nested inside
   `if plugin.supports_streaming:` in `_run_plugin_task`. Python's static
   scope made it a function-local; `nonstream_request(on_retry=on_retry)` on
   the `else` branch raised `UnboundLocalError` on every supports_streaming=
   False plugin. **Fix:** hoist the `def` above the `if` block (and above
   the `for attempt, max_tok in enumerate(token_levels):` loop). Regression
   pinned in `tests/test_cli.py::TestNonStreamingPluginRetry`.
2. **Resume semantics.** `run_model` reads the LATEST result dict per model
   via `BenchmarkState.latest_results()` (i.e. `state.results`); it re-uses a
   plugin's score iff `{pid}_score != "fail"` and the key is present. To
   trigger a per-plugin re-run without restarting the model, **remove** that
   plugin's `{pid}_score` (and timing siblings) from the latest entry in
   `state.results`. Mirroring the deletion in `state.model_info[model]` is
   belt-and-suspenders for the live TUI / `meta.json`. Model keeps
   `status="completed"`. The `purge-results` skill
   (`.agents/skills/purge-results/SKILL.md`) automates this. Never delete
   `responses/*.txt` — regenerated on next run.
3. **Output_Tokens == 0 means EMPTY response.** `count_tokens(text) = max(0, len(text)/4)`
   returns 0 when `text` is empty. Zero-token entries in `results.csv`
   are uniformly 0-byte `.txt` files — model timed out or connection dropped
   silent (no streaming heartbeat reached).
4. **HTTP 429 cleanup-before-sleep invariant.** `_post_request_context`
   cancels the watchdog timer, drops from `_active_requests`, and closes
   `resp` before the interruptible `stop_event.wait(delay)`. Don't reorder —
   Ctrl+C would leak.
5. **Per-plugin max_score varies** — `code-review=15`, `moe-dense=17`,
   `orchestration=16`, `prd-creation=22`, `structured-output=22`,
   `tool-calling=25`, and the remaining plugins use 20. Cross-plugin
   contract tests pin the inventory and native scales; public results are
   normalized once to percentage-v1.
6. **Thinking-truncation auto-escalation.** `_run_plugin_task` in
   `benchmark/core.py` auto-retries once with a doubled `max_tokens` budget
   when a streaming HTTP leg classifies as `thinking-truncation` (empty
   content, large `reasoning_content`, `finish_reason="length"`). The retry
   is capped at 131072 and only fires for HTTP streaming plugins. Regression
   in `tests/test_cli.py::TestThinkingAutoEscalation`.
7. **`faulthandler` is enabled at CLI startup** (`_enable_faulthandler` in
   `benchmark/cli.py`, called first thing in `main()`) and in the
   ChatPlayground worker subprocess. A native crash (SIGSEGV/SIGABRT/…) now
   prints the Python stack to stderr instead of a bare "Segmentation fault",
   and `kill -USR1 <pid>` forces a live stack dump of a wedged run. The
   worker's dump lands in the parent's captured stderr and is surfaced in the
   per-request error. Tests in `tests/test_cli_coverage.py::TestEnableFaulthandler`.
8. **ChatPlayground request timeout must reach the worker.** `_send_request`
   in `benchmark/chatplayground.py` binds `timeout` to a named parameter (for
   the parent's own wait deadline) — it must also be copied into the JSON
   message (`msg["timeout"] = timeout`) or the worker runs with `timeout=0`,
   declares completion instantly, and reads the answer before generation
   finishes (fast empty `(empty response)` legs, score 0, no error).
   Regression in `tests/test_chatplayground.py::TestWorkerSubprocess::test_request_forwards_timeout_to_worker`.

## Freebuff skills

`.agents/skills/` ships reusable, self-contained instructions. Load by name
when the user invokes a skill workflow (`skill <name>`).

- **`purge-results`** — surgically removes `(model, plugin)` entries from
  `benchmark_state.json` so they re-run on the next `ai-benchmark.py`
  run (or `ai-benchmark` console script) resume. Strips score/timing,
  per-plugin judge (`{pid}_judge_*`), residual (`empty_reason`,
  `diagnostics`), and transient keys. Self-discovers plugin
  ids from `state.active_plugins`, default
  dry-run, microsecond backup `benchmark_state.json.pre-purge-<ts>.bak`,
  `--quiet` gates per-pair table + main() banners, `--diff` reads a
  backup to show what was removed.

## Authoritative docs (read on demand)

- `README.md` — quickstart, config, CLI cheatsheet.
- `docs/architecture.md` — execution flow, concurrency model, resume behaviour.
- `docs/cli.md` — full CLI argument reference.
- `docs/configuration.md` — YAML/JSON config, per-source 429 retry knobs.
- `docs/plugins.md` — plugin catalog + lifecycle.
- `docs/development.md` — dev setup + writing plugins + test conventions.
- `docs/scoring-checks.md` — implemented rubric philosophy, section-local and
  typed parsing, executable sandbox checks, adversarial coverage, and the
  remaining deliberately limited qualitative checks.
