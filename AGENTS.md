# AGENTS.md — project context for fresh agent sessions

Multi-source, plugin-based LLM benchmark over OpenAI-compatible APIs. Long-form
reference lives in `docs/` (links at the bottom). Read this file in under 60 s
to know what to grep first.

## Project map

- **`ai-benchmark.py`** — CLI, argparse, TUI, main orchestrator (the entry point).
- **`benchmark_core.py`** — `_run_plugin_task`, `run_model`, `_run_plugins`, plugin dispatch + per-token bookkeeping.
- **`benchmark_http.py`** — `stream_request`, `nonstream_request`, `_post_request_context`, 429 retry/backoff with jitter & `Retry-After`, `_set_429_sleep / _clear_429_sleep / get_429_stats / reset_429_stats`, `build_curl_cmd`, SSE parser.
- **`benchmark_state.py`** — thread-safe `BenchmarkState`: `save_state`, `load_state`, `add_result`, `start_plugin_run`, `finish_plugin_run`, `mark_first_chunk_seen`, `add_bytes_received`, resume semantics.
- **`benchmark_plugin.py`** — abstract base classes (`BenchmarkTaskPlugin`, `BenchmarkOutputPlugin`).
- **`benchmark_outputs.py`** — `gen_markdown / gen_csv / gen_html / gen_pdf`.
- **`shell_completion.py`** — `--generate-shell-completion`.
- **`plugins/challenges/`** — 10 task plugins; auto-discovered. **`plugins/outputs/`** — md/csv/html/pdf output plugins.
- **`tests/`** — unittest suite. **`requirements.txt`** — runtime deps.

## Smoke / test commands (no API key needed)

```sh
python3 ai-benchmark.py --list-plugins              # every discovered plugin
python3 ai-benchmark.py --dump-default-config       # YAML config template
python3 ai-benchmark.py --convert-config in.yml     # convert JSON<->YAML (stdout)
python3 -m pytest tests/ plugins/challenges/ plugins/outputs/ -q
python3 -m mypy benchmark_core.py benchmark_http.py benchmark_state.py benchmark_plugin.py ai-benchmark.py
python3 -m py_compile ai-benchmark.py benchmark_core.py benchmark_http.py benchmark_state.py
```

## Key CLI flags (full reference in `docs/cli.md`)

- `--retry-on-429` / `--no-retry-on-429` — toggle HTTP 429 backoff (default
  ON; per-source `max_429_retries` overrides are preserved either way).
  See `benchmark_http._post_request_context` for the retry math.
- `--no-rerun-failed` — keep `failed` models on resume; default re-runs them.
- `--restart` — discard prior state and start every model fresh.
- `--save-responses` — write per-(model, plugin) response files plus
  `meta.json` with rubric breakdown + `error`/`traceback` on crash.
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
| `_bytes_received`, `_first_chunk_seen`, `_first_tok_ts`, `_start_ts` | runtime | Live TUI; reset on `start_plugin_run` |

`state.results` is the list of result dicts; `latest_results()` keeps the
LAST entry per model.

## Output directory (per run)

- `results.md`, `results.csv`, `results.html`, `results.pdf` — final reports.
- `benchmark_state.json` — resume state.
- `run-info.json` — run metadata; **includes `backoff_429`
  (`{total_retries, per_plugin: {pid: {retries, total_sleep_time}}}`)**
  for post-run 429 analysis (see `_inject_429_stats` in `ai-benchmark.py`).
- `benchmark-config.{yaml,json}` — **copy of the input config persisted
  as a time capsule**; diff against the in-repo config to see what the
  operator actually ran.
- `logs/<model>.log` — curl + response bodies (when `--save-responses`).
- `responses/<model>/<plugin>.txt` — model output; `.prompt.txt` is the
  prompt; `.meta.json` includes the rubric breakdown and `error`/`traceback`
  if `plugin.evaluate()` crashed.

## Built-in plugins (10)

`code-review`, `moe-dense`, `multi-step`, `orchestration`, `prd-creation`,
`rate-limiter`, `software-architecture`, `structured-output`, `tool-calling`,
`wireframes`. Three (`code-review`, `moe-dense`, `structured-output`) set
`supports_streaming=False` — they go through the **non-streaming**
`_post_request_context` path.

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
   `orchestration=16`, `tool-calling=25`, others=20. Tests in
   `tests/test_tui_cells.py` pin score normalisation.

## Freebuff skills

`.agents/skills/` ships reusable, self-contained instructions. Load by name
when the user invokes a skill workflow (`skill <name>`).

- **`purge-results`** — surgically removes `(model, plugin)` entries from
  `benchmark_state.json` so they re-run on the next `ai-benchmark.py`
  resume. Self-discovers plugin ids from `state.active_plugins`, default
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
- `docs/scoring-checks.md` — rubric philosophy + future richer checks (regex,
  structural parsing, Mermaid validation, LLM-as-judge, code-execution).
