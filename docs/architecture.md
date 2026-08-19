# Architecture

This document describes the high-level design of AI Benchmark.

## Components

```
┌─────────────────────────────────────────┐
│         benchmark/cli.py              │
│  CLI parsing, config loading, TUI     │
│  (installed as the `ai-benchmark`     │
│   command; ai-benchmark.py is a       │
│   thin launcher)                      │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│           benchmark/core.py           │
│  State management, API requests,        │
│  plugin execution, output generation   │
└─────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌──────────────┐   ┌─────────────────┐
│   plugins/   │   │  output formats │
│   tasks      │   │  md/csv/html/pdf│
└──────────────┘   └─────────────────┘
```

## Execution Flow

1. **CLI Initialization**
   - Parse arguments with `argparse`.
   - Load and expand config (`load_config`).
   - Discover active plugins (`discover_plugins`).

2. **State Setup**
   - Create or resume `BenchmarkState`.
   - Build per-source model queues.

3. **Runner Scheduling and Worker Threads**
   - `--runner http` uses the existing direct OpenAI-compatible path.
   - `--runner opencode` generates one retained OpenCode config and starts one isolated `opencode run` process per target/plugin.
   - `--runner both` creates one worker per source. That worker runs each target as OpenCode then HTTP before advancing, so OpenCode and HTTP never overlap on the same source; sources still run concurrently with one another.
   - Targets whose OpenCode state is already complete on resume are seeded directly into the HTTP queue.
   - Single-runner modes retain one source worker per source and run `run_model()` for each target in its queue.
   - State identities include the configured target and runner, preventing cross-runner resume reuse.

4. **Plugin Execution**
   - `_run_plugins()` uses `ThreadPoolExecutor` with `plugin_thread_limit` workers.
   - Each plugin task calls `_run_plugin_task()`.
   - HTTP tasks call `stream_request()` or `nonstream_request()`.
   - OpenCode tasks call the subprocess adapter with `opencode run --pure --format json --thinking`, capture stdout/stderr separately, extract the final assistant answer and reasoning from the NDJSON event stream, and score the answer as the response.
   - Each HTTP/OpenCode task uses one scalar `max_tokens` budget and at most one benchmark-level retry. Attempt metadata records `response_nature`, `retry_reason`, `prompt_altered`, token breakdown, failure cause, and the selected attempt; timeout and cancellation are terminal and mark time truncation/cancellation instead of retrying. Provider-level 429 backoff and schema fallbacks remain separate diagnostics.

### Retry layers

The benchmark has separate transport, task-attempt, and judge-attempt layers. A retry at an inner layer does not consume a retry at an outer layer unless the outer layer subsequently classifies the resulting error as retryable.

```text
Benchmark task cell

  _run_plugin_task()                         Judge cell
  up to 2 logical attempts                  judge_response()
          │                                  up to 2 attempts for invalid JSON
          │                                          │
          ▼                                          ▼
  stream_request() /                      stream_request()
  nonstream_request()                              │
          │                                        │
          └──────────────┬─────────────────────────┘
                         ▼
                _post_request_context()
                HTTP transport layer
                         │
                 HTTP 429 backoff/retry
                 (max_429_retries per request)
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
   task response classification     judge JSON parsing
   transport/token/repetition/      valid → persist vote
   timeout/cancellation             invalid → judge retry
          │
   transport/token/repetition
   may trigger task retry
   timeout/cancellation do not
```

For benchmark tasks, `_run_plugin_task()` owns one optional outer retry. A transport failure retries with the original prompt; token exhaustion or repetition adds retry guidance; timeout and cancellation are terminal and preserve partial output. Each logical HTTP attempt can itself contain the initial request plus the configured HTTP 429 retries. OpenCode has the same outer task policy, but replaces the HTTP transport with one isolated subprocess and its own timeout, staleness, step, and text-repetition guards.

Judges do not use `_run_plugin_task()`. Their `judge_response()` loop retries only a received response that fails judge-JSON validation, adding thinking-budget guidance when the failed response consumed at least 80% of the judge budget on reasoning. Judge transport errors return a failed vote immediately after any inner HTTP 429 retries. Judge attempts and benchmark attempts therefore have different retry triggers, even though they share the HTTP transport layer.

5. **Scoring**
   - Plugin `score()` evaluates the response.
   - Results are stored in `BenchmarkState`.

6. **Output Generation**
   - Once at the end of the run (or when the app is stopped), `_save_outputs()`
     generates the CSV/HTML/Markdown/PDF reports from the final state.
   - `gen_pdf()` is called separately for PDF output.

## State Management

`BenchmarkState` is a thread-safe in-memory store:

- `_model_info`: per-model status and metrics
- `results`: list of result dicts
- `_log`: recent error log entries

State persistence and report generation are deliberately separate stages:

- **State snapshot persistence** writes `benchmark_state.json`, which is the
  resume source of truth. Completed judge votes and benchmark tasks accumulate
  in memory and are compacted at most every `flush_interval_seconds` seconds or
  `flush_votes` changes (defaults 60 s / 10). `_BackgroundFlusher.request_flush()`
  is non-blocking; requests arriving mid-flush coalesce, and the snapshot is
  serialized under `persistence_lock` on the background thread.
- **Event journaling** appends compact result and judge updates to
  `results.journal.jsonl`. A successful state flush includes the journal events
  in `benchmark_state.json` and removes only the included journal prefix. If a
  process crashes after a snapshot but before compaction, startup replays the
  journal tail whose sequence is newer than the snapshot. A partial final JSONL
  line is ignored; a corrupt state can use the complete journal for recovery.
- **Report generation** is not part of a persistence flush. `_save_outputs()`
  reads the final in-memory state once after workers and persistence have been
  drained, producing CSV/HTML/Markdown/PDF artifacts. Reports may therefore be
  stale during a live run while `benchmark_state.json` and the journal remain
  current.

Shutdown waits up to `flush_shutdown_timeout_seconds` for the background flusher.
If it does not stop, the run records and prominently prints a persistence failure
then attempts a synchronous final state/journal compaction. Any background or
final-save failure is recorded in `run-info.json` under `persistence_failures`;
a completed report must not be interpreted as proof that the state was durable.
The saved state stores model sources as plain strings; dict-valued model entries
from the config are resolved to their source string before being written.

Every state mutation bumps a monotonic `revision` counter. The live TUI polls it and rebuilds its frame only when something displayed actually changed — revision bumped, terminal resized, operator scrolled, or live elapsed/countdown content (streaming seconds, judge-activity elapsed, 429 sleeps) is ticking — capped at 2 fps (`_TUI_REFRESH_SECONDS = 0.5`). An idle run rebuilds nothing, so the ~0.1 s frame build for 108 models × 18 plugins no longer runs on every 0.2 s tick.

Active benchmark cells and judge activities carry a logical attempt number. Their live thinking/content counters reset when the next logical attempt begins, so the TUI never presents tokens accumulated across retries as if they belonged to the current request. Completed benchmark result fields remain selected-attempt metrics; `total_tokens` is retained as the derived thinking-plus-content report value.

## OpenCode Runner

OpenCode is optional and is never required for the default HTTP mode. When selected, startup resolves the binary in priority order: an on-PATH install that passes the capability preflight; a previously auto-installed local copy at `<project root>/.tools/opencode/opencode`; or a fresh download of the latest official release into that same directory (disabled by `--no-install-opencode`). The preflight (`opencode run --help`) requires `--model`/`--format`/`--agent`/`--pure`/`--thinking` and the `json` format choice; an on-PATH install that fails it is replaced by the local copy or a fresh install automatically. The resolved binary path is threaded into every `opencode run` invocation and recorded in `run-info.json` as `opencode_binary`. The runner resolves every configured target to `{slugified source}/{api_model}`, projects source URL/auth/model settings into a generated config, and stores that config under the OpenCode output namespace. Every target registers an agent so `opencode run --agent` always selects explicit context and OpenCode's built-in default agent prompt ("answer concisely with fewer than 4 lines", all tools enabled) never applies: agent personas keep their explicit system prompt, while plain model targets register a **neutral agent** whose prompt has no conciseness instruction and whose permission map denies every tool family — small function-calling-tuned models therefore see no tool definitions at all and get the same plain "answer the prompt" contract as the HTTP runner. The generated config is retained intentionally; operators must protect the output directory because it contains resolved credentials.

Each OpenCode task runs `opencode run --pure --model <mapped> --format json --thinking --agent benchmark-<target> <prompt>` in an isolated subprocess. OpenCode tasks participate in the same one-retry policy as HTTP tasks; when the subprocess exposes a timeout, cancellation, transport failure, or repetition guard failure, the shared attempt classifier prevents timeout/cancellation retries and retains partial output metadata. `--pure` disables external OpenCode plugins so the benchmark environment, tools, prompts, and event stream are reproducible; `--thinking` is required for OpenCode to emit `reasoning` NDJSON events at all (non-interactive `opencode run` otherwise defaults `thinking` to off); `--agent` selects the per-target registered agent (neutral for plain models, the persona's own prompt for agent targets). The `json` format emits one NDJSON event per line; `_extract_final_text()` joins the `text` events (the model's final answer parts), joins `reasoning` events into `think_text`, and surfaces `error` events as failures. The extracted `think_text` flows through the shared save-responses path, so OpenCode runs produce the same `{plugin}.think.txt` and `<thinking>…</thinking>`-wrapped `{plugin}.txt` sidecars as the HTTP runner when `--save-responses` is enabled. Model entries in the generated config always set both `limit.context` and `limit.output` — OpenCode's schema rejects a `limit` object that omits `context`. The context value is inferred from the benchmark model id's `-NNk`/`-NNm` suffix with a conservative fallback, and the output value comes from `max_tokens`.

OpenCode has buffered final-output semantics in this adapter. Direct HTTP streaming metrics such as TTFT are not fabricated for OpenCode; its response time and estimated output-token count are recorded instead. Timeouts and cancellation terminate the subprocess group where supported.

OpenCode's agent loop has no internal liveness detection: once it emits a step it waits for the provider's next response indefinitely, and a stalled or looping task would otherwise burn the full benchmark timeout with zero diagnostics. `run_process()` therefore runs three loop guards against the live NDJSON stream, each aborting the subprocess early with an actionable error (any partial stdout is retained):

- **Staleness fast-fail** (`sources.<name>.opencode_timeout`, default 300 s): kills the process when no bytes arrive on stdout *or* stderr for that source-specific interval. Catches silent hangs (the provider never answers the initial request — a 0-byte stream) and mid-stream/tool round-trip stalls (`step_start` or a `tool_use` event, then silence). Set the source value to `0` to disable this guard; the outer benchmark timeout remains active.
- **Step budget** (`step_limit`, default 50): kills when `step_finish` events exceed the cap. Catches reasoning/tool planning loops that churn steps (e.g. hundreds of `todowrite` calls) without ever producing a final answer.
- **Text-repetition** (`repeat_threshold` 5 / `repeat_min_len` 20): kills when the same non-trivial text event appears repeatedly. Catches canned-continuation loops (a short canned string fed back into the loop thousands of times).

The defaults are based on observed healthy benchmark streams: they emit `step_start` within seconds, stay below 20 steps, and do not repeat identical text events. Each guard is disabled by passing 0/None, and a process that exits on its own is always honored over the guards.

## API Request Flow

```
_run_plugin_task
    ├── stream_request (if plugin.supports_streaming)
    │       └── fallback to nonstream_request on error
    └── nonstream_request (if not streaming)
```

Both request functions:

1. Build the request body with `model`, `messages`, `max_tokens`, `stream`, and optional `temperature`/`seed`.
2. Apply per-model `drop_params` by removing specified keys.
3. POST to the source's `api_url`.
4. Log the curl command and response.
5. Return text, timing, and usage info.

The HTTP task path also passes per-source live-stream guards (`http._StreamGuards`,
resolved via `core.resolve_stream_guards`) into both request functions. On every
SSE delta the guards compare `reasoning_content` and `content` against their
split token budgets (`max_thinking_tokens` / `max_content_tokens`) and run a
repetition check on both streams; the first violation aborts the request with a
distinct error, retaining the text streamed so far for scoring. Guards apply to
benchmark task requests only - preload probes, judges, and OpenCode are exempt.

## Plugin Scoring

Each plugin's `evaluate()` method receives the raw response text and returns an `EvaluationResult` containing a native task-scale score, rubric criteria, and diagnostics. Plugins may use regex checks, typed parsers, and isolated execution validators; the shared `Rubric` helper records evidence and bounded penalties. The benchmark core normalizes the native score exactly once to the public 0–100 percentage schema and serializes rubric entries as `points`/`total`. `score()` remains the native offline-evaluation convenience method and delegates to `evaluate()` for the built-in challenge plugins.

## Output Generators

- `gen_markdown()`: Markdown report with leaderboards
- `gen_csv()`: CSV data
- `gen_html()`: HTML report with styling
- `gen_pdf()`: PDF report (requires `fpdf2`)

Output generators handle mixed numeric and string scores defensively to avoid errors when a plugin fails.

## Concurrency Model

- Source-level parallelism: one thread per source in all modes; in `both`, that one source worker owns both runner steps.
- Pipeline ordering: target `T` runs OpenCode then HTTP, with no overlap between runners on the source; the source advances to the next target afterward.
- Plugin-level parallelism: controlled by per-source `plugin_thread_limit` (with a top-level fallback).
- `ThreadPoolExecutor` runs plugins for a single model.
- Judge concurrency mirrors the same two tiers per source: `model_thread_limit`
  bounds distinct judge models (each run to completion in discovery order), and
  `plugin_thread_limit` bounds cells scored per judge.

## Resume Behavior

Runner identity is part of the persisted result key. A completed HTTP result cannot satisfy an OpenCode run, and vice versa. In `both`, each source worker skips completed runner steps independently; completed OpenCode targets proceed directly to their pending HTTP step on resume. State and reports retain both variants, while runner-specific response and log artifacts remain under `http/` and `opencode/`.


Saved state includes:

- Active plugin IDs
- Plugin versions
- Per-model status and metrics (sources are stored as strings)
- All results

On resume:

1. Load saved state.
2. Detect plugin set changes.
3. Prompt user to restart or continue.
4. Skip models with status `completed`.
5. Reset models with status `failed` to `pending` so they are re-run.
6. Re-run only the plugins that failed or were missing; successful plugin scores are preserved.

### Disabling automatic rerun

By default, any model that failed in a previous session is re-run on resume. To keep failed models as failed and skip them, pass `--no-rerun-failed`:

```bash
python ai-benchmark.py --no-rerun-failed
```

This is useful when failures are known to be non-transient or when you want to preserve the existing results for reporting.

## Error Handling

- Plugin failures record `"fail"` string scores and preserve error/traceback metadata when response saving is enabled.
- Output generators safely ignore non-numeric scores.
- Historical CSV recovery accepts a known subset of currently discovered plugins, so reports from before a plugin was added remain recoverable; unknown plugin score columns are rejected.
- Worker exceptions are printed but do not stop other workers.
- Ctrl+C closes active requests and saves state.
