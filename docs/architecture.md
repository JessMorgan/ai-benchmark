# Architecture

This document describes the high-level design of AI Benchmark.

## Components

```
┌─────────────────────────────────────────┐
│           ai-benchmark.py             │
│  CLI parsing, config loading, TUI     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│           benchmark_core.py           │
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
   - `--runner both` creates one OpenCode producer and one HTTP consumer per source. Each target's HTTP work is queued as soon as that target's OpenCode work finishes, while later targets continue through OpenCode; there is no global OpenCode/HTTP phase barrier.
   - Targets whose OpenCode state is already complete on resume are seeded directly into the HTTP queue.
   - Single-runner modes retain one source worker per source and run `run_model()` for each target in its queue.
   - State identities include the configured target and runner, preventing cross-runner resume reuse.

4. **Plugin Execution**
   - `_run_plugins()` uses `ThreadPoolExecutor` with `plugin_thread_limit` workers.
   - Each plugin task calls `_run_plugin_task()`.
   - HTTP tasks call `stream_request()` or `nonstream_request()`.
   - OpenCode tasks call the subprocess adapter with `opencode run --format json`, capture stdout/stderr separately, extract the final assistant answer from the NDJSON event stream, and score it as the response.

5. **Scoring**
   - Plugin `score()` evaluates the response.
   - Results are stored in `BenchmarkState`.

6. **Output Generation**
   - After all models complete, `_save_outputs()` generates reports.
   - `gen_pdf()` is called separately for PDF output.

## State Management

`BenchmarkState` is a thread-safe in-memory store:

- `_model_info`: per-model status and metrics
- `results`: list of result dicts
- `_log`: recent error log entries

State is saved to `benchmark_state.json` after each model and on shutdown. The saved state stores model sources as plain strings; dict-valued model entries from the config are resolved to their source string before being written.

## OpenCode Runner

OpenCode is optional and is never required for the default HTTP mode. When selected, startup checks `shutil.which("opencode")` and runs a capability preflight (`opencode run --help`): the CLI must expose `--model`/`--format`/`--agent` and advertise the `json` format choice, otherwise the run fails fast with a clear error. The runner resolves every configured target to `{slugified source}/{api_model}`, projects source URL/auth/model settings into a generated config, and stores that config under the OpenCode output namespace. Existing agent system prompts are registered as OpenCode agents and passed separately from each plugin prompt. The generated config is retained intentionally; operators must protect the output directory because it contains resolved credentials.

Each OpenCode task runs `opencode run --model <mapped> --format json <prompt>` in an isolated subprocess. The `json` format emits one NDJSON event per line; `_extract_final_text()` joins the `text` events (the model's final answer parts) and surfaces `error` events as failures. Model entries in the generated config always set both `limit.context` and `limit.output` — OpenCode's schema rejects a `limit` object that omits `context`. The context value is inferred from the benchmark model id's `-NNk`/`-NNm` suffix with a conservative fallback, and the output value comes from `token_levels`.

OpenCode has buffered final-output semantics in this adapter. Direct HTTP streaming metrics such as TTFT are not fabricated for OpenCode; its response time and estimated output-token count are recorded instead. Timeouts and cancellation terminate the subprocess group where supported.

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

## Plugin Scoring

Each plugin's `score()` method receives the raw response text and returns a float. The benchmark does not interpret the score; it simply records it.

## Output Generators

- `gen_markdown()`: Markdown report with leaderboards
- `gen_csv()`: CSV data
- `gen_html()`: HTML report with styling
- `gen_pdf()`: PDF report (requires `fpdf2`)

Output generators handle mixed numeric and string scores defensively to avoid errors when a plugin fails.

## Concurrency Model

- Source-level parallelism: one thread per source in single-runner modes; in `both`, one OpenCode producer plus one HTTP consumer per source.
- Pipeline ordering: HTTP for target `T` waits only for OpenCode target `T`; it does not wait for other targets' OpenCode work.
- Plugin-level parallelism: controlled by per-source `plugin_thread_limit` (with a top-level fallback).
- `ThreadPoolExecutor` runs plugins for a single model.

## Resume Behavior

Runner identity is part of the persisted result key. A completed HTTP result cannot satisfy an OpenCode run, and vice versa. In `both`, each pipeline side skips only its own completed target/plugin results; completed OpenCode targets are released directly to the HTTP queue on resume. State and reports retain both variants, while runner-specific response and log artifacts remain under `http/` and `opencode/`.


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

- Plugin failures record `"fail"` string scores.
- Output generators safely ignore non-numeric scores.
- Worker exceptions are printed but do not stop other workers.
- Ctrl+C closes active requests and saves state.
