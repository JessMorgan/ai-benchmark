# AI Model Benchmark

Multi-source, plugin-based benchmark for evaluating code generation and general reasoning capabilities of LLMs served through OpenAI-compatible APIs.

Plugins live in the `plugins/` directory. Each plugin defines a prompt, a scoring function, and version metadata. Results are tagged with the plugin versions used.

## Quickstart

Install the package to get the `ai-benchmark` command (or use the `python
ai-benchmark.py` launcher from a checkout — both are the same CLI):

```sh
uv sync
uv run ai-benchmark --dump-default-config > benchmark-config.json
uv run ai-benchmark
```

Or, from a repository checkout:

```sh
# Generate a default config, then run
python ai-benchmark.py --dump-default-config > benchmark-config.json
python ai-benchmark.py

# Run the existing HTTP path explicitly (the default)
python ai-benchmark.py --runner http

# Run configured models and agents through a pre-installed OpenCode CLI
python ai-benchmark.py --runner opencode

# Pipeline OpenCode into HTTP per target (no global phase barrier)
python ai-benchmark.py --runner both

# Or discover models from a running server
python ai-benchmark.py \
  --dump-default-config \
  --base-url http://localhost:11434 \
  --api-key sk-xxx > benchmark-config.json
python ai-benchmark.py

# Run only one plugin
python ai-benchmark.py --plugins-whitelist rate-limiter

# Exclude a plugin
python ai-benchmark.py --plugins-blacklist moe-dense

# Use a fixed seed for reproducibility
python ai-benchmark.py --seed 42
```

## Configuration

All configuration lives in a JSON file (default: `benchmark-config.json`):

```json
{
  "output_dir": "benchmark-results",
  "timeout": 600,
  "max_tokens": 16384,
  "plugins_whitelist": [],
  "plugins_blacklist": [],
  "sources": {
    "OpenCode Zen": {
      "api_url": "https://api.example.com/chat/completions",
      "headers": {"Authorization": "Bearer ${API_KEY}"},
      "opencode_timeout": 300
    }
  },
  "models": { ... }
}
```

| Key | Description |
|---|---|
| `output_dir` | Directory for results and logs |
| `timeout` | API request timeout in seconds |
| `max_tokens` | One generation budget; each task may make at most one policy-controlled retry using the same budget |
| `model_max_tokens` | Scalar per-target max-token overrides keyed by target or `source/api_model` |
| `model_thread_limit` | Positive top-level fallback for concurrent target pipelines per source |
| `plugin_thread_limit` | Top-level fallback for concurrent plugins within one target |
| `plugins_whitelist` | List of plugin IDs to run (empty = all) |
| `plugins_blacklist` | List of plugin IDs to skip (empty = none) |
| `sources` | Named API endpoints with URL, headers, and optional per-source settings such as `opencode_timeout`, `preload`, `model_thread_limit`, `api_protocol`, and `max_429_retries` |
| `models` | Map of model name → source name, or model name → object with `source`, `drop_params`, and optional `max_tokens` |
| `agents` | Optional named targets with a model, source, and system prompt |

### Per-model configuration

The `models` map supports two forms. The simple form maps a model to a source:

```json
"models": {
  "model-a": "Local Server 1"
}
```

The extended form allows per-model settings such as dropping specific API parameters:

```json
"models": {
  "model-a": "Local Server 1",
  "model-b": {
    "source": "Remote Provider 1",
    "drop_params": ["seed"]
  }
}
```

| Key | Description |
|---|---|
| `source` | Source name from the `sources` section |
| `drop_params` | List of request body keys to omit (e.g. `seed`, `temperature`) |
| `max_tokens` | Optional scalar per-target generation budget; takes precedence over the global value |

**API keys** use `${VAR}` or `${VAR:default}` env-var syntax:
```json
"Authorization": "Bearer ${MY_API_KEY:sk-fallback-key}"
```

Variables may come from the process environment or from a `.env` file in the
current directory, which the CLI loads at startup (`python-dotenv`). A missing
`.env` is ignored and real environment variables win over file values.

### 1min.ai sources

Set `api_protocol: "1min"` on a source to talk to the native 1min.ai
`/api/chat-with-ai` endpoint. The source authenticates with an `API-KEY`
header and uses 1min.ai's request/response shape (no system message, no
`max_tokens`/`temperature`/`seed`); see `docs/configuration.md` for details.

```json
"1min.ai": {
  "api_protocol": "1min",
  "api_url": "https://api.1min.ai/api/chat-with-ai",
  "headers": {
    "API-KEY": "${ONEMIN_API_KEY:your-1min-api-key}",
    "Content-Type": "application/json"
  }
}
```

### ChatPlayground.ai sources

Set `api_protocol: "chatplayground"` to drive the ChatPlayground.ai web app
with Playwright using a username/password (Clerk login, no API key). Install
Chromium once with `uv run playwright install chromium`. Each model is
addressed by its `/chat/<slug>` route, so put the slug in `models` (e.g.
`deepseek-v4-pro`, `gpt-5.6-terra`). Answers are buffered (no TTFT/TPS), and
browser work is serialized under one logged-in session. Playwright runs in an
**isolated worker subprocess** (`benchmark/chatplayground_worker.py`): a native
crash in the browser stack surfaces as a per-request error and the next request
spawns a fresh worker, instead of segfaulting the whole benchmark runner.

To enumerate the site's model slugs and emit a ready-to-run config, run
`ai-benchmark --chatplayground-config > benchmark-config.json` with
`CHATPLAYGROUND_EMAIL`/`CHATPLAYGROUND_PASSWORD` set (or use
`python -m benchmark.chatplayground` for raw selector/model diagnostics).

```json
"ChatPlayground": {
  "api_protocol": "chatplayground",
  "base_url": "https://web.chatplayground.ai",
  "email": "${CHATPLAYGROUND_EMAIL}",
  "password": "${CHATPLAYGROUND_PASSWORD}",
  "headless": true,
  "model_thread_limit": 1,
  "plugin_thread_limit": 1
},
"models": {
  "deepseek-v4-pro": "ChatPlayground"
}
```

## CLI Reference

```
aio-benchmark [options]           # installed console script
python ai-benchmark.py [options]  # repository launcher
```

| Argument | Description |
|---|---|
| `--config PATH` | Config file path (default: `benchmark-config.json`) |
| `--restart` | Discard prior state and run all models from scratch |
| `--scripted` | Continue non-interactively instead of prompting on resume/plugin changes |
| `--out DIR` | Override the output directory from config |
| `--timeout SEC` | Override API request timeout |
| `--max-tokens N` | Override the scalar generation budget (the retry reuses this value) |
| `--temperature VAL` | Default plugin temperature |
| `--plugin-temperature ID=VAL [...]` | Override selected plugin temperatures |
| `--plugin-thread-limit N` | Max plugin workers per target; `0` means one per plugin |
| `--dump-default-config` | Print a config template to stdout and exit |
| `--base-url URL` | (with `--dump-default-config`) Discover models from `/v1/models` |
| `--api-key KEY` | (with `--base-url`) API key for model discovery |
| `--chatplayground-config` | Enumerate ChatPlayground.ai models and print a ready-to-run config |
| `--plugins-whitelist ID [ID ...]` | Run only these plugins |
| `--plugins-blacklist ID [ID ...]` | Run all plugins except these |
| `--no-rerun-failed` | Keep failed models failed on resume |
| `--seed INT` | Fixed random seed for all API requests |
| `--runner {http,opencode,both}` | Select the existing HTTP runner (default), the OpenCode runner, or both (per-target OpenCode→HTTP pipeline) |
| `--no-preload` | Disable configured source preload probes |
| `--retry-on-429 / --no-retry-on-429` | Enable or disable default HTTP 429 backoff |
| `--save-responses` | Save prompts, responses, reasoning, and metadata sidecars |
| `--storage {json,sqlite}` | Select run storage; SQLite is the default, JSON remains an explicit compatibility fallback |
| `--storage-profile {compact,debug,portable}` | Select artifact policy; compact omits full successful transcripts |
| `--debug-logs` | Retain redacted full request/response and runner diagnostics as append-only gzip logs |
| `--import-to-sqlite STATE_JSON` | Convert a legacy JSON state file without modifying the source |
| `--import-debug-logs` | Include legacy `.log`/`.log.gz` files during JSON-to-SQLite conversion |
| `--measure-storage` | Print a reproducible synthetic JSON/SQLite size and persistence-latency baseline |
| `--compare-storage JSON SQLITE` | Compare current JSON and SQLite read models without writing either backend |
| `--check-sqlite PATH` | Run read-only SQLite integrity checks |
| `--judge-models MODEL [...]` | Run confidence-weighted semantic judging |
| `--build-judge-queue STATE_FILE` | Build a ranked judge-disagreement queue |
| `--judge-queue-output PATH` | Choose the judge queue output path |
| `--judge-spread-threshold POINTS` | Configure the judge spread queue threshold |
| `--no-judge-spread` | Disable judge spread queueing |
| `--judge-deviation-threshold POINTS` | Configure deterministic/consensus deviation queueing |
| `--no-judge-deviation` | Disable judge deviation queueing |
| `--no-install-opencode` | Do not auto-download OpenCode into `.tools/opencode/` when it is missing or too old; fail with an error instead |
| `--list-plugins` | List discovered plugins and versions |
| `--generate-shell-completion SHELL` | Generate Bash, Zsh, or Fish completion |
| `--convert-config PATH` | Convert JSON/YAML config and print it to stdout |
| `-h, --help` | Show this help message |

## Resume / Continue

By default, re-running resumes from where you left off — completed models are skipped, and failed models are retried. New runs use `run.sqlite3` as the authoritative store; `--storage json` selects the legacy `benchmark_state.json` backend explicitly. Existing JSON runs are adopted into SQLite when the default backend is used and no database exists beside the state file. Use `--restart` to force a clean run. Use `--scripted` to continue automatically without the interactive restart/continue prompt.

If the saved state file is unreadable or fails to load (e.g. a corrupt `benchmark_state.json`), the run **aborts with an error** instead of silently discarding prior results — inspect or repair the state file, or pass `--restart` to explicitly discard it. If the set of active plugins changes between runs, the app detects this and asks whether to **restart** or **continue**. Other newly added plugins are run for models that already completed, and data for removed plugins is preserved but not run again.

## OpenCode runner

OpenCode is an optional runtime dependency. Use `--runner opencode` or `--runner both`. If `opencode` is already installed and available on `PATH`, that binary is used (after a capability check). If it is missing — or is too old to satisfy the required CLI contract — the benchmark downloads the official latest release into a project-local directory (`.tools/opencode/`) and uses that binary instead, printing the resolved path at startup. Pass `--no-install-opencode` to disable the automatic download and fail with an actionable error instead. Either way, an unusable OpenCode fails before benchmark work begins. No OpenCode block is added to the benchmark config.

The runner dynamically generates and retains `<output_dir>/opencode/opencode.generated.json` from the loaded sources and resolved API models. Its model names follow `{strictly-slugified-source}/{api_model}`; for example, `Local Server 1` plus `vendor/model-x` becomes `local-server-1/vendor/model-x`. The generated file contains resolved credentials, so protect it and the output directory.

OpenCode and HTTP artifacts are separated under `<output_dir>/opencode/` and `<output_dir>/http/`. Markdown, CSV, HTML, and PDF reports include runner metadata. Resume matching is runner-aware, so an HTTP result is never reused for an OpenCode task. With `--runner both`, each source has one execution slot and runs a per-target pipeline: OpenCode for a target finishes before its HTTP comparison starts, and the source then advances to the next target. OpenCode and HTTP never overlap on the same source; already-completed OpenCode targets flow directly to HTTP on resume.

Before scheduling any work the runner resolves and preflights the CLI: `opencode run --help` must advertise the `--pure`/`--model`/`--format`/`--agent`/`--thinking` options and the `json` format choice. A previously auto-installed copy under `.tools/opencode/` is reused when it still passes the preflight; the resolved binary path is recorded in `run-info.json` as `opencode_binary`. Each task is invoked as `opencode run --pure --model <slugified-source>/<api_model> --format json --thinking --agent benchmark-<target> <prompt>`; `--pure` prevents external OpenCode plugins from changing the benchmark environment, tools, prompts, or event stream, and `--thinking` makes OpenCode emit the model's `reasoning` NDJSON events so thinking content is preserved alongside the final answer. Every target registers an agent in the generated config so OpenCode never falls back to its built-in default agent prompt: agent personas keep their explicit system prompt, while plain model targets get a **neutral agent** (no "answer concisely" instruction, all tool permissions denied) so small function-calling-tuned models receive the same plain "answer the prompt" contract the HTTP runner provides instead of a tool-fixation prompt. The adapter parses the NDJSON event stream and scores the final assistant answer. Reasoning captured from the OpenCode runner lands in the same per-plugin sidecars the HTTP runner writes (`think.txt` plus a `<thinking>…</thinking>`-wrapped `response.txt`, inside `responses/<target>/<plugin>/`) when `--save-responses` is used. Generated configs always set both `limit.context` (inferred from the model id's `-NNk`/`-NNm` suffix) and `limit.output` (from `max_tokens`), because OpenCode rejects provider models whose `limit` omits `context`.

OpenCode's agent loop has no internal liveness detection, so a stalled or looping task would otherwise burn the full benchmark timeout silently. `run_process()` enforces three data-backed loop guards that kill the subprocess early and surface an actionable error instead (partial stdout is retained): a **staleness fast-fail** (`sources.<name>.opencode_timeout`, 300 s by default, with no output on stdout or stderr — catches silent hangs and mid-stream/tool-round-trip stalls), a **step budget** (50 `step_finish` events — catches reasoning/tool planning loops), and a **text-repetition guard** (same non-trivial text event 5× — catches canned-continuation loops). Set a source's `opencode_timeout` to `0` to disable the staleness guard; the outer benchmark timeout still applies.

## Outputs

After completion the output directory contains:

| File | Format |
|---|---|
| `results.md` | Markdown report |
| `results.csv` | CSV data |
| `results.html` | HTML report |
| `results.pdf` | PDF report (requires `fpdf2`) |
| `run.sqlite3` | Authoritative normalized run store for new runs |
| `logs/*.log.gz` | Redacted append-only debug logs when debug logging is enabled |
| `responses/` | Per-(model, plugin) directories with prompts, final responses, reasoning sidecars, and metadata when `--save-responses` is enabled |
| `benchmark_state.json` | Legacy JSON resume state when `--storage json` is selected |
| `run-info.json` | Run metadata, concurrency/preload data, and HTTP 429 backoff statistics |

## Plugins

Plugins are discovered automatically from `plugins/challenges/`. Each plugin is a Python module containing a `BenchmarkTaskPlugin` subclass. Run `python ai-benchmark.py --list-plugins` for the authoritative inventory. The current built-in plugins are:

| ID | Name | Version | Max score | Streaming |
|---|---|---:|---:|---|
| `code-review` | Code Review | 1.0.0 | 15 | No |
| `debug-consistency` | Debug Report Consistency | 0.1.0 | 20 | Yes |
| `debug-traversal` | Debug Traversal | 1.2.0 | 20 | Yes |
| `decomposition` | Design-Doc Decomposition | 0.1.0 | 20 | Yes |
| `error-recovery` | Error Recovery | 1.4.0 | 20 | Yes |
| `event-processor` | Concurrent Event Processor | 0.3.0 | 20 | Yes |
| `instruction-following` | Instruction Following | 1.0.0 | 20 | Yes |
| `long-context` | Long-Context Retrieval | 0.1.0 | 20 | Yes |
| `moe-dense` | MoE vs Dense | 1.0.1 | 17 | Yes |
| `multi-step` | Multi-Step Instructions | 1.3.0 | 20 | Yes |
| `multi-turn-conversation` | Multi-Turn Conversation | 1.0.0 | 20 | Yes |
| `orchestration` | Orchestration & Workflow | 1.0.0 | 16 | Yes |
| `prd-creation` | PRD Creation | 1.0.0 | 22 | Yes |
| `rate-limiter` | Rate Limiter | 1.3.0 | 20 | Yes |
| `reasoning` | Logical Reasoning | 1.1.0 | 20 | Yes |
| `software-architecture` | Software Architecture | 1.0.0 | 20 | Yes |
| `data-transformation` | Data Transformation | 1.0.2 | 22 | No |
| `tool-calling` | Tool Calling Agent | 1.1.1 | 25 | Yes |
| `wireframes` | Wireframes | 1.0.0 | 20 | Yes |

Each plugin exposes a `version` attribute so results can be correlated to a specific plugin release. Discovery validates required metadata and rejects duplicate IDs before a run starts.

## Tests & Coverage

The `uv sync` above installs the `ai-benchmark` console script plus its dev
dependencies (`pytest`, `coverage`, `mypy`, `ruff`), which live in the `dev`
dependency group of `pyproject.toml`. `uv.lock` pins the full resolved tree.

Run the test suite with pytest:

```sh
uv run pytest tests/ plugins/challenges/ plugins/outputs/ -q
```

Generate a coverage report for the `benchmark/` package and `plugins/`:

```sh
./scripts/run-coverage.sh
```

Coverage configuration lives in `pyproject.toml` (`[tool.coverage]`). This produces a terminal summary, an HTML report in `htmlcov/`, and an XML report in `coverage.xml`.

## Reports

Each model is scored on the active plugins across multiple dimensions:

- **Score**: Normalized integer quality percentage (0–100) based on typed parsers, section-local validators, and executable API checks for code-generation tasks
- **Speed**: Tokens per second (TPS)
- **Latency**: Time to first token (TTFT) for streaming
- **Cost**: Approximate per-model overhead

Results are grouped by phase (code columns first, then general columns). Plugin `max_score` values are internal rubric maxima; public scores and overall scores use the 0–100 percentage schema.
