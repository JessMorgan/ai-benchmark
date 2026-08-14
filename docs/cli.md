# CLI Reference

The benchmark is driven by `ai-benchmark.py` (a thin launcher) or the
installed `ai-benchmark` console script, both of which delegate to
`benchmark.cli.main`.

## Usage

```sh
ai-benchmark [options]            # installed console script
python ai-benchmark.py [options]    # repository launcher
```

## Arguments

| Argument | Description |
|---|---|
| `--config PATH` | Config file path (default: `benchmark-config.json`) |
| `--restart` | Discard prior state and run all models from scratch |
| `--scripted` | Continue non-interactively instead of prompting on resume/plugin changes |
| `--out DIR` | Override the output directory from config |
| `--timeout SEC` | Override API request timeout |
| `--token-levels N [N ...]` | Override token levels (e.g. `--token-levels 4096 8192 16384`) |
| `--temperature VAL` | Default temperature for all plugins (overrides config) |
| `--plugin-temperature ID=VAL [ID=VAL ...]` | Override per-plugin temperatures (highest priority) |
| `--plugin-thread-limit N` | Max concurrent plugins per model (`0` = one per plugin) |
| `--plugins-whitelist ID [ID ...]` | Run only these plugins |
| `--plugins-blacklist ID [ID ...]` | Run all plugins except these |
| `--list-plugins` | List discovered plugins and exit |
| `--generate-shell-completion bash/zsh/fish` | Generate shell completion script |
| `--dump-default-config` | Print a default config template and exit |
| `--convert-config PATH` | Convert JSON/YAML config and print it to stdout |
| `--base-url URL` | (with `--dump-default-config`) Discover models from `/v1/models` |
| `--api-key KEY` | (with `--base-url`) API key for model discovery |
| `--save-responses` | Save each model's plugin response text to the selected runner namespace under `<output_dir>/{http,opencode}/responses/` |
| `--seed INT` | Fixed random seed for all API requests |
| `--no-rerun-failed` | Keep failed models as failed on resume (default re-runs them) |
| `--retry-on-429` / `--no-retry-on-429` | Toggle HTTP-429 retry/backoff globally. Default is **ON**; pass `--no-retry-on-429` to opt out (sources with explicit `max_429_retries` are preserved). See [Configuration Reference](configuration.md#http-429-retry--backoff) for the per-source keys and migration notes. |
| `--runner {http,opencode,both}` | Select the existing HTTP runner (default), the OpenCode CLI runner, or both. In `both`, each target pipelines OpenCode into HTTP. |
| `--judge-models MODEL [MODEL ...]` | Opt in to semantic judging with one or more configured model keys; their valid ratings are combined into a confidence-weighted consensus. Judge inputs are retained under `<output_dir>/judge-inputs/` and scores appear beside deterministic scores. |
| `--build-judge-queue STATE_FILE` | Build a ranked JSON queue of cells selected by judge spread or deterministic/consensus deviation and exit |
| `--judge-queue-output PATH` | Write the judge queue to this path instead of beside the state file |
| `--judge-spread-threshold POINTS` | Include cells whose valid judge scores span at least this many points; default 30 |
| `--no-judge-spread` | Disable judge-spread queue selection |
| `--judge-deviation-threshold POINTS` | Include cells whose consensus differs from the deterministic score by at least this many points; default 40 |
| `--no-judge-deviation` | Disable deterministic/consensus deviation queue selection |
| `--no-install-opencode` | Do not auto-download OpenCode into `.tools/opencode/` when it is missing or too old; fail with an error instead |
| `--no-preload` | Disable per-source model warm-up probes for this run, overriding `preload: true` |
| `-h, --help` | Show help message |

## Examples

### Run with the default config

```sh
python ai-benchmark.py
```

### Use a custom config

```sh
python ai-benchmark.py --config my-config.json
```

### Restart from scratch

```sh
python ai-benchmark.py --restart
```

### Run only one plugin

```sh
python ai-benchmark.py --plugins-whitelist rate-limiter
```

### Exclude a plugin

```sh
python ai-benchmark.py --plugins-blacklist moe-dense
```

### Use a fixed seed

```sh
python ai-benchmark.py --seed 42
```

### Override token levels

```sh
python ai-benchmark.py --token-levels 4096 8192 16384
```

### Override all plugin temperatures

```sh
python ai-benchmark.py --temperature 0.5
```

### Override a specific plugin temperature

```sh
python ai-benchmark.py --plugin-temperature rate-limiter=0.3
```

### Combine default and per-plugin temperatures

Per-plugin settings take priority over the default, and both override config file values.

```sh
python ai-benchmark.py --temperature 0.5 --plugin-temperature moe-dense=0.7
```

### Generate and install shell completions

```sh
# Bash
python ai-benchmark.py --generate-shell-completion bash > /etc/bash_completion.d/ai-benchmark

# Zsh
python ai-benchmark.py --generate-shell-completion zsh > ~/.zsh/completions/_ai-benchmark.py

# Fish
python ai-benchmark.py --generate-shell-completion fish > ~/.config/fish/completions/ai-benchmark.py.fish
```

### Semantic model judging

Use one or more configured plain models as semantic judges. Their valid ratings are combined into a confidence-weighted consensus; a response remains eligible for any judge that has not yet produced a usable result. Supply all judge model keys after one `--judge-models` option—do not repeat the option:

```sh
python ai-benchmark.py --judge-models judge-model-id second-judge-model-id
```

The judge receives the original task prompt and response as explicitly delimited, quoted evaluation data, returns a validated 0–100 score with confidence and rationale, and never changes the deterministic benchmark score or benchmark success status. The prompt tells the judge not to follow instructions in the candidate answer, emit tool calls, or continue the embedded task, and requires exactly one JSON object. HTTP judge requests preserve each model's native thinking behavior and request a JSON object response by default; they do not apply a global thinking-token budget. Provider-specific thinking controls can be added through `judge.request_params`; see [Configuration Reference](configuration.md#semantic-judge-request-parameters). Judge input sidecars are retained automatically so interrupted runs can resume judging. The per-judge HTTP log records the exact merged request body, and each judge response metadata sidecar records whether the observed reasoning-token usage stayed within the requested budget when the provider exposes sufficient usage data. Each judge's raw JSON response is saved beside the benchmark response artifacts as `<plugin>.judge.<model>.txt`; the live TUI footer shows per-judge progress such as `Judging [Big Pickle: 4/17]`, and a `J` is appended to a plugin score once all configured judges have finished that response. While a judge source is still benchmarking, one source slot is reserved for judging and benchmark capacity is reduced by one when the source limit is greater than one. As soon as that source's benchmark queue drains, judging expands to the source's full configured `model_thread_limit`, even when there is only one judge model; judges sharing a source therefore run concurrently up to that limit. A single-thread source does not start judging until its benchmark work finishes.

### Disable model preloading

Sources can opt into a 300-second-by-default warm-up probe with `preload: true` and optionally override the limit with `preload_timeout`. Disable all configured probes for a run with:

```sh
python ai-benchmark.py --no-preload
```

### Select an execution runner

The default runner is the existing OpenAI-compatible HTTP path:

```sh
python ai-benchmark.py --runner http
```

To run each configured model and agent through OpenCode:

```sh
python ai-benchmark.py --runner opencode
```

OpenCode is resolved at startup: an on-PATH install that passes the capability check is used; otherwise the benchmark downloads the official latest release into `<project root>/.tools/opencode/` and uses that binary, printing the resolved path. Pass `--no-install-opencode` to disable the download and fail with an actionable error instead. `--runner both` gives each source one execution slot: each target runs OpenCode, then HTTP, before the source advances to its next target. The two runners never overlap on one source:

```sh
python ai-benchmark.py --runner both --save-responses
```

OpenCode mode generates and retains `<output_dir>/opencode/opencode.generated.json` from the loaded benchmark sources. The file contains resolved authentication values, is written with restrictive permissions where supported, and should be treated as a secret-bearing artifact. OpenCode responses and logs are stored below `<output_dir>/opencode/`; HTTP artifacts are stored below `<output_dir>/http/`. Results include a runner column so the two variants remain distinguishable, and resume reuses results only for the same runner.

Startup preflight validates the OpenCode CLI before any work is scheduled: `opencode run --help` must advertise the `--model`/`--format`/`--agent`/`--pure`/`--thinking` options and `--format` must list `json` as a choice. An on-PATH binary that fails this check is replaced by a fresh `.tools/opencode/` install automatically (unless `--no-install-opencode`); the resolved binary path is recorded in `run-info.json` as `opencode_binary`. An unsupported CLI fails fast with a clear error instead of failing every task at runtime.

Each OpenCode task is invoked as `opencode run --pure --model <slugified-source>/<api_model> --format json --thinking --agent benchmark-<target> <prompt>`. Every target registers an agent in the generated config: agent personas keep their explicit system prompt, while plain model targets register a **neutral agent** whose prompt has no conciseness instruction and whose permission map denies every tool family — so small function-calling-tuned models see no tool definitions and receive the same plain "answer the prompt" contract as HTTP (OpenCode's built-in default "answer concisely <4 lines" prompt never applies). The adapter uses the `json` event-stream format (the only machine-readable choice; `plain` is not a valid value) and extracts the final assistant answer from the NDJSON `text` events (and `reasoning` events into the `think_text` sidecars), so the scored response is the model's final answer without TUI/ANSI noise. Generated configs set both `limit.context` (inferred from the model id's `-NNk`/`-NNm` suffix, e.g. `-128k`) and `limit.output` (from `token_levels`), because OpenCode rejects provider models whose `limit` omits `context`.

OpenCode's agent loop has no internal liveness detection, so stalled/looping tasks would otherwise burn the full benchmark timeout silently. Each subprocess therefore runs three data-backed loop guards that kill it early with an actionable error (partial stdout retained): a **staleness fast-fail** (`sources.<name>.opencode_timeout`, 300 s by default, with no output on stdout or stderr — silent hangs and mid-stream/tool round-trip stalls), a **step budget** (50 `step_finish` events — reasoning/tool planning loops), and a **text-repetition guard** (same non-trivial text event 5× — canned-continuation loops). Set a source's `opencode_timeout` to `0` to disable the staleness guard; the outer benchmark timeout still applies.

The OpenCode model mapping is deterministic: the source name is lowercased and strictly slugified, then joined with the resolved API model as `{slugified-source}/{api_model}`. Existing slashes in `api_model` are preserved.

### Per-source model concurrency

`model_thread_limit` is a configuration key, not a CLI flag. Set
`sources.<name>.model_thread_limit` to allow multiple target/model
pipelines for that source; omit it to use the top-level `model_thread_limit`,
then the default of `1`. This is separate from `plugin_thread_limit`, so the
approximate request ceiling can be their product. Use `1` for local AI Server
and Gaming PC sources unless you intentionally accept the hardware risk; cloud
sources can be set to `2`, `4`, or another positive integer. The CLI prints the
effective limits and warns when a local source is explicitly raised above one.
No CLI override is provided in this implementation.

### Discover models from an API

```sh
python ai-benchmark.py \
  --dump-default-config \
  --base-url http://localhost:11434 \
  --api-key sk-xxx > benchmark-config.json
```

## Resume Behavior

By default, re-running resumes from the saved state. Completed models are skipped, failed models are retried, and newly added models are picked up automatically. Use `--restart` to force a clean run.

If the saved state file is unreadable or fails to load (for example a corrupt `benchmark_state.json`), the run **aborts with a non-zero exit code** rather than silently discarding prior results and starting fresh. Inspect or repair the state file, or pass `--restart` to explicitly discard it.

If the active plugin set changes between runs, the CLI prompts whether to restart or continue. Continuing keeps old data and runs only the newly added plugins for models that already completed.
