# CLI Reference

The benchmark is driven by `ai-benchmark.py`.

## Usage

```sh
python ai-benchmark.py [options]
```

## Arguments

| Argument | Description |
|---|---|
| `--config PATH` | Config file path (default: `benchmark-config.json`) |
| `--restart` | Discard prior state and run all models from scratch |
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
| `--base-url URL` | (with `--dump-default-config`) Discover models from `/v1/models` |
| `--api-key KEY` | (with `--base-url`) API key for model discovery |
| `--save-responses` | Save each model's plugin response text to the selected runner namespace under `<output_dir>/{http,opencode}/responses/` |
| `--seed INT` | Fixed random seed for all API requests |
| `--no-rerun-failed` | Keep failed models as failed on resume (default re-runs them) |
| `--retry-on-429` / `--no-retry-on-429` | Toggle HTTP-429 retry/backoff globally. Default is **ON**; pass `--no-retry-on-429` to opt out (sources with explicit `max_429_retries` are preserved). See [Configuration Reference](configuration.md#http-429-retry--backoff) for the per-source keys and migration notes. |
| `--runner {http,opencode,both}` | Select the existing HTTP runner (default), the OpenCode CLI runner, or both. In `both`, each target pipelines OpenCode into HTTP. |
| `--no-install-opencode` | Do not auto-download OpenCode into `.tools/opencode/` when it is missing or too old; fail with an error instead |
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

Each OpenCode task is invoked as `opencode run --pure --model <slugified-source>/<api_model> --format json --thinking <prompt>` (with `--agent` for agent targets). The adapter uses the `json` event-stream format (the only machine-readable choice; `plain` is not a valid value) and extracts the final assistant answer from the NDJSON `text` events (and `reasoning` events into the `think_text` sidecars), so the scored response is the model's final answer without TUI/ANSI noise. Generated configs set both `limit.context` (inferred from the model id's `-NNk`/`-NNm` suffix, e.g. `-128k`) and `limit.output` (from `token_levels`), because OpenCode rejects provider models whose `limit` omits `context`.

OpenCode's agent loop has no internal liveness detection, so stalled/looping tasks would otherwise burn the full benchmark timeout silently. Each subprocess therefore runs three data-backed loop guards that kill it early with an actionable error (partial stdout retained): a **staleness fast-fail** (120 s with no output on stdout or stderr — silent hangs and mid-stream/tool round-trip stalls), a **step budget** (50 `step_finish` events — reasoning/tool planning loops), and a **text-repetition guard** (same non-trivial text event 5× — canned-continuation loops). All three can be disabled per call by passing 0/None.

The OpenCode model mapping is deterministic: the source name is lowercased and strictly slugified, then joined with the resolved API model as `{slugified-source}/{api_model}`. Existing slashes in `api_model` are preserved.

### Discover models from an API

```sh
python ai-benchmark.py \
  --dump-default-config \
  --base-url http://localhost:11434 \
  --api-key sk-xxx > benchmark-config.json
```

## Resume Behavior

By default, re-running resumes from the saved state. Completed models are skipped, failed models are retried, and newly added models are picked up automatically. Use `--restart` to force a clean run.

If the active plugin set changes between runs, the CLI prompts whether to restart or continue. Continuing keeps old data and runs only the newly added plugins for models that already completed.
