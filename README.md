# AI Model Benchmark

Multi-source, plugin-based benchmark for evaluating code generation and general reasoning capabilities of LLMs served through OpenAI-compatible APIs.

Plugins live in the `plugins/` directory. Each plugin defines a prompt, a scoring function, and version metadata. Results are tagged with the plugin versions used.

## Quickstart

```sh
# Generate a default config, then run
python ai-benchmark.py --dump-default-config > benchmark-config.json
python ai-benchmark.py

# Run the existing HTTP path explicitly (the default)
python ai-benchmark.py --runner http

# Run configured models and agents through a pre-installed OpenCode CLI
python ai-benchmark.py --runner opencode

# Compare OpenCode first, then the existing HTTP path
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
  "token_levels": [16384],
  "plugins_whitelist": [],
  "plugins_blacklist": [],
  "sources": { ... },
  "models": { ... }
}
```

| Key | Description |
|---|---|
| `output_dir` | Directory for results and logs |
| `timeout` | API request timeout in seconds |
| `token_levels` | Max-token limits tried on truncation (ascending order) |
| `plugins_whitelist` | List of plugin IDs to run (empty = all) |
| `plugins_blacklist` | List of plugin IDs to skip (empty = none) |
| `sources` | Named API endpoints with URL and headers |
| `models` | Map of model name → source name, or model name → object with `source` and optional `drop_params` |

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

**API keys** use `${VAR}` or `${VAR:default}` env-var syntax:
```json
"Authorization": "Bearer ${MY_API_KEY:sk-fallback-key}"
```

## CLI Reference

```
python ai-benchmark.py [options]
```

| Argument | Description |
|---|---|
| `--config PATH` | Config file path (default: `benchmark-config.json`) |
| `--restart` | Discard prior state and run all models from scratch |
| `--out DIR` | Override the output directory from config |
| `--timeout SEC` | Override API request timeout |
| `--token-levels N [N ...]` | Override token levels (e.g. `--token-levels 4096 8192 16384`) |
| `--dump-default-config` | Print a config template to stdout and exit |
| `--base-url URL` | (with `--dump-default-config`) Discover models from `/v1/models` |
| `--api-key KEY` | (with `--base-url`) API key for model discovery |
| `--plugins-whitelist ID [ID ...]` | Run only these plugins |
| `--plugins-blacklist ID [ID ...]` | Run all plugins except these |
| `--seed INT` | Fixed random seed for all API requests |
| `--runner {http,opencode,both}` | Select the existing HTTP runner (default), the pre-installed OpenCode runner, or both (OpenCode first) |
| `-h, --help` | Show this help message |

## Resume / Continue

By default, re-running resumes from where you left off — completed models are skipped, and failed models are retried. Saved state is stored in `benchmark_state.json` inside the output directory and is preserved after completion so you can re-run to retry any failures. New models added to the config between runs are picked up automatically. Use `--restart` to force a clean run.

If the set of active plugins changes between runs, the app detects this and asks whether to **restart** or **continue**. If you continue, newly added plugins are run for models that already completed, and data for removed plugins is preserved but not run again.

## OpenCode runner

OpenCode is an optional runtime dependency. Use `--runner opencode` or `--runner both`; the executable must already be installed and available on `PATH`, and startup fails before benchmark work begins if it is missing. No OpenCode block is added to the benchmark config.

The runner dynamically generates and retains `<output_dir>/opencode/opencode.generated.json` from the loaded sources and resolved API models. Its model names follow `{strictly-slugified-source}/{api_model}`; for example, `Local Server 1` plus `vendor/model-x` becomes `local-server-1/vendor/model-x`. The generated file contains resolved credentials, so protect it and the output directory.

OpenCode and HTTP artifacts are separated under `<output_dir>/opencode/` and `<output_dir>/http/`. Markdown, CSV, HTML, and PDF reports include runner metadata. Resume matching is runner-aware, so an HTTP result is never reused for an OpenCode task.

Before scheduling any work the runner preflights the installed CLI: `opencode run --help` must advertise the `--model`/`--format`/`--agent` options and the `json` format choice. Each task is invoked as `opencode run --model <slugified-source>/<api_model> --format json <prompt>`; the adapter parses the NDJSON event stream and scores the final assistant answer. Generated configs always set both `limit.context` (inferred from the model id's `-NNk`/`-NNm` suffix) and `limit.output` (from `token_levels`), because OpenCode rejects provider models whose `limit` omits `context`.

## Outputs

After completion the output directory contains:

| File | Format |
|---|---|
| `results.md` | Markdown report |
| `results.csv` | CSV data |
| `results.html` | HTML report |
| `results.pdf` | PDF report (requires `fpdf2`) |
| `logs/*.log` | Per-model request/response logs |

## Plugins

Plugins are discovered automatically from `plugins/challenges/`. Each plugin is a Python module containing a `BenchmarkTaskPlugin` subclass. Run `python ai-benchmark.py --list-plugins` for the authoritative inventory. The current built-in plugins are:

| ID | Name |
|---|---|
| `code-review` | Code Review |
| `debug-traversal` | Debug Traversal |
| `error-recovery` | Error Recovery |
| `moe-dense` | MoE vs Dense |
| `multi-step` | Multi-Step Instructions |
| `multi-turn-conversation` | Multi-Turn Conversation |
| `orchestration` | Orchestration & Workflow |
| `prd-creation` | PRD Creation |
| `rate-limiter` | Rate Limiter |
| `software-architecture` | Software Architecture |
| `structured-output` | Structured Output |
| `tool-calling` | Tool Calling Agent |
| `wireframes` | Wireframes |

Each plugin exposes a `version` attribute so results can be correlated to a specific plugin release. Discovery validates required metadata and rejects duplicate IDs before a run starts.

## Tests & Coverage

Run the test suite with pytest:

```sh
python -m pytest tests/ plugins/challenges/ plugins/outputs/ -q
```

Generate a coverage report for the plugins:

```sh
./scripts/run-coverage.sh
```

This produces a terminal summary, an HTML report in `htmlcov/`, and an XML report in `coverage.xml`.

## Reports

Each model is scored on the active plugins across multiple dimensions:

- **Score**: Quality rating (0–20) based on rubric keywords
- **Speed**: Tokens per second (TPS)
- **Latency**: Time to first token (TTFT) for streaming
- **Cost**: Approximate per-model overhead

Results are grouped by phase (code columns first, then general columns).
