# AI Model Benchmark

Multi-source, plugin-based benchmark for evaluating code generation and general reasoning capabilities of LLMs served through OpenAI-compatible APIs.

Plugins live in the `plugins/` directory. Each plugin defines a prompt, a scoring function, and version metadata. Results are tagged with the plugin versions used.

## Quickstart

```sh
# Generate a default config, then run
python ai-benchmark.py --dump-default-config > benchmark-config.json
python ai-benchmark.py

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
| `models` | Map of model name → source name |

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
| `-h, --help` | Show this help message |

## Resume / Continue

By default, re-running resumes from where you left off — completed models are skipped, and failed models are retried. Saved state is stored in `benchmark_state.json` inside the output directory and is preserved after completion so you can re-run to retry any failures. New models added to the config between runs are picked up automatically. Use `--restart` to force a clean run.

If the set of active plugins changes between runs, the app detects this and asks whether to **restart** or **continue**. If you continue, newly added plugins are run for models that already completed, and data for removed plugins is preserved but not run again.

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

Plugins are discovered automatically from the `plugins/` directory. Each plugin is a Python module containing a `BenchmarkTaskPlugin` subclass. Built-in plugins:

| ID | Name | Max Score |
|---|---|---|
| `rate-limiter` | Concurrent Rate Limiter | 20 |
| `moe-dense` | MoE vs Dense Architecture | 15 |

Each plugin exposes a `version` attribute so results can be correlated to a specific plugin release.

## Reports

Each model is scored on the active plugins across multiple dimensions:

- **Score**: Quality rating (0–20) based on rubric keywords
- **Speed**: Tokens per second (TPS)
- **Latency**: Time to first token (TTFT) for streaming
- **Cost**: Approximate per-model overhead

Results are grouped by phase (code columns first, then general columns).
