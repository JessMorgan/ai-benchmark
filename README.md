# AI Model Benchmark

Multi-source benchmark for evaluating code generation and general reasoning capabilities of LLMs served through OpenAI-compatible APIs.

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
```

## Configuration

All configuration lives in a JSON file (default: `benchmark-config.json`):

```json
{
  "output_dir": "benchmark-results",
  "timeout": 600,
  "token_levels": [16384],
  "sources": { ... },
  "models": { ... }
}
```

| Key | Description |
|---|---|
| `output_dir` | Directory for results and logs |
| `timeout` | API request timeout in seconds |
| `token_levels` | Max-token limits tried on truncation (ascending order) |
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
| `-h, --help` | Show this help message |

## Resume / Continue

By default, re-running resumes from where you left off — completed models are skipped. Saved state is stored in `benchmark_state.json` inside the output directory and deleted automatically on full completion. Use `--restart` to force a clean run.

## Outputs

After completion the output directory contains:

| File | Format |
|---|---|
| `benchmark-v3-results.md` | Markdown report |
| `benchmark-v3-results.csv` | CSV data |
| `benchmark-v3-results.html` | HTML report |
| `benchmark-v3-results.pdf` | PDF report (requires `fpdf2`) |
| `logs/*.log` | Per-model request/response logs |

## Reports

Each model is scored on two tasks — **Code** (implement a concurrent rate limiter) and **General** (answer a technical ML question) — across multiple dimensions:

- **Score**: Quality rating (0–20) based on rubric keywords
- **Speed**: Tokens per second (TPS)
- **Latency**: Time to first token (TTFT) for streaming
- **Cost**: Approximate per-model overhead

Results are grouped by phase (code columns first, then general columns).
