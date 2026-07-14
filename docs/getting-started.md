# Getting Started

This guide walks you through installing AI Benchmark, configuring it, and running your first benchmark.

## Prerequisites

- Python 3.10 or newer
- `pip` or another Python package manager
- One or more OpenAI-compatible API endpoints

## Installation

1. Clone or download the repository.
2. Install the required packages:

```sh
pip install -r requirements.txt
```

If you want PDF report generation, `fpdf2` is required (already listed in `requirements.txt`).

## Generate a Default Config

The easiest way to start is to dump a default configuration file:

```sh
python ai-benchmark.py --dump-default-config > benchmark-config.json
```

This creates a template with example sources and models. You can also discover models automatically from a running server:

```sh
python ai-benchmark.py \
  --dump-default-config \
  --base-url http://localhost:11434 \
  --api-key sk-xxx > benchmark-config.json
```

## Edit the Config

Open `benchmark-config.json` and update the `sources` and `models` sections for your environment.

Example:

```json
{
  "output_dir": "benchmark-results",
  "timeout": 600,
  "token_levels": [16384],
  "sources": {
    "Local Server": {
      "api_url": "http://localhost:11434/chat/completions",
      "headers": {
        "Authorization": "Bearer ${LOCAL_API_KEY:sk-fallback}",
        "Content-Type": "application/json"
      }
    }
  },
  "models": {
    "llama3:8b": "Local Server"
  }
}
```

See [Configuration](./configuration.md) for the full config reference.

## Run the Benchmark

```sh
python ai-benchmark.py
```

The benchmark will:

1. Load the config.
2. Discover and select plugins.
3. Queue each model against its configured source.
4. Run plugins in parallel (respecting `plugin_thread_limit`).
5. Save state after each model so runs can resume.
6. Generate `results.md`, `results.csv`, `results.html`, and `results.pdf` in the output directory.

## View Results

After the run finishes, open the output directory (default `benchmark-results/`):

```sh
ls benchmark-results/
```

You should see:

- `results.md` — Markdown report
- `results.csv` — CSV data
- `results.html` — HTML report
- `results.pdf` — PDF report
- `logs/` — Per-model request/response logs
- `benchmark_state.json` — Resume state

## Resume a Run

By default, re-running the benchmark resumes from the saved state. Completed models are skipped; failed models are retried. Use `--restart` to discard state and start over.

```sh
# Resume
python ai-benchmark.py

# Restart from scratch
python ai-benchmark.py --restart
```

## Next Steps

- Learn about all config options in [Configuration](./configuration.md).
- Explore the CLI in [CLI Reference](./cli.md).
- Read about the available benchmark tasks in [Plugins](./plugins.md).
