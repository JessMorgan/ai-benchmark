# AI Benchmark Documentation

Welcome to the AI Benchmark documentation. This project is a multi-source, plugin-based benchmark for evaluating code generation and general reasoning capabilities of LLMs served through OpenAI-compatible APIs.

## What is AI Benchmark?

AI Benchmark lets you:

- Run a suite of benchmark tasks (plugins) against multiple LLMs in parallel.
- Compare models across quality scores, latency, throughput, and more.
- Generate Markdown, CSV, HTML, and PDF reports automatically.
- Resume interrupted runs from saved state.
- Configure per-model API sources and request parameters.

## Documentation Index

| Document | Description |
|---|---|
| [Getting Started](./getting-started.md) | Install, configure, and run your first benchmark |
| [Configuration](./configuration.md) | Full reference for `benchmark-config.json` |
| [CLI Reference](./cli.md) | Command-line options and examples |
| [Plugins](./plugins.md) | How the plugin system works and how to select plugins |
| [Plugin: Rate Limiter](./plugins/rate-limiter.md) | Concurrent rate-limiter code generation task |
| [Plugin: MoE vs Dense](./plugins/moe-dense.md) | MoE vs dense architecture analysis task |
| [Plugin: Code Review](./plugins/code-review.md) | Code review and issue identification task |
| [Plugin: Orchestration](./plugins/orchestration.md) | Multi-step workflow orchestration task |
| [Plugin: Tool Calling](./plugins/tool-calling.md) | Tool-use and agent routing task |
| [Plugin: Structured Output](./plugins/structured-output.md) | JSON/YAML structured output task |
| [Development](./development.md) | How to write plugins, run tests, and contribute |
| [Architecture](./architecture.md) | High-level design of the benchmark system |

## Quick Links

- Main entry point: [`ai-benchmark.py`](../ai-benchmark.py)
- Core benchmark logic: [`benchmark_core.py`](../benchmark_core.py)
- Plugin base class: [`benchmark_plugin.py`](../benchmark_plugin.py)
- Built-in plugins: [`plugins/`](../plugins/)
- Unit tests: [`tests/`](../tests/)
