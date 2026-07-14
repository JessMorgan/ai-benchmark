# Plugins

AI Benchmark uses a plugin architecture. Each plugin defines a benchmark task, a prompt, a scoring function, and metadata. Plugins are discovered automatically from the `plugins/` directory.

## Built-In Plugins

| ID | Name | Max Score | Streaming |
|---|---|---|---|
| `rate-limiter` | Rate Limiter | 20 | Yes |
| `moe-dense` | MoE vs Dense | 15 | No |
| `code-review` | Code Review | 15 | No |
| `orchestration` | Orchestration & Workflow | 16 | Yes |
| `tool-calling` | Tool Calling Agent | 25 | Yes |
| `structured-output` | Structured Output | 20 | No |

## Selecting Plugins

By default, all discovered plugins run. You can limit them with:

- `plugins_whitelist` / `--plugins-whitelist` — run only these
- `plugins_blacklist` / `--plugins-blacklist` — run all except these

You cannot use both whitelist and blacklist at the same time.

## Plugin Base Class

All plugins inherit from `BenchmarkTaskPlugin` in `benchmark_plugin.py`:

```python
class BenchmarkTaskPlugin(abc.ABC):
    @property
    def id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def max_score(self) -> float: ...

    @property
    def supports_streaming(self) -> bool: ...

    def get_prompt(self) -> str: ...

    def get_temperature(self, global_config: dict) -> float | None: ...

    def score(self, response_text: str) -> float: ...
```

## Plugin Lifecycle

1. **Discovery**: `discover_plugins()` scans `plugins/` for `BenchmarkTaskPlugin` subclasses.
2. **Selection**: Whitelist/blacklist filters are applied.
3. **Execution**: For each model, the benchmark calls `_run_plugin_task()` for each active plugin.
4. **Scoring**: The plugin's `score()` method evaluates the model's response.
5. **Reporting**: Scores and metrics are aggregated into reports.

## Streaming vs Non-Streaming

- If `supports_streaming` is `True`, the benchmark first tries the streaming API path and falls back to non-streaming if needed.
- If `supports_streaming` is `False`, only the non-streaming path is used.

## Writing a Plugin

See [Development](./development.md#writing-a-plugin) for a step-by-step guide.

## Per-Plugin Documentation

- [Rate Limiter](./plugins/rate-limiter.md)
- [MoE vs Dense](./plugins/moe-dense.md)
- [Code Review](./plugins/code-review.md)
- [Orchestration](./plugins/orchestration.md)
- [Tool Calling](./plugins/tool-calling.md)
- [Structured Output](./plugins/structured-output.md)
