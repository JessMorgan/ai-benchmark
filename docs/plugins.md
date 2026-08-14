# Plugins

AI Benchmark uses a plugin architecture. Each plugin defines a benchmark task, a prompt, a scoring function, and metadata. Challenge plugins are discovered automatically from the `plugins/challenges/` directory and output plugins from the `plugins/outputs/` directory.

## Built-In Plugins

| ID | Name | Version | Internal Max Score | Streaming |
|---|---|---:|---:|---|
| `code-review` | Code Review | 0.8.1 | 15 | No |
| `debug-traversal` | Debug Traversal | 0.5.1 | 20 | Yes |
| `error-recovery` | Error Recovery | 0.8.0 | 20 | Yes |
| `instruction-following` | Instruction Following | 0.1.1 | 20 | Yes |
| `moe-dense` | MoE vs Dense | 0.7.2 | 17 | No |
| `multi-step` | Multi-Step Instructions | 0.9.1 | 20 | Yes |
| `multi-turn-conversation` | Multi-Turn Conversation | 0.6.0 | 20 | Yes |
| `orchestration` | Orchestration & Workflow | 0.8.1 | 16 | Yes |
| `prd-creation` | PRD Creation | 0.7.0 | 22 | Yes |
| `rate-limiter` | Rate Limiter | 0.7.1 | 20 | Yes |
| `reasoning` | Logical Reasoning | 0.1.1 | 20 | Yes |
| `software-architecture` | Software Architecture | 0.8.1 | 20 | Yes |
| `structured-output` | Structured Output | 0.9.0 | 22 | No |
| `tool-calling` | Tool Calling Agent | 0.10.0 | 25 | Yes |
| `wireframes` | Wireframes | 0.7.2 | 20 | Yes |

## Selecting Plugins

By default, all discovered plugins run. You can limit them with:

- `plugins_whitelist` / `--plugins-whitelist` — run only these
- `plugins_blacklist` / `--plugins-blacklist` — run all except these

You cannot use both whitelist and blacklist at the same time.

## Plugin Base Class

All plugins inherit from `BenchmarkTaskPlugin` in `benchmark/plugin.py`. `max_score` is the internal native rubric maximum; plugin `score()` and `evaluate()` continue to return native task-specific values. The benchmark core normalizes only the public persisted benchmark score to an integer percentage from 0 to 100. Persisted rubric entries use `points` and `total`; offline evaluation remains native.

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

    def evaluate(self, response_text: str) -> EvaluationResult: ...

    def score(self, response_text: str) -> float: ...  # native task scale
```

## Plugin Lifecycle

1. **Discovery**: `discover_plugins()` scans `plugins/challenges/` for `BenchmarkTaskPlugin` subclasses and `discover_output_plugins()` scans `plugins/outputs/` for `BenchmarkOutputPlugin` subclasses.
2. **Selection**: Whitelist/blacklist filters are applied.
3. **Execution**: For each model, the benchmark calls `_run_plugin_task()` for each active plugin.
4. **Scoring**: The plugin's `evaluate()` and `score()` methods produce native task-scale values; the benchmark core normalizes the evaluated score to an integer 0–100 public result and serializes rubric entries as points/total.
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
- [Multi-Step Instructions](./plugins/multi-step.md)
- [PRD Creation](./plugins/prd-creation.md)
- [Wireframes](./plugins/wireframes.md)
- [Software Architecture](./plugins/software-architecture.md)
- [Instruction Following](./plugins/instruction-following.md)
- [Logical Reasoning](./plugins/reasoning.md)

The three additional challenge plugins (`debug-traversal`, `error-recovery`, and
`multi-turn-conversation`) currently keep their prompt and rubric documentation
in the corresponding source modules. Use `python ai-benchmark.py --list-plugins`
for the authoritative runtime inventory and versions; this table is a checked-in
snapshot for quick reference.

## Selected Challenge Plugins

### `prd-creation`

Tests a model's ability to act as a product manager by creating a comprehensive Product Requirements Document (PRD) from a product idea.

**Prompt asks for:**
- Executive Summary
- Problem Statement
- Goals & Objectives
- Target Users & Personas
- User Stories
- Functional Requirements
- Non-Functional Requirements
- Success Metrics / KPIs
- Competitive Analysis
- Timeline / Milestones
- Open Questions / Risks

**Scoring:** Up to 22 points based on the presence, structure, and depth of each PRD section.

### `wireframes`

Tests a model's capability as a frontend architect / UX designer by producing text-based wireframes from a PRD.

**Prompt asks for:**
- At least 4 distinct screens
- Screen names and purposes
- Text-based wireframe representations
- Key UI components
- Navigation flows between screens
- Annotations and interaction notes

**Scoring:** Up to 20 points based on screen coverage, structural wireframe content, UI components, navigation, annotations, and PRD feature coverage.

### `software-architecture`

Tests a model's capability as a backend/coding architect by producing a software architecture document from a PRD.

**Prompt asks for:**
- Executive Summary
- Requirements Summary
- Architecture Style
- Component Diagram / Description
- Data Model
- API Design
- Technology Stack
- Deployment Architecture
- Security Considerations
- Scalability & Performance
- Trade-offs & Decisions

**Scoring:** Up to 20 points based on the presence, structure, and technical depth of each architecture section.
