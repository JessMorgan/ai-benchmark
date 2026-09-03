# Plugins

AI Benchmark uses a plugin architecture. Each plugin defines a benchmark task, a prompt, a scoring function, and metadata. Challenge plugins are discovered automatically from the `plugins/challenges/` directory and output plugins from the `plugins/outputs/` directory.

## Built-In Plugins

| ID | Name | Version | Internal Max Score | Streaming |
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

    def get_request_params(self, global_config: dict) -> dict: ...  # default: {}

    def get_response_schema(self) -> dict | None: ...  # default: None

    @property
    def judge_instructions_version(self) -> str: ...  # default: "1.0.0"

    def get_judge_instructions(self) -> str: ...  # default: ""

    def sanitize_for_judge(self, text: str) -> str: ...  # default: identity

    def evaluate(self, response_text: str) -> EvaluationResult: ...

    def score(self, response_text: str) -> float: ...  # native task scale
```

`get_request_params` is an optional hook for task contracts that should be
provided to the model at the API boundary. It defaults to an empty dict so
normal tasks remain unconstrained; `data-transformation` uses it to request a strict JSON schema while scoring schema compliance as only one point beside
its semantic record-processing task. Use `data-transformation` in all new and existing configurations.

`get_response_schema` is optional diagnostic metadata. When present, the core
records response-schema validity and request compatibility separately from the
plugin's primary score; it does not imply that a successful response proves
provider-side enforcement.

`judge_instructions_version` and `get_judge_instructions` are optional hooks
for concise, domain-specific semantic-judge guidance. The shared judge
protocol remains authoritative; plugin guidance must not add hidden scoring
requirements or override the task. The instruction version is kept separate
from the plugin's benchmark version so judge guidance can evolve without
necessarily rerunning deterministic benchmark tasks.

`sanitize_for_judge` is an optional hook applied by the semantic judge
pipeline (`build_judge_prompt` in `benchmark/core.py`) to both the task text
and the candidate answer before they are embedded in a judge prompt.
Plugins whose outputs contain structured fragments that mimic the judge's
required JSON verdict - e.g. the `<tool_call>{...}</tool_call>` blocks in
`tool-calling` - override it to mask those fragments (keeping their
semantic content) so judge models are not hijacked into echoing the format
instead of returning `{"score": ...}`. The default implementation is
identity.

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
- [Data Transformation](./plugins/data-transformation.md)
- [Multi-Step Instructions](./plugins/multi-step.md)
- [PRD Creation](./plugins/prd-creation.md)
- [Wireframes](./plugins/wireframes.md)
- [Software Architecture](./plugins/software-architecture.md)
- [Instruction Following](./plugins/instruction-following.md)
- [Logical Reasoning](./plugins/reasoning.md)
- [Debug Report Consistency](./plugins/debug-consistency.md)
- [Long-Context Retrieval](./plugins/long-context.md)
- [Concurrent Event Processor](./plugins/event-processor.md)
- [Design-Doc Decomposition](./plugins/decomposition.md)

All challenge plugins have dedicated documentation or an explicit source-level
contract. Use `python ai-benchmark.py --list-plugins` for the authoritative
runtime inventory and versions; this table is a checked-in snapshot for quick
reference.

## Data Transformation

`data-transformation` processes a multi-record order feed with historical versions, filtering rules, normalization, deterministic sorting, ranking, and aggregate fields. The strict schema is a one-point contract signal; semantic correctness supplies the remaining 21 points. See [Data Transformation](./plugins/data-transformation.md).

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

**Scoring:** Up to 22 native points. Required headings are matched structurally, and goals, personas, stories, requirements, NFRs, KPIs, competitors, milestones, and risks are checked within their own section; content copied into an unrelated section does not earn that criterion.

### `wireframes`

Tests a model's capability as a frontend architect / UX designer by producing text-based wireframes from a PRD.

**Prompt asks for:**
- At least 4 distinct screens
- Screen names and purposes
- Text-based wireframe representations
- Key UI components
- Navigation flows between screens
- Annotations and interaction notes

**Scoring:** Up to 20 points based on distinct named screen blocks, per-screen structure and components, validated navigation edges, interaction notes, and PRD feature coverage. Global keyword mentions do not substitute for a screen block.

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

**Scoring:** Up to 20 native points: 3 for the required section set and the remainder for section-local architecture/components, data/API, real-time communication, scale/capacity, resiliency, security, and observability checks. Capacity and SLO claims are cross-checked against supporting mechanisms; global keyword mentions do not substitute for the relevant section.
