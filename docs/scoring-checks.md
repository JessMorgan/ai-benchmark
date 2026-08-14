# Scoring checks and evaluator design

Challenge plugins use task-specific native rubrics. The benchmark converts a
plugin's native score to the public 0–100 percentage exactly once at the
`benchmark.plugin` boundary. A rubric's `earned` values and `max` values remain
native task points in offline evaluation and diagnostics.

## Evaluation order

Evaluators should follow this order:

1. Parse the response into a typed representation where the task has one:
   JSON/YAML, Markdown sections, list records, tool-call envelopes, workflow
   edges, Python ASTs, or executable source.
2. Validate the response contract: required sections, exact fields, unique
   records, code-block count, or API definitions.
3. Check each criterion against the smallest relevant unit—the section,
   record, finding, code block, or parsed graph—not the entire response.
4. Run a bounded behavioral harness for code-shaped tasks. Execution evidence
   records `passed`, `failed`, `timeout`, the error, and the isolation mode.
5. Use narrow regular expressions only for syntax recognition or small pieces
   of explanatory prose. A keyword mention by itself should not establish a
   substantive criterion.

`plugins/challenges/_analysis.py` contains Markdown section, list, fence,
and Mermaid-subset helpers. `plugins/challenges/_validators.py` contains
Python AST, structured-data, typed-tool-call, workflow-graph, and section
validators. `plugins/challenges/_rubric.py` keeps score arithmetic, evidence,
negative findings, and execution diagnostics reconstructable.

## Executable code tasks

The code-generation challenges define exact APIs and execute dependency-free
assertion harnesses through
`plugins/challenges/_execution.py::run_python_check`:

- Podman is preferred with `--network=none`, a read-only filesystem, an
  unprivileged user, dropped capabilities, resource limits, and a timeout.
- If Podman is unavailable, a resource-limited local process is used and the
  evidence explicitly says `isolation: local-restricted`. This fallback is
  portable but is not a security boundary for untrusted code.
- A runtime failure is a failed behavior check, not a skipped score.
- The harnesses test both happy and failure paths; a syntactically valid stub
  cannot receive full credit.

The behavior-dominant tasks are:

- **Multi-Step**: exact per-block function contract plus boundary behavior.
- **Rate Limiter**: all three class APIs, independent clients, window limits,
  reset/cleanup, invalid configuration, and concurrent calls.
- **Error Recovery**: injectable asynchronous providers, concurrent attempts,
  exceptions, malformed error payloads, all-success/partial/all-failure, and
  provider details in the terminal exception.
- **Event Processor**: constructor/input validation, concurrent unique events,
  duplicate suppression, transient retry, permanent failure, ordered results,
  and retry counts.

Debug Traversal also executes the proposed threshold fix, while its other
points measure the root-cause explanation and side-effect analysis.

## Structured and sectioned tasks

Sectioned documents are checked using exact normalized headings plus explicitly
documented aliases. Content in a Notes, Examples, or unrelated section does
not satisfy a required section. The main sectioned evaluators are:

- **PRD Creation**: section-local measurable goals, personas, user stories,
  functional requirements, NFR categories, quantitative KPIs, competitors,
  milestones, and risks.
- **Software Architecture**: required document sections, section-local
  architecture/data/API/scale/security/resilience/observability checks, and
  consistency between capacity claims and supporting mechanisms.
- **MoE vs Dense**: required comparison sections, routing/load-balancing
  equations, training/inference implications, named comparisons, references,
  and quantitative trade-offs.
- **Wireframes**: canonical distinct screens, per-screen purpose and visual
  structure, components, known-screen navigation edges, interaction notes,
  and feature coverage from screen content.
- **Multi-Turn Conversation**: three distinct revision blocks, section-local
  feedback incorporation, exact block structure, and a change summary.

The evaluator records missing sections as diagnostics and negative findings;
merely mentioning a heading's vocabulary elsewhere is insufficient.

## Typed and exact-output tasks

- **Instruction Following** parses each order line, verifies filters,
  deterministic tie-breaking, transformed fields, summary arithmetic, and the
  exact output contract.
- **Reasoning** checks the four final fields and independently checks numbered
  deductions, derived assignments, ownership, and the priority chain. The
  puzzle's correct 09:30 answer is Search, owned by Ben, priority P5.
- **Long Context** requires the joined answer and the F02/F05/F09 evidence
  chain from a large deterministic distractor set.
- **Structured Output** parses one JSON/YAML object, validates nested types and
  constraints, rejects multiple candidates, checks exact top-level keys, and
  penalizes explanatory text outside a fenced object.
- **Tool Calling** parses typed `<tool_call>` envelopes, validates names,
  required argument types/values, exact cardinality/order, a complete plan,
  and a final synthesis.
- **Code Review** supports valid JSON and understandable bullet findings, but
  requires independent findings that each connect a concrete defect to its
  remediation and cite the relevant source construct. One finding cannot be
  reused as credit for every defect.

## What remains intentionally limited

Deterministic checks can establish structure, syntax, contracts, and many
behavioral facts; they cannot fully judge the quality of an open-ended product
or architecture argument. The benchmark therefore keeps qualitative criteria
small and explainable rather than pretending that word counts are semantic
understanding. Future improvements should be adversarial probes or an
independent, opt-in judge for the genuinely open-ended remainder—not a return
to global keyword scoring.

Every challenge has adversarial tests for malformed structure, wrong answers,
keyword-only responses, duplicate data, and—where applicable—failed execution.
Run them with:

```sh
uv run pytest tests/ plugins/challenges/ plugins/outputs/ -q
```
