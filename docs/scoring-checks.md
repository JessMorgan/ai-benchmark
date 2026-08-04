# Beyond regex: richer rubric checks for benchmark challenges

This document explains how this codebase scores benchmark responses today
(almost entirely regex over string content) and where we should grow.

## What we do today

All `plugins/challenges/*.py` files expose an `evaluate(response_text)`
method that returns `(score, criteria_list)`.  Internally they almost
exclusively use:

- `plugins/challenges/_rubric.py::Rubric.eval_regex` which adds points
  for each `(regex, points)` pair whose pattern matches anywhere in the
  response.
- Ad-hoc `re.findall` / `re.search` calls that count keyword co-occurrences.

This works for "did the model mention X" style criteria, but is blind to:

- **Quality vs. presence**: a hand-wavy mention of "circuit breaker" earns
  the same point as a correct, justified implementation choice.
- **Numeric reasoning**: regex can detect "30%" in any context but cannot
  tell whether a percentage is a baseline, a target, or unrelated.
- **Structural correctness**: regex cannot tell whether a Mermaid diagram
  parses, whether a user story is in Given/When/Then form, or whether a
  GraphQL schema is well-typed.
- **Inter-criterion consistency**: a claim of "100% SLO uptime" is not
  cross-checked against the failure-mode discussion elsewhere in the doc.

Specifically, see this run: all 8 agents scored **0/3** on
`Multiple screens present` in the Wireframes task — not because every
agent actually wrote only 0 or 1 screens, but because the original regex
only counted headers that contained the literal words `screen` /
`wireframe` / `page`.  A model that wrote `## 1. Dashboard` /
`### Focus Session` was miscounted.  The fix in
`plugins/challenges/wireframes.py` is the same lesson we keep hitting:
regex matches what you point it at, not what you mean.

## Strategies to add, in priority order

### 1. Structural parsing (cheap, deterministic)

We already parse `User Stories` in `prd_creation.py` via regex — that is
the simplest version of structural parsing.  We should grow it into:

- **Mermaid diagrams** (used heavily by `software-architecture`):
  extract the `mermaid` fenced block, parse with a Mermaid tokenizer and
  score on (graph well-formed, components match PRD, edge count &ge; N).
  Library options: write a small ad-hoc tokenizer targeted at the
  `graph TD` / `sequenceDiagram` / `classDiagram` forms used in our
  prompts, or wrap an existing tool such as
  [`mermaid-py`](https://pypi.org/project/mermaid-py/).  Hand-rolled is
  usually enough because our agents emit a constrained subset.
- **JSON / YAML code blocks**: extract fenced ` ```json ` blocks and
  validate against a JSON Schema.  This is cleaner than fuzzing the rest
  of the document.  Library: `jsonschema` (already in `requirements.txt`
  family of schema validators).
- **Markdown tables**: extract pipe-delimited tables, parse into rows,
  and assert structure (e.g., KPI table MUST have columns for `metric`,
  `baseline`, `target`, `measurement`).  No external library needed.
- **Acceptance-Criteria lists**: parse bullet lists under each "User
  Story" heading; assert at least one **negative-path** bullet
  (regex: points containing "when ... fails", "if ... returns error",
  "should NOT", etc.).  This is what the new PM prompt tells the agent
  to produce, so we should be able to *check* it.

### 2. Field extraction + cross-document consistency

Several criteria in `software-architecture.py` are about claims being
*consistent across sections*, not about whether they appear at all:

- **"SLO claims match failover strategy"** — search the doc for the SLA
  percent (e.g., `99.9%`); then verify the Resiliency / Failure Modes
  section mentions at least one concrete mechanism (circuit breaker,
  retry/backoff, dead-letter queue, multi-region failover) that would
  support that target.  This is regex-plus-bool, but it captures the
  intent of an architecture review.
- **"Capacity claim vs scaling mechanism"** — search for `1.?000.?000`
  or `million DAU`; verify Scalability & Capacity Planning mentions
  concrete numbers (RPS per service, sharding strategy, autoscaling
  rules).

### 3. LLM-as-judge (most expensive, highest ceiling)

When a criterion is genuinely about quality, the cheapest faithful
check is to ask another LLM to grade.  Pattern:

1. Reserve a tiny, single-task rubric prompt (no agent persona).
2. Pass `(criterion_text, points, response_excerpt)`.
3. Ask the judge for a strict integer score 0..max with reasoning.
4. Aggregate across criteria into the existing `Rubric.add_criterion`.

This is implemented in several open-source agentic eval frameworks
(Patronus, Promptfoo, RAGAS).  In this codebase, the same
`stream_request` / `nonstream_request` plumbing in `benchmark_http.py`
already works — call it recursively with the criterion as the prompt.

The two big caveats:

- **Cost**: this multiplies every benchmark run by `num_criteria` model
  calls per agent.  We should gate this behind a config flag
  (`judge_model: true` per challenge), and use it only on criteria
  marked `_quality=True` in the rubric.
- **Self-bias**: if the judge is the same model as the agent, scores
  drift upward.  Always use a different source / model for judging.  The
  `sources:` field in `benchmark-agents.yml` already supports this.

### 4. Code-execution checks for code-shaped tasks

For `rate-limiter` and similar, we should actually run the agent's code
in a sandbox and check its behavior.  This codebase already has
`execute_python` for the agents, so there's nothing structural blocking
us from running the *generated* Python in a `pytest` harness and
scoring on `tests passed / total tests`.  Concretely:

- Parse `code` from fenced ```python blocks.
- Write to `/tmp/<ctx>/<model>_<task>.py`.
- `subprocess.run` with a timeout, capture stdout/stderr.
- Run against rubric-baked test cases (concurrency correctness,
  edge-case eviction, type-hint coverage via `mypy --strict`).
- Map `tests_passed` to points.

This replaces `re.findall(...) == 2` heuristic checks for code quality
with real, falsifiable tests.

### 5. Embedding-based similarity (for open-ended "covers X topic" checks)

When we want to award points for covering a topic without rewarding
keyword stuffing, embed both the response and a short canonical
reference and threshold cos-similarity.  Useful for "Did the agent
genuinely talk about Resiliency?" vs "Did it mention the word
"resilience" once?"

Libraries: `sentence-transformers`, `FlagEmbedding`, or the embedding
endpoint that is presumably already exposed by the AI Server.

### 6. Adversarial follow-ups (most discriminating)

The radical move: hold out a small "probe" question asking the agent
to **defend** an architecture under a specific failure, and grade the
defense.  Example probes for the software-architecture task:

- "Spotify returns 502 for 90 minutes.  Walk through what your users
  experience.  Which SLO is violated first?"
- "GDPR deletion request arrives.  Trace the deletion through every
  store you proposed, including the analytics warehouse.  What's the
  RTO?"
- "An engineer adds a new microservice that fans out to every existing
  service for a cache warm.  What fails, in what order, and what is
  your blast radius?"

Each probe is graded with one of the techniques above (regex +
structural parsing are usually enough), and aggregated into the same
`max_score` total.  This separates "architecture that reads well" from
"architecture that survives contact with operations."

## Trade-offs to be honest about

Every richer check has a cost.  Document them so future plugin authors
pick deliberately:

| Strategy | Cost (compute) | Cost (engineering) | Discriminating power | Reproducibility |
|---|---|---|---|---|
| Regex (current) | Trivial | Trivial | Low | High |
| Structural parsing | Low | Medium | Medium | High |
| Field consistency | Low | Medium | Medium-high | High |
| LLM-as-judge | High | Medium | High | Lower (model-dependent) |
| Code execution | Low | Medium (sandbox) | High | High |
| Embedding similarity | Medium | Low | Medium | Medium |
| Adversarial probes | High (extra model call) | Medium | Very high | Medium |

## Recommended rollout

1. **Quick wins (do these first)**:
   - Fix the wireframes `Multiple screens present` regex (already done).
   - Add a Mermaid validator for the architecture diagram criterion.
   - Parse the KPI markdown table in `prd_creation.py` and verify
     baseline + target + measurement columns are present.
2. **Medium effort**:
   - Add `eval_struct_regex` helpers to `_rubric.py` for fenced-block
     extraction.
   - Turn the cross-section consistency checks (SLO vs Resiliency)
     into actual assertions.
3. **Phase 2** — pilot LLM-as-judge on one criterion of one task, behind
   a config flag, and A/B compare against the regex score.  Promote only
   if the judge improves discrimination by a meaningful margin without
   destabilizing the leaderboard.
4. **Phase 3** — add `probe` follow-ups for high-stakes tasks
   (software-architecture, PRD).  These give the most insight into
   whether the agent's reasoning is robust.

## Implementation sketch for a richer plugin

A future plugin would look like:

```python
class SoftwareArchitectureV2Plugin(BenchmarkTaskPlugin):
    def evaluate(self, response_text):
        rubric = Rubric(self.max_score)

        # Existing regex-backed criteria stay.
        rubric.add_criterion("Architecture & Patterns", 3.0,
                             self._score_architecture_patterns(response_text))

        # New structural check: parse Mermaid diagram, assert component count.
        rubric.add_criterion("Diagram is well-formed Mermaid", 1.5,
                             self._score_mermaid_diagram(response_text))

        # New consistency check: SLO claims match Resiliency mechanisms.
        rubric.add_criterion("SLO/Resiliency consistency", 1.5,
                             self._score_slo_resiliency_consistency(response_text))

        # Optional: LLM-as-judge on hardest criterion.
        if self._judge_enabled:
            rubric.add_criterion("Trade-off reasoning quality", 2.0,
                                 self._judge_trade_offs(response_text))

        return rubric.results()

    def _score_mermaid_diagram(self, text):
        from pymeraid import parse
        match = re.search(r'```mermaid\n(.+?)```', text, re.DOTALL)
        if not match:
            return 0.0
        try:
            graph = parse(match.group(1))
            nodes = len(graph.nodes)
            edges = len(graph.edges)
            if nodes >= 6 and edges >= 5:
                return 1.5
            if nodes >= 3:
                return 0.75
            return 0.0
        except ParseError:
            return 0.0
```

This composes with the existing infrastructure: each criterion still
feeds `Rubric.add_criterion`, so `results.md` and `results.csv` keep
their format, and the leaderboard view is unchanged.

## Pointers to the rest of the codebase

- `plugins/challenges/_rubric.py` — extend with `eval_struct`,
  `eval_consistency`, `eval_judge` helpers so every plugin can opt in.
- `benchmark_http.py::stream_request` — already what an LLM-as-judge
  call would use; no new plumbing required.
- `benchmark_core.py::_run_plugin_task` — accepts `system_prompt` and
  `is_agent`; a judge model should pass `is_agent=False` and a
  system_prompt describing the rubric.
- `benchmark_state.py::update` — already stores per-criterion `rubric`
  breakdown; nothing to migrate.

## Final note

Regex is fast and explainable, which are real virtues for a benchmark.
We should keep it as the *first-pass* filter on every criterion.  But
for the criteria we actually want to differentiate agents on
(Resiliency reasoning, Trade-off quality, KPI baseline/target framing),
we should layer at least one of the strategies above on top.  The
fixes in this round already take a first step toward that:
`Multiple screens present` now counts both explicit and named screens,
and `Quantitative trade-off` requires specific numeric, side-by-side
comparisons — both of which are still regex, but with much sharper
intent than the original keyword co-occurrence checks.
