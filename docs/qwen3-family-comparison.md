# Qwen3.x Family Comparison — qwen3.6:35b vs qwen3.8:27b vs qwen3.8-flash-next

Analysis of the local benchmark runs from late August 2026, comparing the
`qwen3.6:35b`, `qwen3.8:27b`, and `qwen3.8-flash-next` model families on the
rubric (deterministic) grade, the judge-panel consensus, and the votes of the
`ornith-1.5` judge in particular.

The judge models themselves — ranking, per-judge strengths/blind spots, and
improvement recommendations — are reviewed in
[`judge-panel-analysis.md`](./judge-panel-analysis.md).

Scores below are 0–100: **det** is the plugin rubric grade (percent-v1),
**panel** is the mean usable vote of the run's judge panel per cell, and
**ornith-1.5** is that judge's own mean. Per-model figures are means over the
plugins with both a completed attempt and a usable vote (n = the number of
such cells out of 18–19).

## Scope and sources

Two runs in the window carried the families with an `ornith-1.5` judge. The
raw run artifacts live under `runs/` (git-ignored); this document is the
durable summary.

| Panel | Run | Store | Models | Judges | Plugins |
|---|---|---|---|---|---|
| **A** (primary) | `runs/2026-08-22-sqlite` | `run.sqlite3`, rev 35 | 128 | muse-glimmer-medium-nas:30b-64k · **ornith-1.5-nas:35b-64k** · qwen3.6:27b-32k · qwen3.8-thinking-medium-nas:27b-64k | 19 |
| **B** (validation) | `runs/2026-08-17-nas-and-more-test-changes` | `benchmark_state.json` | 111 | 17 judges (ox-alpha, nemotron-3-ultra anchors; ornith-1.0/1.5-nas, qwen3.6:27b/35b, qwen3.6-general-thinking, qwen3.5:122b, gpt-oss:20b, kat-coder, agents-a1, muse line, lightning-thinking/instruct, qwen3-coder, hy3, minimax, mistral-small4-reasoning) | 18 |

Naming conventions: a bare family name stands for that model's **default
mode** — for qwen3.6 that is the base release `qwen3.6:35b-32k` (thinking on by
default); for qwen3.8:27b and qwen3.8-flash-next the default is the
**instruct** variant. Variants with the same suffix (`thinking-low`,
`thinking-medium`, …) are compared like-for-like across families. Panel B ran
before the `qwen3.8-flash-next` and `qwen3.8-thinking-xhigh-nas` config
changes, so the cross-panel check covers the 35b and 27b families only.

## Family tables

### Panel A — 4-judge run (sqlite rev 35)

| Model | n | det | panel | ornith-1.5 | others |
|---|---:|---:|---:|---:|---:|
| `qwen3.6:35b-32k` (default) | 18 | 74.7 | 95.5 | 98.5 | 94.1 |
| `qwen3.6-general-instruct-nas:35b-64k` | 19 | 70.4 | **82.3** | 86.2 | 81.0 |
| `qwen3.6-general-thinking-nas:35b-64k` | 19 | 66.7 | 97.6 | 98.6 | 97.3 |
| `qwen3.6-coding-thinking-nas:35b-64k` | 19 | 73.5 | 97.9 | 98.7 | 97.7 |
| `qwen3.8-instruct-nas:27b-64k` (default) | 19 | 67.4 | **86.8** | 87.9 | 86.4 |
| `qwen3.8-thinking-low-nas:27b-64k` | 19 | 70.3 | 98.4 | 98.9 | 98.2 |
| `qwen3.8-thinking-medium-nas:27b-64k` | 19 | 69.7 | 97.9 | 98.5 | 97.7 |
| `qwen3.8-opus-instruct:27b-256k` | 19 | 60.9 | 87.3 | 86.6 | 87.6 |
| `qwen3.8-opus-thinking-low:27b-256k` | 19 | 72.7 | 89.7 | 92.4 | 88.8 |
| `qwen3.8-opus-thinking-medium:27b-256k` | 19 | 69.8 | 88.0 | 88.9 | 87.7 |
| `qwen3.8-opus-thinking-xhigh:27b-256k` | 19 | 63.9 | 89.4 | 91.8 | 88.6 |
| `qwen3.8-flash-next-instruct:177b-64k` (default) | 19 | 70.3 | 90.1 | 91.5 | 89.7 |
| `qwen3.8-flash-next-thinking-low:177b-64k` | 19 | **76.0** | 98.1 | 99.4 | 97.7 |
| `qwen3.8-flash-next-thinking-medium:177b-64k` | 19 | 75.7 | 98.1 | **99.6** | 97.6 |
| `qwen3.8-flash-next-thinking-xhigh:177b-64k` | 16¹ | 60.2 | 87.0 | 89.5 | 86.2 |

¹ Three cells never completed (interrupted run). Panel A coverage is otherwise
complete; ornith-1.5 voted on every listed cell except two of the 18 for the
35b default.

### Panel B — 17-judge run (json state) — cross-panel check

| Model | n | det | panel | ornith-1.5 |
|---|---:|---:|---:|---:|
| `qwen3.6:35b-32k` (default) | 18 | 70.2 | 89.9 | —² |
| `qwen3.6-general-instruct-nas:35b-64k` | 18 | 69.6 | 86.2 | 91.2 |
| `qwen3.6-general-thinking-nas:35b-64k` | 18 | 83.6 | 95.9 | 98.0 |
| `qwen3.6-coding-thinking-nas:35b-64k` | 18 | 75.2 | 95.1 | 97.4 |
| `qwen3.8-instruct-nas:27b-64k` (default) | 18 | 70.9 | 87.5 | 88.5 |
| `qwen3.8-thinking-low-nas:27b-64k` | 18 | 60.5 | 97.4 | 99.6 |
| `qwen3.8-thinking-medium-nas:27b-64k` | 18 | 64.9 | 95.1 | 97.3 |
| `qwen3.8-thinking-xhigh-nas:27b-64k`³ | 18 | 42.5 | **63.6** | 64.7 |
| `qwen3.8-opus-instruct:27b-256k` | 18 | 56.1 | 83.2 | 88.3 |
| `qwen3.8-opus-thinking-low:27b-256k` | 18 | 59.4 | 83.4 | 87.1 |
| `qwen3.8-opus-thinking-medium:27b-256k` | 18 | 52.5 | 79.1 | 80.3 |
| `qwen3.8-opus-thinking-xhigh:27b-256k` | 18 | 69.9 | 92.3 | 95.3 |

² ornith-1.5 did not cover `qwen3.6:35b-32k` in panel B (13–17 judges per cell).
³ The NAS xhigh variant ran only in panel B; it was dropped from the config for
the next run (`#qwen3.8-thinking-xhigh-nas:27b-64k` is commented out).

### Like-for-like slots (Panel A)

| Slot | qwen3.6:35b | qwen3.8:27b | qwen3.8-flash-next |
|---|---|---:|---:|
| default (plain/instruct) | **74.7 / 95.5 / 98.5** | 67.4 / 86.8 / 87.9 | 70.3 / 90.1 / 91.5 |
| thinking-low | — | 70.3 / 98.4 / 98.9 | **76.0 / 98.1 / 99.4** |
| thinking-medium | 66.7 / 97.6 / 98.6 | 69.7 / 97.9 / 98.5 | **75.7 / 98.1 / 99.6** |
| thinking-xhigh | — | 63.9¹ / 89.4 / 91.8 | 60.2 / 87.0 / 89.5 |

¹ `qwen3.8-opus-thinking-xhigh:27b-256k`. Cells show `det / panel / ornith-1.5`.

## Does the ranking hold across judge panels?

Yes — every structural result reproduces with the 17-judge panel, which
included the ox-alpha and nemotron-3-ultra anchors:

- **Thinking ≫ instruct within each family.** Adding thinking to the 27b line
  lifts judged quality from ~87 to ~95–98 in *both* panels; the same holds for
  flash-next (90.1 → 98.1). Ornith-1.5 is not the driver — ox-alpha, nemotron,
  qwen3.6:27b, and the rest of the 17-judge panel place the thinking variants
  on top too.
- **The default/instruct models carry the real, judge-confirmed weaknesses**
  (instruction-following, data-transformation, tool-calling), not just rubric
  misses: qwen3.8-instruct-nas panel 86.8 (A) / 87.5 (B), general-instruct
  82.3 (A) / 86.2 (B) — both are the weakest members of their families in
  both panels.
- **The opus 256k line trails the NAS 64k thinking line** in both panels
  (A: 98 vs 88–90; B: 95–97 vs 79–92). The *internal ordering* of the opus
  variants is noisy across attempts (opus-thinking-xhigh: 89.4 in A vs 92.3
  in B; opus-thinking-medium: 88.0 vs 79.1) — treat intra-opus comparisons as
  provisional; the opus-below-NAS-thinking gap is stable.
- **qwen3.6-general-thinking and coding-thinking top the 35b family in both
  panels** (95–98), general-instruct is last, and the plain default sits in
  between. Ornith-1.5's per-model means reproduce the panel ordering in B
  wherever it voted.
- **`qwen3.8-thinking-xhigh-nas` was a genuine dud** — judged 63.6 with det
  42.5 in panel B — which is why the next run dropped it in favor of the opus
  xhigh variant. This is an example of the judge panel catching what a rubric
  average alone would have obscured.

Rubric (det) numbers are *not* stable across the two runs (single attempts;
plugin sets differ by one plugin and scores like general-thinking's det swing
from 66.7 in A to 83.6 in B). Judge consensus is the stable signal; det should
be read as output-contract conformance on that specific attempt.

## Where each family excelled and found difficulty

**Judge consensus saturates at the top** (all thinking variants 95.5–98.4 in A)
so the rubric grade, not the panel, does the separating. Judges consistently
grade 20–30+ points above det on content that is high quality but fails exact
format/heading contracts; det zeroes are frequently format misses rather than
quality failures (e.g. qwen3.8-thinking-medium's PRD det 0 while the panel and
ornith-1.5 both scored it 100; the response used `## 1. Executive Summary`
where the checker requires `## Executive Summary`).

### qwen3.6:35b family — best default; weakest instruct member
- **Excels:** the plain default has the best rubric discipline of any default
  (74.7) with perfect det on instruction-following, long-context, multi-step,
  data-transformation and rate-limiter, and panel quality on par with the
  thinking variants (95.5, ornith-1.5 98.5). coding-thinking is the most
  balanced member (73.5 / 97.9).
- **Difficulty:** format failures on the debug family (debug-consistency det 0
  at unanimous judge ~99), thin structured/design outputs (orchestration 50,
  software-architecture 46, error-recovery 44 det), and one **genuine
  concurrency bug in event-processor** (det 38; muse 20 and qwen3.8-thinking
  40 identified the broken dedup mapping — ornith-1.5's 95 missed it, see
  below). general-thinking and coding-thinking both det-zero the debug
  tasks. general-instruct is the family's weak point (panel 82.3): judges
  flagged real content failures on long-context (27.5), data-transformation
  (41), and instruction-following (54).

### qwen3.8:27b family — thinking variants excellent, worst format discipline
- **Excels:** thinking-low/medium are top-tier judged quality (98.4/97.9) with
  ornith-1.5 near-unanimous (98.9/98.5) and full det 100s on
  data-transformation, debug-traversal, rate-limiter and instruction-following;
  wireframes/decomposition/error-recovery are rubric-perfect too.
- **Difficulty:** the family has the largest judge–rubric gap in the set.
  Thinking variants det-score 0–22 on the long-form structured plugins
  (prd-creation 0–5, software-architecture 0–22, orchestration 19–63,
  code-review 0, tool-calling 20) despite 90–100 judge consensus — heading and
  section-format misses, not weak prose. The default instruct model's
  weaknesses are genuine: instruction-following 35/34 (det/panel agree),
  data-transformation panel 31 despite det 75 (all four judges name the same
  refunded-order/sorting errors), code-review det 0, multi-turn det 0. The opus
  256k line under-delivers relative to NAS 64k across both rubric and judges.

### qwen3.8-flash-next family — best of the three, and its own cautionary tale
- **Excels:** thinking-low and thinking-medium are the best models in this
  comparison by both measures (det 76.0/75.7, panel 98.1/98.1, ornith-1.5
  99.4/**99.6** — the highest ornith votes anywhere in the tables). Full det
  100s on the contract plugins, error-recovery 100 (vs the 27b-medium's 20),
  code-review 87 (vs 0–60 for the smaller thinking siblings), and *much*
  shallower format damage on long-form outputs than the 27b line (prd 23–41
  vs 0–5, wireframes 80–85 vs 43–55, software-architecture 20–35 vs 0–22).
- **Difficulty:** the default instruct model is mid — judges liked its
  code-review content (83.5 panel, ornith-1.5 92) while det gave 33, and it
  shares the instruct-mode data-transformation failure (det 80, judges 25).
  **thinking-xhigh is the failure mode**: det 60.2, panel 87, only 16/19
  cells — truncated documents (software-architecture cut mid-diagram, judges
  25–42), format zeroes on debug-traversal/long-context, and an all-zero
  rate-limiter cell. Ornith-1.5 still scored xhigh's tool-calling 100 while
  the rest of the panel averaged 63 (see blind-spot section).

## The ornith-1.5 judge: behavior and verified blind spots

Both runs' efficacy analyses rank ornith-1.5 as the best non-anchor judge
(panel A: 0.965 leave-one-out consensus correlation, 0.7% unilateral outliers;
panel B: MAD 3.35 vs the ox-alpha anchor, tighter than the anchors' mutual
agreement, and the fleet's best leaderboard reproduction). On these families
it is consistently the most generous member of the panel — it votes roughly 1–5 points
above the other judges' per-model mean in A and B alike, and about +15 to +32
above det (A: +16…+32), because its requirement-checklist rationales reward
content quality regardless of rubric format. When the panel lands low for a real
reason (data-transformation errors, truncation) ornith-1.5 goes low too: its
divergences are almost all *upward*.

To verify whether that upward bias is a task-level blind spot, its votes were
correlated against the deterministic rubric per plugin across the full panel-A
run (128 models), with panel B as a check (figures: Pearson r, and the share
of rubric-failing cells — det ≤ 40 — that the judge still scored ≥ 85):

| Plugin | ornith-1.5 (A) | best other judge (A) | ornith-1.5 (B) |
|---|---:|---:|---:|
| event-processor | 0.69 · 19% | 0.73 (qwen3.8-thinking) · 12% | 0.75 · 17% |
| tool-calling | **0.46 · 43%** | **0.61 (qwen3.8-thinking) · 31%** | 0.62 · 35% |
| code-review | 0.47 · 24% | 0.62 (qwen3.8-thinking) · 6% | 0.42 · 12% |
| instruction-following | 0.91 · 2% | 0.91 · 2% | 0.92 · 2% |
| rate-limiter | 0.78 · 0% | 0.83 · 0% | 0.87 · 0% |
| data-transformation | 0.73 · 1% | 0.76 · 1% | 0.69 · 1% |

- **event-processor: no run-scale blind spot.** Correlation (0.69 A / 0.75 B)
  is mid-pack and close to the best judge; the flagged case is per-case, not
  per-task: on qwen3.6:35b-32k's buggy event-processor, ornith-1.5 scored 95
  and its rationale praised dedup code that demonstrably broke, while muse (20)
  and qwen3.8-thinking (40) plus the rubric (38) caught it.
- **tool-calling: a moderate task-level blind spot, confirmed.** Ornith-1.5
  tracks the rubric worst of the panel (0.46 in A) and is the most
  over-generous (43% of rubric-failing tool-calling cells ≥ 85, vs 31% for the
  best judge; 35% in panel B). Named cases: qwen3.8-flash-next-thinking-xhigh
  (ornith 100 vs panel 63.3, det 40) and tiel-coder-qwen3.6 (95 vs 63.3, det
  20). The pattern is over-crediting well-written responses that miss exact
  tool-call/format requirements — the reverse of muse's known hallucinated
  "no tool calls present" false zero on the same plugin.
- **code-review is a secondary weak spot** (0.47/24% vs 0.62/6% for the best
  judge) in the same direction.

Structural judge–rubric decoupling on software-architecture (r ≈ 0.11–0.18,
53–75% of rubric-failing cells ≥ 85 for *every* judge), prd-creation,
debug-traversal and orchestration is systemic — format-heavy long-form plugins
— and is not ornith-specific.

## Bottom line

1. **qwen3.8-flash-next thinking-low/medium is the best family** — top rubric
   discipline *and* top judged quality, with the smallest format penalty of
   the thinking variants. Avoid its xhigh and instruct defaults.
2. **qwen3.6:35b's plain default is the best as-delivered model** (best det of
   any default, judge quality beside the thinking tier); coding-thinking is
   its balanced alternative. general-instruct is the weakest member of any
   family judged.
3. **qwen3.8:27b thinking-low/medium match the top tier in content quality**
   but pay the largest format-conformance penalty; its instruct default has
   the genuine task-following failures, and the opus 256k line trails its NAS
   64k siblings.
4. **Ranking conclusions are robust to the judge panel** — all structural
   results (thinking ≫ instruct, opus < NAS thinking, general-instruct
   weakness, thinking-xhigh dud) reproduce with the 17-judge panel.
5. **ornith-1.5 is a safe primary judge** but should be paired with a
   rubric-disciplined coder (qwen3.8-thinking-medium) when grading
   tool-calling and code-review cells, where its generosity is systematic.

## Caveats

- Single attempt per cell; deterministic grades vary run-to-run even for the
  same model. Judge consensus is the stable ranking signal.
- Panel A's final leg was interrupted: judge coverage is 3–4 votes per cell
  (some cells lack qwen3.6:27b or a completed attempt), and
  qwen3.8-flash-next-thinking-xhigh is missing three cells.
- Flash-next/opus models ran on the AI Server, NAS qwen3.6/3.8 variants on the
  NAS, and qwen3.6:35b-32k on the Gaming PC — hardware/quantization differ
  between sources.
- Det zeroes are sometimes pure heading/format artifacts; read each cell's
  rubric JSON before treating a zero as a quality failure.
- `runs/` is git-ignored; provenance reports (`judge-efficacy-report.md`,
  `persona-model-report.md`, `task-model-rankings.md`) live next to the run
  stores referenced above and are not checked in.
