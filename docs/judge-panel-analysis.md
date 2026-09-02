# LLM Judge Fleet Analysis — August 2026 runs

A consolidated review of the semantic-judge models used in the late-August 2026
benchmark runs: an ordered ranking of the top five judges, what each of the
rest of the pool is good for, per-judge blind spots, and concrete improvements
for efficacy (agreement, calibration, auditability) and discrimination (score
spread, rank fidelity).

Companion to [`qwen3-family-comparison.md`](./qwen3-family-comparison.md),
which uses this fleet to compare the qwen3.x families.

## Scope, panels, and metrics

Two runs were analyzed. Raw artifacts live under `runs/` (git-ignored); both
efficacy analyses read the runtime's own stored votes and validity semantics.

| Panel | Run | Judges | Cells/votes |
|---|---|---|---|
| **A** | `runs/2026-08-22-sqlite` (`run.sqlite3`, rev 35) | 4 standing: muse-glimmer-medium-nas:30b-64k, **ornith-1.5-nas:35b-64k**, qwen3.6:27b-32k, qwen3.8-thinking-medium-nas:27b-64k — plus ox-alpha, ornith-1.0-nas, qwen3.6:35b-32k and minimax-m2.7 across earlier legs | 13,250 judge attempts → 12,860 usable votes; 1,988 det-scored cells |
| **B** | `runs/2026-08-17-nas-and-more-test-changes` (`benchmark_state.json`) | 17 judges; anchors **ox-alpha** + **nemotron-3-ultra-openrouter** | 33,839 usable votes across ~35,034 attempts |

Metrics used below:

- **Consensus correlation / MAD / unilateral outliers** — leave-one-out
  agreement with the mean of the other judges on shared cells (A), or
  agreement with the two trusted anchors (B). Unilateral outlier = vote more
  than 10 pts outside the range the other judges spanned on that cell.
- **vs-deterministic r** — cell-level Pearson between a judge's score and the
  plugin rubric grade (percent-v1). Low values are structural (judges grade
  holistic quality; rubrics check exact artifacts), but *relative* differences
  between judges on the same plugin are meaningful.
- **Reliability** — usable rate, failed attempts, retries.
- **Auditability** — criteria-per-vote and % of votes carrying a criteria
  breakdown + evidence.
- **Leaderboard fidelity (LB Pearson / Spearman)** — correlation between a
  judge's per-model mean and the deterministic per-model mean.
- **Calibration profile** — mean score and share of votes at 100; generosity
  offset vs the deterministic rubric.

The blind-spot checks in §3 were computed for this report by correlating each
judge's votes against the rubric per plugin across the full panel-A run (128
models at rev 35), cross-checked on panel B.

## 1. The top five judges

Ordered list and the reasoning behind it: **ornith-1.5-nas:35b-64k (#1)** is
the anchor-class workhorse; **qwen3.8-thinking-medium-nas:27b-64k (#2)** is the
discriminator; **openrouter/ox-alpha (#3)** is the gold-standard auditor;
**qwen3.6:27b-32k (#4)** is the value full-coverage ranker; **qwen3.5:122b-128k
(#5)** is the near-twin with more context. The first two never shared a panel
(ornith-1.5 was #1 in both runs; qwen3.8-thinking judged only panel A), so the
order reflects the union of the evidence, not a single head-to-head.

| # | Judge | Consensus / anchor agreement | vs rubric (cell r) | Reliability | Calibration |
|--:|---|---|---:|---|---|
| 1 | ornith-1.5-nas:35b-64k | 0.965 LOO corr · MAD 3.8 · 0.7% outliers (A); MAD 3.35 vs ox-alpha, composite 86.9 (B) | 0.482 (best in B) | 100% usable (B); 1.5% fail (A) | generous: mean 86.6, 49% at 100 |
| 2 | qwen3.8-thinking-medium-nas:27b-64k | 0.961 · MAD 4.7 · 1.9% outliers (A) | **0.426 (best in A)** | **0% fail, 0 retries** | harshest: 82.7, **25% at 100** (widest spread) |
| 3 | openrouter/ox-alpha | 0.959 · MAD 4.2 · 0.7% outliers (A); anchor (B) | 0.461 (best in B) | 5.0% fail (API) | gold-standard depth |
| 4 | qwen3.6:27b-32k | 0.950 · MAD 5.4 · 2.2% outliers (A); composite 85.1 (B) | 0.376 | **8.8% fail (worst)** | generous: 74% at 100, terse |
| 5 | qwen3.5:122b-128k | MAD ≈5 · r ≈0.93 vs anchors (B) | 0.440 | 99.7% usable | slightly generous, mid-depth |

### #1 ornith-1.5-nas:35b-64k — the anchor-class workhorse

**Excels**
- Tightest consensus agreement of any judge in either run (LOO 0.965 in A; in
  B it matched the ox-alpha anchor at MAD 3.35 — closer than the two anchors
  matched each other) and the best leaderboard reproduction (LB Pearson 0.826).
- Exhaustive requirement-checklist rationales; every vote carries criteria +
  evidence; 100% usable in panel B, 1.5% failure in A.
- Reliably *right about ordering*: it never manufactures agreement downward —
  when the panel lands low for a genuine reason (transform errors,
  truncation), ornith-1.5 goes low too.

**Struggles**
- *Level, not order:* the most generous judge in the pool (49–53% of votes at
  100; ~+15 to +32 over the rubric) — it floats above rubric failures instead
  of reflecting them.
- Verified task-level blind spots (see §3): **tool-calling** (r 0.46 vs rubric;
  43% of rubric-failing cells still scored ≥85) and **code-review** (0.47;
  24%) — it over-credits well-written responses that miss exact tool-call or
  format requirements.
- Case-level misses: praised demonstrably broken dedup code in an
  event-processor answer (95 vs rubric 38 and 20/40 from the two coding
  judges), and it missed half the `<think>`-prologue-as-prose cells in A.

**Improve**
- On artifact-heavy plugins, prompt it to *verify artifacts before judgment*
  — enumerate tool calls, citations, signatures — and score against that list.
- Force score spread ("assume a real share of answers deserve <70; state what
  the top answers still miss").
- Apply a det-anchored calibration offset or normalize its scores to
  percentile ranks before consensus.
- De-weight it on tool-calling/code-review cells and pair it with
  qwen3.8-thinking-medium, which catches exactly what it over-credits.

### #2 qwen3.8-thinking-medium-nas:27b-64k — the discriminator

**Excels**
- Best discrimination in the fleet: only 25% of votes at 100 (vs 49–91% for
  the rest), most criteria per vote (8.0), best evidence accuracy, and the
  **highest correlation with the deterministic rubric of any judge in A**
  (0.426 cell; 0.480 on its best plugins).
- The only judge whose scores don't just ride on top of rubric failures — it
  is the natural counterweight to the generous judges and prevents consensus
  saturation.
- Caught the think-block-prose cases 3/3; zero failures, zero retries; cheap
  (1,084 thinking tokens/vote).

**Struggles**
- Calibrated ~3–4 pts harsher than the anchors (arguably correct, but it makes
  its absolute scores look pessimistic next to everyone else's).
- Coverage: judged only ~68 of 112 models in panel A and sat out panel B
  entirely, so its model-level figures are subset-based.

**Improve**
- Promote to a standing full-coverage judge — local, reliable, zero-failure.
- Weight it 1:1 against generous judges in consensus instead of letting
  mean-voting wash out its spread.
- Route coding/verification-heavy plugins (code-review, tool-calling,
  event-processor, rate-limiter) to it as the specialist.

### #3 openrouter/ox-alpha — the gold standard, infrastructure pending

**Excels**
- Best rationale quality in the fleet (997 chars, 7.7 criteria, 153-char
  evidence strings), anchor-grade agreement (0.959 in A; trusted anchor in B
  with 1,968 cells), strongest leaderboard fidelity (0.814/0.715), and fully
  audit-grade output — the reference every other judge is measured against.

**Struggles**
- API fragility: ~5% attempt loss in A to prompt-injection-filter false blocks
  (the benchmark's own judge prompts trip it) plus free-tier rate limits;
  cost and latency make full-run judging expensive.

**Improve**
- Retry on prompt-injection-block responses.
- Keep as a spot anchor on a representative subset or a fixed golden set per
  run, and use its per-cell evidence to calibrate the local judges.
- Mine its rationales as few-shot exemplars for the smaller judges.

### #4 qwen3.6:27b-32k — the value full-coverage ranker

**Excels**
- Best *cheap* judge: full coverage at near-anchor ranking fidelity (Spearman
  0.973/0.981 vs anchors in A — essentially indistinguishable), 100% criteria
  compliance, tiny per-vote cost; composite #2 in the 17-judge run (85.1).

**Struggles**
- Terse rationales (466 chars — the weakest audit trail of the top tier),
  very generous (74% at 100), **highest failure rate of any good judge
  (8.8%)**, and it burns 2,137 thinking tokens per vote for the shallowest
  output — the fleet's worst thinking-to-output efficiency.

**Improve**
- Schema-forced JSON output with retry-on-parse-failure to kill the 8.8% loss.
- Mandatory "why not higher" sentence and score-distribution prompting to push
  below its 74%-at-100 ceiling.
- Cap thinking and redirect the budget into evidence strings.
- Keep as the default full-coverage judge, but exclude its failed attempts
  from consensus rather than re-running blindly.

### #5 qwen3.5:122b-128k — the near-twin with more context

**Excels**
- Effectively qwen3.6:27b with more headroom: near-anchor agreement (MAD ≈5,
  r ≈0.93 vs both anchors in B), 99.7% usable, full criteria compliance, 128k
  context for judging long documents; composite #3 in B (84.8).

**Struggles**
- Weakest rank-order fidelity of the top group (LB Spearman 0.640 vs 0.659–
  0.728), fewer criteria per vote (median 5), 56-word median rationales — it
  discriminates points better than it discriminates order.

**Improve**
- Emphasize rank-order reasoning and require ≥6 criteria with evidence.
- Reserve it for where document length matters (long-context, PRD,
  architecture).
- Same score-spread and generosity fixes as qwen3.6:27b.

## 2. The rest of the pool

### Tier A — worth keeping in a rotation

**muse-glimmer-medium-nas:30b-64k** (nearest miss to the top five)
- **Excels:** ranks models as well as the anchors (Spearman 0.967/0.979 vs the
  anchors' mutual 0.979); the *only* judge that modulates confidence (27% of
  votes "medium"), making it the sole source of a usable human-review flag;
  harshness (mean 81.4, panel-A lowest) prevents saturation.
- **Struggles:** harshest grader (−4.8 vs ox-alpha, −5.2 vs ornith-1.5), most
  unilateral low outliers of the good judges (6.6%), and one outright
  hallucination — scored a tool-calling answer 0 claiming no plan/tool
  calls/final response existed when six other judges confirmed all three
  (92–100).
- **Slot:** ranker with −5 calibration and mandatory audit of low scores;
  trust "medium" confidence as a review trigger.

**nemotron-3-ultra-openrouter** (B anchor)
- **Excels:** widest, most even coverage (2,042 cells) and anchor-grade
  accuracy (r 0.920 mutual with ox-alpha); LB fidelity 0.801/0.680; no
  per-plugin blind spot recorded.
- **Struggles:** omits the criteria block in 38% of votes (hard to audit
  criterion-by-criterion); losses to free-tier rate limits and invalid JSON.
- **Slot:** consensus anchor; move off the free tier, enforce criteria output.

**gpt-oss:20b-32k**
- **Excels:** the only judge that runs harsh (mean 74.9, widest SD) with the
  smallest deviation from the rubric (+18.7 vs +19…+33 for everyone else) —
  the closest thing to a rubric-aligned grader; only judge using the full
  confidence scale (85% H / 12% M / 3% L).
- **Struggles:** strictness without accuracy — mid anchor agreement (r 0.830),
  ~1 criterion/vote with ~50% gaps.
- **Slot:** conformance-harsh counterweight on format-sensitive plugins; not a
  holistic-quality anchor.

**kat-coder-v2.5-thinking-nas:35b-64k**
- **Excels:** 100% usable; substantive free-text rationales (96-word median);
  strong leaderboard reproduction (0.806/0.680).
- **Struggles:** *never* emits a criteria array (0% over 2,007 votes) — a
  score-only judge with no audit trail; second-tier agreement (MAD ≈7.7–8.2).
- **Slot:** second-opinion ranker only.

**qwen3.6-general-thinking-nas:35b-64k** (as judge)
- **Excels:** genuinely good judgment — anchor agreement r 0.904, LB
  0.808/0.681 — proving the capability is there.
- **Struggles:** emits criteria in ~1–2% of votes (worst contract compliance,
  tied with kat-coder); generous (+27 over det).
- **Slot:** a top-tier judge hamstrung purely by schema non-compliance — fix
  the schema-forcing (its JSON-parse error history says it's a format issue,
  not a judgment issue) and it graduates.

**ornith-1.0:35b-64k**
- **Excels:** best rank-order fidelity of the entire 17-judge fleet (LB
  Spearman 0.728, Pearson 0.820) — the most faithful model-level sorter if all
  you need is "who beats whom".
- **Struggles:** everything else — worst consensus agreement in A by a wide
  margin (0.781, MAD 7.6, 6.7% outliers); systematic per-plugin mis-scoring
  (rate-limiter det-corr 0.17, event-processor 0.34, debug-traversal 0.10);
  103 cells scored ≤45 against unanimous anchor ≥85; rationales that
  contradict their own scores; ~50% criteria omission.
- **Slot:** rank signal with ~0 weight in consensus; never as a grader; never
  on rate-limiter/event-processor/debug tasks.

**nemotron-3.5-lightning-thinking-nas:30b-64k**
- **Excels:** second-best agreement structure after the anchors in B (W10
  84.5%, r 0.873) with near-perfect criteria compliance (98.5%).
- **Struggles:** loses 360 attempts to invalid JSON (93.5% usable); weak
  rank-order fidelity (LB 0.689/0.405); most generous grader (+32.7 over det).
- **Slot:** fix the JSON retry path and recalibrate — then a viable mid-tier
  judge.

### Tier B — promising but under-tested

**minimax-m2.7:229b-192k**
- **Excels:** excellent on its small slice (composite 84.5, MAD 4.3–4.8 vs
  anchors, r ≥0.91) with well-structured rationales (7.8 criteria/vote).
- **Struggles:** judged only ~40–77 cells; panel A exposed a specific blind
  spot — **instruction-following** (det-corr 0.35 vs 0.76–0.91 for everyone
  else; 5 of its 10 worst over-harsh calls were IF cells the anchors scored
  100); most generous outlier profile (5.2% >10 pts above the anchor band).
- **Slot:** complete its coverage, then use with an explicit
  instruction-following exclusion.

**hy3-iq2-xs:295b-32k**
- **Excels:** good agreement (r ≈0.87) and the best leaderboard Spearman among
  partial-coverage judges (0.775).
- **Struggles:** only 640 cells / 36 targets — unrankable as a full judge.
- **Slot:** worth completing coverage.

### Tier C — retire or don't bother

- **nemotron-3.5-lightning-instruct-nas** — weak agreement (W10 57.4%,
  r 0.64), compressed score spread (SD 22.5 — the fleet's worst
  discrimination), bottom-tier rank fidelity (0.676/0.543).
- **mistral-small4-reasoning** — format-perfect output that is content-noise
  (r 0.07–0.17 vs anchors); hallucinates disqualifications (scored a correct
  reasoning answer 0 on a fabricated contradiction). Its small mean deviation
  (+11.4) is a trap: zero correlation means it is wrong in both directions.
- **nemotron-3-super** — 66.2% usable, deterministic correlation 0.049, 51
  cells. Nothing to salvage.
- **qwen3-coder:30b-32k** — weakest full-coverage judge of the ranked set
  (r 0.719, LB 0.706/0.571) despite the coder branding; being a code model did
  not make it a good code judge.

### Special case: qwen3.6:35b-32k (as judge)

Never ranked (only ~56 models judged in A), but the most extreme profile in
the fleet: **a perfect pure ranker and a terrible grader**. It reproduced the
anchors' model ordering at 0.996/0.997 on its subset — best of anyone — while
being the most generous judge (mean 88.5, 75% at 100), writing the shallowest
rationales (6.4 criteria, 497 chars), burning the most thinking per vote
(2,651 tokens), and tracking the rubric worst on code-review (0.26),
error-recovery (0.19) and prd-creation (0.12). Use for ordering a subset;
never for scores or reasons.

## 3. Blind-spot verification (judge vs deterministic rubric, per plugin)

Judge votes were correlated against the rubric grade per plugin across the
full panel-A run (128 models), with panel B as a check. Figures: Pearson r and
the share of rubric-failing cells (det ≤ 40) that the judge still scored ≥ 85.

**ornith-1.5** (the most heavily used judge — the question was whether its
generosity is a task-level blind spot):

| Plugin | ornith-1.5 (A) | best other judge (A) | ornith-1.5 (B) |
|---|---:|---:|---:|
| event-processor | 0.69 · 19% | 0.73 (qwen3.8-thinking) · 12% | 0.75 · 17% |
| tool-calling | **0.46 · 43%** | **0.61 (qwen3.8-thinking) · 31%** | 0.62 · 35% |
| code-review | 0.47 · 24% | 0.62 (qwen3.8-thinking) · 6% | 0.42 · 12% |
| instruction-following | 0.91 · 2% | 0.91 · 2% | 0.92 · 2% |
| rate-limiter | 0.78 · 0% | 0.83 · 0% | 0.87 · 0% |
| data-transformation | 0.73 · 1% | 0.76 · 1% | 0.69 · 1% |

- **event-processor: no run-scale blind spot.** Mid-pack correlation (0.69 A /
  0.75 B). The flagged case is per-case: on qwen3.6:35b-32k's buggy
  event-processor, ornith-1.5 scored 95 and its rationale praised dedup code
  that demonstrably broke, while muse (20) and qwen3.8-thinking (40) plus the
  rubric (38) caught it.
- **tool-calling: a moderate task-level blind spot, confirmed.** Worst rubric
  tracking of the panel (0.46 A) and the most over-generous (43% of
  rubric-failing cells ≥85 vs 31% for the best judge; 35% in B). Named cases:
  qwen3.8-flash-next-thinking-xhigh (ornith 100 vs panel 63.3, det 40) and
  tiel-coder-qwen3.6 (95 vs 63.3, det 20). Over-credits coherent narratives
  that miss exact tool-call/format requirements — the mirror image of muse's
  hallucinated "no tool calls present" false zero on the same plugin.
- **code-review: a secondary weak spot** (0.47/24% vs 0.62/6%) in the same
  direction.

**Known per-judge, per-plugin blind spots** (from both efficacy reports):

| Judge | Blind spot | Evidence |
|---|---|---|
| minimax-m2.7 | instruction-following | det-corr 0.35 vs 0.76–0.91; scored a correct IF answer 0 on a fabricated order-count error (all others 100) |
| ornith-1.0 | rate-limiter, event-processor, debug-traversal | det-corrs 0.17 / 0.34 / 0.10; scored a rate-limiter answer 18 while its own rationale said "all three required methods implemented" |
| qwen3.6:35b (as judge) | code-review, error-recovery, prd-creation | det-corrs 0.26 / 0.19 / 0.12 |
| muse-glimmer-medium | low-score calls (direction: over-harsh) | 6.6% unilateral outliers; hallucinated a full-response absence in tool-calling |
| ornith-1.0 & ornith-1.5 | think-block prose detection | scored an answer with a `<think>` prologue 0, claiming "no prose included" (det 90, ox-alpha 92) — both factually wrong in the same direction |
| mistral-small4-reasoning | everything (noise) | r 0.07–0.17 vs anchors; hallucinated disqualifications |

**Panel-wide structural decoupling** (not judge-specific): on
software-architecture (r 0.13; 73% of rubric-failing cells scored ≥85 by
*every* judge), prd-creation (0.17; 42%), debug-traversal (0.25) and
orchestration (0.27), judges and the rubric measure different things — judges
reward content quality, rubrics enforce exact output contracts. This is
expected, and is why rubric and consensus are consumed as two separate
signals rather than averaged.

## 4. The area map — who to assign where

| Need | Assign |
|---|---|
| Broad, audit-grade holistic judgment | ornith-1.5 · ox-alpha · qwen3.8-thinking-medium |
| Cheap full-coverage ranking | qwen3.6:27b |
| Rubric/conformance-aligned strictness | gpt-oss:20b · qwen3.8-thinking-medium |
| Harsh counterweight with review flags | muse (audit lows, −5 calibration) |
| Pure model ordering (not grading) | kat-coder · ornith-1.0 · qwen3.6:35b (subsets) |
| Never on instruction-following | minimax |
| Never as a grader | mistral-small4-reasoning · nemotron-3-super · qwen3-coder |

The recommended standing panel is **ornith-1.5 + qwen3.8-thinking-medium +
qwen3.6:27b** as the always-on local trio, **ox-alpha** as the calibrating
anchor, **muse** retained only as a flagged ranker — with tool-calling and
code-review cells deliberately assigned to qwen3.8-thinking-medium rather than
ornith-1.5.

## 5. Cross-cutting improvements for efficacy and discrimination

1. **Per-judge × per-plugin weights in consensus** instead of flat means:
   down-weight ornith-1.5 on tool-calling/code-review; audit muse's low
   scores; exclude minimax from instruction-following; weight ornith-1.0 ~0.
2. **A fixed golden cell set per run** — known-good answers plus adversarial
   format cases (think-block prose, tool-call structure, transform refund
   ordering) — to measure each judge's drift and bias every run, then
   normalize to percentile ranks before consensus.
3. **Score-spread prompting and real confidence.** Confidence is saturated at
   "high" for 16 of 17 judges and adds no information. Elicit genuine
   uncertainty (muse's 27% "medium" is the only usable flag in the fleet) or
   drop confidence weighting and use outlier/spread metrics instead.
4. **Keep the two-signal split.** Judges rank content quality; the rubric
   checks output contracts. Best discrimination comes from feeding both into
   decisions — judges to order, rubric to filter — never averaging the two
   scales.
5. **Reliability plumbing:** schema-forced JSON + retries for the NAS judges
   (lightning-thinking lost 360 attempts, qwen3.6:27b 47, ornith-1.5
   non-compliant criteria for kat-coder/qwen3.6-general-thinking); move
   nemotron-3-ultra off the OpenRouter free tier; retry ox-alpha on
   prompt-injection-block responses.
6. **Retire the noise floor:** mistral-small4-reasoning, nemotron-3-super and
   lightning-instruct add no signal; ornith-1.0 is actively harmful as a
   grader.

## Caveats

- Panel A's run was interrupted; judge coverage is 3–4 votes per cell on most
  cells and a few judges judged subsets only (qwen3.6:35b 56 models,
  qwen3.8-thinking 68, minimax 40) — their model-level numbers are indicative,
  not leaderboard-grade.
- Judges 2 and 5 never co-occurred with every other top judge in one run; the
  top-five order is a synthesis, not a single-run ranking.
- Cell-level judge–rubric correlations are structurally low (0.34–0.48) for
  every judge; only *relative* comparisons per plugin are meaningful.
- Deterministic grades are single-attempt and vary run to run; judge consensus
  is the stable signal.
