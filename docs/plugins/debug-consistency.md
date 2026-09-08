# Debug Report Consistency Plugin

| Property | Value |
|---|---|
| ID | `debug-consistency` |
| Version | `0.2.0` |
| Max Score | 20 |
| Streaming | Yes |

This challenge tests whether a model verifies a reported failure against the supplied code instead of inventing a root cause. The correct response reproduces the sample, explains that the report is inconsistent with the implementation, requests missing evidence, and recommends investigation rather than an unjustified patch.

Scoring (v0.2.0): all criteria are section-local and direction-aware.

- **Reproduction trace (4.0)** — requires the positive trace: the value `abc` and the actual output list `['abc']` (quote style and inner whitespace tolerated). Quoting the report's claimed `[]` output no longer zeroes the criterion, and a hallucinated empty-output trace earns no credit.
- **Consistency conclusion (5.0)** — word-bounded, direction-aware signals (`correct`, `consistent`, `not reproducible`, `returns ['abc']`, `does not follow`, `no bug`); the negated forms `incorrect`/`inconsistent` do not earn the criterion.
- **Non-hallucinated diagnosis (4.0)** — requires specific positive no-bug signals ("no bug/defect/issue", "cannot/could not reproduce", "cannot confirm", "behaves as specified", "works correctly", "the code is correct", "the report is incorrect/wrong/invalid", …); generic words (`report`, `environment`, `input`) do not earn it.
- **Evidence request (3.0)** — requires an evidence keyword (stack/version/actual/input/log/repro/environment/trace) **and** a reference to the actual trace/output (`abc` or `['abc']`).
- **Actionable recommendation (2.0)** — a verify/reproduce/investigate step (`do not`, `not enough`, `collect`, `reproduce`, `instrument`, `verify`).
- **Required report structure (2.0)** — all five sections must have content beyond the heading (substantive body, ≥ 20 chars).
