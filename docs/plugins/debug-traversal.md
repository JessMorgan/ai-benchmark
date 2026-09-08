# Debug Traversal Plugin

| Property | Value |
|---|---|
| ID | `debug-traversal` |
| Version | `1.3.0` |
| Max Score | 20 |
| Streaming | Yes |

The supplied function incorrectly uses `count > 2` when the requirement is at least two occurrences. The response must trace the exact sample, identify the comparison, provide corrected code, include a test, and discuss side effects. The corrected code is executed in the restricted harness.

Scoring (1.3.0):

- **Executable fix verification (3.0)** gates the lexical criteria. The gate
  requires `harness_ok` (the assertion harness ran to completion and printed
  its sentinel), not merely `status == "passed"` — a response that exits early
  (`sys.exit` / `SystemExit`) before the harness does not pass. When the gate
  fails or the corrected block is absent, the three lexical criteria below are
  scaled down proportionally (to 0) and a negative finding is emitted, so
  prose alone cannot earn them.
- **Depth of analysis (3.0)** requires both the defective-comparison
  identification (word-bounded, so `count > 20` does not match) *and* a
  corrective statement — a corrective modal (`should`/`must`/…) followed by
  the correct comparison (`>= 2`, `at least 2`, `count > 1`, …). An inverted
  diagnosis that names the buggy comparison without the remedy earns no depth
  credit.
- **Proposed fix / corrected code (3.0)** accepts the equivalent corrected
  forms `>= 2` and `count > 1`.
- **Systematic trace / code walkthrough (3.0)**: the empty/return hit must
  co-reference the specific `count=2` value (within ~80 chars), not any
  "returns" mention.
- **Test code provided (3.0)**: the membership check must sit in a
  test-assertion context (`assert x in y`), not any "in " fragment.
- **Structured RCA sections (2.0)** is linear: each of the 5 sections earns
  0.4 and full credit requires >=4 of 5.
