# Logical Reasoning Plugin

| Property | Value |
|---|---|
| ID | `reasoning` |
| Version | `1.2.0` |
| Max Score | 20 |
| Streaming | Yes |

The task is a constrained scheduling puzzle. The evaluator parses the four
exact final fields (the LAST occurrence of each `LABEL:` line wins, so a
tentative mid-text line cannot shadow the final answer) and independently
checks numbered time-chain deductions, derived time assignments, ownership
deductions, and the priority chain.

Scoring gates and tolerance (1.2.0):

- The four reasoning-point criteria are capped at half their max when the
  final answer lines are wrong or absent, so restating the clue wording
  cannot outscore a correct answer.
- Ownership patterns accept `owns`/`owned`/`owner` phrasings, and the
  priority chain accepts `>` comparison chains in addition to the word
  "higher".
- The time-chain deduction is negation-guarded: a negation word in the gap
  ("Auth is NOT immediately before Search") earns no point.
- The P5 association check spans newlines, so a sparse final block
  (`Search` / `09:30` / `P5` on separate lines) matches.
- Numbered prose after the last `FAILED_SERVICE` line breaks the "exactly
  four final lines" contract and is penalized; mid-text "Step 1:" style
  deductions do not trigger the penalty.
- All service/time pairs in the response are compared against the unique
  solution, and the final `TIME` is checked against the solution time of
  the final `FAILED_SERVICE`; any disagreement is penalized. The final
  answer alone cannot receive full credit.

The correct result is:

```text
FAILED_SERVICE: Search
OWNER: Ben
PRIORITY: P5
TIME: 09:30
```

P4 belongs to Profile; the ordering constraints leave Upload at P6, Search at
P5, and Billing at P3.
