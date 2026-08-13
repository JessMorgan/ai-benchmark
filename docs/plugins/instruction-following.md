# Instruction Following Plugin

| Property | Value |
|---|---|
| ID | `instruction-following` |
| Name | Instruction Following |
| Version | `0.1.1` |
| Max Score | 20 |
| Streaming | Yes |

## Task

The model receives nine order records and must apply all of the following:

- keep only paid, non-refunded orders;
- require an amount of at least 50.00;
- exclude the APAC region and the internal channel;
- sort by amount descending, then customer name alphabetically for ties;
- uppercase customer names while preserving IDs and two-decimal amounts;
- calculate an exact count, total, and top order;
- emit only four order lines and one summary line in the required format.

The records contain distractors that each fail a different filter, and the two 120.00 orders require the customer-name tie-break rather than input order or ID order.

## Scoring Rubric

| Criterion | Max | Description |
|---|---:|---|
| All filters applied | 4 | Retains exactly the four eligible records and excludes every distractor |
| Sort and tie-break order | 4 | Uses amount descending and customer-name ascending for the tie |
| Transformed order lines | 4 | Preserves IDs/amounts and uppercases customer names exactly |
| Summary arithmetic and format | 4 | Correct count, total, top order, and syntax |
| Exact response discipline | 4 | No headings, bullets, fences, reasoning, blank extra output, or filtered records |

## Temperature

Default temperature can be set with:

```json
"instruction_following_temperature": 0.2
```
