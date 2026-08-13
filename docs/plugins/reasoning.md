# Logical Reasoning Plugin

| Property | Value |
|---|---|
| ID | `reasoning` |
| Name | Logical Reasoning |
| Version | `0.1.1` |
| Max Score | 20 |
| Streaming | Yes |

## Task

The model solves a six-service incident puzzle with six timestamps, six owners, and six priority levels. The clues form interacting chains:

- Profile before an adjacent Auth/Search time pair;
- Upload after Search;
- Billing between Upload and Notifications;
- owner assignments linked to relative positions;
- fixed P1/P2 anchors;
- a three-way priority ordering among Upload, Search, and Billing.

The model must show numbered deductions and identify the service, owner, priority, and time of the 09:30 incident. The answer is not awarded full credit from the final labels alone: the supporting time, ownership, and priority deductions must also be present. Solving the time chain requires placing all six services; the target service is not named by a direct timestamp clue.

## Scoring Rubric

| Criterion | Max | Description |
|---|---:|---|
| Final answer | 8 | Correct service, owner, priority, and time |
| Time-chain deductions | 4 | Applies the adjacency and before/after constraints |
| Derived time assignments | 4 | Establishes the six resulting timestamps |
| Ownership deductions | 2 | Uses the owner clues to identify the 09:30 owner |
| Priority-chain deductions | 2 | Uses P1/P2 anchors and the strict priority ordering |

A response without numbered deductions loses one point from the time-chain criterion even if its final labels are correct.

## Temperature

Default temperature can be set with:

```json
"reasoning_temperature": 0.1
```
