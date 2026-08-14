# Debug Report Consistency Plugin

| Property | Value |
|---|---|
| ID | `debug-consistency` |
| Version | `0.1.0` |
| Max Score | 20 |
| Streaming | Yes |

This challenge tests whether a model verifies a reported failure against the supplied code instead of inventing a root cause. The correct response reproduces the sample, explains that the report is inconsistent with the implementation, requests missing evidence, and recommends investigation rather than an unjustified patch.

Scoring checks section-local reproduction, consistency conclusion, non-hallucinated diagnosis, evidence requests, recommendation, and required structure.
