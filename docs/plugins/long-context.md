# Long-Context Retrieval Plugin

| Property | Value |
|---|---|
| ID | `long-context` |
| Version | `0.2.0` |
| Max Score | 20 |
| Streaming | Yes |

The prompt contains a large set of relevant and distracting incident records. The model must retrieve the matching incident and owner, follow references across separate facts to derive its escalation channel, cite evidence IDs, and use the exact response headings. This measures long-context retrieval plus cross-reference reasoning rather than simple final-answer guessing.

Scoring (v0.2.0): the incident ID is the primary criterion (6.0 pts); the evidence (4.0), cross-reference (3.0), and owner (2.0) criteria are gated on it, while exact answer (4.0) and response contract (1.0) are not. Evidence credit is per-ID for the correct chain {F02, F05, F09, F13} (1.0 each). Incident-ID and P1 tokens are word-bounded and negation-aware, so "NOT I-17", "I-170", and "P12" do not earn credit.
