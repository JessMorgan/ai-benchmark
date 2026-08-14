# Long-Context Retrieval Plugin

| Property | Value |
|---|---|
| ID | `long-context` |
| Version | `0.1.0` |
| Max Score | 20 |
| Streaming | Yes |

The prompt contains a large set of relevant and distracting incident records. The model must retrieve the matching incident and owner, follow references across separate facts to derive its escalation channel, cite evidence IDs, and use the exact response headings. This measures long-context retrieval plus cross-reference reasoning rather than simple final-answer guessing.
