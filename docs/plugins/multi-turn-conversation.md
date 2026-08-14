# Multi-Turn Conversation Plugin

| Property | Value |
|---|---|
| ID | `multi-turn-conversation` |
| Version | `1.0.0` |
| Max Score | 20 |
| Streaming | Yes |

This challenge evaluates stateful revision behavior. The response simulates
three user turns and must provide one typed JSON state object per turn. Turn 2
disables music and adds a `deep-work` label while preserving the start time and
calendar event. Turn 3 preserves those decisions, changes duration to 50
minutes, and adds a five-minute notification. A typed state summary must
explain both transitions.

The evaluator parses each JSON object, validates exact keys and value types,
checks requested updates and preserved state, and rejects prose outside the
state blocks. It does not award full credit for three merely different pieces
of prose.

This is intentionally distinct from `long-context`: Multi-Turn tests
cross-turn state retention and minimal updates; Long Context tests retrieval
and multi-record cross-reference over a large distractor set.
