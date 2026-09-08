# Multi-Turn Conversation Plugin

| Property | Value |
|---|---|
| ID | `multi-turn-conversation` |
| Version | `1.1.0` |
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

## 1.1.0 scoring changes

- Each turn is graded against its own expected state, not the final state:
  Turn 1 expects music on, 25 minutes, and no labels; Turn 2 expects music
  toggled off; Turn 3 expects the `deep-work` label to persist. Copying the
  final state into every turn no longer earns preservation credit.
- The State-change summary requires the transition claims to match the
  transitions that actually happened: a negated disable claim, an "enabled
  music" claim, or a "shortened the duration" claim (the duration was
  lengthened from 25 to 50) voids the summary credit.
- The summary accepts the `5-minute` / `5 minute` paraphrase as equivalent to
  "5 minutes".
- Turn slots are graded by index: a broken intermediate turn no longer shifts
  a later turn against the wrong expected state.
- Untyped values are checked against the canonical types (`music: 1` no
  longer matches `music: true`), and label checks require a canonical list of
  strings.

The decorated-headings behavior is unchanged (deferred, prompt-aligned):
decorated headings such as `## **Turn 1**` are still normalized and accepted.

This is intentionally distinct from `long-context`: Multi-Turn tests
cross-turn state retention and minimal updates; Long Context tests retrieval
and multi-record cross-reference over a large distractor set.
