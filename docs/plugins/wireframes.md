# Wireframes Plugin

| Property | Value |
|---|---|
| ID | `wireframes` |
| Version | `1.1.0` |
| Max Score | 20 |
| Streaming | Yes |

The evaluator parses Markdown screen blocks, requires at least four distinct named FlowState screens, and checks purpose, structural content, controls, known-screen navigation edges, interaction notes, and feature coverage per screen. Repeated headings and global component keyword lists do not count as a complete wireframe set.

Since 1.1.0 the scoring is gaming-resistant:

- Position keywords (`top`, `bottom`, `header`, ...) and component
  keywords (`nav`, `tab`, ...) require word boundaries, so substrings
  like "s**top**s", "navigation", and "tabletop" no longer match.
- Box-drawing/structural lines (`+---+`, `|  |`, `─`, `│`, `┌`, `└`)
  count as wireframe-diagram evidence on their own; position words
  otherwise require co-occurring structural content.
- Navigation edges are normalized, deduplicated, and self-loops are
  rejected — only distinct edges between known screens count.
- The PRD-feature criterion references only the features the prompt
  names (focus, calendar, schedule/planning, timer/session, settings).
- Partial credit is proportional and strictly below the full score
  whenever the criterion gate is not met.
