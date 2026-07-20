# Wireframes Plugin

| Property | Value |
|---|---|
| ID | `wireframes` |
| Name | Wireframes |
| Version | `0.1.0` |
| Max Score | 20 |
| Streaming | Yes |

## Task

The model acts as a UX designer and produces a set of low-fidelity wireframes for the FlowState mobile app based on a PRD.

The wireframe set must include:

- At least 4 distinct screens
- A clear screen name and purpose for each screen
- A text-based wireframe representation (ASCII art, box-drawing, or structured component list)
- Key UI components on each screen
- Navigation flows between screens
- Annotations or notes explaining interactions

## Scoring Rubric

| Criterion | Max | Description |
|---|---|---|
| Multiple screens present | 3 | At least 4 distinct screens |
| Screen names and purposes | 3 | Named screens related to the PRD |
| Visual/structural wireframe | 4 | ASCII/box drawing or structured layout |
| Key UI components | 4 | Buttons, cards, lists, nav, etc. |
| Navigation flows | 3 | Arrows, transitions, or explicit flows |
| Annotations and interaction notes | 2 | Notes explaining interactions |
| Coverage of PRD features | 1 | Focus, calendar, music, AI, settings, etc. |

## Temperature

Default temperature can be set with:

```json
"wireframes_temperature": 0.5
```

## Tips for Models

- Use box-drawing characters or clear component labels.
- Label each screen and its purpose.
- Show navigation with arrows or explicit transitions.
- Include annotations for important interactions.
