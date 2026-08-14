# Error Recovery Plugin

| Property | Value |
|---|---|
| ID | `error-recovery` |
| Version | `1.0.0` |
| Max Score | 20 |
| Streaming | Yes |

The model implements an exact injectable asynchronous `WeatherClient.fetch` API and `AllProvidersFailedError`. The evaluator checks fallback after exceptions and error payloads, exact exception type, and bounded provider behavior in an isolated harness. Lexical design credit is secondary to executable behavior.
