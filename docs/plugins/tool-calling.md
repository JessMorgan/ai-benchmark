# Tool Calling Agent Plugin

| Property | Value |
|---|---|
| ID | `tool-calling` |
| Version | `1.0.0` |
| Max Score | 25 |
| Streaming | Yes |

The model must emit exactly one valid call for each required tool, in weather, flight, hotel, stock, currency, email order. Arguments are type-checked and dates accept either ISO dates or ISO datetimes representing the requested date. The final response must include all requested results and a numeric JPY amount. Duplicate, missing, unknown, or malformed calls do not receive full contract credit.
