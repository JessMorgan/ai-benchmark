# Tool Calling Agent Plugin

| Property | Value |
|---|---|
| ID | `tool-calling` |
| Version | `1.2.0` |
| Max Score | 25 |
| Streaming | Yes |

The model must emit exactly one valid call for each required tool, in weather, flight, hotel, stock, currency, email order. Arguments are type-checked and dates accept either ISO dates or ISO datetimes representing the requested date. The final response must include all requested results and a numeric JPY amount. Duplicate, missing, unknown, or malformed calls do not receive full contract credit.

## Exact argument keys

Each call is a typed JSON object `{"name": ..., "args": {...}}`. The `args`
object must use these exact keys (the prompt signals them for
`convert_currency`):

| Tool | Required `args` keys |
|---|---|
| `get_weather` | `location` (str); optional `unit` (str) |
| `search_flights` | `origin` (str), `destination` (str), `date` (str) |
| `book_hotel` | `city` (str), `check_in` (str), `check_out` (str), `guests` (int) |
| `get_stock_price` | `ticker` (str) |
| `convert_currency` | `amount` (int/float), `from_curr` (str), `to_curr` (str) |
| `send_email` | `to` (str), `subject` (str), `body` (str) |

In particular, `convert_currency` takes its currencies under the keys
`from_curr` and `to_curr` (not `from`/`to` or `source`/`target`), and
`send_email` takes the recipient under `to`.
