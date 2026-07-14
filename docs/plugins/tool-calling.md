# Tool Calling Agent Plugin

| Property | Value |
|---|---|
| ID | `tool-calling` |
| Name | Tool Calling Agent |
| Version | `0.2.0` |
| Max Score | 25 |
| Streaming | Yes |

## Task

The model is a travel-planning agent with access to these tools:

- `get_weather(location, unit='celsius')`
- `search_flights(origin, destination, date)`
- `book_hotel(city, check_in, check_out, guests)`
- `get_stock_price(ticker)`
- `convert_currency(amount, from_curr, to_curr)`
- `send_email(to, subject, body)`

The model must:

1. Plan tool calls inside a `<plan>...</plan>` block.
2. Output each tool call inside a `<tool_call>...</tool_call>` block as JSON.
3. Synthesize a final response with hypothetical return values.

## Scoring Rubric

| Criterion | Max | Description |
|---|---|---|
| Output format compliance | 3 | Valid `<tool_call>` blocks |
| Planning section | 2 | `<plan>` block before tool calls |
| Required tools present | 5 | All six tools called |
| Correct arguments | 8 | Correct arguments per tool |
| Correct ordering | 3 | Expected tool call order |
| Synthesis | 4 | Final response with all required elements |

## Temperature

Default temperature can be set with:

```json
"tool-calling_temperature": 0.2
```

## Tips for Models

- Use exact JSON inside `<tool_call>` blocks.
- Follow the expected order: weather, flights, hotel, stock, currency, email.
- Include all required arguments.
