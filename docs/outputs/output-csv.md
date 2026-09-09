# CSV Output Plugin

| Property | Value |
|---|---|
| ID | `output-csv` |
| Name | CSV Data |
| Extension | `csv` |

## Description

The CSV output plugin generates a `results.csv` file containing the raw benchmark data. It is useful for importing results into spreadsheets, notebooks, or other analysis tools.

## Output Columns

Columns are emitted in this order: `Model`, `Runner`, `Source`, `TTFT_s`,
then — when judging is enabled — `Judge_Models` and `Judge_Status`, then the
per-plugin block for each active plugin, and finally the overall columns.

| Column | Description |
|---|---|
| `Model` | Model name |
| `Runner` | Runner that produced the row (`http`, `opencode`, or `pi`) |
| `Source` | Source identifier |
| `TTFT_s` | Time to first token (seconds) |
| `Judge_Models` | Comma-separated judge model list (judge runs only) |
| `Judge_Status` | Judge status for the row (judge runs only) |
| `<plugin>_Response_s` | Response time per plugin (seconds) |
| `<plugin>_Thinking_Tokens` | Thinking/reasoning tokens per plugin |
| `<plugin>_Content_Tokens` | Content (final answer) tokens per plugin |
| `<plugin>_Total_Tokens` | Total tokens (thinking + content) per plugin |
| `<plugin>_TPS` | Tokens per second per plugin |
| `<plugin>_Score_100` | Normalized score per plugin (0–100, or `fail`) |
| `<plugin>_Max_Tokens` | Max-token budget used for the plugin |
| `<plugin>_Attempt_Count` | Number of logical attempts |
| `<plugin>_Retry_Count` | Number of retries |
| `<plugin>_Retried` | Whether the plugin was retried |
| `<plugin>_Selected_Attempt` | Attempt selected for scoring |
| `<plugin>_Retry_Reason` | Last retry reason |
| `<plugin>_Retry_Reasons_JSON` | JSON list of retry reasons |
| `<plugin>_Prompt_Altered` | Retry prompt-alteration label (e.g. `thinking_50_percent`) |
| `<plugin>_Response_Nature` | Machine-observable response nature (e.g. `token_limit`, `completed`) |
| `<plugin>_Finish_Reason` | Provider finish reason |
| `<plugin>_Truncated_Due_To_Time` | Whether the response was truncated by the request deadline |
| `<plugin>_Failure_Cause` | Failure cause for failed cells |
| `<plugin>_Attempts_JSON` | JSON per-attempt records |
| `<plugin>_Judge_Score_100` | Judge consensus score (judge runs only) |
| `<plugin>_Judge_Confidence` | Judge consensus confidence (judge runs only) |
| `<plugin>_Judge_Error` | Judge error state (judge runs only) |
| `<plugin>_Judge_Votes` | JSON judge votes (judge runs only) |
| `<plugin>_Judge_Criteria_JSON` | JSON criterion-level interpretations (judge runs only) |
| `<plugin>_Judge_Consensus_By_Contract_JSON` | JSON per-contract consensus (judge runs only) |
| `<plugin>_Empty_Reason` | Empty-response classification (`error`/`thinking-truncation`/`thinking-only`/`max-tokens`/`empty`) |
| `Overall_Score_100` | Half-up mean of available plugin percentages |
| `Overall_Scored_Plugins` | Number of active plugins with a numeric public score |
| `Time_s` | Total benchmark time |
| `Mode` | `stream` or `non-streaming` |
| `Status` | `OK` or `FAIL` |
| `Error` | Error message if failed |

Detailed rubric data in saved metadata and in-memory results uses native
`points` and `total` fields for each criterion; it is not converted to
percentages.

## Generated File

The CSV is one of the report formats selected with `--output-format csv`.
When `output_dir` is provided, the plugin writes:

```
<output_dir>/results.csv
```

If `output_dir` is omitted, the CSV content is returned as a string.
