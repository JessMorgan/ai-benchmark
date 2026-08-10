# CSV Output Plugin

| Property | Value |
|---|---|
| ID | `output-csv` |
| Name | CSV Data |
| Extension | `csv` |

## Description

The CSV output plugin generates a `results.csv` file containing the raw benchmark data. It is useful for importing results into spreadsheets, notebooks, or other analysis tools.

## Output Columns

| Column | Description |
|---|---|
| Model | Model name |
| Source | Source identifier |
| TTFT_s | Time to first token (seconds) |
| `<plugin>_Response_s` | Response time per plugin |
| `<plugin>_Output_Tokens` | Output tokens per plugin |
| `<plugin>_TPS` | Tokens per second per plugin |
| `<plugin>_Score_100` | Normalized integer score per plugin (0–100) |
| Overall_Score_100 | Half-up mean of available plugin percentages |
| Overall_Scored_Plugins | Number of active plugins contributing to the overall score |
| Time_s | Total benchmark time |
| Mode | `stream` or `non-streaming` |
| Status | `OK` or `FAIL` |
| Error | Error message if failed |

## Generated File

When `output_dir` is provided, the plugin writes:

```
<output_dir>/results.csv
```

If `output_dir` is omitted, the CSV content is returned as a string.
