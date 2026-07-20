# Markdown Output Plugin

| Property | Value |
|---|---|
| ID | `output-markdown` |
| Name | Markdown Report |
| Extension | `md` |

## Description

The Markdown output plugin generates a `results.md` report with tables, leaderboards, and rubric breakdowns in Markdown format.

## Features

- Complete results table with per-plugin metrics
- Leaderboards for fastest TTFT, best score per plugin, and best combined total
- Links to saved response text files when `output_dir` is provided
- Detailed rubric breakdown when rubric data is available
- Session seed display

## Generated File

When `output_dir` is provided, the plugin writes:

```
<output_dir>/results.md
```

If `output_dir` is omitted, the Markdown content is returned as a string.
