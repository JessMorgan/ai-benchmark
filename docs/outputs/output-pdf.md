# PDF Output Plugin

| Property | Value |
|---|---|
| ID | `output-pdf` |
| Name | PDF Report |
| Extension | `pdf` |

## Description

The PDF output plugin generates a `results.pdf` report with summary statistics, complete results, leaderboards, and rubric breakdowns.

## Features

- Summary header with task list and model counts
- Complete results table
- Leaderboards for fastest TTFT and best score per plugin
- Detailed rubric breakdown when rubric data is available
- Session seed display

## Requirements

The PDF plugin requires the `fpdf2` package:

```bash
pip install fpdf2
```

If `fpdf2` is not installed, the plugin silently returns `None`.

## Generated File

When `output_dir` is provided and `fpdf2` is installed, the plugin writes:

```
<output_dir>/results.pdf
```
