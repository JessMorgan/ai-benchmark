# HTML Output Plugin

| Property | Value |
|---|---|
| ID | `output-html` |
| Name | HTML Report |
| Extension | `html` |

## Description

The HTML output plugin generates a styled `results.html` report with leaderboards, complete results, and detailed rubric breakdowns.

## Features

- Dark-themed responsive HTML table
- Leaderboards for fastest TTFT and best score per plugin
- Links to saved response text files when `output_dir` is provided
- Detailed rubric breakdown when rubric data is available
- Session seed display

## Generated File

When `output_dir` is provided, the plugin writes:

```
<output_dir>/results.html
```

If `output_dir` is omitted, the HTML content is returned as a string.
