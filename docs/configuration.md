# Configuration Reference

All benchmark configuration lives in a single JSON file (default: `benchmark-config.json`). The file is passed to `ai-benchmark.py` with `--config`.

## Top-Level Keys

| Key | Type | Default | Description |
|---|---|---|---|
| `output_dir` | string | `benchmark-results` | Directory for reports, logs, and state |
| `timeout` | integer | `600` | API request timeout in seconds |
| `token_levels` | list[int] | `[16384]` | Max-token limits tried in ascending order |
| `plugin_thread_limit` | integer | `1` | Top-level fallback for `sources.*.plugin_thread_limit` |
| `plugins_whitelist` | list[string] | `[]` | Run only these plugin IDs (empty = all) |
| `plugins_blacklist` | list[string] | `[]` | Skip these plugin IDs (empty = none) |
| `sources` | object | required | Named API endpoint definitions |
| `models` | object | required | Model-to-source mapping |

## Sources

Each entry under `sources` defines an API endpoint. The key is the source name used in `models`.

```json
{
  "sources": {
    "Local Server": {
      "api_url": "http://localhost:11434/chat/completions",
      "headers": {
        "Authorization": "Bearer ${LOCAL_API_KEY:sk-fallback}",
        "Content-Type": "application/json"
      },
      "plugin_thread_limit": 1
    }
  }
}
```

### Per-Source Plugin Concurrency

Each source can define `plugin_thread_limit` to control how many plugins run concurrently for models against that source. The top-level `plugin_thread_limit` is used as a fallback for sources that do not define their own value. The CLI `--plugin-thread-limit` overrides all sources.

### Environment Variable Expansion

Header values support `${VAR}` and `${VAR:default}` syntax:

```json
"Authorization": "Bearer ${OPENAI_API_KEY:sk-fallback-key}"
```

The value is replaced with the environment variable value, or the default if the variable is unset.

## Models

The `models` map supports two forms.

### Simple Form

```json
"models": {
  "llama3:8b": "Local Server"
}
```

The value is the source name from the `sources` section.

### Extended Form

```json
"models": {
  "llama3:8b": {
    "source": "Local Server",
    "drop_params": ["seed"]
  }
}
```

| Key | Type | Description |
|---|---|---|
| `source` | string | Source name from `sources` |
| `drop_params` | list[string] | Request body keys to omit for this model |

### Per-Model Parameter Dropping

Use `drop_params` to omit parameters that a particular model or provider does not support. Common examples:

```json
"models": {
  "model-a": {
    "source": "Remote Provider",
    "drop_params": ["seed"]
  },
  "model-b": {
    "source": "Another Provider",
    "drop_params": ["seed", "temperature"]
  }
}
```

## Per-Plugin Temperature

You can set the temperature for each plugin using either of these config keys:

```json
{
  "rate-limiter_temperature": 0.2,
  "moe-dense_temperature": 0.7,
  "code-review_temperature": 0.3,
  "orchestration_temperature": 0.5,
  "tool-calling_temperature": 0.2,
  "structured-output_temperature": 0.2
}
```

## Complete Example

```json
{
  "output_dir": "benchmark-results",
  "timeout": 600,
  "token_levels": [16384],
  "rate-limiter_temperature": 0.2,
  "moe-dense_temperature": 0.7,
  "plugins_whitelist": [],
  "plugins_blacklist": [],
  "sources": {
    "Local Server": {
      "api_url": "http://localhost:11434/chat/completions",
      "headers": {
        "Authorization": "Bearer ${LOCAL_API_KEY:sk-fallback}",
        "Content-Type": "application/json"
      },
      "plugin_thread_limit": 1
    },
    "Remote Provider": {
      "api_url": "https://api.example.com/v1/chat/completions",
      "headers": {
        "Authorization": "Bearer ${REMOTE_API_KEY}",
        "Content-Type": "application/json"
      }
    }
  },
  "models": {
    "llama3:8b": "Local Server",
    "gpt-oss:120b-128k": {
      "source": "Remote Provider",
      "drop_params": ["seed"]
    }
  }
}
```

## Notes

- `token_levels` are tried in order. If a response is truncated, the next level is used.
- `plugin_thread_limit` controls how many plugins run concurrently for each model against a given source. Set to `1` for sequential execution or `0` for maximum parallelism. Define it per-source or as a top-level fallback.
