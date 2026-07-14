# CLI Reference

The benchmark is driven by `ai-benchmark.py`.

## Usage

```sh
python ai-benchmark.py [options]
```

## Arguments

| Argument | Description |
|---|---|
| `--config PATH` | Config file path (default: `benchmark-config.json`) |
| `--restart` | Discard prior state and run all models from scratch |
| `--out DIR` | Override the output directory from config |
| `--timeout SEC` | Override API request timeout |
| `--token-levels N [N ...]` | Override token levels (e.g. `--token-levels 4096 8192 16384`) |
| `--temperature VAL` | Default temperature for all plugins (overrides config) |
| `--plugin-temperature ID=VAL [ID=VAL ...]` | Override per-plugin temperatures (highest priority) |
| `--plugin-thread-limit N` | Max concurrent plugins per model (`0` = one per plugin) |
| `--plugins-whitelist ID [ID ...]` | Run only these plugins |
| `--plugins-blacklist ID [ID ...]` | Run all plugins except these |
| `--list-plugins` | List discovered plugins and exit |
| `--generate-shell-completion bash/zsh/fish` | Generate shell completion script |
| `--dump-default-config` | Print a default config template and exit |
| `--base-url URL` | (with `--dump-default-config`) Discover models from `/v1/models` |
| `--api-key KEY` | (with `--base-url`) API key for model discovery |
| `--save-responses` | Save each model's plugin response text to `<output_dir>/responses/` |
| `--seed INT` | Fixed random seed for all API requests |
| `-h, --help` | Show help message |

## Examples

### Run with the default config

```sh
python ai-benchmark.py
```

### Use a custom config

```sh
python ai-benchmark.py --config my-config.json
```

### Restart from scratch

```sh
python ai-benchmark.py --restart
```

### Run only one plugin

```sh
python ai-benchmark.py --plugins-whitelist rate-limiter
```

### Exclude a plugin

```sh
python ai-benchmark.py --plugins-blacklist moe-dense
```

### Use a fixed seed

```sh
python ai-benchmark.py --seed 42
```

### Override token levels

```sh
python ai-benchmark.py --token-levels 4096 8192 16384
```

### Override all plugin temperatures

```sh
python ai-benchmark.py --temperature 0.5
```

### Override a specific plugin temperature

```sh
python ai-benchmark.py --plugin-temperature rate-limiter=0.3
```

### Combine default and per-plugin temperatures

Per-plugin settings take priority over the default, and both override config file values.

```sh
python ai-benchmark.py --temperature 0.5 --plugin-temperature moe-dense=0.7
```

### Generate and install shell completions

```sh
# Bash
python ai-benchmark.py --generate-shell-completion bash > /etc/bash_completion.d/ai-benchmark

# Zsh
python ai-benchmark.py --generate-shell-completion zsh > ~/.zsh/completions/_ai-benchmark.py

# Fish
python ai-benchmark.py --generate-shell-completion fish > ~/.config/fish/completions/ai-benchmark.py.fish
```

### Discover models from an API

```sh
python ai-benchmark.py \
  --dump-default-config \
  --base-url http://localhost:11434 \
  --api-key sk-xxx > benchmark-config.json
```

## Resume Behavior

By default, re-running resumes from the saved state. Completed models are skipped, failed models are retried, and newly added models are picked up automatically. Use `--restart` to force a clean run.

If the active plugin set changes between runs, the CLI prompts whether to restart or continue. Continuing keeps old data and runs only the newly added plugins for models that already completed.
