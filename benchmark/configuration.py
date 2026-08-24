"""Canonical benchmark configuration loading, overrides, and target resolution."""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any

import yaml

from .http import fetch_models_v1
from .opencode import OPENCODE_NO_OUTPUT_GRACE

FLUSH_INTERVAL_SECONDS = 120.0
FLUSH_MAX_VOTES = 50
PERSISTENCE_SHUTDOWN_TIMEOUT = 10.0
JUDGE_DEFAULT_MAX_TOKENS = 4096
JUDGE_DEFAULT_REQUEST_PARAMS = {
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "benchmark_judge_result",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "score": {"type": "integer", "minimum": 0, "maximum": 100},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "rationale": {"type": "string"},
                    "criteria": {"type": "array"},
                },
                "required": ["score", "confidence", "rationale", "criteria"],
            },
        },
    },
}

PRELOAD_DEFAULT_TIMEOUT = 300
DEFAULT_MAX_THINKING_TOKENS = 32768
DEFAULT_MAX_CONTENT_TOKENS = 16384

def _expand_env(val: Any) -> Any:
    """Recursively expand ${VAR} or ${VAR:default} in strings."""
    if isinstance(val, str):
        parts = []
        raw = val
        i = 0
        while i < len(raw):
            start = raw.find("${", i)
            if start == -1:
                parts.append(raw[i:])
                break
            end = raw.find("}", start)
            if end == -1:
                parts.append(raw[i:])
                break
            expr = raw[start+2:end]
            default = None
            if ":" in expr:
                var, default = expr.split(":", 1)
            else:
                var = expr
            parts.append(raw[i:start])
            parts.append(os.environ.get(var, default or ""))
            i = end + 1
        return "".join(parts)
    if isinstance(val, dict):
        return {k: _expand_env(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_expand_env(v) for v in val]
    return val


def load_dotenv_file(path: str | None = None) -> bool:
    """Load environment variables from a ``.env`` file into ``os.environ``.

    The file defaults to ``.env`` in the current working directory. A missing
    file is ignored, and variables already present in the environment take
    precedence over file values (dotenv's ``override=False`` default). Returns
    ``True`` when a file was found and loaded, ``False`` otherwise.
    """
    from dotenv import load_dotenv

    return bool(load_dotenv(dotenv_path=path if path is not None else ".env", override=False))


def load_config(path: str) -> Any:
    """Load benchmark config from a JSON or YAML file. Returns the full config dict."""
    with open(path) as f:
        if path.lower().endswith((".yaml", ".yml")):
            data = yaml.safe_load(f)
            if data is None:
                raise ValueError(f"YAML config file is empty: {path}")
        else:
            data = json.load(f)
    data = _expand_env(data)
    legacy_paths = []

    def find_legacy(value: Any, path: str = "config") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"token_levels", "model_token_levels"}:
                    legacy_paths.append(f"{path}.{key}")
                find_legacy(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                find_legacy(child, f"{path}[{index}]")

    find_legacy(data)
    if legacy_paths:
        raise ValueError(
            "Removed token configuration key(s): "
            + ", ".join(legacy_paths)
            + ". Use scalar max_tokens instead."
        )
    return data


def parse_plugin_temperatures(cfg: dict[str, Any]) -> dict[str, Any]:
    """Parse per-plugin temperature settings from a config dict.

    Keys ending in ``_temperature`` are mapped to plugin IDs by replacing
    underscores with hyphens (e.g. ``rate-limiter_temperature`` →
    ``rate-limiter``).
    """
    plugin_temperatures = {}
    for key, value in cfg.items():
        if key.endswith("_temperature"):
            plugin_id = key[:-len("_temperature")].replace("_", "-")
            plugin_temperatures[plugin_id] = value
    return plugin_temperatures


def resolve_model_sources(models: dict[str, Any]) -> dict[str, str]:
    """Resolve model entries to source strings.

    Model entries may be either a source string or a dict with a
    ``source`` key (and optional per-model settings such as ``drop_params``
    and ``plugins_blacklist``).
    Missing/invalid entries default to ``"Default"``.
    """
    resolved = {}
    for name, val in models.items():
        if isinstance(val, dict):
            resolved[name] = val.get("source", "Default")
        elif isinstance(val, str):
            resolved[name] = val
        else:
            resolved[name] = "Default"
    return resolved


_PI_TOOL_NAMES = {"read", "bash", "edit", "write", "grep", "find", "ls"}
_PI_CONFIG_KEYS = {
    "tools", "permissions", "system_prompt", "reasoning", "thinking_budgets",
    "max_tool_calls", "compat", "max_tokens",
}


def _resolve_pi_config(target_name: str, value: Any) -> dict[str, Any]:
    """Validate the small, deterministic Pi configuration surface."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(  # noqa: TRY004 - config validation uses ValueError in this project
            f"Target '{target_name}' pi configuration must be an object"
        )
    unknown = sorted(set(value) - _PI_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            f"Target '{target_name}' pi configuration has unsupported key(s): {', '.join(unknown)}"
        )
    tools = value.get("tools", [])
    if not isinstance(tools, list) or any(not isinstance(item, str) for item in tools):
        raise ValueError(f"Target '{target_name}' pi.tools must be a list of strings")
    unsupported = sorted(set(tools) - _PI_TOOL_NAMES)
    if unsupported:
        raise ValueError(
            f"Target '{target_name}' pi.tools has unsupported tool(s): {', '.join(unsupported)}"
        )
    permissions = value.get("permissions", {})
    if not isinstance(permissions, dict) or any(
        not isinstance(key, str) or value not in {"allow", "deny"}
        for key, value in permissions.items()
    ):
        raise ValueError(
            f"Target '{target_name}' pi.permissions must map tool names to 'allow' or 'deny'"
        )
    unknown_permissions = sorted(set(permissions) - _PI_TOOL_NAMES)
    if unknown_permissions:
        raise ValueError(
            f"Target '{target_name}' pi.permissions has unsupported tool(s): "
            f"{', '.join(unknown_permissions)}"
        )
    system_prompt = value.get("system_prompt")
    if system_prompt is not None and not isinstance(system_prompt, str):
        raise ValueError(f"Target '{target_name}' pi.system_prompt must be a string or null")
    reasoning = value.get("reasoning", False)
    if not isinstance(reasoning, bool):
        raise ValueError(  # noqa: TRY004 - config validation uses ValueError in this project
            f"Target '{target_name}' pi.reasoning must be boolean"
        )
    max_tool_calls = value.get("max_tool_calls", 50)
    if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int) or max_tool_calls < 0:
        raise ValueError(f"Target '{target_name}' pi.max_tool_calls must be a non-negative integer")
    max_tokens = value.get("max_tokens")
    if max_tokens is not None and (
        isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0
    ):
        raise ValueError(f"Target '{target_name}' pi.max_tokens must be a positive integer")
    compat = value.get("compat", {})
    if not isinstance(compat, dict):
        raise ValueError(  # noqa: TRY004 - config validation uses ValueError in this project
            f"Target '{target_name}' pi.compat must be an object"
        )
    thinking_budgets = value.get("thinking_budgets")
    if thinking_budgets is not None and not isinstance(thinking_budgets, dict):
        raise ValueError(f"Target '{target_name}' pi.thinking_budgets must be an object or null")
    return copy.deepcopy(value)


def resolve_targets(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve models and agents into a unified target map.

    Each target contains:
    - ``source``: API source name
    - ``api_model``: actual model string sent to the API
    - ``system_prompt``: optional system prompt for the agent
    - ``is_agent``: whether this target is an agent
    - ``drop_params``: per-target params to drop from API requests
    - ``plugins_blacklist``: per-target plugins to skip
    - ``max_tokens``: per-target max-token override (``None`` = use the
      global ``max_tokens`` / ``--max-tokens``)
    """
    models = cfg.get("models", {})
    agents = cfg.get("agents", {})
    if "token_levels" in cfg or "model_token_levels" in cfg:
        raise ValueError("Removed token_levels configuration; use scalar max_tokens instead")
    # Per-target max-token overrides for thinking-heavy models whose entire
    # ``max_tokens`` budget can be consumed by ``reasoning_content`` before a
    # single content token lands (see ``empty-content-investigation.md``).
    # Keys are target names or ``"{source}/{api_model}"``; scalar values beat
    # the global ``max_tokens`` for that target.
    model_max_tokens = cfg.get("model_max_tokens") or {}
    targets = {}

    def _normalize_max_tokens(value: Any) -> int | None:
        """Coerce one configured positive max-token value to an int."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
        return value

    def _resolve_target_max_tokens(name: str, source: str, api_model: str, val: Any) -> int | None:
        """Return a per-target scalar max-token override, if configured."""
        if isinstance(val, dict):
            token_value = _normalize_max_tokens(val.get("max_tokens"))
            if token_value is not None:
                return token_value
            pi_value = val.get("pi")
            if isinstance(pi_value, dict):
                token_value = _normalize_max_tokens(pi_value.get("max_tokens"))
                if token_value is not None:
                    return token_value
        for key in (name, f"{source}/{api_model}"):
            token_value = _normalize_max_tokens(model_max_tokens.get(key))
            if token_value is not None:
                return token_value
        return None
    for name, val in models.items():
        if isinstance(val, dict):
            targets[name] = {
                "source": val.get("source", "Default"),
                "api_model": name,
                "system_prompt": None,
                "is_agent": False,
                "drop_params": val.get("drop_params", []),
                "plugins_blacklist": list(val.get("plugins_blacklist", [])),
            }
        elif isinstance(val, str):
            targets[name] = {
                "source": val,
                "api_model": name,
                "system_prompt": None,
                "is_agent": False,
                "drop_params": [],
                "plugins_blacklist": [],
            }
        else:
            targets[name] = {
                "source": "Default",
                "api_model": name,
                "system_prompt": None,
                "is_agent": False,
                "drop_params": [],
                "plugins_blacklist": [],
            }
    for name, val in agents.items():
        if not isinstance(val, dict):
            # TRY004 would suggest TypeError, but config-validation errors are
            # ValueError throughout this codebase (and tests pin it).
            raise ValueError(  # noqa: TRY004 - config errors are ValueError throughout
                f"Agent '{name}' must be an object with at least 'model' and 'system_prompt' keys"
            )
        if "model" not in val:
            raise ValueError(f"Agent '{name}' must specify a 'model' key")
        if "system_prompt" not in val:
            raise ValueError(f"Agent '{name}' must specify a 'system_prompt' key")
        targets[name] = {
            "source": val.get("source", "Default"),
            "api_model": val["model"],
            "system_prompt": val["system_prompt"],
            "is_agent": True,
            "drop_params": val.get("drop_params", []),
            "plugins_blacklist": val.get("plugins_blacklist", []),
        }
    # Populate per-target ``max_tokens`` after both loops.
    for name, info in targets.items():
        val = models[name] if name in models else agents.get(name)
        info["max_tokens"] = _resolve_target_max_tokens(
            name, info["source"], info["api_model"], val)
        info["pi"] = _resolve_pi_config(
            name,
            val.get("pi", {}) if isinstance(val, dict) else {},
        )
    return targets


def get_target_plugins_blacklist(targets: dict[str, Any], target_name: str) -> list[str]:
    """Get the plugins blacklist for a specific model or agent."""
    val = targets.get(target_name)
    if isinstance(val, dict):
        result = val.get("plugins_blacklist", [])
        if isinstance(result, list):
            return [str(p) for p in result]
    return []

# Backward-compatible alias.
get_model_plugins_blacklist = get_target_plugins_blacklist


def _apply_http_retry_default(cfg: dict[str, Any], retry_on_429: bool) -> None:
    """Mutate ``cfg`` so HTTP 429 retries align with a global toggle.

    When ``retry_on_429`` is True (the default), this function is a no-op —
    per-source ``max_429_retries`` defaults to 2 inside ``_post_request_context``
    and per-source overrides remain in force. When ``retry_on_429`` is False
    (the user passed ``--no-retry-on-429``), every source that did NOT explicitly
    set ``max_429_retries`` is flipped to ``0`` here so the opt-out propagates
    globally without forcing operators to edit every per-source config block.
    Explicit per-source ``max_429_retries`` values are preserved regardless of
    the global flag — a source that opted in to 5 retries keeps its 5 even
    when the global flag is ``--no-retry-on-429``.

    Mutating ``cfg`` in place is intentional: ``load_config`` returns a fresh
    dict every call, and downstream consumers (``resolve_targets``,
    ``run_model``) read the same object.
    """
    if retry_on_429:
        return
    sources = cfg.get("sources") or {}
    for src_cfg in sources.values():
        if isinstance(src_cfg, dict) and "max_429_retries" not in src_cfg:
            src_cfg["max_429_retries"] = 0


def dump_default_config() -> None:
    """Print the default config JSON to stdout."""
    cfg = {
        "output_dir": "benchmark-output-dir",
        "timeout": 1200,
        "max_tokens": 16384,
        "flush_interval_seconds": FLUSH_INTERVAL_SECONDS,
        "flush_votes": FLUSH_MAX_VOTES,
        "flush_shutdown_timeout_seconds": PERSISTENCE_SHUTDOWN_TIMEOUT,
        "judge": {
            "max_tokens": JUDGE_DEFAULT_MAX_TOKENS,
            "request_params": copy.deepcopy(JUDGE_DEFAULT_REQUEST_PARAMS),
        },
        # Per-target max-token overrides for thinking models; keys are target
        # names or "{source}/{api_model}", values beat the global max_tokens.
        "model_max_tokens": {},
        "model_thread_limit": 1,
        "rate-limiter_temperature": 0.2,
        "moe-dense_temperature": 0.7,
        "plugins_whitelist": [],
        "plugins_blacklist": [],
        "sources": {
            "Local Server 1": {
                "api_url": "http://local.server:11434/chat/completions",
                "headers": {
                    "Authorization": "Bearer ${AI_SERVER_API_KEY:sk-your-key-here}",
                    "Content-Type": "application/json"
                },
                "plugin_thread_limit": 1,
                "model_thread_limit": 1,
                "preload": False,
                "preload_timeout": PRELOAD_DEFAULT_TIMEOUT,
                "opencode_timeout": int(OPENCODE_NO_OUTPUT_GRACE),
                # Live stream watchdog: abort requests whose reasoning or
                # final content exceed these budgets, or whose content/
                # thinking starts repeating itself.
                "max_thinking_tokens": DEFAULT_MAX_THINKING_TOKENS,
                "max_content_tokens": DEFAULT_MAX_CONTENT_TOKENS,
                "repetition_guard": True
            },
            "Local Server 2": {
                "api_url": "http://other.server:11434/chat/completions",
                "headers": {
                    "Authorization": "Bearer ${GAMING_PC_API_KEY:sk-your-key-here}",
                    "Content-Type": "application/json"
                },
                "plugin_thread_limit": 1,
                "model_thread_limit": 1,
                "preload": False,
                "preload_timeout": PRELOAD_DEFAULT_TIMEOUT,
                "opencode_timeout": int(OPENCODE_NO_OUTPUT_GRACE),
                # Live stream watchdog: abort requests whose reasoning or
                # final content exceed these budgets, or whose content/
                # thinking starts repeating itself.
                "max_thinking_tokens": DEFAULT_MAX_THINKING_TOKENS,
                "max_content_tokens": DEFAULT_MAX_CONTENT_TOKENS,
                "repetition_guard": True
            },
            "Remote Provider 1": {
                "api_url": "http://remote.provider:11434/chat/completions",
                "headers": {
                    "Authorization": "Bearer ${REMOTE_API_KEY:sk-your-key-here}",
                    "Content-Type": "application/json"
                },
                "plugin_thread_limit": 1,
                "model_thread_limit": 1,
                "preload": False,
                "preload_timeout": PRELOAD_DEFAULT_TIMEOUT,
                "opencode_timeout": int(OPENCODE_NO_OUTPUT_GRACE),
                # Live stream watchdog: abort requests whose reasoning or
                # final content exceed these budgets, or whose content/
                # thinking starts repeating itself.
                "max_thinking_tokens": DEFAULT_MAX_THINKING_TOKENS,
                "max_content_tokens": DEFAULT_MAX_CONTENT_TOKENS,
                "repetition_guard": True
            },
            "1min.ai": {
                "api_protocol": "1min",
                "api_url": "https://api.1min.ai/api/chat-with-ai",
                "headers": {
                    "API-KEY": "${ONEMIN_API_KEY:your-1min-api-key}",
                    "Content-Type": "application/json"
                },
                "plugin_thread_limit": 1,
                "model_thread_limit": 1,
                "preload": False,
                "preload_timeout": PRELOAD_DEFAULT_TIMEOUT,
                "opencode_timeout": int(OPENCODE_NO_OUTPUT_GRACE),
                # Live stream watchdog: abort requests whose reasoning or
                # final content exceed these budgets, or whose content/
                # thinking starts repeating itself.
                "max_thinking_tokens": DEFAULT_MAX_THINKING_TOKENS,
                "max_content_tokens": DEFAULT_MAX_CONTENT_TOKENS,
                "repetition_guard": True
            },
            "ChatPlayground": {
                "api_protocol": "chatplayground",
                "base_url": "https://web.chatplayground.ai",
                "email": "${CHATPLAYGROUND_EMAIL:you@example.com}",
                "password": "${CHATPLAYGROUND_PASSWORD:your-password}",
                "headless": True,
                "plugin_thread_limit": 1,
                "model_thread_limit": 1,
                "preload": False,
                "preload_timeout": PRELOAD_DEFAULT_TIMEOUT,
                "opencode_timeout": int(OPENCODE_NO_OUTPUT_GRACE)
            }
        },
        "models": {
            "example-model-1": "Local Server 1",
            "example-model-2": "Remote Provider 1",
            "example-model-3": {
                "source": "Local Server 2",
                "drop_params": ["seed"],
                "max_tokens": 32768
            }
        },
        "agents": {
            "example-agent": {
                "model": "gpt-4",
                "source": "Remote Provider 1",
                "system_prompt": "You are a helpful coding assistant. Be concise and accurate."
            }
        }
    }
    print(json.dumps(cfg, indent=2))


def generate_config_from_api(base_url: str, api_key: str | None = None) -> dict[str, Any]:
    """Build a benchmark config dict by discovering models via the /v1/models endpoint."""
    model_ids = fetch_models_v1(base_url, api_key)
    if not model_ids:
        raise RuntimeError("No models returned by /v1/models endpoint.")

    source_name = "Default"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return {
        "output_dir": "benchmark-results",
        "timeout": 600,
        "max_tokens": 16384,
        "plugins_whitelist": [],
        "plugins_blacklist": [],
        "sources": {
            source_name: {
                "api_url": base_url.rstrip("/") + "/chat/completions",
                "headers": headers,
            }
        },
        "models": {mid: source_name for mid in model_ids},
    }


@dataclass
class Configuration:
    """Immutable-at-boundary normalized configuration for a benchmark run."""

    raw: dict[str, Any]
    targets: dict[str, dict[str, Any]] = field(default_factory=dict)
    plugin_temperatures: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str, args: Any | None = None) -> Configuration:
        loaded = load_config(path)
        if not isinstance(loaded, dict):
            raise ValueError(f"Configuration must be an object: {path}")
        return cls.from_mapping(loaded, args)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any], args: Any | None = None) -> Configuration:
        raw = copy.deepcopy(mapping)
        if args is not None:
            cls._apply_cli_overrides(raw, args)
        return cls(raw, resolve_targets(raw), parse_plugin_temperatures(raw))

    @staticmethod
    def _apply_cli_overrides(raw: dict[str, Any], args: Any) -> None:
        retry = getattr(args, "retry_on_429", None)
        if retry is not None:
            _apply_http_retry_default(raw, retry)
        for key in ("timeout", "max_tokens", "temperature"):
            value = getattr(args, key, None)
            if value is not None:
                raw[key] = value
        limit = getattr(args, "plugin_thread_limit", None)
        if limit is not None:
            for source in (raw.get("sources") or {}).values():
                if isinstance(source, dict):
                    source["plugin_thread_limit"] = limit
        for item in getattr(args, "plugin_temperature", []) or []:
            plugin, separator, value = item.partition("=")
            if not separator:
                raise ValueError(f"Invalid --plugin-temperature value: {item}")
            try:
                raw[f"{plugin}_temperature"] = float(value)
            except ValueError as exc:
                raise ValueError(f"Invalid plugin temperature: {item}") from exc

    def source_config(self) -> dict[str, Any]:
        value = self.raw.get("sources", {})
        return value if isinstance(value, dict) else {}

    def models(self) -> dict[str, Any]:
        value = self.raw.get("models", {})
        return value if isinstance(value, dict) else {}

    def value(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]
