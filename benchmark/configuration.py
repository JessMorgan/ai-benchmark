"""Normalized benchmark configuration."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .core import (
    _apply_http_retry_default,
    load_config,
    parse_plugin_temperatures,
    resolve_targets,
)


@dataclass
class Configuration:
    """Configuration shared by commands and benchmark execution.

    ``raw`` is the mutable, fully expanded config mapping. CLI overrides are
    applied once in :meth:`from_file`, while derived targets and temperatures
    are exposed consistently to callers.
    """

    raw: dict[str, Any]
    targets: dict[str, dict[str, Any]] = field(default_factory=dict)
    plugin_temperatures: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str, args: Any | None = None) -> Configuration:
        loaded = load_config(path)
        if not isinstance(loaded, dict):
            raise ValueError(f"Configuration must be an object: {path}")
        raw = deepcopy(loaded)
        if args is not None:
            cls._apply_cli_overrides(raw, args)
        return cls(
            raw=raw,
            targets=resolve_targets(raw),
            plugin_temperatures=parse_plugin_temperatures(raw),
        )

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any], args: Any | None = None) -> Configuration:
        raw = deepcopy(mapping)
        if args is not None:
            cls._apply_cli_overrides(raw, args)
        return cls(raw=raw, targets=resolve_targets(raw),
                   plugin_temperatures=parse_plugin_temperatures(raw))

    @staticmethod
    def _apply_cli_overrides(raw: dict[str, Any], args: Any) -> None:
        """Apply only configuration overrides, not command-selection flags."""
        retry = getattr(args, "retry_on_429", None)
        if retry is not None:
            _apply_http_retry_default(raw, retry)
        if getattr(args, "timeout", None) is not None:
            raw["timeout"] = args.timeout
        if getattr(args, "max_tokens", None) is not None:
            raw["max_tokens"] = args.max_tokens
        if getattr(args, "temperature", None) is not None:
            raw["temperature"] = args.temperature
        if getattr(args, "plugin_thread_limit", None) is not None:
            for source in (raw.get("sources") or {}).values():
                if isinstance(source, dict):
                    source["plugin_thread_limit"] = args.plugin_thread_limit
        if getattr(args, "plugin_temperature", None):
            for item in args.plugin_temperature:
                plugin_id, separator, value = item.partition("=")
                if not separator:
                    raise ValueError(f"Invalid --plugin-temperature value: {item}")
                try:
                    raw[f"{plugin_id}_temperature"] = float(value)
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
