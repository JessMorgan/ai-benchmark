"""Shared internal typing primitives for benchmark production modules."""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]
JSONList: TypeAlias = list[JSONValue]
ConfigValue: TypeAlias = JSONValue
ConfigMap: TypeAlias = dict[str, ConfigValue]
StringCallback: TypeAlias = Callable[[str], None]
NoArgCallback: TypeAlias = Callable[[], None]
