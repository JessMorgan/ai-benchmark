"""Canonical judge contract identity and serialization."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JudgeContract:
    """Complete immutable definition of one judge request contract."""

    contract_id: str
    plugin_id: str
    plugin_version: str
    prompt_version: str
    instructions_version: str
    response_schema_hash: str
    contract_json: str
    contract_hash: str

    @classmethod
    def from_definition(
        cls,
        *,
        plugin_id: str,
        plugin_version: str,
        prompt_version: str,
        instructions_version: str,
        response_schema: Mapping[str, Any],
        instructions: str,
    ) -> JudgeContract:
        definition = {
            "schema": "judge-contract-v1",
            "prompt_version": prompt_version,
            "response_schema": response_schema,
            "plugin_id": plugin_id,
            "plugin_version": plugin_version,
            "judge_instructions_version": instructions_version,
            "judge_instructions": instructions,
        }
        contract_json = json.dumps(
            definition, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        contract_hash = hashlib.sha256(contract_json.encode("utf-8")).hexdigest()
        response_schema_json = json.dumps(
            response_schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        response_schema_hash = hashlib.sha256(response_schema_json.encode("utf-8")).hexdigest()
        return cls(
            contract_id=f"judge-contract-v1:{contract_hash}",
            plugin_id=plugin_id,
            plugin_version=plugin_version,
            prompt_version=prompt_version,
            instructions_version=instructions_version,
            response_schema_hash=response_schema_hash,
            contract_json=contract_json,
            contract_hash=contract_hash,
        )

    @classmethod
    def from_json(
        cls,
        *,
        contract_id: str,
        plugin_id: str,
        plugin_version: str,
        prompt_version: str,
        instructions_version: str,
        response_schema_hash: str,
        contract_json: str,
        contract_hash: str,
    ) -> JudgeContract:
        """Build a value from persisted data, rejecting hash drift."""
        actual_hash = hashlib.sha256(contract_json.encode("utf-8")).hexdigest()
        if actual_hash != contract_hash:
            raise ValueError(f"judge contract hash mismatch: {contract_id}")
        return cls(
            contract_id=contract_id,
            plugin_id=plugin_id,
            plugin_version=plugin_version,
            prompt_version=prompt_version,
            instructions_version=instructions_version,
            response_schema_hash=response_schema_hash,
            contract_json=contract_json,
            contract_hash=contract_hash,
        )

    def as_spec(self) -> dict[str, Any]:
        """Return storage-neutral fields for adapters and importers."""
        return {
            "contract_id": self.contract_id,
            "plugin_id": self.plugin_id,
            "plugin_version": self.plugin_version,
            "prompt_version": self.prompt_version,
            "instructions_version": self.instructions_version,
            "response_schema_hash": self.response_schema_hash,
            "contract": self.contract_json,
            "contract_hash": self.contract_hash,
        }
