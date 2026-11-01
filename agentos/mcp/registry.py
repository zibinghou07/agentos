from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from agentos.models import ToolMetadata
from agentos.runtime.routing import lexical_similarity


@dataclass
class RegisteredTool:
    metadata: ToolMetadata
    function: Callable[..., Any]
    calls: int = 0
    successes: int = 0
    total_latency_ms: float = 0


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, RegisteredTool] = {}
        self._operation_cache: dict[str, Any] = {}

    def register(self, metadata: ToolMetadata, function: Callable[..., Any]) -> str:
        name = metadata.name
        if name in self._tools:
            name = f"{metadata.server_name}.{name}"
            metadata = metadata.model_copy(update={"name": name})
        self._tools[name] = RegisteredTool(metadata, function)
        return name

    def list_tools(self) -> list[ToolMetadata]:
        return [entry.metadata for entry in self._tools.values() if entry.metadata.enabled]

    def search(self, query: str, top_k: int = 5) -> list[ToolMetadata]:
        scored = []
        for entry in self._tools.values():
            if not entry.metadata.enabled:
                continue
            history = (entry.successes + 1) / (entry.calls + 2)
            score = 0.8 * lexical_similarity(query, f"{entry.metadata.name} {entry.metadata.description}") + 0.2 * history
            scored.append((score, entry.metadata))
        return [tool for _, tool in sorted(scored, key=lambda x: (-x[0], x[1].name))[:top_k]]

    def infer_arguments(self, name: str, query: str) -> dict[str, Any]:
        schema = self._tools[name].metadata.input_schema
        args = {}
        for key, spec in schema.get("properties", {}).items():
            if key in {"query", "keyword"}:
                args[key] = query
            elif key == "top_k":
                args[key] = min(5, spec.get("maximum", 5))
            elif "default" in spec:
                args[key] = spec["default"]
        return args

    async def call(self, name: str, arguments: dict[str, Any], operation_id: str | None = None) -> Any:
        if operation_id and operation_id in self._operation_cache:
            return self._operation_cache[operation_id]
        if name not in self._tools or not self._tools[name].metadata.enabled:
            raise KeyError(f"unknown or disabled tool: {name}")
        entry = self._tools[name]
        schema = entry.metadata.input_schema
        Draft202012Validator(schema).validate(arguments)
        started = time.perf_counter()
        entry.calls += 1
        try:
            outcome = entry.function(**arguments)
            if inspect.isawaitable(outcome):
                outcome = await asyncio.wait_for(outcome, timeout=entry.metadata.timeout)
            entry.successes += 1
            if operation_id:
                self._operation_cache[operation_id] = outcome
            return outcome
        finally:
            entry.total_latency_ms += (time.perf_counter() - started) * 1000

