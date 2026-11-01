from __future__ import annotations

from typing import Any

from agentos.llm.base import BaseLLM, ChatResult, SchemaT, Usage
from agentos.mcp.provider import ToolProvider
from agentos.models import ToolMetadata
from agentos.taskbench.schema import FaultSpec


class MeteredLLM(BaseLLM):
    """Records the provider-reported usage of every call made during one task run."""

    def __init__(self, inner: BaseLLM):
        self.inner = inner
        self.usages: list[Usage] = []

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResult:
        result = await self.inner.chat(messages, **kwargs)
        self.usages.append(result.usage)
        return result

    async def chat_json(
        self, messages: list[dict[str, Any]], schema: type[SchemaT], **kwargs: Any
    ) -> tuple[SchemaT, Usage]:
        # Count the call even when the JSON fails to validate: the tokens were spent.
        result = await self.inner.chat(messages, **kwargs)
        self.usages.append(result.usage)
        from agentos.llm.base import extract_json

        return schema.model_validate_json(extract_json(result.content)), result.usage

    def totals(self) -> dict[str, Any]:
        return {
            "llm_calls": len(self.usages),
            "input_tokens": sum(u.input_tokens for u in self.usages),
            "output_tokens": sum(u.output_tokens for u in self.usages),
            "reasoning_tokens": sum(u.reasoning_tokens for u in self.usages),
            "usage_estimated": any(u.usage_estimated for u in self.usages),
        }


class InjectedFault(RuntimeError):
    pass


class InstrumentedProvider:
    """Counts tool calls and injects the task's fault identically for every method."""

    def __init__(self, inner: ToolProvider, fault: FaultSpec | None = None):
        self.inner = inner
        self.fault = fault
        self.calls = 0
        self.errors = 0
        self.side_effect_calls = 0
        self.fault_hits = 0
        self._idempotent: dict[str, bool] = {}

    async def list_tools(self) -> list[ToolMetadata]:
        tools = await self.inner.list_tools()
        self._idempotent = {tool.name: tool.idempotent for tool in tools}
        return tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any], operation_id: str | None = None
    ) -> Any:
        self.calls += 1
        if not self._idempotent.get(name, True):
            self.side_effect_calls += 1
        fault = self.fault
        if fault and fault.tool == name and (fault.mode == "outage" or self.fault_hits < fault.failures):
            self.fault_hits += 1
            self.errors += 1
            raise InjectedFault(f"503 service unavailable: injected {fault.mode} fault in {name}")
        try:
            return await self.inner.call_tool(name, arguments, operation_id=operation_id)
        except Exception:
            self.errors += 1
            raise
