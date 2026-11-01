from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from uuid import uuid4


from agentos.llm.base import BaseLLM, ChatResult, SchemaT, Usage, extract_json


class MockProvider(BaseLLM):
    """Deterministic provider for offline integration and benchmark tests."""

    def __init__(self, responses: list[str | dict[str, Any]] | None = None, latency_ms: float = 1.0):
        self.responses = list(responses or ["mock response"])
        self.latency_ms = latency_ms
        self.index = 0

    def _next(self) -> str | dict[str, Any]:
        value = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return value

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResult:
        started = time.perf_counter()
        await asyncio.sleep(self.latency_ms / 1000)
        value = self._next()
        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        input_tokens = sum(len(str(m.get("content") or "").split()) for m in messages)
        usage = Usage(
            request_id=f"mock-{uuid4()}", model_name="mock-v1", input_tokens=input_tokens,
            output_tokens=len(content.split()), latency_ms=(time.perf_counter() - started) * 1000,
            usage_estimated=False,
        )
        return ChatResult(content=content, usage=usage, raw={"mock": True})

    async def chat_json(self, messages: list[dict[str, Any]], schema: type[SchemaT], **kwargs: Any) -> tuple[SchemaT, Usage]:
        result = await self.chat(messages, **kwargs)
        return schema.model_validate_json(extract_json(result.content)), result.usage

