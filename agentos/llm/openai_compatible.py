from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from uuid import uuid4

import httpx

from agentos.llm.base import BaseLLM, ChatResult, SchemaT, Usage, extract_json


class OpenAICompatibleProvider(BaseLLM):
    """OpenAI-compatible chat API, including Qwen DashScope compatibility mode."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30,
        max_concurrency: int = 4,
        retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
        enable_thinking: bool | None = None,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key, self.base_url, self.model = api_key, base_url.rstrip("/"), model
        self.timeout, self.retries = timeout, retries
        self.transport = transport
        # DashScope-compatible thinking switch; None leaves the provider default.
        self.enable_thinking = enable_thinking
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResult:
        started = time.perf_counter()
        payload = {"model": self.model, "messages": messages, **kwargs}
        if self.enable_thinking is not None:
            payload.setdefault("enable_thinking", self.enable_thinking)
        error: Exception | None = None
        async with self.semaphore:
            for attempt in range(self.retries + 1):
                try:
                    async with httpx.AsyncClient(
                        timeout=self.timeout, transport=self.transport
                    ) as client:
                        response = await client.post(
                            f"{self.base_url}/chat/completions",
                            headers={"Authorization": f"Bearer {self.api_key}"}, json=payload,
                        )
                        response.raise_for_status()
                    raw = response.json()
                    usage_raw = raw.get("usage") or {}
                    # A function-calling response can legally have null content and only
                    # expose message.tool_calls.
                    content = raw["choices"][0]["message"].get("content") or ""
                    exact = bool(usage_raw)
                    usage = Usage(
                        request_id=raw.get("id", str(uuid4())), model_name=raw.get("model", self.model),
                        input_tokens=usage_raw.get("prompt_tokens", 0), output_tokens=usage_raw.get("completion_tokens", 0),
                        reasoning_tokens=(usage_raw.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
                        latency_ms=(time.perf_counter() - started) * 1000, usage_estimated=not exact,
                    )
                    return ChatResult(content=content, usage=usage, raw=raw)
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    error = exc
                    retryable = not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code in {429, 500, 502, 503, 504}
                    if attempt >= self.retries or not retryable:
                        break
                    await asyncio.sleep(0.25 * 2**attempt)
        assert error is not None
        raise error

    async def chat_json(self, messages: list[dict[str, Any]], schema: type[SchemaT], **kwargs: Any) -> tuple[SchemaT, Usage]:
        kwargs.setdefault("response_format", {"type": "json_object"})
        result = await self.chat(messages, **kwargs)
        return schema.model_validate(json.loads(extract_json(result.content))), result.usage

