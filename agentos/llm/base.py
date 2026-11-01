from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel, Field

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class Usage(BaseModel):
    request_id: str
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    # Thinking-mode models count hidden reasoning inside output_tokens; keep it visible.
    reasoning_tokens: int = 0
    latency_ms: float
    estimated_cost: float = 0.0
    usage_estimated: bool = False
    success: bool = True
    error_type: str | None = None


class ChatResult(BaseModel):
    content: str
    usage: Usage
    raw: dict[str, Any] = Field(default_factory=dict)


def extract_json(content: str) -> str:
    """Return the first complete JSON value, ignoring Markdown fences, prose or trailing text."""
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)(?:```|$)", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        return text
    start = min(starts)
    try:
        _, end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError:
        return text[start:]
    return text[start:end]


class BaseLLM(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResult: ...

    @abstractmethod
    async def chat_json(self, messages: list[dict[str, Any]], schema: type[SchemaT], **kwargs: Any) -> tuple[SchemaT, Usage]: ...

