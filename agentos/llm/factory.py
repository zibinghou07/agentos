from __future__ import annotations

from agentos.config import Settings, get_settings
from agentos.llm.bailian import BailianProvider
from agentos.llm.base import BaseLLM
from agentos.llm.mock import MockProvider
from agentos.llm.openai_compatible import OpenAICompatibleProvider


def create_llm(settings: Settings | None = None) -> BaseLLM:
    settings = settings or get_settings()
    if settings.provider == "mock":
        return MockProvider()
    if settings.provider == "bailian":
        return BailianProvider(settings)
    return OpenAICompatibleProvider(
        api_key=settings.api_key,
        base_url=settings.resolved_base_url(),
        model=settings.model,
        timeout=settings.timeout_seconds,
        max_concurrency=settings.max_concurrency,
        enable_thinking=settings.enable_thinking,
    )
