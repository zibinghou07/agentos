from __future__ import annotations

import httpx

from agentos.config import Settings
from agentos.llm.openai_compatible import OpenAICompatibleProvider


class BailianProvider(OpenAICompatibleProvider):
    """Bailian/DashScope Chat Completions with credential/endpoint guardrails."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        key = settings.api_key.strip()
        if not key:
            raise ValueError(
                "Missing API key: set DASHSCOPE_API_KEY in .env (never commit it)"
            )
        is_token_plan_key = key.startswith("sk-sp-")
        effective_mode = settings.effective_bailian_mode()
        endpoint = settings.resolved_base_url()
        if effective_mode == "token_plan" and not is_token_plan_key:
            raise ValueError(
                "Token Plan requires an sk-sp-... key; do not mix it with a pay-as-you-go key"
            )
        if effective_mode != "token_plan" and is_token_plan_key:
            raise ValueError(
                "An sk-sp-... Token Plan key must use AGENTOS_BAILIAN_MODE=token_plan"
            )
        if effective_mode == "token_plan" and "token-plan." not in endpoint:
            raise ValueError(
                "Token Plan keys must use the token-plan.cn-beijing.maas.aliyuncs.com endpoint"
            )
        if effective_mode != "token_plan" and "token-plan." in endpoint:
            raise ValueError("The Token Plan endpoint requires AGENTOS_BAILIAN_MODE=token_plan")
        super().__init__(
            api_key=key,
            base_url=endpoint,
            model=settings.model,
            timeout=settings.timeout_seconds,
            max_concurrency=settings.max_concurrency,
            transport=transport,
            enable_thinking=settings.enable_thinking,
        )

