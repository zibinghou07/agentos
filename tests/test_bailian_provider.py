import httpx
import pytest

from agentos.config import Settings, TOKEN_PLAN_BASE_URL
from agentos.llm.bailian import BailianProvider


def token_plan_settings(**overrides):
    values = {
        "_env_file": None,
        "AGENTOS_API_KEY": "sk-sp-not-a-real-key",
        "provider": "bailian",
        "model": "qwen3.8-flash",
    }
    values.update(overrides)
    return Settings(**values)


def test_token_plan_endpoint_and_key_guardrail():
    settings = token_plan_settings()
    assert settings.resolved_base_url() == TOKEN_PLAN_BASE_URL
    assert settings.api_key.startswith("sk-sp-")
    free_quota = token_plan_settings(AGENTOS_API_KEY="sk-free-quota-example")
    assert free_quota.effective_bailian_mode() == "pay_as_you_go"
    assert free_quota.resolved_base_url() == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    BailianProvider(free_quota)
    token_plan = token_plan_settings(bailian_mode="pay_as_you_go")
    assert token_plan.effective_bailian_mode() == "token_plan"
    assert token_plan.resolved_base_url() == TOKEN_PLAN_BASE_URL
    with pytest.raises(ValueError, match="Token Plan keys must use"):
        BailianProvider(
            token_plan_settings(
                AGENTOS_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        )


async def test_bailian_chat_parses_exact_usage_without_leaking_key():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer sk-sp-not-a-real-key"
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(
            200,
            json={
                "id": "req-test",
                "model": "qwen3.8-flash",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    provider = BailianProvider(
        token_plan_settings(), transport=httpx.MockTransport(handler)
    )
    result = await provider.chat([{"role": "user", "content": "test"}])
    assert result.content == "ok"
    assert result.usage.input_tokens == 3
    assert result.usage.output_tokens == 1
    assert result.usage.usage_estimated is False


async def test_bailian_retries_protocol_disconnect():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.RemoteProtocolError("server disconnected", request=request)
        return httpx.Response(
            200,
            json={
                "id": "req-retry",
                "model": "qwen3.8-flash",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    provider = BailianProvider(
        token_plan_settings(), transport=httpx.MockTransport(handler)
    )
    result = await provider.chat([{"role": "user", "content": "test"}])
    assert result.content == "ok"
    assert calls == 2
