from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

TOKEN_PLAN_BASE_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
LEGACY_BEIJING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
REGION_DOMAINS = {
    "cn-beijing": "cn-beijing.maas.aliyuncs.com",
    "ap-southeast-1": "ap-southeast-1.maas.aliyuncs.com",
    "us-east-1": "us-east-1.maas.aliyuncs.com",
    "eu-central-1": "eu-central-1.maas.aliyuncs.com",
    "ap-northeast-1": "ap-northeast-1.maas.aliyuncs.com",
    "cn-hongkong": "cn-hongkong.maas.aliyuncs.com",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AGENTOS_", extra="ignore")

    provider: Literal["mock", "bailian", "openai_compatible"] = "mock"
    model: str = "qwen3.8-flash"
    api_key: str = Field(
        default="",
        validation_alias=AliasChoices("AGENTOS_API_KEY", "DASHSCOPE_API_KEY"),
        repr=False,
    )
    base_url: str = Field(
        default="",
        validation_alias=AliasChoices("AGENTOS_BASE_URL", "DASHSCOPE_BASE_URL"),
    )
    bailian_mode: Literal["token_plan", "pay_as_you_go", "custom"] = "token_plan"
    bailian_workspace_id: str = ""
    bailian_region: Literal[
        "cn-beijing",
        "ap-southeast-1",
        "us-east-1",
        "eu-central-1",
        "ap-northeast-1",
        "cn-hongkong",
    ] = "cn-beijing"
    timeout_seconds: float = 30.0
    max_concurrency: int = 4
    # None keeps the model default; false disables Qwen3 thinking for lower latency/cost.
    enable_thinking: bool | None = None
    database: Path = Path("data/agentos.db")
    world_database: Path = Path("data/world.db")
    routing_epsilon: float = 0.05
    log_level: str = "INFO"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    def effective_bailian_mode(self) -> Literal["token_plan", "pay_as_you_go", "custom"]:
        """Infer the endpoint family from the credential when available.

        New-user free quota uses an ordinary sk- key. It is a billing credit on
        the standard API surface, not a Token Plan credential.
        """
        if self.bailian_mode == "custom":
            return "custom"
        key = self.api_key.strip()
        if key:
            return "token_plan" if key.startswith("sk-sp-") else "pay_as_you_go"
        return self.bailian_mode

    def resolved_base_url(self) -> str:
        """Resolve an endpoint without ever deriving or exposing credentials."""
        if self.base_url:
            return self.base_url.rstrip("/")
        effective_mode = self.effective_bailian_mode()
        if effective_mode == "token_plan":
            return TOKEN_PLAN_BASE_URL
        if effective_mode == "custom":
            raise ValueError("AGENTOS_BASE_URL is required when AGENTOS_BAILIAN_MODE=custom")
        if self.bailian_workspace_id:
            domain = REGION_DOMAINS[self.bailian_region]
            return (
                f"https://{self.bailian_workspace_id}.{domain}/compatible-mode/v1"
            )
        if self.bailian_region == "cn-beijing":
            return LEGACY_BEIJING_BASE_URL
        raise ValueError(
            "AGENTOS_BAILIAN_WORKSPACE_ID is required outside cn-beijing"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()

