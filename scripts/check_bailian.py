from __future__ import annotations

import argparse
import asyncio
import json

from agentos.config import get_settings
from agentos.llm import BailianProvider


async def main(dry_run: bool) -> None:
    settings = get_settings()
    endpoint = settings.resolved_base_url()
    safe = {
        "provider": settings.provider,
        "configured_mode": settings.bailian_mode,
        "effective_mode": settings.effective_bailian_mode(),
        "model": settings.model,
        "base_url": endpoint,
        "api_key_configured": bool(settings.api_key),
    }
    print(json.dumps(safe, ensure_ascii=False, indent=2))
    if dry_run:
        return
    provider = BailianProvider(settings)
    result = await provider.chat(
        [{"role": "user", "content": "只回复：AgentOS API OK"}],
        temperature=0,
        max_tokens=32,
    )
    print(result.content)
    print(result.usage.model_dump_json(indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bailian connection check")
    parser.add_argument(
        "--dry-run", action="store_true", help="show sanitized configuration without an API call"
    )
    asyncio.run(main(parser.parse_args().dry_run))
