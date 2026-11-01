from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agentos.api import build_runtime_service
from agentos.config import get_settings


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run one complete AgentOS goal")
    parser.add_argument(
        "--goal",
        default=(
            "Search the local documents for how AgentOS handles recovery and return "
            "the most relevant evidence."
        ),
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--database",
        type=Path,
        help="checkpoint database (use /tmp for disposable smoke runs)",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.database:
        settings = settings.model_copy(update={"database": args.database})
    service = build_runtime_service(settings)
    if args.dry_run:
        tools = await service.executor.tool_provider.list_tools()
        print(
            json.dumps(
                {
                    "provider": settings.provider,
                    "model": settings.model,
                    "effective_mode": settings.effective_bailian_mode(),
                    "api_key_configured": bool(settings.api_key),
                    "goal": args.goal,
                    "discovered_tools": [tool.name for tool in tools],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    record = await service.submit(args.goal)
    await asyncio.wait_for(service.wait(record.task_id), timeout=args.timeout)
    finished = await service.get_task(record.task_id)
    events = await service.get_trace(record.task_id)
    operations = await service.get_operations(record.task_id)
    report = {
        "task": finished.model_dump(mode="json") if finished else None,
        "events": [event.model_dump(mode="json") for event in events],
        "operations": operations,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"saved: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
