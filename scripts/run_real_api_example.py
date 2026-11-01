from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agentos.benchmarks.real_api import build_request, evaluate_api_result
from agentos.benchmarks.real_data import OpenSourceTask, read_tasks
from agentos.config import get_settings
from agentos.llm.bailian import BailianProvider


def select_task(
    tasks: list[OpenSourceTask], source: str, index: int, task_id: str | None
) -> OpenSourceTask:
    if task_id:
        for task in tasks:
            if task.task_id == task_id:
                return task
        raise ValueError(f"task ID not found: {task_id}")
    selected = [task for task in tasks if source == "any" or task.source.dataset.startswith(source)]
    if index < 0 or index >= len(selected):
        raise ValueError(f"index {index} is outside 0..{len(selected) - 1} for source={source}")
    return selected[index]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Call Bailian on one real benchmark task")
    parser.add_argument("--input", type=Path, default=Path("benchmarks/real_data/test.jsonl"))
    parser.add_argument("--source", choices=["bfcl", "hotpotqa", "any"], default="bfcl")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--task-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/results/real_api_example.json")
    )
    args = parser.parse_args()

    settings = get_settings()
    task = select_task(read_tasks(args.input), args.source, args.index, args.task_id)
    messages, request_kwargs = build_request(task)
    mapping = request_kwargs.pop("_name_mapping", None)
    preview = {
        "dry_run": args.dry_run,
        "task_id": task.task_id,
        "source": task.source.dataset,
        "model": settings.model,
        "configured_mode": settings.bailian_mode,
        "effective_mode": settings.effective_bailian_mode(),
        "base_url": settings.resolved_base_url(),
        "api_key_configured": bool(settings.api_key),
        "prompt": task.prompt,
        "candidate_tools": [tool.get("name") for tool in task.tools],
    }
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return
    if settings.provider != "bailian":
        raise SystemExit("Set AGENTOS_PROVIDER=bailian in .env before making a real API call")
    if not settings.api_key:
        raise SystemExit("Set DASHSCOPE_API_KEY in .env before making a real API call")

    provider = BailianProvider(settings)
    result = await provider.chat(
        messages,
        temperature=0,
        max_tokens=512,
        **request_kwargs,
    )
    report = evaluate_api_result(task, result, mapping)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
