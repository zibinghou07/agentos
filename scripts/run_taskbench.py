"""Run TaskBench end-to-end with a real LLM provider and write runs + summary.

Examples:
  .venv/bin/python scripts/run_taskbench.py --dry-run
  .venv/bin/python scripts/run_taskbench.py --split dev --per-family 1 --methods react,agentos
  .venv/bin/python scripts/run_taskbench.py --split test
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections import Counter
from pathlib import Path

from agentos.config import get_settings
from agentos.llm import create_llm
from agentos.taskbench.analysis import summarize
from agentos.taskbench.generator import load_tasks, write_taskbench
from agentos.taskbench.hotpot import hotpot_tasks
from agentos.taskbench.runner import METHODS, dump_json, load_records, run_suite

BENCH_DIR = Path("benchmarks/taskbench")
HOTPOT = Path("benchmarks/real_data/all.jsonl")


def select(tasks, families: set[str] | None, per_family: int | None, skip_per_family: int = 0):
    """Pick tasks per family in file order; ``skip_per_family`` skips tasks already inspected."""
    chosen, seen = [], Counter()
    for task in tasks:
        if families and task.family not in families:
            continue
        seen[task.family] += 1
        if seen[task.family] <= skip_per_family:
            continue
        if per_family is not None and seen[task.family] > skip_per_family + per_family:
            continue
        chosen.append(task)
    return chosen


async def main() -> None:
    # Per-request HTTP and MCP tool-error logs would drown the per-task progress lines.
    for name in ("httpx", "mcp", "mcp.server"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--methods", default="react,fixed_roles,agentos")
    parser.add_argument("--families", default="")
    parser.add_argument("--per-family", type=int)
    parser.add_argument(
        "--skip-per-family", type=int, default=0,
        help="skip the first N tasks of each family (e.g. ones whose failures were already analysed)",
    )
    parser.add_argument("--no-hotpot", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--token-budget", type=int,
        help="stop starting new runs once this invocation has spent this many tokens",
    )
    parser.add_argument("--dry-run", action="store_true", help="show the task mix; send no request")
    parser.add_argument("--allow-mock", action="store_true", help="mechanics smoke test only")
    args = parser.parse_args()

    if not (BENCH_DIR / f"{args.split}.jsonl").exists():
        write_taskbench(BENCH_DIR)
    tasks = load_tasks(BENCH_DIR / f"{args.split}.jsonl")
    if not args.no_hotpot and HOTPOT.exists():
        tasks += hotpot_tasks(HOTPOT, args.split)
    families = {f for f in args.families.split(",") if f} or None
    tasks = select(tasks, families, args.per_family, args.skip_per_family)
    methods = [m for m in args.methods.split(",") if m]
    unknown = set(methods) - set(METHODS)
    if unknown:
        parser.error(f"unknown methods {sorted(unknown)}; known={sorted(METHODS)}")

    settings = get_settings()
    mix = Counter(task.family for task in tasks)
    print(f"split={args.split} tasks={len(tasks)} methods={methods} repeats={args.repeats}")
    print(f"provider={settings.provider} model={settings.model} enable_thinking={settings.enable_thinking}")
    print("families:", dict(sorted(mix.items())))
    if args.dry_run:
        return
    if settings.provider == "mock" and not args.allow_mock:
        sys.exit("Refusing to report mock results as an evaluation. Configure a real provider in .env.")

    output = args.output or Path(f"experiments/results/taskbench/{args.split}/runs.jsonl")
    llm = create_llm(settings.model_copy(update={"timeout_seconds": max(settings.timeout_seconds, 120.0)}))

    def progress(record) -> None:
        mark = "ok " if record.success else "FAIL"
        print(
            f"[{mark}] {record.method:24} {record.task_id:48} score={record.score:.2f} "
            f"tokens={record.input_tokens + record.output_tokens:6d} {record.latency_ms / 1000:6.1f}s"
            + (f" error={record.error[:80]}" if record.error else ""),
            flush=True,
        )

    await run_suite(
        tasks, methods, llm, output,
        repeats=args.repeats, concurrency=args.concurrency, timeout=args.timeout,
        provider_name=settings.provider, model_name=settings.model, on_record=progress,
        token_budget=args.token_budget,
    )
    wanted = {(r, m, t.task_id) for r in range(args.repeats) for m in methods for t in tasks}
    records = [r for r in load_records(output) if (r.run_id, r.method, r.task_id) in wanted]
    # Only tasks finished by every method enter the comparison (budget may stop a run early).
    keys = {(r.run_id, r.method, r.task_id) for r in records}
    complete = {
        (r.run_id, r.task_id)
        for r in records
        if all((r.run_id, m, r.task_id) in keys for m in methods)
    }
    dropped = len({(r.run_id, r.task_id) for r in records} - complete)
    records = [r for r in records if (r.run_id, r.task_id) in complete]
    if dropped:
        print(f"excluded {dropped} task(s) not finished by every method")
    summary = summarize(records)
    summary["config"] = {
        "split": args.split, "tasks": len(tasks), "methods": methods, "repeats": args.repeats,
        "provider": settings.provider, "model": settings.model,
        "enable_thinking": settings.enable_thinking, "families": dict(mix),
        "token_budget": args.token_budget, "complete_tasks": len(complete),
    }
    dump_json(output.with_name("summary.json"), summary)
    for method, row in summary["methods"].items():
        print(
            f"{method:24} success={row['task_success_rate']:.3f} score={row['avg_check_score']:.3f} "
            f"tokens={row['avg_total_tokens']:.0f} p95={row['p95_latency_ms'] / 1000:.1f}s"
        )
    print(f"saved: {output} and {output.with_name('summary.json')}")


if __name__ == "__main__":
    asyncio.run(main())
