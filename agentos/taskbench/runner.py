"""Run methods end-to-end on TaskBench tasks with identical worlds, tools, faults and LLM."""

from __future__ import annotations

import asyncio
import functools
import json
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agentos.agents import AgentExecutor, Planner, Verifier
from agentos.agents.synthesizer import Synthesizer
from agentos.baselines import run_fixed_roles, run_react
from agentos.llm.base import BaseLLM
from agentos.mcp import MCPToolProvider
from agentos.models import TaskStatus
from agentos.runtime.checkpoint import CheckpointStore
from agentos.runtime.recovery import RecoveryManager
from agentos.runtime.routing import AdaptiveRouter
from agentos.runtime.service import AgentRuntimeService
from agentos.taskbench.grading import grade, token_f1
from agentos.taskbench.instrument import InstrumentedProvider, MeteredLLM
from agentos.taskbench.schema import RunRecord, TaskSpec
from agentos.world import World, build_mcp_servers

MethodFn = Callable[[TaskSpec, BaseLLM, InstrumentedProvider, CheckpointStore], Awaitable[tuple[str | None, dict[str, Any]]]]


async def _agentos(
    task: TaskSpec,
    llm: BaseLLM,
    provider: InstrumentedProvider,
    store: CheckpointStore,
    *,
    adaptive: bool = True,
    recovery: bool = True,
    concurrency: int = 4,
    seed: int = 42,
) -> tuple[str | None, dict[str, Any]]:
    manager = (
        RecoveryManager(retries=2, repairs=1, replans=1, base_delay=0.2, seed=seed)
        if recovery
        else RecoveryManager(retries=0, repairs=0, replans=0)
    )
    executor = AgentExecutor(
        provider,
        llm,
        manager,
        router=AdaptiveRouter(epsilon=0.05 if adaptive else 0.0, seed=seed),
        stats=store,
        adaptive=adaptive,
        selection_repairs=1 if recovery else 0,
    )
    service = AgentRuntimeService(
        Planner(llm),
        executor,
        Verifier(),
        store,
        max_concurrency=concurrency,
        max_replans=manager.replans,
        node_repairs=1 if recovery else 0,
        synthesizer=Synthesizer(llm),
        lease_seconds=600,
    )
    record = await service.submit(task.goal)
    await service.wait(record.task_id)
    finished = await service.get_task(record.task_id)
    assert finished is not None
    events = await service.get_trace(record.task_id)
    final = finished.final_result or {}
    created = next((e for e in events if e.event == "PLAN_CREATED"), None)
    extra = {
        "runtime_status": finished.status.value,
        "plan_version": finished.plan_version,
        "replans": sum(e.event == "PLAN_REPLACED" for e in events),
        "node_reexecutions": sum(e.event == "NODE_REEXECUTING" for e in events),
        "plan_shape": created.payload.get("shape") if created else None,
        "planned_agents": created.payload.get("agents") if created else None,
        "executed_agents": {
            node.task_id: (node.result or {}).get("agent") for node in finished.plan.subtasks
        },
        "error": finished.error,
        "answer_error": final.get("answer_error"),
    }
    answer = final.get("answer") if finished.status == TaskStatus.SUCCESS else None
    return answer, extra


async def _react(task, llm, provider, store):
    del store
    return await run_react(task.goal, llm, provider)


async def _fixed(task, llm, provider, store):
    del store
    return await run_fixed_roles(task.goal, llm, provider)


METHODS: dict[str, MethodFn] = {
    "react": _react,
    "fixed_roles": _fixed,
    "agentos": _agentos,
    "agentos_static_routing": functools.partial(_agentos, adaptive=False),
    "agentos_no_recovery": functools.partial(_agentos, recovery=False),
    "agentos_sequential": functools.partial(_agentos, concurrency=1),
}


async def run_one(
    task: TaskSpec,
    method: str,
    inner_llm: BaseLLM,
    store: CheckpointStore,
    *,
    run_id: int,
    workdir: Path,
    timeout: float,
    provider_name: str,
    model_name: str,
) -> RunRecord:
    world = World.create(workdir / f"{method}-{run_id}-{task.task_id}.db", task.documents)
    servers = build_mcp_servers(world)
    provider = InstrumentedProvider(MCPToolProvider(servers), task.fault)
    llm = MeteredLLM(inner_llm)
    started = time.perf_counter()
    answer, extra, error = None, {}, None
    try:
        answer, extra = await asyncio.wait_for(METHODS[method](task, llm, provider, store), timeout)
    except Exception as exc:  # a crashed or timed-out run is a failed task, not a benchmark error
        error = f"{type(exc).__name__}: {exc}"
    latency = (time.perf_counter() - started) * 1000
    checks = grade(task, answer, world)
    if task.family == "hotpotqa_agentic":
        extra["answer_f1"] = token_f1(answer or "", str(task.source.get("gold_answer", "")))
    totals = llm.totals()
    return RunRecord(
        run_id=run_id,
        method=method,
        task_id=task.task_id,
        family=task.family,
        split=task.split,
        synthetic=task.synthetic,
        success=all(check.passed for check in checks),
        score=sum(check.passed for check in checks) / len(checks),
        checks=checks,
        answer=answer,
        tool_calls=provider.calls,
        tool_errors=provider.errors,
        side_effect_calls=provider.side_effect_calls,
        latency_ms=latency,
        error=error,
        fault=f"{task.fault.mode}:{task.fault.tool}" if task.fault else None,
        extra=extra,
        provider=provider_name,
        model=model_name,
        **totals,
    )


async def run_suite(
    tasks: list[TaskSpec],
    methods: list[str],
    llm: BaseLLM,
    output: Path,
    *,
    repeats: int = 1,
    concurrency: int = 4,
    timeout: float = 600,
    provider_name: str,
    model_name: str,
    on_record: Callable[[RunRecord], None] | None = None,
    token_budget: int | None = None,
) -> list[RunRecord]:
    """Run every (repeat, task, method); records stream to ``output`` and existing ones are reused.

    Jobs are ordered task-major (all methods of a task before the next task) so a
    token budget that runs out still leaves complete method pairs. ``token_budget``
    counts tokens spent by this invocation; once reached, no new job starts and
    jobs already running finish normally.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise KeyError(f"unknown methods {unknown}; known={sorted(METHODS)}")
    done: dict[tuple[int, str, str], RunRecord] = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = RunRecord.model_validate_json(line)
                done[(record.run_id, record.method, record.task_id)] = record
    records = list(done.values())
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    spent = 0
    skipped = 0
    with tempfile.TemporaryDirectory(prefix="taskbench-") as tmp:
        workdir = Path(tmp)
        for run_id in range(repeats):
            # One store per (run, method): routing statistics accumulate across that
            # method's tasks only, exactly as they would in a long-running deployment.
            stores: dict[str, CheckpointStore] = {}
            for method in methods:
                store_path = output.parent / "stores" / f"{method}-run{run_id}.db"
                if not any(key[:2] == (run_id, method) for key in done):
                    # Fresh (run, method): never inherit statistics from an older run.
                    for stale in store_path.parent.glob(f"{store_path.name}*"):
                        stale.unlink()
                stores[method] = CheckpointStore(store_path)

            async def one(task: TaskSpec, method: str, run_id: int = run_id) -> None:
                nonlocal spent, skipped
                if (run_id, method, task.task_id) in done:
                    return
                async with semaphore:
                    if token_budget is not None and spent >= token_budget:
                        skipped += 1
                        return
                    record = await run_one(
                        task, method, llm, stores[method], run_id=run_id, workdir=workdir,
                        timeout=timeout, provider_name=provider_name, model_name=model_name,
                    )
                async with lock:
                    spent += record.input_tokens + record.output_tokens
                    records.append(record)
                    with output.open("a", encoding="utf-8") as handle:
                        handle.write(record.model_dump_json() + "\n")
                    if on_record:
                        on_record(record)

            await asyncio.gather(*(one(task, method) for task in tasks for method in methods))
    if skipped:
        print(f"token budget reached: spent={spent} budget={token_budget}; skipped {skipped} jobs")
    return records


def load_records(path: Path) -> list[RunRecord]:
    return [
        RunRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
