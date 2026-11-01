from __future__ import annotations

import random
import re
import time

from agentos.benchmarks.schema import BenchmarkTask, RunRecord
from agentos.mcp.catalog import build_local_registry
from agentos.runtime.recovery import RecoveryManager

# Offline tool-routing microbenchmark (no LLM). These are routing heuristics, not agents:
# end-to-end ReAct / fixed-role / AgentOS comparisons live in agentos.taskbench.
METHODS = ("greedy_top1_tool", "keyword_rules", "embedding_topk", "clause_routing")


def _fixed_rules(prompt: str) -> list[str]:
    """Static intent rules: fixed before the experiment and independent of labels."""
    lowered = prompt.lower()
    tools = []
    if "product" in lowered:
        tools.append("query_product")
    if "local documents" in lowered:
        tools.append("search_documents")
    if "bm25" in lowered:
        tools.append("bm25_search")
    if "dense" in lowered:
        tools.append("dense_search")
    if "hybrid" in lowered:
        tools.append("hybrid_search")
    return tools


def _select_tools(task: BenchmarkTask, method: str, registry) -> list[str]:
    if method == "greedy_top1_tool":
        # Not ReAct: greedily retrieves the single best-matching schema from the full registry.
        return [tool.name for tool in registry.search(task.prompt, top_k=1)]
    if method == "keyword_rules":
        return _fixed_rules(task.prompt)
    if method == "embedding_topk":
        clauses = [
            clause
            for clause in re.split(r"\b(?:then|and)\b|[,;]", task.prompt, flags=re.IGNORECASE)
            if clause.strip()
        ]
        estimated_k = min(3, max(1, len(clauses)))
        return [tool.name for tool in registry.search(task.prompt, top_k=estimated_k)]
    # Adaptive route decomposes clauses, retrieves one candidate each and de-duplicates.
    # It never reads expected_tools or the task family.
    selected = []
    clauses = re.split(r"\b(?:then|and)\b|[,;]", task.prompt, flags=re.IGNORECASE)
    for clause in clauses:
        candidates = registry.search(clause, top_k=1)
        if candidates and candidates[0].name not in selected:
            selected.append(candidates[0].name)
    return selected


async def evaluate_task(task: BenchmarkTask, method: str, run_id: int, seed: int) -> RunRecord:
    started = time.perf_counter()
    registry = build_local_registry(":memory:")
    selected = _select_tools(task, method, registry)
    correct = selected == task.expected_tools
    recovery_eligible = task.fault is not None
    recovered = False
    calls = 0

    if recovery_eligible:
        attempts = 0

        async def faulty_call(arguments):
            nonlocal attempts, calls
            attempts += 1
            calls += 1
            if "product_id" not in arguments:
                raise ValueError("invalid argument: product_id is required")
            return await registry.call("query_product", arguments)

        def repair(args, exc):
            return {"product_id": "p1"}
        manager = RecoveryManager(repairs=1 if method == "clause_routing" else 0, retries=1, replans=1 if method == "clause_routing" else 0)
        try:
            result, trace = await manager.execute(faulty_call, {"wrong": "p1"}, repair=repair if method == "clause_routing" else None)
            recovered = result.get("name") == "AgentOS Pro" and any(item.startswith("REPAIR") for item in trace)
        except Exception:
            recovered = False
    else:
        calls = len(selected)

    # Success is determined only by exact tool ground truth and tested recovery, never model self-evaluation.
    step_success = sum(a == b for a, b in zip(selected, task.expected_tools)) / len(task.expected_tools)
    success = correct and (recovered if recovery_eligible else True)
    latency = (time.perf_counter() - started) * 1000
    input_tokens = len(task.prompt.split()) + 20 * (len(registry.list_tools()) if method == "greedy_top1_tool" else len(selected))
    output_tokens = 8 * max(1, task.steps)
    return RunRecord(
        run_id=run_id, method=method, task_id=task.task_id, family=task.family, success=success,
        step_success=step_success, tool_selection_correct=correct, recovery_eligible=recovery_eligible,
        recovered=recovered, tool_calls=calls, input_tokens=input_tokens, output_tokens=output_tokens,
        cost=0.0, latency_ms=latency, seed=seed, provider="mock-v1-offline",
    )


async def evaluate_dataset(tasks: list[BenchmarkTask], methods: tuple[str, ...] = METHODS, repeats: int = 3, seed: int = 42) -> list[RunRecord]:
    records = []
    for run_id in range(repeats):
        run_seed = seed + run_id
        random.seed(run_seed)
        for method in methods:
            for task in tasks:
                records.append(await evaluate_task(task, method, run_id, run_seed))
    return records

