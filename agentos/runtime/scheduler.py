from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any

from agentos.models import ExecutionEvent, TaskNode, TaskPlan, TaskStatus
from agentos.runtime.checkpoint import CheckpointStore
from agentos.runtime.dag import ready_nodes
from agentos.runtime.recovery import (
    AmbiguousOperationError,
    RecoveryExhausted,
    classify_error,
)

NodeExecutor = Callable[
    [TaskNode, dict[str, dict[str, Any]]], Awaitable[dict[str, Any]]
]


class DAGScheduler:
    def __init__(
        self,
        executor: NodeExecutor,
        checkpoint: CheckpointStore,
        max_concurrency: int = 4,
    ):
        self.executor = executor
        self.checkpoint = checkpoint
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def cancel(self, task_id: str) -> None:
        await self.checkpoint.arequest_cancel(task_id)

    async def _event(
        self,
        event: str,
        plan: TaskPlan,
        node: TaskNode | None = None,
        **payload: Any,
    ) -> None:
        await self.checkpoint.aappend_event(
            ExecutionEvent(
                event=event,
                task_id=plan.task_id,
                node_id=node.task_id if node else None,
                payload=payload,
            )
        )

    async def _run_node(
        self,
        plan: TaskPlan,
        node: TaskNode,
        results: dict[str, dict[str, Any]],
        *,
        resume: bool,
    ) -> None:
        operation = await self.checkpoint.aoperation(node.operation_id)
        if operation and operation["status"] == "SUCCESS":
            node.result, node.status = operation["result"], TaskStatus.SUCCESS
            results[node.task_id] = operation["result"]
            await self._event("NODE_REUSED", plan, node, operation_id=node.operation_id)
            return
        if operation and operation["status"] == "RUNNING" and resume and not node.idempotent:
            exc = AmbiguousOperationError(
                "non-idempotent operation was RUNNING at interruption; manual reconciliation required"
            )
            node.result = {
                "error": type(exc).__name__,
                "message": str(exc),
                "error_type": classify_error(exc),
                "recovery_action": "manual",
            }
            node.status = TaskStatus.FAILED
            await self.checkpoint.arecord_operation(
                node.operation_id,
                plan.task_id,
                node.task_id,
                "AMBIGUOUS",
                error=node.result,
            )
            await self._event("NODE_AMBIGUOUS", plan, node, **node.result)
            return

        async with self.semaphore:
            node.status = TaskStatus.RUNNING
            await self.checkpoint.asave(
                plan, plan.task_id, node.task_id, TaskStatus.RUNNING
            )
            await self.checkpoint.arecord_operation(
                node.operation_id, plan.task_id, node.task_id, "RUNNING"
            )
            await self._event("NODE_STARTED", plan, node, operation_id=node.operation_id)
            try:
                result = await self.executor(node, deepcopy(results))
                node.result, node.status = result, TaskStatus.SUCCESS
                results[node.task_id] = result
                await self.checkpoint.arecord_operation(
                    node.operation_id,
                    plan.task_id,
                    node.task_id,
                    "SUCCESS",
                    result=result,
                )
                await self._event(
                    "NODE_SUCCEEDED",
                    plan,
                    node,
                    tool=result.get("tool"),
                    recovery_trace=result.get("recovery_trace", []),
                    llm_usage=result.get("llm_usage", {}),
                )
            except Exception as exc:
                if isinstance(exc, RecoveryExhausted):
                    error_type, action = exc.error_type, exc.action
                else:
                    error_type, action = classify_error(exc), "fail"
                node.result = {
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "error_type": error_type,
                    "recovery_action": action,
                }
                node.status = TaskStatus.FAILED
                await self.checkpoint.arecord_operation(
                    node.operation_id,
                    plan.task_id,
                    node.task_id,
                    "FAILED",
                    error=node.result,
                )
                await self._event("NODE_FAILED", plan, node, **node.result)
            finally:
                await self.checkpoint.asave(
                    plan, plan.task_id, node.task_id, TaskStatus.RUNNING
                )

    async def run(self, plan: TaskPlan, resume: bool = False) -> TaskPlan:
        if resume:
            plan = await self.checkpoint.alatest(plan.task_id) or plan
            for node in plan.subtasks:
                if node.status in {
                    TaskStatus.RUNNING,
                    TaskStatus.RETRYING,
                    TaskStatus.VERIFYING,
                    TaskStatus.READY,
                }:
                    node.status = TaskStatus.PENDING
        results = {
            node.task_id: node.result
            for node in plan.subtasks
            if node.status == TaskStatus.SUCCESS and node.result
        }
        await self.checkpoint.asave(plan, plan.task_id, None, TaskStatus.RUNNING)
        await self._event("SCHEDULER_STARTED", plan, resume=resume, version=plan.version)
        # Event-driven: a node starts as soon as its own dependencies succeed, instead of
        # waiting for the slowest member of a whole "wave" to finish.
        running: dict[asyncio.Task[None], str] = {}
        launched: set[str] = set()
        while True:
            if await self.checkpoint.acancel_requested(plan.task_id):
                if running:
                    await asyncio.gather(*running, return_exceptions=True)
                for node in plan.subtasks:
                    if node.status in {TaskStatus.PENDING, TaskStatus.READY}:
                        node.status = TaskStatus.CANCELLED
                await self.checkpoint.asave(
                    plan, plan.task_id, None, TaskStatus.CANCELLED
                )
                await self._event("TASK_CANCELLED", plan)
                return plan
            for node in ready_nodes(plan.subtasks):
                if node.task_id in launched:
                    continue
                launched.add(node.task_id)
                job = asyncio.create_task(
                    self._run_node(plan, node, results, resume=resume),
                    name=f"agentos-node:{node.task_id}",
                )
                running[job] = node.task_id
            if not running:
                break
            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for job in done:
                running.pop(job)
                job.result()
        terminal = (
            TaskStatus.SUCCESS
            if all(node.status == TaskStatus.SUCCESS for node in plan.subtasks)
            else TaskStatus.FAILED
        )
        await self.checkpoint.asave(plan, plan.task_id, None, terminal)
        await self._event("SCHEDULER_FINISHED", plan, status=terminal)
        return plan
