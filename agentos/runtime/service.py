from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any
from uuid import uuid4

from agentos.agents.executor import AgentExecutor
from agentos.agents.planner import Planner
from agentos.agents.synthesizer import Synthesizer
from agentos.agents.verifier import Verifier
from agentos.models import ExecutionEvent, TaskNode, TaskPlan, TaskRecord, TaskStatus
from agentos.runtime.checkpoint import CheckpointStore
from agentos.runtime.dag import plan_shape
from agentos.runtime.recovery import ErrorType, RecoveryExhausted, VerificationError
from agentos.runtime.scheduler import DAGScheduler


class AgentRuntimeService:
    """Application service for planning, execution, verification, recovery, and resume."""

    def __init__(
        self,
        planner: Planner,
        executor: AgentExecutor,
        verifier: Verifier,
        checkpoint: CheckpointStore,
        *,
        max_concurrency: int = 4,
        max_replans: int = 1,
        lease_seconds: float = 30,
        node_repairs: int = 1,
        synthesizer: Synthesizer | None = None,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.verifier = verifier
        self.checkpoint = checkpoint
        self.max_concurrency = max_concurrency
        self.max_replans = max_replans
        self.lease_seconds = lease_seconds
        self.node_repairs = node_repairs
        self.synthesizer = synthesizer
        self.jobs: dict[str, asyncio.Task[None]] = {}
        self._job_lock = asyncio.Lock()

    async def _event(
        self,
        event: str,
        task_id: str,
        node_id: str | None = None,
        **payload: Any,
    ) -> None:
        await self.checkpoint.aappend_event(
            ExecutionEvent(
                event=event,
                task_id=task_id,
                node_id=node_id,
                payload=payload,
            )
        )

    async def submit(self, goal: str) -> TaskRecord:
        task_id = str(uuid4())
        placeholder = TaskPlan(task_id=task_id, goal=goal, subtasks=[])
        await self.checkpoint.aupsert_task(placeholder, TaskStatus.PLANNING)
        await self.checkpoint.asave(
            placeholder, task_id, None, TaskStatus.PLANNING
        )
        await self._event("TASK_CREATED", task_id, goal=goal)
        await self.start(task_id)
        record = await self.checkpoint.atask_record(task_id)
        assert record is not None
        return record

    async def start(self, task_id: str, *, resume: bool = False) -> bool:
        async with self._job_lock:
            existing = self.jobs.get(task_id)
            if existing and not existing.done():
                return False
            self.jobs[task_id] = asyncio.create_task(
                self._run(task_id, resume=resume), name=f"agentos:{task_id}"
            )
            return True

    async def _execute_node(
        self, task_id: str, node: TaskNode, context: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        feedback: list[str] | None = None
        for attempt in range(self.node_repairs + 1):
            result = await self.executor.execute(node, context, feedback=feedback)
            verification = self.verifier.verify_node(node, result)
            await self._event(
                "NODE_VERIFIED",
                task_id,
                node.task_id,
                passed=verification.passed,
                failures=verification.failures,
                action=verification.action,
                attempt=attempt,
                agent=result.get("agent"),
            )
            await self.checkpoint.arecord_agent_outcome(
                result["agent"],
                verification.passed,
                float(result.get("llm_tokens", 0)),
                float(result.get("latency_ms", 0.0)),
            )
            if verification.passed:
                return {**result, "verification": verification.model_dump(), "verify_attempts": attempt + 1}
            if not result.get("idempotent", True):
                # The side effect already happened; re-running or replanning could repeat it.
                raise RecoveryExhausted(
                    ErrorType.LOGIC_FAILURE,
                    "manual",
                    VerificationError(verification.failures, verification.action),
                )
            if verification.action not in {"repair", "retry"} or attempt >= self.node_repairs:
                break
            # Honour the Verifier: re-run this node locally with the failures as feedback
            # (cheap) before escalating to a whole-plan replan (expensive).
            feedback = verification.failures
            await self._event(
                "NODE_REEXECUTING", task_id, node.task_id, action=verification.action, failures=feedback
            )
        raise RecoveryExhausted(
            ErrorType.LOGIC_FAILURE,
            "replan",
            VerificationError(verification.failures, verification.action),
        )

    async def _heartbeat(self, task_id: str, runner_id: str, stop: asyncio.Event) -> None:
        interval = max(1.0, self.lease_seconds / 3)
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except TimeoutError:
                await self.checkpoint.aacquire_lease(
                    task_id, runner_id, self.lease_seconds
                )

    async def _run(self, task_id: str, *, resume: bool) -> None:
        runner_id = str(uuid4())
        acquired = await self.checkpoint.aacquire_lease(
            task_id, runner_id, self.lease_seconds
        )
        if not acquired:
            await self._event("RUNNER_REJECTED", task_id, reason="active_lease")
            return
        stop = asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat(task_id, runner_id, stop))
        try:
            plan = await self.checkpoint.alatest(task_id)
            if plan is None:
                return
            if not plan.subtasks:
                tools = await self.executor.tool_provider.list_tools()
                generated = await self.planner.plan(plan.goal, tools)
                plan = TaskPlan.model_validate(
                    generated.model_copy(
                        update={"task_id": task_id, "goal": plan.goal, "version": 1}
                    ).model_dump()
                )
                await self.checkpoint.aupsert_task(plan, TaskStatus.PENDING)
                await self.checkpoint.asave(plan, task_id, None, TaskStatus.PENDING)
                await self._event(
                    "PLAN_RECOVERED" if resume else "PLAN_CREATED",
                    task_id,
                    version=plan.version,
                    nodes=[node.task_id for node in plan.subtasks],
                    shape=plan_shape(plan.subtasks),
                    agents={node.task_id: node.agent_type for node in plan.subtasks},
                )
            replan_count = 0
            current_resume = resume
            while True:
                await self.checkpoint.aupsert_task(plan, TaskStatus.RUNNING)
                scheduler = DAGScheduler(
                    lambda node, context: self._execute_node(
                        task_id, node, context
                    ),
                    self.checkpoint,
                    self.max_concurrency,
                )
                plan = await scheduler.run(plan, resume=current_resume)
                current_resume = False
                if await self.checkpoint.acancel_requested(task_id):
                    await self.checkpoint.aupsert_task(plan, TaskStatus.CANCELLED)
                    return
                if all(node.status == TaskStatus.SUCCESS for node in plan.subtasks):
                    final = {
                        "goal": plan.goal,
                        "plan_version": plan.version,
                        "nodes": {
                            node.task_id: node.result for node in plan.subtasks
                        },
                    }
                    if self.synthesizer is not None:
                        final.update(await self._synthesize(task_id, plan))
                    await self.checkpoint.aupsert_task(
                        plan, TaskStatus.SUCCESS, final_result=final
                    )
                    await self._event(
                        "TASK_SUCCEEDED", task_id, version=plan.version
                    )
                    return
                failed = next(
                    (node for node in plan.subtasks if node.status == TaskStatus.FAILED),
                    None,
                )
                can_replan = (
                    failed is not None
                    and failed.result is not None
                    and failed.result.get("recovery_action") == "replan"
                    and replan_count < self.max_replans
                )
                if not can_replan:
                    error = failed.result if failed and failed.result else {
                        "error": "PlanFailed",
                        "message": "DAG terminated without all nodes succeeding",
                    }
                    await self.checkpoint.aupsert_task(
                        plan, TaskStatus.FAILED, error=error
                    )
                    await self._event("TASK_FAILED", task_id, **error)
                    return
                replan_count += 1
                await self.checkpoint.aupsert_task(plan, TaskStatus.REPLANNING)
                await self._event(
                    "REPLAN_STARTED",
                    task_id,
                    failed.task_id,
                    from_version=plan.version,
                    attempt=replan_count,
                )
                tools = await self.executor.tool_provider.list_tools()
                plan = await self.planner.replan(plan, failed, failed.result, tools)
                await self.checkpoint.asave(
                    plan, task_id, None, TaskStatus.REPLANNING
                )
                await self._event(
                    "PLAN_REPLACED",
                    task_id,
                    from_version=plan.version - 1,
                    to_version=plan.version,
                    nodes=[node.task_id for node in plan.subtasks],
                )
        except Exception as exc:
            plan = await self.checkpoint.alatest(task_id)
            if plan is not None:
                error = {"error": type(exc).__name__, "message": str(exc)}
                await self.checkpoint.aupsert_task(plan, TaskStatus.FAILED, error=error)
                await self._event("RUNTIME_FAILED", task_id, **error)
        finally:
            stop.set()
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            await self.checkpoint.arelease_lease(task_id, runner_id)

    async def _synthesize(self, task_id: str, plan: TaskPlan) -> dict[str, Any]:
        assert self.synthesizer is not None
        try:
            answer, usage = await self.synthesizer.synthesize(plan)
        except Exception as exc:
            # Node results are verified and kept; only the prose summary is missing.
            await self._event("ANSWER_FAILED", task_id, error=type(exc).__name__, message=str(exc))
            return {"answer": None, "answer_error": f"{type(exc).__name__}: {exc}"}
        await self._event("ANSWER_COMPOSED", task_id, evidence_nodes=answer.evidence_nodes)
        return {"answer": answer.answer, "evidence_nodes": answer.evidence_nodes, "answer_usage": usage.model_dump()}

    async def resume(self, task_id: str) -> bool:
        record = await self.checkpoint.atask_record(task_id)
        if record is None:
            raise KeyError(task_id)
        if record.status in {
            TaskStatus.SUCCESS,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            return False
        return await self.start(task_id, resume=True)

    async def recover_incomplete(self) -> list[str]:
        task_ids = await self.checkpoint.arecoverable_task_ids()
        started = []
        for task_id in task_ids:
            if await self.start(task_id, resume=True):
                started.append(task_id)
        return started

    async def cancel(self, task_id: str) -> bool:
        record = await self.checkpoint.atask_record(task_id)
        if record is None or record.status in {
            TaskStatus.SUCCESS,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            return False
        requested = await self.checkpoint.arequest_cancel(task_id)
        if requested:
            await self._event("CANCEL_REQUESTED", task_id)
        return requested

    async def get_task(self, task_id: str) -> TaskRecord | None:
        return await self.checkpoint.atask_record(task_id)

    async def get_trace(self, task_id: str) -> list[ExecutionEvent]:
        return await self.checkpoint.aevents(task_id)

    async def get_operations(self, task_id: str) -> list[dict[str, Any]]:
        return await self.checkpoint.aoperations(task_id)

    async def wait(self, task_id: str) -> None:
        job = self.jobs.get(task_id)
        if job:
            await job


async def demo_executor(
    node: TaskNode, context: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Explicit test/demo helper; never used by the production API task route."""
    await asyncio.sleep(0)
    return {
        "description": node.description,
        "agent": node.agent_type,
        "dependencies": sorted(context),
    }
