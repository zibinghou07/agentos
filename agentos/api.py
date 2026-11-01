from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agentos.agents import AgentExecutor, Planner, Verifier
from agentos.config import Settings, get_settings
from agentos.llm import create_llm
from agentos.mcp import MCPToolProvider
from agentos.models import ExecutionEvent, TaskRecord
from agentos.runtime.checkpoint import CheckpointStore
from agentos.runtime.recovery import RecoveryManager
from agentos.runtime.service import AgentRuntimeService


class CreateTask(BaseModel):
    goal: str = Field(min_length=1, max_length=10_000)


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    temperature: float = 0.0
    max_tokens: int = 512


def build_runtime_service(settings: Settings | None = None) -> AgentRuntimeService:
    settings = settings or get_settings()
    # Structured planning/tool decisions are non-streaming and can take longer than
    # an interactive chat request. Keep /api/chat user-configurable while giving the
    # orchestration path a safe minimum.
    runtime_settings = settings.model_copy(
        update={"timeout_seconds": max(settings.timeout_seconds, 120.0)}
    )
    llm = create_llm(runtime_settings)
    # Imports are delayed so importing data models never initializes MCP servers.
    from agentos.agents.synthesizer import Synthesizer
    from agentos.runtime.routing import AdaptiveRouter
    from agentos.world import World, build_mcp_servers

    world = World.create(settings.world_database, reset=False)
    provider = MCPToolProvider(build_mcp_servers(world))
    checkpoint = CheckpointStore(settings.database)
    recovery = RecoveryManager(retries=2, repairs=1, replans=1)
    executor = AgentExecutor(
        provider,
        llm,
        recovery,
        router=AdaptiveRouter(epsilon=settings.routing_epsilon),
        stats=checkpoint,
    )
    return AgentRuntimeService(
        planner=Planner(llm),
        executor=executor,
        verifier=Verifier(),
        checkpoint=checkpoint,
        max_concurrency=settings.max_concurrency,
        max_replans=recovery.replans,
        synthesizer=Synthesizer(llm),
    )


def create_app(
    runtime: AgentRuntimeService | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    runtime = runtime or build_runtime_service(resolved_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await runtime.recover_incomplete()
        yield

    application = FastAPI(title="AgentOS", version="0.2.0", lifespan=lifespan)
    application.state.runtime = runtime
    application.state.settings = resolved_settings

    @application.get("/api/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "provider": resolved_settings.provider,
            "runtime": "agent-runtime-service",
        }

    @application.post("/api/chat")
    async def chat(request: ChatRequest) -> dict[str, Any]:
        result = await create_llm(resolved_settings).chat(
            request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return {"content": result.content, "usage": result.usage.model_dump()}

    @application.post("/api/tasks", status_code=202)
    async def create_task(request: CreateTask) -> dict[str, str]:
        record = await runtime.submit(request.goal)
        return {"task_id": record.task_id, "status": record.status}

    @application.get("/api/tasks/{task_id}", response_model=TaskRecord)
    async def get_task(task_id: str) -> TaskRecord:
        record = await runtime.get_task(task_id)
        if record is None:
            raise HTTPException(404, "task not found")
        return record

    @application.get("/api/tasks/{task_id}/trace")
    async def get_trace(task_id: str) -> dict[str, Any]:
        if await runtime.get_task(task_id) is None:
            raise HTTPException(404, "task not found")
        events: list[ExecutionEvent] = await runtime.get_trace(task_id)
        return {
            "task_id": task_id,
            "events": [event.model_dump(mode="json") for event in events],
        }

    @application.get("/api/tasks/{task_id}/operations")
    async def get_operations(task_id: str) -> dict[str, Any]:
        if await runtime.get_task(task_id) is None:
            raise HTTPException(404, "task not found")
        return {
            "task_id": task_id,
            "operations": await runtime.get_operations(task_id),
            "delivery_semantics": (
                "confirmed successes are deduplicated; external side effects are not "
                "transactionally exactly-once"
            ),
        }

    @application.post("/api/tasks/{task_id}/resume", status_code=202)
    async def resume_task(task_id: str) -> dict[str, str]:
        try:
            started = await runtime.resume(task_id)
        except KeyError as exc:
            raise HTTPException(404, "task not found") from exc
        if not started:
            raise HTTPException(409, "task is terminal or already has an active runner")
        return {"task_id": task_id, "status": "RESUMING"}

    @application.post("/api/tasks/{task_id}/cancel", status_code=202)
    async def cancel_task(task_id: str) -> dict[str, str]:
        if not await runtime.cancel(task_id):
            raise HTTPException(404, "task not found")
        return {"task_id": task_id, "status": "CANCEL_REQUESTED"}

    @application.get("/api/metrics")
    async def metrics() -> dict[str, Any]:
        return {
            "tasks_active_in_process": sum(
                not job.done() for job in runtime.jobs.values()
            ),
            "jobs_known_in_process": len(runtime.jobs),
            "persisted_tasks": await runtime.checkpoint.atask_counts(),
        }

    return application


settings = get_settings()
app = create_app(settings=settings)
