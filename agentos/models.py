from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStatus(StrEnum):
    PLANNING = "PLANNING"
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    VERIFYING = "VERIFYING"
    REPLANNING = "REPLANNING"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"


class TaskNode(BaseModel):
    task_id: str
    description: str
    agent_type: str
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: dict[str, Any] | None = None
    retry_count: int = 0
    operation_id: str = Field(default_factory=lambda: str(uuid4()))
    preferred_tools: list[str] = Field(default_factory=list)
    required_output_keys: list[str] = Field(default_factory=list)
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    idempotent: bool = True


class TaskPlan(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str
    subtasks: list[TaskNode]
    version: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dependencies(self) -> "TaskPlan":
        ids = [node.task_id for node in self.subtasks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate task_id")
        known = set(ids)
        for node in self.subtasks:
            missing = set(node.dependencies) - known
            if missing:
                raise ValueError(f"unknown dependencies for {node.task_id}: {sorted(missing)}")
            if node.task_id in node.dependencies:
                raise ValueError(f"self dependency: {node.task_id}")
        from agentos.runtime.dag import topological_sort

        topological_sort(self.subtasks)
        return self


class AgentProfile(BaseModel):
    name: str
    capabilities: list[str]
    enabled: bool = True
    average_cost: float = 0.0
    average_latency_ms: float = 0.0
    successes: int = 0
    failures: int = 0
    # MCP server names this agent may use; empty means every discovered tool.
    tool_servers: list[str] = Field(default_factory=list)
    instructions: str = ""

    def can_use(self, tool: "ToolMetadata") -> bool:
        return not self.tool_servers or tool.server_name in self.tool_servers

    def with_stats(self, stats: dict[str, float] | None) -> "AgentProfile":
        """Return a copy whose success/cost/latency reflect persisted runtime outcomes."""
        if not stats:
            return self
        runs = int(stats["successes"] + stats["failures"])
        return self.model_copy(
            update={
                "successes": int(stats["successes"]),
                "failures": int(stats["failures"]),
                "average_cost": stats["total_cost"] / runs if runs else 0.0,
                "average_latency_ms": stats["total_latency_ms"] / runs if runs else 0.0,
            }
        )


class RouteDecision(BaseModel):
    selected_agent: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)


class ToolMetadata(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str
    timeout: float = 10.0
    enabled: bool = True
    synthetic: bool = False
    output_schema: dict[str, Any] = Field(default_factory=dict)
    idempotent: bool = True


class ExecutionEvent(BaseModel):
    event_id: int | None = None
    event: str
    task_id: str
    node_id: str | None = None
    timestamp: str = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)


class TaskRecord(BaseModel):
    task_id: str
    goal: str
    status: TaskStatus
    plan_version: int
    cancel_requested: bool = False
    final_result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: str
    updated_at: str
    plan: TaskPlan

