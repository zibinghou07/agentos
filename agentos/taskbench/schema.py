from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

Family = Literal[
    "lookup",
    "parallel_cross_source",
    "multi_hop",
    "analytics",
    "policy_action",
    "logistics",
    "fault_transient",
    "fault_outage",
    "hotpotqa_agentic",
]


class NumberCheck(BaseModel):
    kind: Literal["number"] = "number"
    label: str
    value: float
    rel_tol: float = 0.005
    abs_tol: float = 0.01


class TextCheck(BaseModel):
    kind: Literal["text"] = "text"
    label: str
    aliases: list[str]


class DbCheck(BaseModel):
    """A side-effect assertion evaluated on the task's world database after the run."""

    kind: Literal["db"] = "db"
    label: str
    sql: str
    params: list[Any] = Field(default_factory=list)
    expected: float
    tolerance: float = 0.01


Check = Annotated[NumberCheck | TextCheck | DbCheck, Field(discriminator="kind")]


class FaultSpec(BaseModel):
    tool: str
    # transient: the first `failures` calls fail with a retryable 503;
    # outage: every call fails, so the agent must switch tools or plans.
    mode: Literal["transient", "outage"]
    failures: int = 2


class TaskSpec(BaseModel):
    task_id: str
    family: Family
    split: Literal["dev", "test"]
    goal: str
    checks: list[Check]
    fault: FaultSpec | None = None
    # When set, the task runs on this corpus instead of the default world documents.
    documents: dict[str, str] | None = None
    synthetic: bool
    source: dict[str, Any] = Field(default_factory=dict)
    # Minimum number of tool calls a correct solution needs (documented, not graded).
    reference_steps: int = Field(ge=1)


class CheckResult(BaseModel):
    label: str
    kind: str
    passed: bool
    detail: str = ""


class RunRecord(BaseModel):
    run_id: int
    method: str
    task_id: str
    family: str
    split: str
    synthetic: bool
    success: bool
    score: float
    checks: list[CheckResult]
    answer: str | None
    llm_calls: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    usage_estimated: bool
    tool_calls: int
    tool_errors: int
    side_effect_calls: int
    latency_ms: float
    error: str | None = None
    fault: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    provider: str
    model: str
