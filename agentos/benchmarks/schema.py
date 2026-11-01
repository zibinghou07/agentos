from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BenchmarkTask(BaseModel):
    task_id: str
    family: Literal["single_tool", "multi_tool", "multi_hop", "dependency_dag", "fault_recovery", "long_horizon"]
    split: Literal["dev", "test"]
    prompt: str
    expected_tools: list[str]
    expected_answer: Any
    steps: int = Field(ge=1)
    synthetic: bool = True
    fault: str | None = None
    template_id: str


class RunRecord(BaseModel):
    run_id: int
    method: str
    task_id: str
    family: str
    success: bool
    step_success: float
    tool_selection_correct: bool
    recovery_eligible: bool = False
    recovered: bool = False
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float
    seed: int
    provider: str = "mock-v1"

