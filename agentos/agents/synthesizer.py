from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from agentos.agents.executor import compact_result
from agentos.llm.base import BaseLLM, Usage
from agentos.models import TaskPlan


class FinalAnswer(BaseModel):
    answer: str
    evidence_nodes: list[str] = Field(default_factory=list)

    @field_validator("answer", mode="before")
    @classmethod
    def _stringify(cls, value: Any) -> Any:
        # Models sometimes return the answer as structured fields; keep the content.
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, (int, float)):
            return str(value)
        return value

    @field_validator("evidence_nodes", mode="before")
    @classmethod
    def _node_ids(cls, value: Any) -> Any:
        # Some models cite whole node objects instead of their ids.
        if not isinstance(value, list):
            return [] if value is None else [value]
        return [
            str(item.get("task_id") or item.get("id") or "") if isinstance(item, dict) else str(item)
            for item in value
        ]


class Synthesizer:
    """Compose the user-facing answer strictly from verified node outputs."""

    def __init__(self, llm: BaseLLM, max_tokens: int = 1000, max_repairs: int = 1):
        self.llm, self.max_tokens, self.max_repairs = llm, max_tokens, max_repairs

    async def synthesize(self, plan: TaskPlan) -> tuple[FinalAnswer, Usage]:
        evidence: dict[str, Any] = {
            node.task_id: {"description": node.description, **compact_result(node.result or {})}
            for node in plan.subtasks
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Write the final answer to the goal using only the verified subtask "
                    "outputs. State exact numbers and identifiers, report any actions that "
                    "were performed (ticket or refund ids), and say so if evidence is "
                    'missing. Return only JSON where "answer" is one plain-text string: {"answer": "...", "evidence_nodes": [...]}.'
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"goal": plan.goal, "subtasks": evidence}, ensure_ascii=False, default=str),
            },
        ]
        error: Exception | None = None
        for _ in range(self.max_repairs + 1):
            try:
                return await self.llm.chat_json(
                    messages, FinalAnswer, temperature=0, max_tokens=self.max_tokens
                )
            except (ValueError, ValidationError) as exc:  # malformed JSON or schema mismatch
                error = exc
                messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "Your reply was not valid FinalAnswer JSON "
                            f"({type(exc).__name__}). Reply again with one short JSON object only: "
                            '{"answer": "<plain text>", "evidence_nodes": ["<task_id>", ...]}.'
                        ),
                    },
                ]
        assert error is not None
        raise error
