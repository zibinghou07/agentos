from __future__ import annotations

from typing import Any

from agentos.baselines.common import LoopResult, tool_calling_loop
from agentos.llm.base import BaseLLM
from agentos.mcp.provider import ToolProvider

REACT_SYSTEM = (
    "You are a ReAct agent (Yao et al., 2023) with access to tools. Work step by step: "
    "before each action write a short Thought about what you know and what you need next, "
    "then call exactly the tool that makes progress. Read each Observation carefully; if a "
    "tool fails, try another way. Perform side-effecting actions (tickets, refunds) at most "
    "once and only when the goal's conditions require them. When you have enough "
    "information, reply with the final answer (no tool call), stating exact numbers and "
    "identifiers."
)


async def run_react(
    goal: str, llm: BaseLLM, provider: ToolProvider, *, max_turns: int = 12
) -> tuple[str, dict[str, Any]]:
    """Single agent, full tool registry, LLM-driven Thought/Action/Observation loop."""
    tools = [tool for tool in await provider.list_tools() if tool.enabled]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user", "content": goal},
    ]
    result: LoopResult = await tool_calling_loop(llm, messages, tools, provider, max_turns=max_turns)
    return result.final, {"steps": result.steps, "stopped": result.stopped, "tool_trace": result.tool_calls}
