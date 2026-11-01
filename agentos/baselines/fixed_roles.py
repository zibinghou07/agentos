from __future__ import annotations

from typing import Any

from agentos.agents.profiles import DEFAULT_AGENTS
from agentos.baselines.common import tool_calling_loop
from agentos.llm.base import BaseLLM
from agentos.mcp.provider import ToolProvider

ROLE_ORDER = ("researcher", "data_analyst", "support_agent")


async def run_fixed_roles(
    goal: str, llm: BaseLLM, provider: ToolProvider, *, turns_per_role: int = 4
) -> tuple[str, dict[str, Any]]:
    """Fixed-role sequential crew: every role runs in a fixed order, then a writer answers.

    There is no planner, no dynamic routing, no DAG parallelism, no programmatic
    verification and no runtime recovery; it isolates what AgentOS adds on top
    of role specialisation alone.
    """
    tools = [tool for tool in await provider.list_tools() if tool.enabled]
    profiles = {agent.name: agent for agent in DEFAULT_AGENTS}
    notes: list[str] = []
    trace: list[dict[str, Any]] = []
    for role in ROLE_ORDER:
        agent = profiles[role]
        scoped = [tool for tool in tools if agent.can_use(tool)]
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    f"{agent.instructions} You are one member of a fixed team "
                    f"({', '.join(ROLE_ORDER)}, then a writer). Use only your own tools to "
                    "contribute facts the goal needs. If nothing in your area is needed, reply "
                    "'nothing to add'. Finish with concise notes containing exact values."
                ),
            },
            {
                "role": "user",
                "content": f"Goal: {goal}\n\nNotes from earlier teammates:\n" + ("\n".join(notes) or "(none)"),
            },
        ]
        result = await tool_calling_loop(llm, messages, scoped, provider, max_turns=turns_per_role)
        notes.append(f"[{role}] {result.final}")
        trace.append({"role": role, "steps": result.steps, "tool_trace": result.tool_calls})
    writer = await llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are the writer. Using only the team notes, answer the goal with exact "
                    "numbers and identifiers and report any actions that were performed."
                ),
            },
            {"role": "user", "content": f"Goal: {goal}\n\nTeam notes:\n" + "\n".join(notes)},
        ],
        temperature=0,
        max_tokens=700,
    )
    return writer.content, {"roles": trace}
