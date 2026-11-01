from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentos.llm.base import BaseLLM
from agentos.mcp.provider import ToolProvider
from agentos.models import ToolMetadata

OBSERVATION_CHARS = 3000


def openai_function(tool: ToolMetadata) -> dict[str, Any]:
    parameters = tool.input_schema or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": parameters},
    }


@dataclass
class LoopResult:
    final: str
    steps: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stopped: str = "final_answer"


async def tool_calling_loop(
    llm: BaseLLM,
    messages: list[dict[str, Any]],
    tools: list[ToolMetadata],
    provider: ToolProvider,
    *,
    max_turns: int,
    max_tokens: int = 800,
) -> LoopResult:
    """Native function-calling loop: Thought (content) -> Action (tool_calls) -> Observation."""
    functions = [openai_function(tool) for tool in tools]
    allowed = {tool.name for tool in tools}
    calls: list[dict[str, Any]] = []
    for turn in range(1, max_turns + 1):
        kwargs: dict[str, Any] = {"temperature": 0, "max_tokens": max_tokens}
        if functions:
            kwargs.update({"tools": functions, "tool_choice": "auto"})
        result = await llm.chat(messages, **kwargs)
        message = (result.raw.get("choices") or [{}])[0].get("message") or {}
        requested = message.get("tool_calls") or []
        if not requested:
            return LoopResult(final=result.content, steps=turn, tool_calls=calls)
        messages.append(
            {"role": "assistant", "content": message.get("content") or "", "tool_calls": requested}
        )
        for request in requested:
            function = request.get("function") or {}
            name = function.get("name", "")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
                if name not in allowed:
                    raise KeyError(f"unknown tool {name}")
                output = await provider.call_tool(name, arguments)
                observation = json.dumps(output, ensure_ascii=False, default=str)
                ok = True
            except Exception as exc:  # observations, including errors, go back to the model
                arguments = function.get("arguments")
                observation = f"ERROR {type(exc).__name__}: {exc}"
                ok = False
            calls.append({"tool": name, "arguments": arguments, "ok": ok})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": request.get("id", name),
                    "content": observation[:OBSERVATION_CHARS],
                }
            )
    # Turn budget exhausted: ask once for a best-effort answer without tools.
    messages.append({"role": "user", "content": "Stop calling tools and give your final answer now."})
    result = await llm.chat(messages, temperature=0, max_tokens=max_tokens)
    return LoopResult(final=result.content, steps=max_turns + 1, tool_calls=calls, stopped="turn_budget")
