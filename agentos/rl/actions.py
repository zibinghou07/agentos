"""Parse one assistant turn into an action.

Text actions use Qwen's Hermes tool-call format, which is also what verl's
``hermes`` parser expects:
    <tool_call>{"name": "bm25_search", "arguments": {"query": "..."}}</tool_call>
The final answer is written as <answer>...</answer>. Structured actions from an
OpenAI-compatible API (``tool_calls``) are accepted as well.

Rules mirror verl's ToolAgentLoop so training and evaluation agree: a tool call
that is not decodable JSON with "name" and "arguments" is dropped; if no call
survives, the turn has no tool call and the episode ends there.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass
class Action:
    kind: str  # "tool_call" | "answer" | "malformed_tool_call" | "no_action"
    text: str = ""
    name: str | None = None
    arguments: Any = None
    answer: str | None = None
    error: str | None = None
    extra_tool_calls: int = 0
    malformed_calls: int = 0


def _decode_call(payload: str) -> tuple[str | None, Any, str | None]:
    try:
        call = json.loads(payload)
    except json.JSONDecodeError as exc:
        return None, None, f"tool call is not valid JSON: {exc.msg}"
    if not isinstance(call, dict) or not isinstance(call.get("name"), str) or "arguments" not in call:
        return None, None, 'tool call must be {"name": ..., "arguments": {...}}'
    # A string "arguments" is passed through as-is (as verl does); the tool rejects it.
    return call["name"], call["arguments"], None


def parse_action(turn: str | dict[str, Any]) -> Action:
    """A tool call takes precedence over an answer in the same turn (as in verl).

    Only the first decodable tool call is executed; extra calls are counted.
    """
    if isinstance(turn, dict):
        text = turn.get("content") or ""
        payloads = []
        for call in turn.get("tool_calls") or []:
            function = call.get("function", call)
            arguments = function.get("arguments", "{}")
            # OpenAI-compatible APIs send arguments as a JSON string; keep raw text if it is broken.
            payloads.append(
                f'{{"name": {json.dumps(function.get("name"))}, "arguments": {arguments}}}'
                if isinstance(arguments, str)
                else json.dumps({"name": function.get("name"), "arguments": arguments})
            )
    else:
        text = turn
        payloads = TOOL_CALL.findall(text)
    decoded, errors = [], []
    for payload in payloads:
        name, arguments, error = _decode_call(payload)
        (errors if error else decoded).append(error or (name, arguments))
    if decoded:
        name, arguments = decoded[0]
        return Action("tool_call", text, name, arguments, extra_tool_calls=len(decoded) - 1, malformed_calls=len(errors))
    if errors:
        return Action("malformed_tool_call", text, error=errors[0], malformed_calls=len(errors))
    answers = ANSWER.findall(text)
    if answers:
        return Action("answer", text, answer=answers[-1].strip())
    return Action("no_action", text)
