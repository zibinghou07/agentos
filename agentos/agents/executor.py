from __future__ import annotations

import json
import time
from typing import Any, Protocol

from jsonschema import Draft202012Validator
from pydantic import AliasChoices, BaseModel, Field, ValidationError

from agentos.agents.profiles import DEFAULT_AGENTS
from agentos.llm.base import BaseLLM, Usage
from agentos.mcp.provider import ToolProvider
from agentos.models import AgentProfile, TaskNode, ToolMetadata
from agentos.runtime.recovery import (
    ErrorType,
    RecoveryExhausted,
    RecoveryManager,
    ToolCapabilityError,
)
from agentos.runtime.routing import AdaptiveRouter, beta_success_rate, rank_tools

CONTEXT_CHARS = 3000


class ToolCallDecision(BaseModel):
    tool_name: str = Field(validation_alias=AliasChoices("tool_name", "tool", "name"))
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("arguments", "args", "parameters"),
    )
    reason: str = ""


class RepairedArguments(BaseModel):
    arguments: dict[str, Any] = Field(
        validation_alias=AliasChoices("arguments", "args", "parameters")
    )


class RoutingStats(Protocol):
    async def aagent_stats(self) -> dict[str, dict[str, float]]: ...

    async def atool_stats(self) -> dict[str, dict[str, float]]: ...

    async def arecord_tool_outcome(self, tool_name: str, success: bool, latency_ms: float) -> None: ...

    async def arecord_agent_outcome(
        self, agent_name: str, success: bool, cost: float, latency_ms: float
    ) -> None: ...


def compact_result(result: Any) -> Any:
    """Keep only what a downstream agent needs from an upstream node, bounded in size."""
    if isinstance(result, dict) and "output" in result:
        result = {"tool": result.get("tool"), "arguments": result.get("arguments"), "output": result["output"]}
    encoded = json.dumps(result, ensure_ascii=False, default=str)
    if len(encoded) <= CONTEXT_CHARS:
        return result
    return {"truncated": True, "json_prefix": encoded[:CONTEXT_CHARS]}


def usage_tokens(usages: list[Usage]) -> int:
    return sum(item.input_tokens + item.output_tokens for item in usages)


class AgentExecutor:
    """Route a node to a specialised agent, pick a tool from its scope, call it with recovery.

    With ``adaptive=True`` both decisions use persisted outcomes: agents are scored
    on relevance, success rate, token cost and latency; tools on relevance,
    success rate and latency. ``adaptive=False`` is the static ablation that
    follows the planner's agent hint and ranks tools by relevance only.
    """

    def __init__(
        self,
        tool_provider: ToolProvider,
        llm: BaseLLM,
        recovery: RecoveryManager,
        candidate_count: int = 5,
        router: AdaptiveRouter | None = None,
        agent_profiles: list[AgentProfile] | None = None,
        stats: RoutingStats | None = None,
        *,
        adaptive: bool = True,
        selection_repairs: int = 1,
        min_route_evidence: int = 3,
    ):
        self.tool_provider = tool_provider
        self.llm = llm
        self.recovery = recovery
        self.candidate_count = candidate_count
        self.router = router or AdaptiveRouter(epsilon=0)
        self.agent_profiles = list(agent_profiles or DEFAULT_AGENTS)
        self.stats = stats
        self.adaptive = adaptive
        self.selection_repairs = selection_repairs
        self.min_route_evidence = min_route_evidence

    async def _stats(self) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        if not self.adaptive or self.stats is None:
            return {}, {}
        return await self.stats.aagent_stats(), await self.stats.atool_stats()

    def _route(
        self,
        node: TaskNode,
        tools: list[ToolMetadata],
        agent_stats: dict[str, dict[str, float]],
    ) -> tuple[AgentProfile, dict[str, Any]]:
        by_name = {tool.name: tool for tool in tools}
        preferred = [by_name[name] for name in node.preferred_tools if name in by_name]
        profiles = [profile.with_stats(agent_stats.get(profile.name)) for profile in self.agent_profiles]
        # Capability filter: an agent is eligible only if it may use the planner's tools
        # (or, without a preference, at least one discovered tool).
        eligible = [
            p
            for p in profiles
            if all(p.can_use(tool) for tool in preferred) and any(p.can_use(tool) for tool in tools)
        ] or profiles
        # A generalist (no tool scope) is a fallback, not a peer: it stays eligible only
        # when no specialist qualifies or every eligible specialist is proven unreliable.
        specialists = [p for p in eligible if p.tool_servers]
        healthy = [
            p for p in specialists
            if p.successes + p.failures < self.min_route_evidence
            or beta_success_rate(p.successes, p.failures) >= 0.5
        ]
        if healthy:
            eligible = specialists
        if not self.adaptive:
            chosen = next((p for p in eligible if p.name == node.agent_type), None) or next(
                (p for p in eligible if not p.tool_servers), eligible[0]
            )
            return chosen, {"selected_agent": chosen.name, "reason": "static planner hint"}
        query = " ".join([node.description, *node.preferred_tools])
        decision = self.router.route(query, eligible, hint=node.agent_type)
        chosen = next(p for p in eligible if p.name == decision.selected_agent)
        return chosen, decision.model_dump()

    async def discover(
        self,
        node: TaskNode,
        agent: AgentProfile | None = None,
        tool_stats: dict[str, dict[str, float]] | None = None,
    ) -> tuple[list[ToolMetadata], dict[str, float]]:
        tools = [tool for tool in await self.tool_provider.list_tools() if tool.enabled]
        scoped = [tool for tool in tools if agent is None or agent.can_use(tool)]
        ranked = rank_tools(
            " ".join([node.description, *node.preferred_tools]),
            scoped,
            tool_stats,
            node.preferred_tools,
        )
        candidates = ranked[: self.candidate_count]
        if not candidates:
            # No usable tool means the plan step is infeasible as written: ask for a new plan.
            raise RecoveryExhausted(
                ErrorType.LOGIC_FAILURE, "replan", ToolCapabilityError("no enabled tools discovered")
            )
        return [tool for tool, _ in candidates], {tool.name: round(score, 4) for tool, score in candidates}

    async def select_tool(
        self,
        node: TaskNode,
        context: dict[str, Any],
        candidates: list[ToolMetadata],
        agent: AgentProfile | None = None,
        feedback: list[str] | None = None,
    ) -> tuple[ToolCallDecision, list[Usage]]:
        catalog = [
            {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
            for tool in candidates
        ]
        instructions = agent.instructions if agent else ""
        payload: dict[str, Any] = {
            "task": node.description,
            "dependency_results": context,
            "tools": catalog,
        }
        if feedback:
            payload["previous_attempt_feedback"] = feedback
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    f"{instructions} Choose exactly one supplied tool and produce arguments "
                    "that satisfy its JSON Schema. Use concrete values from dependency_results "
                    'when needed. Return only JSON: {"tool_name": ..., "arguments": {...}, '
                    '"reason": ...}.'
                ).strip(),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ]
        allowed = {tool.name for tool in candidates}
        usages: list[Usage] = []
        error: Exception | None = None
        for _ in range(self.selection_repairs + 1):
            try:
                decision, usage = await self.llm.chat_json(
                    messages, ToolCallDecision, temperature=0, max_tokens=512
                )
                usages.append(usage)
                if decision.tool_name not in allowed:
                    raise ToolCapabilityError(
                        f"model selected unavailable tool {decision.tool_name}; allowed={sorted(allowed)}"
                    )
                return decision, usages
            except (ToolCapabilityError, ValidationError, json.JSONDecodeError) as exc:
                error = exc
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous answer was rejected: {exc}. Choose one tool from "
                            f"{sorted(allowed)} and return valid ToolCallDecision JSON."
                        ),
                    }
                )
        assert error is not None
        # Selection failures escalate like any other exhausted node so the runtime can replan.
        raise RecoveryExhausted(ErrorType.LOGIC_FAILURE, "replan", error) from error

    async def _repair_arguments(
        self,
        node: TaskNode,
        tool: ToolMetadata,
        current: dict[str, Any],
        exc: Exception,
        context: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Usage]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Repair tool arguments to satisfy the supplied JSON Schema and fix the "
                    'reported error. Return only JSON: {"arguments": {...}}.'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": node.description,
                        "tool": tool.name,
                        "schema": tool.input_schema,
                        "current_arguments": current,
                        "validation_error": str(exc),
                        "dependency_results": context or {},
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ]
        repaired, usage = await self.llm.chat_json(
            messages, RepairedArguments, temperature=0, max_tokens=512
        )
        return repaired.arguments, usage

    async def execute(
        self,
        node: TaskNode,
        context: dict[str, dict[str, Any]],
        feedback: list[str] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        agent_stats, tool_stats = await self._stats()
        tools = [tool for tool in await self.tool_provider.list_tools() if tool.enabled]
        agent, route = self._route(node, tools, agent_stats)
        usages: list[Usage] = []
        try:
            candidates, candidate_scores = await self.discover(node, agent, tool_stats)
            visible = {key: compact_result(context[key]) for key in node.dependencies if key in context}
            decision, selection_usages = await self.select_tool(node, visible, candidates, agent, feedback)
            usages.extend(selection_usages)
            tool = next(item for item in candidates if item.name == decision.tool_name)
            used_arguments = dict(decision.arguments)
            repair_usages: list[dict[str, Any]] = []

            async def call(arguments: dict[str, Any]) -> Any:
                nonlocal used_arguments
                Draft202012Validator(tool.input_schema).validate(arguments)
                used_arguments = dict(arguments)
                call_started = time.perf_counter()
                try:
                    output = await self.tool_provider.call_tool(
                        tool.name, arguments, operation_id=node.operation_id
                    )
                except Exception:
                    await self._record_tool(tool.name, False, call_started)
                    raise
                await self._record_tool(tool.name, True, call_started)
                return output

            async def repair(arguments: dict[str, Any], exc: Exception) -> dict[str, Any]:
                repaired, usage = await self._repair_arguments(node, tool, arguments, exc, visible)
                usages.append(usage)
                repair_usages.append(usage.model_dump())
                return repaired

            output, trace = await self.recovery.execute(
                call,
                decision.arguments,
                repair=repair,
                allow_empty_list=tool.output_schema.get("type") == "array",
            )
        except Exception:
            if self.stats is not None:
                await self.stats.arecord_agent_outcome(
                    agent.name, False, usage_tokens(usages), (time.perf_counter() - started) * 1000
                )
            raise
        return {
            "agent": agent.name,
            "planned_agent": node.agent_type,
            "route": route,
            "candidates": candidate_scores,
            "tool": tool.name,
            "arguments": used_arguments,
            "output": output,
            "tool_output_schema": tool.output_schema,
            "idempotent": tool.idempotent,
            "recovery_trace": trace,
            "llm_usage": {
                "tool_selection": [usage.model_dump() for usage in selection_usages],
                "repairs": repair_usages,
            },
            "llm_tokens": usage_tokens(usages),
            "latency_ms": (time.perf_counter() - started) * 1000,
            "context_nodes": sorted(visible),
        }

    async def _record_tool(self, name: str, success: bool, started: float) -> None:
        if self.stats is not None:
            await self.stats.arecord_tool_outcome(name, success, (time.perf_counter() - started) * 1000)
