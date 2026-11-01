from __future__ import annotations

import json
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field

from agentos.agents.profiles import DEFAULT_AGENTS, agent_catalog
from agentos.llm.base import BaseLLM
from agentos.models import AgentProfile, TaskNode, TaskPlan, TaskStatus, ToolMetadata
from agentos.runtime.dag import plan_shape

PLANNING_RULES = (
    "You are the planner of a multi-agent runtime. Decompose the goal into a DAG of "
    "subtasks and return PlanDraft JSON. Rules: "
    "(1) Each subtask is ONE tool call by ONE agent from available_agents. preferred_tools "
    "names exactly one tool from available_tools whose server is in that agent's "
    "tool_servers, or two alternative tools when the subtask is a conditional action. "
    "(2) Subtasks that do not need each other's outputs must NOT depend on each other, so "
    "the scheduler runs them in parallel. Add a dependency only when a subtask needs a "
    "concrete value (an identifier, name, date or number) produced upstream. "
    "(3) Write each description so the agent can act on it alone, naming which values to "
    "take from its dependencies. "
    "(4) Include every step needed to finish the goal, including side-effecting actions "
    "(tickets, refunds) the goal asks for. A conditional action is a single subtask that "
    "depends on the checks it needs and whose description states the condition and both "
    "outcomes. Never schedule the same side effect twice. "
    "(5) Do not add a final summary subtask; the runtime composes the final answer from "
    "subtask outputs. "
    "(6) Leave expected_output_schema empty unless you know it; do not mark nodes successful."
)


class PlanNodeDraft(BaseModel):
    task_id: str
    description: str
    agent_type: str = "general_executor"
    dependencies: list[str] = Field(default_factory=list)
    preferred_tools: list[str] = Field(default_factory=list)
    required_output_keys: list[str] = Field(default_factory=list)
    expected_output_schema: dict = Field(default_factory=dict)
    idempotent: bool = True


class PlanDraft(BaseModel):
    subtasks: list[PlanNodeDraft]


class Planner:
    def __init__(
        self,
        llm: BaseLLM,
        max_repairs: int = 1,
        agents: tuple[AgentProfile, ...] | list[AgentProfile] = DEFAULT_AGENTS,
        max_tokens: int = 2400,
    ):
        self.llm, self.max_repairs = llm, max_repairs
        self.agents = list(agents)
        self.max_tokens = max_tokens

    async def _generate(
        self,
        messages: list[dict[str, str]],
        goal: str,
        kept: list[TaskNode] | None = None,
    ) -> TaskPlan:
        """Ask for a PlanDraft and validate it as a DAG; ``kept`` nodes may be depended on."""
        kept = kept or []
        kept_ids = {node.task_id for node in kept}
        error: Exception | None = None
        for _ in range(self.max_repairs + 1):
            try:
                draft, usage = await self.llm.chat_json(
                    messages, PlanDraft, temperature=0, max_tokens=self.max_tokens
                )
                new_nodes = [
                    TaskNode.model_validate(node.model_dump())
                    for node in draft.subtasks
                    if node.task_id not in kept_ids
                ]
                if not new_nodes:
                    raise ValueError("plan must contain at least one new subtask")
                return TaskPlan(
                    goal=goal,
                    subtasks=[*kept, *new_nodes],
                    metadata={"planner_usage": usage.model_dump()},
                )
            except Exception as exc:
                error = exc
                if not isinstance(exc, httpx.HTTPError):
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Repair the plan and return a complete PlanDraft JSON. "
                                f"Validation error: {type(exc).__name__}: {exc!r}"
                            ),
                        }
                    )
        assert error is not None
        raise ValueError(
            "planner failed after repair budget: "
            f"{type(error).__name__}: {error!r}"
        )

    @staticmethod
    def _signature(schema: dict) -> dict:
        """Compact field summary of a JSON Schema: planning needs names and types, not $defs."""
        if not schema:
            return {}
        definitions = schema.get("$defs", {})
        if schema.get("type") == "array":
            item = schema.get("items", {})
            ref = item.get("$ref", "")
            item = definitions.get(ref.rsplit("/", 1)[-1], item) if ref else item
            return {"list_of": Planner._signature({**item, "$defs": definitions})}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        return {
            name: f"{spec.get('type', 'any')}{'' if name in required else '?'}"
            for name, spec in properties.items()
        }

    @classmethod
    def _tool_catalog(cls, tools: list[ToolMetadata]) -> list[dict]:
        return [
            {
                "name": tool.name,
                "server": tool.server_name,
                "description": tool.description,
                "inputs": cls._signature(tool.input_schema),
                "outputs": cls._signature(tool.output_schema),
                "side_effect": not tool.idempotent,
            }
            for tool in tools
        ]

    @staticmethod
    def _reset_node(node: TaskNode) -> TaskNode:
        return node.model_copy(
            update={
                "status": TaskStatus.PENDING,
                "result": None,
                "retry_count": 0,
                "operation_id": str(uuid4()),
            }
        )

    def _ground_agent(self, node: TaskNode, by_name: dict[str, ToolMetadata]) -> str:
        """Keep the planner's agent if it may use the node's tools, else pick one that can."""
        tools = [by_name[name] for name in node.preferred_tools if name in by_name]
        known = {agent.name: agent for agent in self.agents}
        current = known.get(node.agent_type)
        if current and all(current.can_use(tool) for tool in tools):
            return current.name
        for agent in self.agents:
            if agent.tool_servers and tools and all(agent.can_use(tool) for tool in tools):
                return agent.name
        if current:
            return current.name
        generalist = next((agent for agent in self.agents if not agent.tool_servers), None)
        return generalist.name if generalist else node.agent_type

    def _ground_contracts(
        self, nodes: list[TaskNode], tools: list[ToolMetadata]
    ) -> list[TaskNode]:
        by_name = {tool.name: tool for tool in tools}
        grounded = []
        for original in nodes:
            # Acceptance keys come from real tool contracts, never from the model's guess.
            node = self._reset_node(original).model_copy(
                update={"required_output_keys": [], "expected_output_schema": {}}
            )
            if by_name:
                known = [name for name in node.preferred_tools if name in by_name]
                node = node.model_copy(
                    update={"preferred_tools": known, "agent_type": self._ground_agent(node, by_name)}
                )
            if (
                len(node.preferred_tools) == 1
                and node.preferred_tools[0] in by_name
                and by_name[node.preferred_tools[0]].output_schema
            ):
                tool = by_name[node.preferred_tools[0]]
                schema = tool.output_schema
                required = (
                    list(schema.get("required", []))
                    if schema.get("type") == "object"
                    else []
                )
                node = node.model_copy(
                    update={
                        "expected_output_schema": schema,
                        "required_output_keys": required,
                        "idempotent": node.idempotent and tool.idempotent,
                    }
                )
            grounded.append(node)
        return grounded

    async def plan(self, goal: str, tools: list[ToolMetadata] | None = None) -> TaskPlan:
        schema = PlanDraft.model_json_schema()
        messages = [
            {"role": "system", "content": PLANNING_RULES},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "goal": goal,
                        "available_agents": agent_catalog(self.agents),
                        "available_tools": self._tool_catalog(tools or []),
                        "plan_draft_schema": schema,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        generated = await self._generate(messages, goal)
        subtasks = self._ground_contracts(generated.subtasks, tools or [])
        return generated.model_copy(
            update={
                "goal": goal,
                "version": 1,
                "subtasks": subtasks,
                "metadata": {**generated.metadata, "shape": plan_shape(subtasks)},
            }
        )

    async def replan(
        self,
        current: TaskPlan,
        failed_node: TaskNode,
        error: dict,
        tools: list[ToolMetadata],
    ) -> TaskPlan:
        from agentos.agents.executor import compact_result

        kept = [node for node in current.subtasks if node.status == TaskStatus.SUCCESS]
        summary = [
            {
                "task_id": node.task_id,
                "description": node.description,
                "agent_type": node.agent_type,
                "dependencies": node.dependencies,
                "preferred_tools": node.preferred_tools,
                "status": node.status.value,
                **({"output": compact_result(node.result or {}).get("output")} if node in kept else {}),
            }
            for node in current.subtasks
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    PLANNING_RULES
                    + " You are now REPAIRING a plan. Successful subtasks are kept automatically "
                    "and their outputs are shown. Return a PlanDraft containing ONLY the new or "
                    "replacement subtasks (fresh task_ids) needed to finish the goal; they may "
                    "depend on successful task_ids. Avoid the reported failure, e.g. resolve a "
                    "missing identifier with a lookup tool, or use an alternative tool or agent."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "goal": current.goal,
                        "current_plan": summary,
                        "failed_node": failed_node.task_id,
                        "failure": {k: error.get(k) for k in ("error", "message", "error_type")},
                        "available_agents": agent_catalog(self.agents),
                        "available_tools": self._tool_catalog(tools),
                        "plan_draft_schema": PlanDraft.model_json_schema(),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ]
        generated = await self._generate(messages, current.goal, kept=kept)
        kept_ids = {node.task_id for node in kept}
        new_nodes = self._ground_contracts(
            [node for node in generated.subtasks if node.task_id not in kept_ids], tools
        )
        merged = [*kept, *new_nodes]
        return TaskPlan(
            task_id=current.task_id,
            goal=current.goal,
            subtasks=merged,
            version=current.version + 1,
            metadata={**generated.metadata, "shape": plan_shape(merged)},
        )
