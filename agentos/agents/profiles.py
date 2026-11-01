from __future__ import annotations

from agentos.models import AgentProfile

DEFAULT_AGENTS: tuple[AgentProfile, ...] = (
    AgentProfile(
        name="researcher",
        capabilities=[
            "search read documents policy postmortem incident runbook product sheet evidence "
            "warranty refund sla shipping knowledge retrieval"
        ],
        tool_servers=["search", "knowledge"],
        instructions=(
            "You are the research agent. Find and read the documents that contain the "
            "evidence the task needs; prefer exact document identifiers when known."
        ),
    ),
    AgentProfile(
        name="data_analyst",
        capabilities=[
            "product price category inventory stock units order orders revenue sales "
            "aggregate calculate percent change quarter statistics numbers"
        ],
        tool_servers=["catalog", "sales", "analytics"],
        instructions=(
            "You are the data analyst. Query structured product, inventory and sales data "
            "and compute exact numbers; never estimate when a tool can compute."
        ),
    ),
    AgentProfile(
        name="support_agent",
        capabilities=[
            "customer account tier ticket refund issue create warehouse shipping delivery "
            "support escalation"
        ],
        tool_servers=["crm", "ops"],
        instructions=(
            "You are the customer-operations agent. Look up customers, warehouses and "
            "shipping, and perform side-effecting actions (tickets, refunds) exactly once."
        ),
    ),
    AgentProfile(
        name="general_executor",
        capabilities=["general tool execution task"],
        instructions="You are a general executor that may use any available tool.",
    ),
)


def agent_catalog(agents: tuple[AgentProfile, ...] | list[AgentProfile]) -> list[dict]:
    return [
        {
            "name": agent.name,
            "capabilities": agent.capabilities,
            "tool_servers": agent.tool_servers or ["*"],
        }
        for agent in agents
    ]
