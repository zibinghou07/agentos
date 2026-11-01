import json
import time
import uuid

import pytest

from agentos.llm.base import BaseLLM, ChatResult, Usage, extract_json
from agentos.mcp import MCPToolProvider
from agentos.runtime.checkpoint import CheckpointStore
from agentos.taskbench.generator import generate_tasks
from agentos.taskbench.grading import contains_phrase, extract_numbers, grade, token_f1
from agentos.taskbench.hotpot import hotpot_tasks
from agentos.taskbench.instrument import InjectedFault, InstrumentedProvider
from agentos.taskbench.runner import run_one
from agentos.taskbench.schema import FaultSpec
from agentos.world import World, build_mcp_servers
from agentos.world import data


@pytest.fixture
def world(tmp_path):
    return World.create(tmp_path / "world.db")


def test_world_is_rich_and_deterministic(world):
    assert len(world.documents) >= 40
    assert len(data.ORDERS) == 120 and len(data.PRODUCTS) == 12 and len(data.CUSTOMERS) == 16
    again = data._generate_orders()
    assert again == data.ORDERS


async def test_mcp_servers_expose_structured_tools_and_side_effect_hints(world):
    provider = MCPToolProvider(build_mcp_servers(world))
    tools = {tool.name: tool for tool in await provider.list_tools()}
    assert len(tools) >= 23
    assert tools["issue_refund"].idempotent is False and tools["create_ticket"].idempotent is False
    assert tools["get_product"].idempotent is True
    assert tools["get_product"].output_schema["required"][:2] == ["id", "name"]
    with pytest.raises(RuntimeError, match="product not found: p99"):
        await provider.call_tool("get_product", {"product_id": "p99"})


async def test_tool_outputs_match_generated_gold(world):
    provider = MCPToolProvider(build_mcp_servers(world))
    analytics = next(t for t in generate_tasks("test") if t.task_id == "analytics-revenue-hardware")
    summary = await provider.call_tool(
        "aggregate_revenue", {"start_date": "2026-01-01", "end_date": "2026-03-31", "category": "Hardware"}
    )
    assert summary["revenue"] == analytics.checks[0].value


def test_splits_are_entity_disjoint_and_synthetic():
    dev, test = generate_tasks("dev"), generate_tasks("test")
    assert all(task.synthetic for task in dev + test)
    dev_text = " ".join(task.goal for task in dev)
    for buyer in data.CUSTOMERS[8:]:
        assert buyer.name not in dev_text
    policy = [t for t in test if t.family == "policy_action"]
    assert {t.source["eligible"] for t in policy} == {True, False}
    assert any(t.fault and t.fault.mode == "outage" for t in test)


def test_graders_are_programmatic(world):
    assert extract_numbers("Revenue was $17,847.00 (+12.5%)") == [17847.0, 12.5]
    assert contains_phrase("The affected product is the Sensor Hub.", "Sensor Hub")
    assert not contains_phrase("Sensor", "Sensor Hub")
    assert token_f1("The Conversation", "The Conversation") == 1.0
    task = next(t for t in generate_tasks("test") if t.family == "policy_action" and t.source["eligible"])
    results = {r.label: r.passed for r in grade(task, "done", world)}
    assert results["refund_count"] is False and results["denied_ticket_count"] is True


async def test_fault_injection_is_identical_for_every_method(world):
    inner = MCPToolProvider(build_mcp_servers(world))
    transient = InstrumentedProvider(inner, FaultSpec(tool="get_customer", mode="transient", failures=2))
    await transient.list_tools()
    for _ in range(2):
        with pytest.raises(InjectedFault, match="503 service unavailable"):
            await transient.call_tool("get_customer", {"customer_id": "c1"})
    assert (await transient.call_tool("get_customer", {"customer_id": "c1"}))["tier"] == "gold"
    outage = InstrumentedProvider(inner, FaultSpec(tool="get_customer", mode="outage"))
    for _ in range(3):
        with pytest.raises(InjectedFault):
            await outage.call_tool("get_customer", {"customer_id": "c1"})
    assert outage.errors == 3


def test_hotpot_tasks_become_agentic_with_private_corpus():
    tasks = hotpot_tasks("benchmarks/real_data/all.jsonl", "test")
    assert len(tasks) == 25 and not any(t.synthetic for t in tasks)
    assert all(len(t.documents) == 10 for t in tasks)
    assert all(t.checks[0].aliases[0] == t.source["gold_answer"] for t in tasks)


class ScriptedLLM(BaseLLM):
    """Stage-aware fake model: answers planner/executor/synthesizer prompts and tool calls."""

    def __init__(self, script):
        self.script, self.calls = script, 0

    def _usage(self):
        return Usage(request_id=str(uuid.uuid4()), model_name="scripted", input_tokens=10,
                     output_tokens=5, latency_ms=0.1)

    async def chat(self, messages, **kwargs):
        self.calls += 1
        reply = self.script(messages, kwargs)
        if isinstance(reply, dict) and "tool_calls" in reply:
            raw = {"choices": [{"message": {"content": "", "tool_calls": reply["tool_calls"]}}]}
            return ChatResult(content="", usage=self._usage(), raw=raw)
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return ChatResult(content=content, usage=self._usage(), raw={"choices": [{"message": {"content": content}}]})

    async def chat_json(self, messages, schema, **kwargs):
        result = await self.chat(messages, **kwargs)
        return schema.model_validate_json(extract_json(result.content)), result.usage


def _call(name, arguments):
    return {"tool_calls": [{"id": f"call-{time.time_ns()}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)}}]}


async def test_agentos_end_to_end_parallel_plan_with_transient_fault(tmp_path):
    task = next(t for t in generate_tasks("dev") if t.task_id == "parallel-c1-transient-list_orders")
    q1 = sum(1 for o in data.ORDERS if o.customer_id == "c1" and "2026-01-01" <= o.order_date <= "2026-03-31")

    def script(messages, kwargs):
        system = messages[0]["content"]
        if "planner of a multi-agent runtime" in system:
            return {"subtasks": [
                {"task_id": "customer", "description": "find Aurora Logistics", "agent_type": "support_agent",
                 "preferred_tools": ["find_customer"]},
                {"task_id": "sla", "description": "read SLA policy", "agent_type": "researcher",
                 "preferred_tools": ["get_document"]},
                {"task_id": "orders", "description": "Q1 orders for the customer id", "agent_type": "data_analyst",
                 "dependencies": ["customer"], "preferred_tools": ["list_orders"]},
            ]}
        if "Choose exactly one supplied tool" in system:
            task_text = json.loads(messages[1]["content"])["task"]
            if "Aurora" in task_text:
                return {"tool_name": "find_customer", "arguments": {"name": "Aurora"}}
            if "SLA" in task_text:
                return {"tool_name": "get_document", "arguments": {"document_id": "policy_sla"}}
            return {"tool_name": "list_orders", "arguments": {
                "customer_id": "c1", "start_date": "2026-01-01", "end_date": "2026-03-31"}}
        if "Write the final answer" in system:
            evidence = json.loads(messages[1]["content"])["subtasks"]
            count = len(evidence["orders"]["output"])
            return {"answer": f"Gold tier: 4 hours first response; {count} orders in Q1.", "evidence_nodes": ["orders"]}
        raise AssertionError(system)

    record = await run_one(
        task, "agentos", ScriptedLLM(script), CheckpointStore(tmp_path / "store.db"),
        run_id=0, workdir=tmp_path, timeout=30, provider_name="scripted", model_name="scripted",
    )
    assert record.success, record.checks
    assert record.extra["plan_shape"]["parallelism"] == 1.5
    assert record.extra["planned_agents"] == {"customer": "support_agent", "sla": "researcher", "orders": "data_analyst"}
    # epsilon-greedy routing may explore, but only among agents allowed to use the tool
    assert record.extra["executed_agents"]["orders"] in {"data_analyst", "general_executor"}
    assert record.tool_errors == 2 and record.extra["replans"] == 0  # retried locally
    assert f"{q1} orders" in record.answer


async def test_react_baseline_runs_a_real_tool_calling_loop(tmp_path):
    task = next(t for t in generate_tasks("dev") if t.task_id == "lookup-p1")
    seen = []

    def script(messages, kwargs):
        assert "tools" in kwargs and len(kwargs["tools"]) >= 23
        tool_messages = [m for m in messages if m["role"] == "tool"]
        seen.append(len(tool_messages))
        if not tool_messages:
            return _call("search_products", {"keyword": "AgentOS"})
        record = json.loads(tool_messages[-1]["content"])[0]
        return f"Warranty {record['warranty_months']} months, price {record['price']} USD."

    record = await run_one(
        task, "react", ScriptedLLM(script), CheckpointStore(tmp_path / "s.db"),
        run_id=0, workdir=tmp_path, timeout=30, provider_name="scripted", model_name="scripted",
    )
    assert record.success and record.tool_calls == 1 and record.llm_calls == 2
    assert seen == [0, 1]


async def test_fixed_roles_baseline_scopes_tools_per_role(tmp_path):
    task = next(t for t in generate_tasks("dev") if t.task_id == "lookup-p1")
    scopes = []

    def script(messages, kwargs):
        if "tools" not in kwargs:
            return "Warranty 12 months; price 99.0 USD."
        scopes.append({tool["function"]["name"] for tool in kwargs["tools"]})
        return "nothing to add"

    record = await run_one(
        task, "fixed_roles", ScriptedLLM(script), CheckpointStore(tmp_path / "s.db"),
        run_id=0, workdir=tmp_path, timeout=30, provider_name="scripted", model_name="scripted",
    )
    assert record.success and record.llm_calls == 4
    assert "search_documents" in scopes[0] and "get_product" not in scopes[0]
    assert "get_product" in scopes[1] and "issue_refund" in scopes[2]


async def test_token_budget_stops_new_runs_and_keeps_task_major_order(tmp_path):
    from agentos.taskbench.runner import run_suite

    tasks = [t for t in generate_tasks("dev") if t.family == "lookup"][:3]

    def script(messages, kwargs):
        return "Warranty 12 months; price 99.0 USD."

    output = tmp_path / "runs.jsonl"
    records = await run_suite(
        tasks, ["react", "fixed_roles"], ScriptedLLM(script), output,
        concurrency=1, provider_name="scripted", model_name="scripted", token_budget=40,
    )
    order = [(r.task_id, r.method) for r in records]
    # react = 1 call (15 tokens), fixed_roles = 4 calls (60 tokens): budget hit after task 1
    assert order == [(tasks[0].task_id, "react"), (tasks[0].task_id, "fixed_roles")]
