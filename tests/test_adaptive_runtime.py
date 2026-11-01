import asyncio
import time

import httpx
import pytest

from agentos.agents import AgentExecutor, Planner, Verifier
from agentos.llm.mock import MockProvider
from agentos.models import AgentProfile, TaskNode, TaskPlan, TaskStatus, ToolMetadata
from agentos.runtime.checkpoint import CheckpointStore
from agentos.runtime.dag import plan_shape
from agentos.runtime.recovery import (
    ErrorType,
    RecoveryExhausted,
    RecoveryManager,
    classify_error,
)
from agentos.runtime.routing import AdaptiveRouter, rank_tools
from agentos.runtime.scheduler import DAGScheduler
from agentos.runtime.service import AgentRuntimeService


class FakeProvider:
    def __init__(self, tools, functions):
        self.tools, self.functions, self.calls = tools, functions, []

    async def list_tools(self):
        return self.tools

    async def call_tool(self, name, arguments, operation_id=None):
        self.calls.append((name, arguments))
        return self.functions[name](arguments)


def tool(name, server="fake", *, idempotent=True, output_schema=None):
    return ToolMetadata(
        name=name,
        description=f"{name} tool",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        server_name=server,
        idempotent=idempotent,
        output_schema=output_schema or {},
    )


def node(task_id="n", tools=("a",), agent="general_executor", deps=()):
    return TaskNode(
        task_id=task_id, description=f"{task_id} task", agent_type=agent,
        dependencies=list(deps), preferred_tools=list(tools),
    )


def test_router_uses_persisted_success_cost_and_latency(tmp_path):
    store = CheckpointStore(tmp_path / "s.db")
    agents = [
        AgentProfile(name="alpha", capabilities=["search documents"]),
        AgentProfile(name="beta", capabilities=["search documents"]),
    ]
    router = AdaptiveRouter(epsilon=0)
    first = router.route("search documents", [a.with_stats(store.agent_stats().get(a.name)) for a in agents])
    assert first.selected_agent == "alpha"  # tie broken by order
    for _ in range(4):
        store.record_agent_outcome("alpha", False, cost=900, latency_ms=4000)
        store.record_agent_outcome("beta", True, cost=200, latency_ms=500)
    stats = store.agent_stats()
    assert stats["alpha"]["failures"] == 4 and stats["beta"]["successes"] == 4
    hydrated = [a.with_stats(stats.get(a.name)) for a in agents]
    assert hydrated[0].average_cost == 900 and hydrated[1].average_latency_ms == 500
    decision = router.route("search documents", hydrated)
    assert decision.selected_agent == "beta"
    assert decision.scores["beta"] > decision.scores["alpha"]


def test_rank_tools_demotes_a_failing_preferred_tool():
    a, b = tool("search_documents"), tool("bm25_search")
    healthy = rank_tools("search documents", [a, b], {}, ["search_documents"])
    assert healthy[0][0].name == "search_documents"
    failing = {
        "search_documents": {"calls": 30, "successes": 0, "total_latency_ms": 30 * 2000.0},
        "bm25_search": {"calls": 30, "successes": 30, "total_latency_ms": 30 * 10.0},
    }
    ranked = rank_tools("search documents", [a, b], failing, ["search_documents"])
    assert ranked[0][0].name == "bm25_search"


async def test_executor_scopes_tools_to_routed_agent_and_records_stats(tmp_path):
    store = CheckpointStore(tmp_path / "s.db")
    provider = FakeProvider(
        [tool("lookup", "catalog"), tool("search", "search")],
        {"lookup": lambda args: {"value": 1}, "search": lambda args: [{"hit": 1}]},
    )
    agents = [
        AgentProfile(name="analyst", capabilities=["lookup product price"], tool_servers=["catalog"]),
        AgentProfile(name="researcher", capabilities=["search documents"], tool_servers=["search"]),
    ]
    llm = MockProvider([{"tool_name": "lookup", "arguments": {}}], latency_ms=0)
    executor = AgentExecutor(provider, llm, RecoveryManager(base_delay=0), agent_profiles=agents, stats=store)
    result = await executor.execute(
        TaskNode(task_id="n", description="lookup product price", agent_type="analyst", preferred_tools=["lookup"]),
        {},
    )
    assert result["agent"] == "analyst"
    assert set(result["candidates"]) == {"lookup"}  # researcher's tool is out of scope
    assert store.tool_stats()["lookup"]["successes"] == 1


async def test_invalid_tool_selection_is_repaired_then_escalates_to_replan():
    provider = FakeProvider([tool("a")], {"a": lambda args: {"ok": True}})
    llm = MockProvider([{"tool_name": "ghost"}, {"tool_name": "a", "arguments": {}}], latency_ms=0)
    executor = AgentExecutor(provider, llm, RecoveryManager(base_delay=0))
    result = await executor.execute(node(), {})
    assert result["tool"] == "a" and len(result["llm_usage"]["tool_selection"]) == 2

    stubborn = AgentExecutor(
        provider, MockProvider([{"tool_name": "ghost"}], latency_ms=0), RecoveryManager(base_delay=0)
    )
    with pytest.raises(RecoveryExhausted) as caught:
        await stubborn.execute(node(), {})
    assert caught.value.action == "replan"


def test_auth_errors_fail_fast_instead_of_replanning():
    request = httpx.Request("POST", "https://example.invalid")
    exc = httpx.HTTPStatusError("x", request=request, response=httpx.Response(401, request=request))
    assert classify_error(exc) == ErrorType.AUTH_ERROR


async def test_auth_error_exhaustion_returns_fail_action():
    async def call(arguments):
        raise RuntimeError("401 unauthorized: invalid api key")

    with pytest.raises(RecoveryExhausted) as caught:
        await RecoveryManager(base_delay=0).execute(call, {})
    assert caught.value.action == "fail"


def _service(tmp_path, llm, provider, **kwargs):
    store = CheckpointStore(tmp_path / "runtime.db")
    executor = AgentExecutor(provider, llm, RecoveryManager(retries=0, repairs=0, base_delay=0), stats=store)
    return AgentRuntimeService(Planner(llm), executor, Verifier(), store, **kwargs), store


async def test_verifier_repair_action_reexecutes_node_before_replanning(tmp_path):
    schema = {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}
    provider = FakeProvider(
        [tool("bad", output_schema=schema), tool("good", output_schema=schema)],
        {"bad": lambda args: {"value": "text"}, "good": lambda args: {"value": 3}},
    )
    plan = {"subtasks": [{"task_id": "n", "description": "get value", "agent_type": "general_executor"}]}
    llm = MockProvider(
        [plan, {"tool_name": "bad", "arguments": {}}, {"tool_name": "good", "arguments": {}}], latency_ms=0
    )
    service, store = _service(tmp_path, llm, provider)
    record = await service.submit("get value")
    await service.wait(record.task_id)
    finished = await service.get_task(record.task_id)
    assert finished.status == TaskStatus.SUCCESS and finished.plan_version == 1
    events = [e.event for e in await service.get_trace(record.task_id)]
    assert "NODE_REEXECUTING" in events and "REPLAN_STARTED" not in events
    stats = store.agent_stats()["general_executor"]
    assert stats["failures"] == 1 and stats["successes"] == 1


async def test_failed_verification_of_side_effect_requires_manual_review(tmp_path):
    schema = {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}
    provider = FakeProvider(
        [tool("refund", idempotent=False, output_schema=schema)], {"refund": lambda args: {"id": "x"}}
    )
    plan = {"subtasks": [{"task_id": "n", "description": "refund", "agent_type": "general_executor"}]}
    llm = MockProvider([plan, {"tool_name": "refund", "arguments": {}}], latency_ms=0)
    service, _ = _service(tmp_path, llm, provider)
    record = await service.submit("refund")
    await service.wait(record.task_id)
    finished = await service.get_task(record.task_id)
    assert finished.status == TaskStatus.FAILED
    assert finished.error["recovery_action"] == "manual"
    assert len(provider.calls) == 1  # never repeated


async def test_scheduler_is_event_driven_not_wave_based(tmp_path):
    plan = TaskPlan(
        goal="g",
        subtasks=[node("slow", ()), node("fast", ()), node("after_fast", (), deps=("fast",))],
    )
    started: dict[str, float] = {}
    origin = time.perf_counter()

    async def execute(item, context):
        started[item.task_id] = time.perf_counter() - origin
        await asyncio.sleep(0.4 if item.task_id == "slow" else 0.05)
        return {"output": item.task_id}

    finished = await DAGScheduler(execute, CheckpointStore(tmp_path / "s.db"), 4).run(plan)
    assert all(n.status == TaskStatus.SUCCESS for n in finished.subtasks)
    # A wave scheduler would wait for "slow" (0.4s) before starting "after_fast".
    assert started["after_fast"] < 0.3


def test_plan_shape_measures_parallelism():
    nodes = [node("a", ()), node("b", ()), node("c", (), deps=("a", "b"))]
    shape = plan_shape(nodes)
    assert shape == {"nodes": 3, "edges": 2, "depth": 2, "max_width": 2, "parallelism": 1.5}


async def test_planner_grounds_agent_and_unknown_tools():
    tools = [tool("get_product", "catalog"), tool("search_documents", "search")]
    draft = {
        "subtasks": [
            {"task_id": "a", "description": "find doc", "agent_type": "data_analyst", "preferred_tools": ["search_documents"]},
            {"task_id": "b", "description": "price", "agent_type": "made_up", "preferred_tools": ["get_product", "nope"]},
        ]
    }
    plan = await Planner(MockProvider([draft], latency_ms=0)).plan("goal", tools)
    by_id = {n.task_id: n for n in plan.subtasks}
    assert by_id["a"].agent_type == "researcher"
    assert by_id["b"].agent_type == "data_analyst"
    assert by_id["b"].preferred_tools == ["get_product"]
    assert plan.metadata["shape"]["parallelism"] == 2.0


async def test_replan_returns_only_new_nodes_that_may_depend_on_kept_ones():
    tools = [tool("search_products", "catalog"), tool("get_inventory", "catalog")]
    done = node("doc", ("search_products",), agent="data_analyst")
    done.status, done.result = TaskStatus.SUCCESS, {"tool": "search_products", "output": [{"id": "p9"}]}
    failed = node("stock", ("get_inventory",), agent="data_analyst", deps=("doc",))
    failed.status = TaskStatus.FAILED
    current = TaskPlan(goal="g", subtasks=[done, failed])
    draft = {"subtasks": [
        {"task_id": "stock2", "description": "inventory for p9", "agent_type": "data_analyst",
         "dependencies": ["doc"], "preferred_tools": ["get_inventory"], "required_output_keys": ["invented"]},
    ]}
    llm = MockProvider([draft], latency_ms=0)
    spy = []
    original = llm.chat

    async def chat(messages, **kwargs):
        spy.append(messages[1]["content"])
        return await original(messages, **kwargs)

    llm.chat = chat
    plan = await Planner(llm).replan(current, failed, {"error": "x", "message": "not found"}, tools)
    assert [n.task_id for n in plan.subtasks] == ["doc", "stock2"] and plan.version == 2
    assert plan.subtasks[0] is done or plan.subtasks[0].status == TaskStatus.SUCCESS
    assert plan.subtasks[1].required_output_keys == []  # model-invented keys are dropped
    assert '"output": [{"id": "p9"}]' in spy[0] and "operation_id" not in spy[0]


def test_extract_json_ignores_trailing_text_and_unclosed_fences():
    from agentos.llm.base import extract_json

    assert extract_json('{"answer": "x"}\n}') == '{"answer": "x"}'
    assert extract_json('```json\n{"answer": "y"}') == '{"answer": "y"}'
    assert extract_json('Sure:\n```json\n{"a": [1]}\n``` done') == '{"a": [1]}'


async def test_synthesizer_coerces_node_objects_and_repairs_bad_json():
    from agentos.agents.synthesizer import FinalAnswer, Synthesizer

    cited = FinalAnswer.model_validate({"answer": "ok", "evidence_nodes": [{"id": "t1", "output": {}}, "t2"]})
    assert cited.evidence_nodes == ["t1", "t2"]

    class Flaky(MockProvider):
        calls = 0

        async def chat_json(self, messages, schema, **kwargs):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise ValueError("Invalid JSON: trailing characters")
            return await super().chat_json(messages, schema, **kwargs)

    plan = TaskPlan(goal="g", subtasks=[node("t1")])
    answer, _ = await Synthesizer(Flaky([{"answer": "42"}], latency_ms=0)).synthesize(plan)
    assert answer.answer == "42" and Flaky.calls == 2


async def test_empty_list_is_a_valid_answer_only_for_list_contracts():
    recovery = RecoveryManager(retries=0, repairs=0, base_delay=0)

    async def empty(arguments):
        return []

    output, _ = await recovery.execute(empty, {}, allow_empty_list=True)
    assert output == []
    with pytest.raises(RecoveryExhausted):
        await recovery.execute(empty, {})
    verifier = Verifier()
    assert verifier.verify({"output": []}, output_schema={"type": "array"}).passed
    assert verifier.verify({"output": []}).action == "retry"


def test_generalist_is_only_a_fallback_for_unreliable_specialists():
    analyst = AgentProfile(name="analyst", capabilities=["lookup"], tool_servers=["catalog"])
    generalist = AgentProfile(name="general_executor", capabilities=["lookup anything"])
    executor = AgentExecutor(
        FakeProvider([], {}), MockProvider([], latency_ms=0), RecoveryManager(),
        agent_profiles=[analyst, generalist],
    )
    tools = [tool("lookup", "catalog")]
    chosen, _ = executor._route(node(tools=("lookup",), agent="analyst"), tools, {})
    assert chosen.name == "analyst"
    broken = {"analyst": {"successes": 0, "failures": 6, "total_cost": 0.0, "total_latency_ms": 0.0}}
    eligible_names = set()
    for _ in range(3):
        chosen, decision = executor._route(node(tools=("lookup",), agent="analyst"), tools, broken)
        eligible_names |= set(decision["scores"])
    assert "general_executor" in eligible_names
