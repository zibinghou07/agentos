import asyncio

from agentos.agents import AgentExecutor, Planner, Verifier
from agentos.agents.executor import ToolCallDecision
from agentos.llm.mock import MockProvider
from agentos.mcp.provider import ToolProvider
from agentos.models import TaskNode, TaskPlan, TaskStatus, ToolMetadata
from agentos.runtime.checkpoint import CheckpointStore
from agentos.runtime.recovery import RecoveryManager
from agentos.runtime.service import AgentRuntimeService


class FakeToolProvider(ToolProvider):
    def __init__(self, tools, functions):
        self.tools = tools
        self.functions = functions
        self.calls = []

    async def list_tools(self):
        return self.tools

    async def call_tool(self, name, arguments, operation_id=None):
        self.calls.append((name, arguments, operation_id))
        value = self.functions[name](arguments)
        return await value if asyncio.iscoroutine(value) else value


def tool(name, schema, *, idempotent=True):
    return ToolMetadata(
        name=name,
        description=f"{name} tool",
        input_schema=schema,
        server_name="fake",
        idempotent=idempotent,
    )


def runtime(tmp_path, responses, provider, *, replans=1):
    llm = MockProvider(responses, latency_ms=0)
    recovery = RecoveryManager(
        retries=1, repairs=1, replans=replans, base_delay=0, jitter=0
    )
    store = CheckpointStore(tmp_path / "runtime.db")
    service = AgentRuntimeService(
        Planner(llm),
        AgentExecutor(provider, llm, recovery),
        Verifier(),
        store,
        max_replans=replans,
    )
    return service, store


async def test_end_to_end_repairs_schema_arguments_and_verifies(tmp_path):
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 10}},
        "required": ["n"],
        "additionalProperties": False,
    }
    provider = FakeToolProvider(
        [tool("double", schema)], {"double": lambda args: {"value": args["n"] * 2}}
    )
    plan = {
        "goal": "double two",
        "subtasks": [
            {
                "task_id": "calculate",
                "description": "double two",
                "agent_type": "calculator",
                "preferred_tools": ["double"],
                "required_output_keys": ["value"],
                "expected_output_schema": {
                    "type": "object",
                    "properties": {"value": {"const": 4}},
                    "required": ["value"],
                },
            }
        ],
    }
    decision = {"tool_name": "double", "arguments": {"n": 99}}
    repair = {"arguments": {"n": 2}}
    service, _ = runtime(tmp_path, [plan, decision, repair], provider)
    record = await service.submit("double two")
    await service.wait(record.task_id)
    finished = await service.get_task(record.task_id)
    assert finished and finished.status == TaskStatus.SUCCESS
    result = finished.final_result["nodes"]["calculate"]
    assert result["arguments"] == {"n": 2}
    assert result["output"] == {"value": 4}
    assert result["recovery_trace"][0].startswith("REPAIR:INVALID_SCHEMA")
    assert len(provider.calls) == 1
    assert provider.calls[0][1] == {"n": 2}


async def test_empty_tool_result_triggers_versioned_replan(tmp_path):
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    provider = FakeToolProvider(
        [tool("bad", schema), tool("good", schema)],
        {"bad": lambda args: {}, "good": lambda args: {"answer": "ok"}},
    )
    first_plan = {
        "goal": "get an answer",
        "subtasks": [
            {
                "task_id": "bad-node",
                "description": "try unavailable path",
                "agent_type": "worker",
                "preferred_tools": ["bad"],
            }
        ],
    }
    second_plan = {
        "goal": "get an answer",
        "subtasks": [
            {
                "task_id": "good-node",
                "description": "use working path",
                "agent_type": "worker",
                "preferred_tools": ["good"],
                "required_output_keys": ["answer"],
            }
        ],
    }
    responses = [
        first_plan,
        {"tool_name": "bad", "arguments": {}},
        second_plan,
        {"tool_name": "good", "arguments": {}},
    ]
    service, _ = runtime(tmp_path, responses, provider)
    record = await service.submit("get an answer")
    await service.wait(record.task_id)
    finished = await service.get_task(record.task_id)
    assert finished and finished.status == TaskStatus.SUCCESS
    assert finished.plan_version == 2
    assert [node.task_id for node in finished.plan.subtasks] == ["good-node"]
    events = await service.get_trace(record.task_id)
    assert "PLAN_REPLACED" in {event.event for event in events}
    assert [call[0] for call in provider.calls] == ["bad", "good"]


async def test_resume_reuses_confirmed_operation_without_tool_replay(tmp_path):
    schema = {"type": "object", "properties": {}}
    provider = FakeToolProvider([tool("unused", schema)], {"unused": lambda args: {}})
    service, store = runtime(tmp_path, [], provider)
    node = TaskNode(
        task_id="done", description="done", agent_type="worker", status=TaskStatus.RUNNING
    )
    plan = TaskPlan(task_id="resume-task", goal="resume", subtasks=[node])
    store.upsert_task(plan, TaskStatus.RUNNING)
    store.save(plan, plan.task_id, node.task_id, TaskStatus.RUNNING)
    store.record_operation(
        node.operation_id,
        plan.task_id,
        node.task_id,
        "SUCCESS",
        {"tool": "already-called", "output": {"ok": True}},
    )
    assert await service.resume(plan.task_id)
    await service.wait(plan.task_id)
    finished = await service.get_task(plan.task_id)
    assert finished and finished.status == TaskStatus.SUCCESS
    assert provider.calls == []
    assert "NODE_REUSED" in {event.event for event in await service.get_trace(plan.task_id)}


async def test_resume_does_not_replay_ambiguous_non_idempotent_operation(tmp_path):
    schema = {"type": "object", "properties": {}}
    provider = FakeToolProvider([tool("charge", schema)], {"charge": lambda args: {}})
    service, store = runtime(tmp_path, [], provider)
    node = TaskNode(
        task_id="charge",
        description="charge card",
        agent_type="worker",
        status=TaskStatus.RUNNING,
        idempotent=False,
    )
    plan = TaskPlan(task_id="ambiguous-task", goal="charge", subtasks=[node])
    store.upsert_task(plan, TaskStatus.RUNNING)
    store.save(plan, plan.task_id, node.task_id, TaskStatus.RUNNING)
    store.record_operation(node.operation_id, plan.task_id, node.task_id, "RUNNING")
    assert await service.resume(plan.task_id)
    await service.wait(plan.task_id)
    finished = await service.get_task(plan.task_id)
    assert finished and finished.status == TaskStatus.FAILED
    assert finished.error["recovery_action"] == "manual"
    assert provider.calls == []


async def test_persistent_cancel_and_single_runner_guard(tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(_):
        started.set()
        await release.wait()
        return {"ok": True}

    schema = {"type": "object", "properties": {}}
    provider = FakeToolProvider([tool("slow", schema)], {"slow": slow})
    plan = {
        "goal": "slow task",
        "subtasks": [
            {
                "task_id": "slow-node",
                "description": "slow task",
                "agent_type": "worker",
                "preferred_tools": ["slow"],
                "required_output_keys": ["ok"],
            }
        ],
    }
    service, store = runtime(
        tmp_path, [plan, {"tool_name": "slow", "arguments": {}}], provider
    )
    record = await service.submit("slow task")
    await started.wait()
    assert await service.start(record.task_id, resume=True) is False
    assert await service.cancel(record.task_id)
    assert store.cancel_requested(record.task_id)
    release.set()
    await service.wait(record.task_id)
    finished = await service.get_task(record.task_id)
    assert finished and finished.status == TaskStatus.CANCELLED


def test_sqlite_lease_blocks_a_second_process_runner(tmp_path):
    store_a = CheckpointStore(tmp_path / "lease.db")
    store_b = CheckpointStore(tmp_path / "lease.db")
    plan = TaskPlan(task_id="leased", goal="goal", subtasks=[])
    store_a.upsert_task(plan, TaskStatus.PENDING)
    assert store_a.acquire_lease(plan.task_id, "runner-a", 60)
    assert not store_b.acquire_lease(plan.task_id, "runner-b", 60)
    store_a.release_lease(plan.task_id, "runner-a")
    assert store_b.acquire_lease(plan.task_id, "runner-b", 60)


async def test_restart_recovers_task_interrupted_during_planning(tmp_path):
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    provider = FakeToolProvider(
        [tool("recover", schema)], {"recover": lambda args: {"ok": True}}
    )
    plan_response = {
        "subtasks": [
            {
                "task_id": "recovered-node",
                "description": "recover work",
                "preferred_tools": ["recover"],
                "required_output_keys": ["ok"],
            }
        ]
    }
    service, store = runtime(
        tmp_path,
        [plan_response, {"tool": "recover", "parameters": {}}],
        provider,
    )
    placeholder = TaskPlan(task_id="planning-crash", goal="recover work", subtasks=[])
    store.upsert_task(placeholder, TaskStatus.PLANNING)
    store.save(placeholder, placeholder.task_id, None, TaskStatus.PLANNING)
    assert await service.resume(placeholder.task_id)
    await service.wait(placeholder.task_id)
    finished = await service.get_task(placeholder.task_id)
    assert finished and finished.status == TaskStatus.SUCCESS
    assert "PLAN_RECOVERED" in {
        event.event for event in await service.get_trace(placeholder.task_id)
    }


def test_tool_decision_accepts_provider_field_aliases():
    decision = ToolCallDecision.model_validate(
        {"tool": "search", "parameters": {"query": "recovery"}}
    )
    assert decision.tool_name == "search"
    assert decision.arguments == {"query": "recovery"}
