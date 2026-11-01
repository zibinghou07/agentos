import asyncio

from agentos.models import TaskNode, TaskPlan, TaskStatus
from agentos.runtime.checkpoint import CheckpointStore
from agentos.runtime.scheduler import DAGScheduler


async def test_parallel_scheduler_and_checkpoint(tmp_path):
    active = maximum = 0

    async def execute(node, context):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.02)
        active -= 1
        return {"id": node.task_id}

    plan = TaskPlan(goal="parallel", subtasks=[TaskNode(task_id="a", description="a", agent_type="r"), TaskNode(task_id="b", description="b", agent_type="r"), TaskNode(task_id="c", description="c", agent_type="v", dependencies=["a", "b"])])
    store = CheckpointStore(tmp_path / "state.db")
    result = await DAGScheduler(execute, store, 2).run(plan)
    assert maximum == 2
    assert all(node.status == TaskStatus.SUCCESS for node in result.subtasks)
    assert store.latest(plan.task_id) is not None


async def test_resume_skips_completed_operation(tmp_path):
    calls = []

    async def execute(node, context):
        calls.append(node.task_id)
        return {"ok": node.task_id}

    plan = TaskPlan(goal="resume", subtasks=[TaskNode(task_id="a", description="a", agent_type="r"), TaskNode(task_id="b", description="b", agent_type="r", dependencies=["a"])])
    store = CheckpointStore(tmp_path / "resume.db")
    await DAGScheduler(execute, store).run(plan)
    snapshot = store.latest(plan.task_id)
    assert snapshot
    await DAGScheduler(execute, store).run(snapshot, resume=True)
    assert calls == ["a", "b"]

