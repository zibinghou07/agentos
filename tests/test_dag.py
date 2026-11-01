import pytest

from agentos.models import TaskNode, TaskStatus
from agentos.runtime.dag import CycleError, ready_nodes, topological_sort


def test_topological_sort_multilayer():
    nodes = [TaskNode(task_id="a", description="a", agent_type="r"), TaskNode(task_id="b", description="b", agent_type="r"), TaskNode(task_id="c", description="c", agent_type="a", dependencies=["a", "b"])]
    order = topological_sort(nodes)
    assert order.index("a") < order.index("c") and order.index("b") < order.index("c")


def test_cycle_rejected():
    nodes = [TaskNode(task_id="a", description="a", agent_type="r", dependencies=["b"]), TaskNode(task_id="b", description="b", agent_type="r", dependencies=["a"])]
    with pytest.raises(CycleError):
        topological_sort(nodes)


def test_failed_branch_does_not_block_independent_branch():
    failed = TaskNode(task_id="a", description="a", agent_type="r", status=TaskStatus.FAILED)
    dependent = TaskNode(task_id="b", description="b", agent_type="r", dependencies=["a"])
    independent = TaskNode(task_id="c", description="c", agent_type="r")
    assert [node.task_id for node in ready_nodes([failed, dependent, independent])] == ["c"]
    assert dependent.status == TaskStatus.SKIPPED

