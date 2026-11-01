from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

from agentos.models import TaskNode, TaskStatus


class CycleError(ValueError):
    pass


def topological_sort(nodes: Iterable[TaskNode]) -> list[str]:
    node_list = list(nodes)
    ids = {n.task_id for n in node_list}
    indegree = {n.task_id: 0 for n in node_list}
    children: dict[str, list[str]] = defaultdict(list)
    for node in node_list:
        for dep in node.dependencies:
            if dep not in ids:
                raise ValueError(f"unknown dependency: {dep}")
            children[dep].append(node.task_id)
            indegree[node.task_id] += 1
    queue = deque(sorted(k for k, v in indegree.items() if v == 0))
    ordered: list[str] = []
    while queue:
        current = queue.popleft()
        ordered.append(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(ordered) != len(node_list):
        raise CycleError("task graph contains a cycle")
    return ordered


def ready_nodes(nodes: Iterable[TaskNode]) -> list[TaskNode]:
    node_list = list(nodes)
    by_id = {node.task_id: node for node in node_list}
    ready: list[TaskNode] = []
    for node in node_list:
        if node.status not in {TaskStatus.PENDING, TaskStatus.READY}:
            continue
        dependencies = [by_id[dep].status for dep in node.dependencies]
        if any(status in {TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.CANCELLED} for status in dependencies):
            node.status = TaskStatus.SKIPPED
        elif all(status == TaskStatus.SUCCESS for status in dependencies):
            node.status = TaskStatus.READY
            ready.append(node)
    return ready



def plan_shape(nodes: Iterable[TaskNode]) -> dict[str, float]:
    """Structural metrics: depth is the critical-path length, width the largest level."""
    node_list = list(nodes)
    if not node_list:
        return {"nodes": 0, "edges": 0, "depth": 0, "max_width": 0, "parallelism": 0.0}
    by_id = {node.task_id: node for node in node_list}
    level: dict[str, int] = {}
    for task_id in topological_sort(node_list):
        deps = by_id[task_id].dependencies
        level[task_id] = 1 + max((level[dep] for dep in deps), default=0)
    depth = max(level.values())
    widths = defaultdict(int)
    for value in level.values():
        widths[value] += 1
    return {
        "nodes": len(node_list),
        "edges": sum(len(node.dependencies) for node in node_list),
        "depth": depth,
        "max_width": max(widths.values()),
        # Average nodes runnable per level; 1.0 means a purely sequential chain.
        "parallelism": round(len(node_list) / depth, 3),
    }
