from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agentos.models import TaskNode, TaskPlan
from agentos.runtime.checkpoint import CheckpointStore
from agentos.runtime.scheduler import DAGScheduler


async def demo() -> None:
    plan = TaskPlan(
        goal="Research AgentOS and compare recovery strategies",
        subtasks=[
            TaskNode(task_id="research", description="research checkpoint", agent_type="researcher"),
            TaskNode(task_id="retrieve", description="retrieve recovery evidence", agent_type="researcher"),
            TaskNode(task_id="analyze", description="compare evidence", agent_type="analyst", dependencies=["research", "retrieve"]),
            TaskNode(task_id="verify", description="verify output", agent_type="verifier", dependencies=["analyze"]),
        ],
    )

    async def execute(node, context):
        await asyncio.sleep(0.02)
        return {"node": node.task_id, "received": sorted(context)}

    path = Path("data/demo.db")
    result = await DAGScheduler(execute, CheckpointStore(path), 2).run(plan)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["demo"])
    args = parser.parse_args()
    if args.command == "demo":
        asyncio.run(demo())


if __name__ == "__main__":
    main()

