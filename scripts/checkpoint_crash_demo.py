from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from agentos.models import TaskNode, TaskPlan
from agentos.runtime.checkpoint import CheckpointStore
from agentos.runtime.scheduler import DAGScheduler


async def main(phase: str, database: Path, log: Path) -> None:
    store = CheckpointStore(database)
    id_path = database.with_suffix(".task_id")
    if phase == "crash":
        plan = TaskPlan(goal="real process crash recovery", subtasks=[
            TaskNode(task_id="one", description="one", agent_type="worker"),
            TaskNode(task_id="two", description="two", agent_type="worker", dependencies=["one"]),
            TaskNode(task_id="three", description="three", agent_type="worker", dependencies=["two"]),
        ])
        id_path.write_text(plan.task_id, encoding="utf-8")
    else:
        task_id = id_path.read_text(encoding="utf-8")
        plan = store.latest(task_id)
        if plan is None:
            raise RuntimeError("checkpoint missing")

    async def execute(node, context):
        with log.open("a", encoding="utf-8") as handle:
            handle.write(node.task_id + "\n")
            handle.flush()
        if phase == "crash" and node.task_id == "three":
            os._exit(73)
        return {"node": node.task_id}

    result = await DAGScheduler(execute, store, 1).run(plan, resume=phase == "resume")
    print(json.dumps({"status": [node.status for node in result.subtasks], "log": log.read_text().splitlines()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["crash", "resume"])
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.phase, args.database, args.log))
