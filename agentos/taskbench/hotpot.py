"""Turn HotpotQA distractor questions into full agentic tasks.

The ten paragraphs of each question become the task's private document corpus,
so an agent has to search, read and combine evidence through MCP tools instead
of receiving the passages in its prompt. Data is real (``synthetic: false``).
"""

from __future__ import annotations

import re
from pathlib import Path

from agentos.benchmarks.real_data import OpenSourceTask
from agentos.taskbench.schema import TaskSpec, TextCheck


def _doc_id(title: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:48] or "doc"
    candidate, index = base, 2
    while candidate in used:
        candidate, index = f"{base}_{index}", index + 1
    used.add(candidate)
    return candidate


def hotpot_tasks(path: str | Path, split: str) -> list[TaskSpec]:
    tasks = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = OpenSourceTask.model_validate_json(line)
            if item.task_type != "multi_hop_qa" or item.split != split:
                continue
            used: set[str] = set()
            documents = {
                _doc_id(paragraph["title"], used): f"{paragraph['title']}. " + " ".join(paragraph["sentences"])
                for paragraph in item.context
            }
            tasks.append(
                TaskSpec(
                    task_id=f"hotpot-{item.source.source_id}",
                    family="hotpotqa_agentic",
                    split=split,
                    goal=(
                        f"{item.prompt} Use the document search tools to find the evidence, "
                        "then answer concisely."
                    ),
                    checks=[TextCheck(label="answer", aliases=[str(item.ground_truth["answer"])])],
                    documents=documents,
                    synthetic=False,
                    source={
                        **item.source.model_dump(),
                        "question_type": item.ground_truth.get("question_type"),
                        "level": item.ground_truth.get("level"),
                        "gold_answer": item.ground_truth["answer"],
                    },
                    reference_steps=2,
                )
            )
    return tasks
