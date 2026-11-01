from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    dataset: Literal["bfcl_v4_live_multiple", "hotpotqa_distractor"]
    source_id: str
    source_partition: str
    source_url: str
    license: str
    raw_sha256: str
    ground_truth_url: str | None = None
    ground_truth_sha256: str | None = None


class OpenSourceTask(BaseModel):
    task_id: str
    split: Literal["dev", "test"]
    task_type: Literal["tool_calling", "multi_hop_qa"]
    prompt: str
    tools: list[dict[str, Any]] = Field(default_factory=list)
    context: list[dict[str, Any]] = Field(default_factory=list)
    ground_truth: dict[str, Any]
    source: SourceInfo
    synthetic: Literal[False] = False
    conversion_version: str = "1.0"


def stable_order(value: str) -> str:
    """Deterministic sampling order independent of upstream row order."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def flatten_messages(turns: Any) -> str:
    """Flatten BFCL's nested conversation representation without changing text."""
    messages: list[dict[str, Any]] = []
    for turn in turns if isinstance(turns, list) else []:
        if isinstance(turn, list):
            messages.extend(item for item in turn if isinstance(item, dict))
        elif isinstance(turn, dict):
            messages.append(turn)
    return "\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "user" and message.get("content")
    )


def expected_function_names(ground_truth: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(ground_truth, list):
        return names
    for call in ground_truth:
        if isinstance(call, dict):
            names.extend(str(name) for name in call)
    return names


def convert_bfcl_rows(
    prompts: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    *,
    count: int,
    dev_count: int,
    source_url: str,
    raw_sha256: str,
    ground_truth_url: str | None = None,
    ground_truth_sha256: str | None = None,
) -> list[OpenSourceTask]:
    answer_by_id = {str(row["id"]): row["ground_truth"] for row in answers}
    joined = [row for row in prompts if str(row.get("id")) in answer_by_id]
    selected = sorted(joined, key=lambda row: stable_order(str(row["id"])))[:count]
    if len(selected) != count:
        raise ValueError(f"BFCL requested {count} rows, only {len(selected)} matched answers")

    tasks = []
    for index, row in enumerate(selected):
        source_id = str(row["id"])
        ground_truth = answer_by_id[source_id]
        tasks.append(
            OpenSourceTask(
                task_id=f"real-bfcl-{source_id}",
                split="dev" if index < dev_count else "test",
                task_type="tool_calling",
                prompt=flatten_messages(row["question"]),
                tools=row.get("function", []),
                ground_truth={
                    "calls": ground_truth,
                    "function_names": expected_function_names(ground_truth),
                },
                source=SourceInfo(
                    dataset="bfcl_v4_live_multiple",
                    source_id=source_id,
                    source_partition="live_multiple",
                    source_url=source_url,
                    license="Apache-2.0",
                    raw_sha256=raw_sha256,
                    ground_truth_url=ground_truth_url,
                    ground_truth_sha256=ground_truth_sha256,
                ),
            )
        )
    return tasks


def convert_hotpot_rows(
    rows: list[dict[str, Any]],
    *,
    count: int,
    dev_count: int,
    source_url: str,
    raw_sha256: str,
) -> list[OpenSourceTask]:
    selected = sorted(rows, key=lambda row: stable_order(str(row["_id"])))[:count]
    if len(selected) != count:
        raise ValueError(f"HotpotQA requested {count} rows, only {len(selected)} available")

    tasks = []
    for index, row in enumerate(selected):
        source_id = str(row["_id"])
        supporting = [
            {"title": str(title), "sentence_id": int(sentence_id)}
            for title, sentence_id in row["supporting_facts"]
        ]
        context = [
            {"title": str(title), "sentences": [str(sentence) for sentence in sentences]}
            for title, sentences in row["context"]
        ]
        tasks.append(
            OpenSourceTask(
                task_id=f"real-hotpot-{source_id}",
                split="dev" if index < dev_count else "test",
                task_type="multi_hop_qa",
                prompt=str(row["question"]),
                context=context,
                ground_truth={
                    "answer": str(row["answer"]),
                    "supporting_facts": supporting,
                    "level": row.get("level"),
                    "question_type": row.get("type"),
                },
                source=SourceInfo(
                    dataset="hotpotqa_distractor",
                    source_id=source_id,
                    source_partition="dev_distractor_v1",
                    source_url=source_url,
                    license="CC-BY-SA-4.0",
                    raw_sha256=raw_sha256,
                ),
            )
        )
    return tasks


def read_tasks(path: Path) -> list[OpenSourceTask]:
    with path.open(encoding="utf-8") as handle:
        return [OpenSourceTask.model_validate_json(line) for line in handle if line.strip()]


def write_tasks(path: Path, tasks: list[OpenSourceTask]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.model_dump(), ensure_ascii=False) + "\n")
