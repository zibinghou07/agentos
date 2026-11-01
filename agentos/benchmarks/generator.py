from __future__ import annotations

import json
from pathlib import Path

from agentos.benchmarks.schema import BenchmarkTask

COUNTS = {"single_tool": 30, "multi_tool": 30, "multi_hop": 30, "dependency_dag": 30, "fault_recovery": 30, "long_horizon": 20}


def generate_tasks() -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    global_index = 0
    for family, count in COUNTS.items():
        for index in range(count):
            global_index += 1
            split = "test" if index >= count - (9 if family != "long_horizon" else 5) else "dev"
            if family == "single_tool":
                prompt = "Look up the exact product p1" if split == "dev" else "Retrieve the product record p1"
                tools, answer, steps = ["query_product"], "AgentOS Pro", 1
            elif family == "multi_tool":
                prompt = "Look up the exact product p1 then search local documents about AgentOS architecture" if split == "dev" else "Retrieve product p1, then search documents for AgentOS architecture"
                tools, answer, steps = ["query_product", "search_documents"], "AgentOS Pro", 2
            elif family == "multi_hop":
                prompt = "Run BM25 keyword retrieval then hybrid retrieval to find reciprocal rank fusion" if split == "dev" else "First use BM25 search; then use hybrid search for reciprocal rank fusion"
                tools, answer, steps = ["bm25_search", "hybrid_search"], "RRF", 2
            elif family == "dependency_dag":
                prompt = "In parallel look up exact product p1, run BM25 keyword retrieval, then hybrid retrieval" if split == "dev" else "Retrieve product p1 and BM25 evidence concurrently; then perform hybrid retrieval"
                tools, answer, steps = ["query_product", "bm25_search", "hybrid_search"], "SUCCESS", 4
            elif family == "fault_recovery":
                prompt = "Look up exact product p1; an injected invalid argument must be repaired" if split == "dev" else "Repair the malformed parameter and retrieve product p1"
                tools, answer, steps = ["query_product"], "AgentOS Pro", 2
            else:
                prompt = "Run BM25 keyword retrieval, dense semantic retrieval, and hybrid retrieval before analysis and verification" if split == "dev" else "Use BM25 search; then dense semantic search; then hybrid search before verification"
                tools, answer, steps = ["bm25_search", "dense_search", "hybrid_search"], "SUCCESS", 8
            tasks.append(BenchmarkTask(task_id=f"bench_{global_index:03d}", family=family, split=split, prompt=f"{prompt} #{index}", expected_tools=tools, expected_answer=answer, steps=steps, fault="INVALID_ARGUMENT" if family == "fault_recovery" else None, template_id=f"{family}_{split}_v1"))
    return tasks


def write_dataset(directory: str | Path) -> dict[str, int]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tasks = generate_tasks()
    counts = {}
    for split in ("dev", "test"):
        selected = [task for task in tasks if task.split == split]
        with (directory / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for task in selected:
                handle.write(task.model_dump_json() + "\n")
        counts[split] = len(selected)
    with (directory / "all.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(task.model_dump_json() + "\n")
    # There is intentionally no supervised train split in the synthetic system
    # benchmark; dev is for tuning and test is locked evaluation.
    (directory / "train.jsonl").write_text("", encoding="utf-8")
    (directory / "task_schema.json").write_text(
        json.dumps(BenchmarkTask.model_json_schema(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (directory / "METADATA.json").write_text(
        json.dumps(
            {
                "name": "AgentOS-Bench Synthetic v1",
                "source": "generated locally by agentos.benchmarks.generator",
                "license": "same as this repository",
                "synthetic": True,
                "counts": {"dev": counts["dev"], "test": counts["test"], "total": len(tasks)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return counts


def load_dataset(path: str | Path) -> list[BenchmarkTask]:
    with Path(path).open(encoding="utf-8") as handle:
        return [BenchmarkTask.model_validate_json(line) for line in handle if line.strip()]

