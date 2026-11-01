from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

from agentos.benchmarks.generator import load_dataset, write_dataset
from agentos.benchmarks.metrics import bootstrap_ci, summarize
from agentos.benchmarks.runner import evaluate_dataset


async def main(repeats: int) -> None:
    write_dataset("benchmarks/data")
    tasks = load_dataset("benchmarks/data/test.jsonl")
    records = await evaluate_dataset(tasks, repeats=repeats)
    directory = Path("experiments/results/benchmark")
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([record.model_dump() for record in records])
    frame.to_csv(directory / "runs.csv", index=False)
    report = {}
    for method, group in frame.groupby("method"):
        rows = [record for record in records if record.method == method]
        metrics = summarize(rows)
        task_level_success = group.groupby("task_id")["success"].mean().tolist()
        metrics["unique_tasks"] = group["task_id"].nunique()
        metrics["success_ci95"] = bootstrap_ci(task_level_success)
        report[method] = metrics
    (directory / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    asyncio.run(main(parser.parse_args().repeats))

