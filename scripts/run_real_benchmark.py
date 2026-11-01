from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from agentos.benchmarks.real_data import read_tasks
from agentos.benchmarks.real_runner import evaluate_real_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline baselines on open-source tasks")
    parser.add_argument(
        "--input", type=Path, default=Path("benchmarks/real_data/test.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/results/real_benchmark")
    )
    args = parser.parse_args()

    records, summary = evaluate_real_dataset(read_tasks(args.input))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **record,
                    "prediction": json.dumps(record["prediction"], ensure_ascii=False),
                    "expected": json.dumps(record["expected"], ensure_ascii=False),
                }
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
