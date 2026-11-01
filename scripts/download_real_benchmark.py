from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import httpx

from agentos.benchmarks.real_data import (
    OpenSourceTask,
    convert_bfcl_rows,
    convert_hotpot_rows,
    write_tasks,
)

BFCL_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_BASE = (
    f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{BFCL_REVISION}/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)
BFCL_PROMPTS = f"{BFCL_BASE}/BFCL_v4_live_multiple.json"
BFCL_ANSWERS = f"{BFCL_BASE}/possible_answer/BFCL_v4_live_multiple.json"
HOTPOT_DEV = "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
HOTPOT_MIRROR = (
    "https://datasets-server.huggingface.co/rows?dataset=hotpotqa%2Fhotpot_qa"
    "&config=distractor&split=validation&offset=0&length=100"
)


def download(url: str, target: Path) -> str:
    digest = hashlib.sha256()
    with httpx.stream("GET", url, follow_redirects=True, timeout=180.0) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
                digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_hotpot_payload(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError("Unrecognized HotpotQA payload")
    normalized = []
    for wrapper in payload["rows"]:
        row = wrapper["row"]
        facts = row["supporting_facts"]
        context = row["context"]
        normalized.append(
            {
                "_id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "type": row.get("type"),
                "level": row.get("level"),
                "supporting_facts": list(zip(facts["title"], facts["sent_id"])),
                "context": list(zip(context["title"], context["sentences"])),
            }
        )
    return normalized


def download_hotpot(target: Path, count: int) -> tuple[str, str]:
    try:
        return download(HOTPOT_DEV, target), HOTPOT_DEV
    except httpx.HTTPError as exc:
        if count > 100:
            raise RuntimeError(
                "Official HotpotQA download failed and mirror fallback supports at most 100 rows"
            ) from exc
        print(f"official HotpotQA download failed ({exc}); using Hugging Face mirror")
        return download(HOTPOT_MIRROR, target), HOTPOT_MIRROR


def validate(tasks: list[OpenSourceTask], bfcl_count: int, hotpot_count: int) -> None:
    expected = bfcl_count + hotpot_count
    if len(tasks) != expected or len({task.task_id for task in tasks}) != expected:
        raise ValueError("Converted task count or task IDs are invalid")
    if any(task.synthetic for task in tasks):
        raise ValueError("Real benchmark contains a task marked synthetic")
    for task in tasks:
        if not task.prompt or not task.source.source_id or not task.source.raw_sha256:
            raise ValueError(f"Incomplete provenance for {task.task_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert 200 open-source tasks")
    parser.add_argument("--bfcl-count", type=int, default=100)
    parser.add_argument("--hotpot-count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/real_data"))
    args = parser.parse_args()
    if args.bfcl_count < 4 or args.hotpot_count < 4:
        parser.error("each source needs at least four tasks")

    output = args.output
    with tempfile.TemporaryDirectory(prefix="agentos-real-data-") as temporary:
        work = Path(temporary)
        bfcl_prompts_path = work / "bfcl_prompts.jsonl"
        bfcl_answers_path = work / "bfcl_answers.jsonl"
        hotpot_path = work / "hotpot.json"
        bfcl_prompt_sha = download(BFCL_PROMPTS, bfcl_prompts_path)
        bfcl_answer_sha = download(BFCL_ANSWERS, bfcl_answers_path)
        hotpot_sha, hotpot_url = download_hotpot(hotpot_path, args.hotpot_count)

        bfcl_dev = round(args.bfcl_count * 0.75)
        hotpot_dev = round(args.hotpot_count * 0.75)
        bfcl = convert_bfcl_rows(
            read_jsonl(bfcl_prompts_path),
            read_jsonl(bfcl_answers_path),
            count=args.bfcl_count,
            dev_count=bfcl_dev,
            source_url=BFCL_PROMPTS,
            raw_sha256=bfcl_prompt_sha,
            ground_truth_url=BFCL_ANSWERS,
            ground_truth_sha256=bfcl_answer_sha,
        )
        hotpot = convert_hotpot_rows(
            normalize_hotpot_payload(json.loads(hotpot_path.read_text(encoding="utf-8"))),
            count=args.hotpot_count,
            dev_count=hotpot_dev,
            source_url=hotpot_url,
            raw_sha256=hotpot_sha,
        )

    tasks = bfcl + hotpot
    validate(tasks, args.bfcl_count, args.hotpot_count)
    write_tasks(output / "all.jsonl", tasks)
    write_tasks(output / "dev.jsonl", [task for task in tasks if task.split == "dev"])
    write_tasks(output / "test.jsonl", [task for task in tasks if task.split == "test"])
    (output / "task_schema.json").write_text(
        json.dumps(OpenSourceTask.model_json_schema(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "name": "AgentOS-Real-200",
        "generated_at": datetime.now(UTC).isoformat(),
        "count": len(tasks),
        "splits": {
            "dev": bfcl_dev + hotpot_dev,
            "test": len(tasks) - bfcl_dev - hotpot_dev,
        },
        "sampling": "ascending SHA-256(source_id); first N; first 75% local dev",
        "sources": {
            "bfcl_v4_live_multiple": {
                "count": len(bfcl),
                "revision": BFCL_REVISION,
                "prompts_url": BFCL_PROMPTS,
                "answers_url": BFCL_ANSWERS,
                "prompts_sha256": bfcl_prompt_sha,
                "answers_sha256": bfcl_answer_sha,
                "license": "Apache-2.0",
            },
            "hotpotqa_distractor": {
                "count": len(hotpot),
                "url": hotpot_url,
                "canonical_url": HOTPOT_DEV,
                "sha256": hotpot_sha,
                "license": "CC-BY-SA-4.0",
            },
        },
        "limitations": [
            "Local dev/test are deterministic subsets of public evaluation/development data.",
            "Do not tune on test.jsonl.",
            "BFCL functions are evaluated as schemas; external APIs are not executed.",
        ],
    }
    (output / "METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    test_count = len(tasks) - bfcl_dev - hotpot_dev
    print(
        f"wrote {len(tasks)} real tasks to {output} "
        f"({bfcl_dev + hotpot_dev} dev, {test_count} test)"
    )


if __name__ == "__main__":
    main()
