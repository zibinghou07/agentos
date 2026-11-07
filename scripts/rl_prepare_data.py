"""Build train/dev/test/OOD splits, retrieval pools and the paragraph corpus.

Download the raw parquet files first (see docs/rl_runbook.md), then:
  .venv/bin/python scripts/rl_prepare_data.py
Outputs go to data/rl/ (git-ignored); the split and leakage report goes to results/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from agentos.rl.data import (
    attach_pools,
    corpus_records,
    leakage_report,
    load_raw,
    normalize_answer,
    split_summary,
    stratified_sample,
    write_jsonl,
)

RAW_FILES = {
    "hotpot_train": ("hotpot_train_0.parquet", "hotpotqa"),
    "hotpot_validation": ("hotpot_validation.parquet", "hotpotqa"),
    "2wiki_validation": ("2wiki_validation.parquet", "2wiki"),
    "musique_validation": ("musique_validation.parquet", "musique"),
}
SOURCES = {
    "hotpotqa": "hotpotqa/hotpot_qa (distractor), CC BY-SA 4.0",
    "2wiki": "framolfese/2WikiMultihopQA (mirror of 2WikiMultihopQA); check upstream license",
    "musique": "dgslibisey/MuSiQue (answerable), CC BY 4.0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/rl"))
    parser.add_argument("--report", type=Path, default=Path("results/rl_data_report.json"))
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--train", type=int, default=8000)
    parser.add_argument("--dev", type=int, default=300)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--ood", type=int, default=300)
    args = parser.parse_args()

    raw = {name: load_raw(args.raw / file, source) for name, (file, source) in RAW_FILES.items()}

    # dev and test come from the HotpotQA validation set and never overlap.
    held_out = stratified_sample(raw["hotpot_validation"], args.dev + args.test, args.seed)
    dev = stratified_sample(held_out, args.dev, args.seed + 1)
    dev_ids = {item.qid for item in dev}
    test = [item for item in held_out if item.qid not in dev_ids]

    # Drop train questions that duplicate an evaluation question word for word.
    eval_questions = {normalize_answer(item.question) for item in held_out}
    train_pool = [
        item for item in raw["hotpot_train"] if normalize_answer(item.question) not in eval_questions
    ]
    train = stratified_sample(train_pool, args.train, args.seed + 2, key=lambda i: f"{i.qtype}/{i.level}")

    splits = {
        "train": train,
        "dev": dev,
        "test": test,
        "ood_2wiki": stratified_sample(raw["2wiki_validation"], args.ood, args.seed + 3),
        "ood_musique": stratified_sample(raw["musique_validation"], args.ood, args.seed + 4),
    }
    for offset, items in enumerate(splits.values()):
        attach_pools(items, args.seed + 10 + offset)

    counts = {name: write_jsonl(args.out / f"{name}.jsonl", (i.to_record() for i in items)) for name, items in splits.items()}
    counts["corpus"] = write_jsonl(args.out / "corpus.jsonl", corpus_records(splits))

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "pool_size": len(train[0].pool),
        "counts": counts,
        "splits": {name: split_summary(items) for name, items in splits.items()},
        "leakage": leakage_report(splits),
        "raw_files": {
            name: {"file": file, "sha256": sha256(args.raw / file), "source": SOURCES[source]}
            for name, (file, source) in RAW_FILES.items()
        },
        "train_candidates_after_dedup": len(train_pool),
    }
    (args.out / "metadata.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"counts": counts, "leakage": report["leakage"]}, indent=2))


if __name__ == "__main__":
    main()
