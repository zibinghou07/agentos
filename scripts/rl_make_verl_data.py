"""Convert data/rl splits into verl parquet files (data/rl/verl/<split>.parquet).

Each row carries the chat prompt (system + question), gold answers for the reward,
and tools_kwargs so every tool call searches that question's own pool.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agentos.rl.data import read_jsonl
from agentos.rl.env import initial_messages
from agentos.rl.tools import TOOL_NAMES, Corpus

SPLITS = ("train", "dev", "test", "ood_2wiki", "ood_musique")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/rl"))
    args = parser.parse_args()
    corpus = Corpus.load(args.data / "corpus.jsonl")
    out = args.data / "verl"
    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        rows = []
        for index, record in enumerate(read_jsonl(args.data / f"{split}.jsonl")):
            support = set(record["support_titles"])
            rows.append(
                {
                    "data_source": record["source"],
                    "prompt": initial_messages(record["question"]),
                    "ability": "search_qa",
                    "reward_model": {"style": "rule", "ground_truth": record["answers"]},
                    "agent_name": "tool_agent",
                    "extra_info": {
                        "index": index,
                        "split": split,
                        "qid": record["qid"],
                        "qtype": record["qtype"],
                        "support_doc_ids": [d for d in record["own_doc_ids"] if corpus[d].title in support],
                        "need_tools_kwargs": True,
                        "tools_kwargs": {name: {"create_kwargs": {"qid": record["qid"]}} for name in TOOL_NAMES},
                    },
                }
            )
        pd.DataFrame(rows).to_parquet(out / f"{split}.parquet")
        print(f"{split}: {len(rows)} rows -> {out / f'{split}.parquet'}")


if __name__ == "__main__":
    main()
