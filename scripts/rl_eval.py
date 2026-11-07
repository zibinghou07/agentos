"""Evaluate a chat model in the search environment over several seeds.

Examples:
  # pre-RL policy served by vLLM on the GPU machine
  .venv/bin/python scripts/rl_eval.py --split test --base-url http://127.0.0.1:8000/v1 \
      --model qwen2.5-1.5b-instruct --api-key EMPTY --seeds 0 1 2 --tag pre_rl

  # prompt-only large model on Bailian (key from .env)
  .venv/bin/python scripts/rl_eval.py --split test --model qwen3.8-max --seeds 0 1 2 --tag large_prompt_only

Outputs results/rl_eval/<tag>/<split>/{seed<k>.jsonl, summary.json}.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from agentos.config import get_settings
from agentos.llm.openai_compatible import OpenAICompatibleProvider
from agentos.rl.data import read_jsonl
from agentos.rl.env import SearchEnv, dumps
from agentos.rl.evaluate import aggregate, run_split, summarize
from agentos.rl.reward import RewardConfig
from agentos.rl.tools import Corpus, EmbeddingStore, HashingEncoder, SentenceTransformerEncoder


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/rl"))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, help="evaluate a fixed random subset (seeded)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint; default: the .env provider")
    parser.add_argument("--api-key", help="default: key from .env")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--reward", default="v11", help="RewardConfig preset (EM/F1 do not depend on it)")
    parser.add_argument("--closed-book", action="store_true")
    parser.add_argument("--dense", default="hashing", help="'hashing' (offline) or a sentence-transformers model")
    parser.add_argument("--enable-thinking", choices=("true", "false"), help="send enable_thinking (Qwen3 APIs only)")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", type=Path, default=Path("results/rl_eval"))
    args = parser.parse_args()

    settings = get_settings()
    llm = OpenAICompatibleProvider(
        api_key=args.api_key or settings.api_key,
        base_url=(args.base_url or settings.resolved_base_url()).rstrip("/"),
        model=args.model,
        timeout=120,
        max_concurrency=args.concurrency,
        enable_thinking=None if args.enable_thinking is None else args.enable_thinking == "true",
    )
    records = {r["qid"]: r for r in read_jsonl(args.data / f"{args.split}.jsonl")}
    qids = sorted(records)
    if args.limit:
        qids = sorted(random.Random(f"{args.split}|limit").sample(qids, args.limit))
    corpus = Corpus.load(args.data / "corpus.jsonl")
    if args.dense == "hashing":
        embeddings = EmbeddingStore(HashingEncoder())
    else:
        embeddings = EmbeddingStore.load(SentenceTransformerEncoder(args.dense), args.data)
    config = RewardConfig.preset(args.reward)

    def make_env() -> SearchEnv:
        return SearchEnv(records, corpus, embeddings, config, closed_book=args.closed_book)

    out = args.out / args.tag / args.split
    out.mkdir(parents=True, exist_ok=True)
    per_seed = []
    for seed in args.seeds:
        results = await run_split(make_env, llm, qids, seed, args.temperature, args.max_tokens, args.concurrency)
        with (out / f"seed{seed}.jsonl").open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(dumps(result.trajectory) + "\n")
        summary = summarize(results)
        per_seed.append(summary)
        print(f"seed={seed} EM={summary['em']:.3f} ({summary['em_count']}) calls={summary['avg_tool_calls']:.2f} "
              f"valid_calls={summary['valid_call_rate']:.3f} no_answer={summary['no_answer_rate']:.3f}")
    report = {
        "config": {**{k: str(v) for k, v in vars(args).items() if k not in {"api_key"}}, "dense_encoder": embeddings.encoder.name},
        "per_seed": per_seed,
        "aggregate": aggregate(per_seed),
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
