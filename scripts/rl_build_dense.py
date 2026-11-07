"""Precompute dense vectors for every corpus paragraph (run on the GPU machine).

  python scripts/rl_build_dense.py --model BAAI/bge-small-en-v1.5
Writes data/rl/dense.npy, dense_ids.txt and dense_meta.json, which
EmbeddingStore.load checks against the query encoder's model name.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from agentos.rl.tools import Corpus, SentenceTransformerEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/rl"))
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    corpus = Corpus.load(args.data / "corpus.jsonl")
    ids = sorted(corpus.documents)
    encoder = SentenceTransformerEncoder(args.model, device=args.device)
    started = time.perf_counter()
    chunks = []
    for start in range(0, len(ids), args.batch_size * 16):
        batch = [corpus[d] for d in ids[start : start + args.batch_size * 16]]
        chunks.append(encoder.encode([f"{doc.title}. {doc.text}" for doc in batch]))
        print(f"{min(start + len(batch), len(ids))}/{len(ids)}", flush=True)
    matrix = np.concatenate(chunks).astype(np.float16)
    np.save(args.data / "dense.npy", matrix)
    (args.data / "dense_ids.txt").write_text("\n".join(ids) + "\n")
    meta = {"model": args.model, "count": len(ids), "dim": int(matrix.shape[1]), "seconds": round(time.perf_counter() - started, 1)}
    (args.data / "dense_meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
