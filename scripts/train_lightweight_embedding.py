"""A real CPU-local metric-learning experiment, explicitly not a BGE substitute."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "data/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIM = 128
TRAIN = [
    ("resume interrupted task", "checkpoint restart preserves completed task state", "schema repair changes malformed arguments"),
    ("repair malformed input", "schema repair changes malformed arguments", "checkpoint restart preserves completed task state"),
    ("combine sparse semantic retrieval", "hybrid search fuses bm25 dense rankings with rrf", "mcp discovers remote tools"),
    ("discover protocol tools", "mcp discovers remote tools through schemas", "hybrid search fuses rankings"),
]
TEST_DOCS = {
    "checkpoint": "checkpoint restart preserves completed task state",
    "repair": "schema repair changes malformed arguments",
    "hybrid": "hybrid search fuses bm25 dense rankings with rrf",
    "mcp": "mcp discovers remote tools through schemas",
}
TEST = {
    "resume state after interruption": "checkpoint",
    "fix malformed schema input": "repair",
    "sparse dense rank fusion": "hybrid",
    "protocol schema tool discovery": "mcp",
}


def vector(text: str) -> np.ndarray:
    output = np.zeros(DIM)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        # Stable FNV-1a rather than Python's randomized hash.
        value = 2166136261
        for byte in token.encode():
            value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
        output[value % DIM] += 1
    norm = np.linalg.norm(output)
    return output / norm if norm else output


def score(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    a, b = left * weights, right * weights
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def evaluate(weights: np.ndarray) -> dict[str, float]:
    ranks = []
    for query, relevant in TEST.items():
        ordered = sorted(TEST_DOCS, key=lambda key: score(vector(query), vector(TEST_DOCS[key]), weights), reverse=True)
        ranks.append(ordered.index(relevant) + 1)
    return {"recall@1": float(np.mean([rank == 1 for rank in ranks])), "mrr@4": float(np.mean([1 / rank for rank in ranks]))}


def main() -> None:
    weights = np.ones(DIM)
    before = evaluate(weights)
    losses = []
    learning_rate, margin = 0.05, 0.5
    for _ in range(120):
        epoch_loss = 0.0
        for query, positive, negative in TRAIN:
            q, p, n = vector(query), vector(positive), vector(negative)
            qw, pw, nw = q * weights, p * weights, n * weights
            positive_score, negative_score = qw @ pw, qw @ nw
            loss = max(0.0, margin - positive_score + negative_score)
            if loss:
                gradient = 2 * weights * (q * n - q * p)
                weights -= learning_rate * gradient
                weights = np.clip(weights, 0.1, 3.0)
            epoch_loss += loss
        losses.append(epoch_loss / len(TRAIN))
    after = evaluate(weights)
    directory = Path("experiments/results/embedding_lightweight")
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "weights.npz", weights=weights)
    pd.DataFrame([{"model": "hashing_before", **before}, {"model": "metric_learned", **after}]).to_csv(directory / "metrics.csv", index=False)
    (directory / "train_log.json").write_text(json.dumps({"epochs": 120, "seed": 42, "train_pairs": len(TRAIN), "test_queries": len(TEST), "final_loss": losses[-1], "synthetic": True}, indent=2), encoding="utf-8")
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Triplet loss")
    plt.tight_layout()
    plt.savefig(directory / "training_curve.png", dpi=160)
    print(json.dumps({"before": before, "after": after, "final_loss": losses[-1]}, indent=2))


if __name__ == "__main__":
    main()
