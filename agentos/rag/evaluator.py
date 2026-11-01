from __future__ import annotations

import math
from statistics import mean


def evaluate_rankings(rankings: dict[str, list[str]], qrels: dict[str, set[str]], k: int = 10) -> dict[str, float]:
    recalls, reciprocal, ndcgs, hits = [], [], [], []
    for query_id, relevant in qrels.items():
        retrieved = rankings.get(query_id, [])[:k]
        found = [1 if doc in relevant else 0 for doc in retrieved]
        recalls.append(sum(found) / len(relevant) if relevant else 0)
        reciprocal.append(next((1 / rank for rank, hit in enumerate(found, 1) if hit), 0))
        dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(found, 1))
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
        ndcgs.append(dcg / ideal if ideal else 0)
        hits.append(float(any(found)))
    return {f"recall@{k}": mean(recalls), f"mrr@{k}": mean(reciprocal), f"ndcg@{k}": mean(ndcgs), f"hit_rate@{k}": mean(hits)}
