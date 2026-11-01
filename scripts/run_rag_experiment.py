from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from agentos.mcp.catalog import DOCUMENTS
from agentos.rag import BM25Retriever, DenseRetriever, HybridRetriever
from agentos.rag.evaluator import evaluate_rankings

QUERIES = {
    "q1": ("DAG concurrent scheduling", {"doc_agent"}),
    "q2": ("Retry Repair Replan", {"doc_recovery"}),
    "q3": ("reciprocal rank fusion hybrid", {"doc_rag"}),
    "q4": ("protocol tool discovery host", {"doc_mcp"}),
}

retrievers = {
    "bm25": BM25Retriever(DOCUMENTS),
    "dense_hashing": DenseRetriever(DOCUMENTS),
}
retrievers["hybrid_rrf"] = HybridRetriever(retrievers["bm25"], retrievers["dense_hashing"])
rows, top_k = [], {}
for method, retriever in retrievers.items():
    rankings, latencies = {}, []
    for query_id, (query, _) in QUERIES.items():
        started = time.perf_counter()
        results = retriever.search(query, 4)
        latencies.append((time.perf_counter() - started) * 1000)
        rankings[query_id] = [result.document_id for result in results]
        top_k[f"{method}:{query_id}"] = [result.__dict__ for result in results]
    metrics = evaluate_rankings(rankings, {qid: qrel for qid, (_, qrel) in QUERIES.items()})
    metrics.update({"method": method, "latency_ms": sum(latencies) / len(latencies)})
    rows.append(metrics)
directory = Path("experiments/results/rag")
directory.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
(directory / "top_k.json").write_text(json.dumps(top_k, indent=2), encoding="utf-8")
print(pd.DataFrame(rows).to_string(index=False))

