from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


@dataclass(frozen=True)
class SearchResult:
    document_id: str
    text: str
    score: float


class BM25Retriever:
    def __init__(self, documents: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.documents, self.k1, self.b = documents, k1, b
        self.tokens = {key: tokenize(text) for key, text in documents.items()}
        self.avgdl = sum(map(len, self.tokens.values())) / max(len(self.tokens), 1)
        self.df = Counter(term for terms in self.tokens.values() for term in set(terms))

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        qterms, n = tokenize(query), len(self.documents)
        scores = []
        for doc_id, terms in self.tokens.items():
            counts, length, score = Counter(terms), len(terms), 0.0
            for term in qterms:
                freq = counts[term]
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                score += idf * freq * (self.k1 + 1) / (freq + self.k1 * (1 - self.b + self.b * length / max(self.avgdl, 1)))
            scores.append(SearchResult(doc_id, self.documents[doc_id], score))
        return sorted(scores, key=lambda x: (-x.score, x.document_id))[:top_k]


class DenseRetriever:
    """Offline hashing-vector baseline; replace encoder with SentenceTransformer in full mode."""

    def __init__(self, documents: dict[str, str], dimensions: int = 256):
        self.documents, self.dimensions = documents, dimensions
        self.vectors = {key: self.embed(text) for key, text in documents.items()}

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            vector[value % self.dimensions] += 1 if value & 1 else -1
        norm = math.sqrt(sum(v * v for v in vector)) or 1
        return [v / norm for v in vector]

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        q = self.embed(query)
        scored = [SearchResult(key, text, sum(a * b for a, b in zip(q, self.vectors[key]))) for key, text in self.documents.items()]
        return sorted(scored, key=lambda x: (-x.score, x.document_id))[:top_k]


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60):
        self.bm25, self.dense, self.rrf_k = bm25, dense, rrf_k

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        depth = max(top_k * 3, 10)
        sparse, dense = self.bm25.search(query, depth), self.dense.search(query, depth)
        scores: dict[str, float] = Counter()
        texts = {}
        for ranking in (sparse, dense):
            for rank, result in enumerate(ranking, 1):
                scores[result.document_id] += 1 / (self.rrf_k + rank)
                texts[result.document_id] = result.text
        return [SearchResult(key, texts[key], score) for key, score in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]]

