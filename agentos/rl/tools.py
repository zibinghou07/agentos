"""Retrieval tools over one question's paragraph pool.

The same code serves the local environment, the evaluation harness and the verl
tool wrappers, so a tool call returns identical text everywhere.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from jsonschema import Draft202012Validator

from agentos.rag.retrieval import BM25Retriever, tokenize

SNIPPET_CHARS = 200
DOCUMENT_CHARS = 1500
MAX_K = 5

_SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1, "description": "Search query."},
        "k": {"type": "integer", "minimum": 1, "maximum": MAX_K, "description": "Number of results (default 3)."},
    },
    "required": ["query"],
    "additionalProperties": False,
}
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bm25_search",
            "description": "Keyword search over the paragraphs. Best for exact names, titles and rare words. "
            f"Returns doc ids, titles and the first {SNIPPET_CHARS} characters.",
            "parameters": _SEARCH_PARAMETERS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dense_search",
            "description": "Semantic search over the paragraphs. Best for paraphrased or descriptive queries. "
            f"Returns doc ids, titles and the first {SNIPPET_CHARS} characters.",
            "parameters": _SEARCH_PARAMETERS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Read the full text of one paragraph by its doc id (costs more tokens than a search).",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": {"type": "string", "pattern": "^d[0-9a-f]{10}$"}},
                "required": ["doc_id"],
                "additionalProperties": False,
            },
        },
    },
]
TOOL_NAMES = tuple(schema["function"]["name"] for schema in TOOL_SCHEMAS)


def prompt_tool_schemas() -> list[dict[str, Any]]:
    """Tool definitions exactly as verl renders them into the training prompt.

    verl's OpenAIFunctionToolSchema keeps only type/description/enum per parameter and
    type/properties/required per object, dropping minimum, maximum, pattern, minLength
    and additionalProperties. Evaluation must show the model the same text it was
    trained on; full schemas are still enforced when a call is executed.
    """
    rendered = []
    for schema in TOOL_SCHEMAS:
        function, parameters = schema["function"], schema["function"]["parameters"]
        properties = {
            name: {key: spec[key] for key in ("type", "description", "enum") if key in spec}
            for name, spec in parameters["properties"].items()
        }
        rendered.append(
            {
                "type": schema["type"],
                "function": {
                    "name": function["name"],
                    "description": function["description"],
                    "parameters": {"type": parameters["type"], "properties": properties, "required": parameters.get("required", [])},
                },
            }
        )
    return rendered
_VALIDATORS = {
    schema["function"]["name"]: Draft202012Validator(schema["function"]["parameters"]) for schema in TOOL_SCHEMAS
}
DOC_ID_IN_TEXT = re.compile(r"\[(d[0-9a-f]{10})\]")


def approx_tokens(text: str) -> int:
    """Tokenizer-free length estimate (~4 characters per token), used identically everywhere."""
    return math.ceil(len(text) / 4)


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


class Corpus:
    def __init__(self, documents: dict[str, Document]):
        self.documents = documents

    @classmethod
    def load(cls, path: str | Path) -> Corpus:
        documents = {}
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    documents[row["doc_id"]] = Document(row["doc_id"], row["title"], row["text"])
        return cls(documents)

    def __getitem__(self, doc_id: str) -> Document:
        return self.documents[doc_id]


class Encoder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEncoder:
    """Deterministic bag-of-words hashing vectors: offline tests only, not a semantic model."""

    name = "hashing-256"

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                value = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
                matrix[row, value % self.dimensions] += 1.0 if value & 1 else -1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-8)


class SentenceTransformerEncoder:
    """Real dense encoder (e.g. BAAI/bge-small-en-v1.5); needs sentence-transformers."""

    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, device: str | None = None, query_prefix: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        # bge English models expect this instruction on short queries (not on passages).
        if query_prefix is None:
            query_prefix = self.BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode([self.query_prefix + text for text in texts])


class EmbeddingStore:
    """Document vectors: precomputed for the whole corpus, or computed lazily per document."""

    def __init__(self, encoder: Encoder, matrix: np.ndarray | None = None, ids: Sequence[str] | None = None):
        self.encoder = encoder
        self.matrix = matrix
        self.row = {doc_id: index for index, doc_id in enumerate(ids or [])}
        self.cache: dict[str, np.ndarray] = {}

    @classmethod
    def load(cls, encoder: Encoder, directory: str | Path) -> EmbeddingStore:
        directory = Path(directory)
        meta = json.loads((directory / "dense_meta.json").read_text())
        if meta["model"] != encoder.name:
            raise ValueError(f"embeddings were built with {meta['model']}, encoder is {encoder.name}")
        ids = (directory / "dense_ids.txt").read_text().split()
        return cls(encoder, np.load(directory / "dense.npy", mmap_mode="r"), ids)

    def vectors(self, documents: Sequence[Document]) -> np.ndarray:
        if self.matrix is not None and all(doc.doc_id in self.row for doc in documents):
            return np.asarray(self.matrix[[self.row[doc.doc_id] for doc in documents]], dtype=np.float32)
        missing = [doc for doc in documents if doc.doc_id not in self.cache]
        if missing:
            for doc, vector in zip(missing, self.encoder.encode([f"{d.title}. {d.text}" for d in missing])):
                self.cache[doc.doc_id] = vector
        return np.stack([self.cache[doc.doc_id] for doc in documents])

    def query(self, text: str) -> np.ndarray:
        encode = getattr(self.encoder, "encode_queries", self.encoder.encode)
        return encode([text])[0]


@dataclass
class ToolResult:
    ok: bool
    text: str
    doc_ids: tuple[str, ...] = ()
    error_kind: str | None = None


class PoolSearcher:
    """Executes the three tools against one question's pool."""

    def __init__(self, corpus: Corpus, pool: Sequence[str], embeddings: EmbeddingStore):
        self.documents = [corpus[doc_id] for doc_id in pool]
        self.by_id = {doc.doc_id: doc for doc in self.documents}
        self.bm25 = BM25Retriever({doc.doc_id: f"{doc.title}. {doc.text}" for doc in self.documents})
        self.embeddings = embeddings
        self._doc_vectors: np.ndarray | None = None

    def _format_hits(self, doc_ids: Sequence[str]) -> ToolResult:
        lines = []
        for rank, doc_id in enumerate(doc_ids, 1):
            doc = self.by_id[doc_id]
            snippet = doc.text[:SNIPPET_CHARS] + ("..." if len(doc.text) > SNIPPET_CHARS else "")
            lines.append(f"{rank}. [{doc_id}] {doc.title} | {snippet}")
        return ToolResult(True, "\n".join(lines) or "No results.", tuple(doc_ids))

    def call(self, name: str, arguments: Any) -> ToolResult:
        if name not in _VALIDATORS:
            return ToolResult(False, f"Error: unknown tool '{name}'. Available tools: {list(TOOL_NAMES)}", error_kind="unknown_tool")
        if not isinstance(arguments, dict):
            return ToolResult(False, f"Error: arguments for '{name}' must be a JSON object.", error_kind="invalid_arguments")
        errors = sorted(_VALIDATORS[name].iter_errors(arguments), key=lambda e: list(e.path))
        if errors:
            return ToolResult(False, f"Error: invalid arguments for '{name}': {errors[0].message}", error_kind="invalid_arguments")
        if name == "get_document":
            doc = self.by_id.get(arguments["doc_id"])
            if doc is None:
                return ToolResult(False, f"Error: doc_id {arguments['doc_id']} is not in this question's paragraphs.", error_kind="unknown_document")
            text = doc.text[:DOCUMENT_CHARS] + ("..." if len(doc.text) > DOCUMENT_CHARS else "")
            return ToolResult(True, f"[{doc.doc_id}] {doc.title} | {text}", (doc.doc_id,))
        k = arguments.get("k", 3)
        if name == "bm25_search":
            return self._format_hits([hit.document_id for hit in self.bm25.search(arguments["query"], k)])
        if self._doc_vectors is None:
            self._doc_vectors = self.embeddings.vectors(self.documents)
        # numpy 2.x on Apple Accelerate raises spurious FP warnings in matmul, even on random data.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = self._doc_vectors @ self.embeddings.query(arguments["query"])
        order = sorted(range(len(self.documents)), key=lambda i: (-float(scores[i]), self.documents[i].doc_id))
        return self._format_hits([self.documents[i].doc_id for i in order[:k]])
