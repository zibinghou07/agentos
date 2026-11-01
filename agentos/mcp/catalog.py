from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from agentos.mcp.registry import ToolRegistry
from agentos.models import ToolMetadata
from agentos.rag import BM25Retriever, DenseRetriever, HybridRetriever

DOCUMENTS = {
    "doc_agent": "AgentOS uses DAG scheduling to run independent task nodes concurrently and persists checkpoints in SQLite.",
    "doc_recovery": "Retry handles transient failures, Repair corrects invalid arguments, and Replan replaces an infeasible plan.",
    "doc_rag": "Hybrid retrieval combines sparse BM25 and dense semantic retrieval using reciprocal rank fusion.",
    "doc_mcp": "Model Context Protocol standardizes tool discovery and invocation between a host client and tool servers.",
}

BUSINESS_SCHEMA = """
CREATE TABLE IF NOT EXISTS products(id TEXT PRIMARY KEY,name TEXT NOT NULL,price REAL NOT NULL);
CREATE TABLE IF NOT EXISTS orders(id TEXT PRIMARY KEY,product_id TEXT NOT NULL,quantity INTEGER NOT NULL,order_date TEXT NOT NULL);
INSERT OR IGNORE INTO products VALUES('p1','AgentOS Pro',99.0),('p2','RAG Toolkit',49.0),('p3','MCP Adapter',29.0);
INSERT OR IGNORE INTO orders VALUES('o1','p1',2,'2026-01-10'),('o2','p2',3,'2026-02-12'),('o3','p1',1,'2026-03-05');
"""


def initialize_business_database(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(BUSINESS_SCHEMA)


def build_local_registry(database: str | Path = "data/business.db") -> ToolRegistry:
    if str(database) == ":memory:":
        connection = sqlite3.connect(":memory:")
        connection.executescript(BUSINESS_SCHEMA)
    else:
        initialize_business_database(database)
        connection = sqlite3.connect(database)
    bm25 = BM25Retriever(DOCUMENTS)
    dense = DenseRetriever(DOCUMENTS)
    hybrid = HybridRetriever(bm25, dense)
    registry = ToolRegistry()

    def add(name: str, description: str, schema: dict[str, Any], server: str, fn: Any) -> None:
        registry.register(ToolMetadata(name=name, description=description, input_schema=schema, server_name=server), fn)

    query_schema = {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 3, "maximum": 10}}, "required": ["query"]}
    add("search_documents", "search local documents for evidence", query_schema, "search", lambda query, top_k=3: [r.__dict__ for r in bm25.search(query, top_k)])
    add("bm25_search", "keyword sparse BM25 retrieval", query_schema, "knowledge", lambda query, top_k=3: [r.__dict__ for r in bm25.search(query, top_k)])
    add("dense_search", "semantic dense document retrieval", query_schema, "knowledge", lambda query, top_k=3: [r.__dict__ for r in dense.search(query, top_k)])
    add("hybrid_search", "hybrid semantic and keyword retrieval", query_schema, "knowledge", lambda query, top_k=3: [r.__dict__ for r in hybrid.search(query, top_k)])

    def query_product(product_id: str) -> dict[str, Any] | None:
        row = connection.execute(
            "SELECT id,name,price FROM products WHERE id=?", (product_id,)
        ).fetchone()
        return dict(zip(("id", "name", "price"), row)) if row else None

    def search_products(keyword: str) -> list[dict[str, Any]]:
        rows = connection.execute(
            "SELECT id,name,price FROM products WHERE name LIKE ? ORDER BY id", (f"%{keyword}%",)
        ).fetchall()
        return [dict(zip(("id", "name", "price"), row)) for row in rows]

    add("query_product", "look up an exact product identifier", {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]}, "database", query_product)
    add("search_products", "find products by a name keyword", {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]}, "database", search_products)
    return registry

