from mcp.server import MCPServer

from agentos.mcp.catalog import DOCUMENTS
from agentos.rag import BM25Retriever, DenseRetriever, HybridRetriever

mcp = MCPServer("agentos-knowledge")
bm25 = BM25Retriever(DOCUMENTS)
dense = DenseRetriever(DOCUMENTS)
hybrid = HybridRetriever(bm25, dense)


@mcp.tool()
def bm25_search(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve documents with BM25."""
    return [result.__dict__ for result in bm25.search(query, top_k)]


@mcp.tool()
def dense_search(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve documents with the offline hashing dense baseline."""
    return [result.__dict__ for result in dense.search(query, top_k)]


@mcp.tool()
def hybrid_search(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve documents using reciprocal rank fusion."""
    return [result.__dict__ for result in hybrid.search(query, top_k)]


@mcp.tool()
def rerank(query: str, documents: list[str]) -> list[dict]:
    """Rerank supplied documents with lexical relevance in offline mode."""
    local = {str(index): text for index, text in enumerate(documents)}
    return [result.__dict__ for result in BM25Retriever(local).search(query, len(local))]


if __name__ == "__main__":
    mcp.run()

