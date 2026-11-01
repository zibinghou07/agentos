from mcp.server import MCPServer

from agentos.mcp.catalog import DOCUMENTS
from agentos.rag import BM25Retriever

mcp = MCPServer("agentos-search")
retriever = BM25Retriever(DOCUMENTS)


@mcp.tool()
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """Search the bundled document corpus."""
    return [result.__dict__ for result in retriever.search(query, top_k)]


@mcp.tool()
def search_by_keyword(query: str) -> list[dict]:
    """Search local documents with BM25 keyword matching."""
    return search_documents(query)


@mcp.tool()
def get_document(document_id: str) -> dict:
    """Read an exact local document identifier."""
    if document_id not in DOCUMENTS:
        raise ValueError("document not found")
    return {"document_id": document_id, "text": DOCUMENTS[document_id]}


if __name__ == "__main__":
    mcp.run()

