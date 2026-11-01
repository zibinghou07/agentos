from agentos.mcp.provider import MCPToolProvider
from agentos.mcp.servers.search_server import mcp


async def test_mcp_provider_discovers_and_calls_through_protocol():
    provider = MCPToolProvider({"search": mcp})
    tools = await provider.list_tools()
    assert "search_documents" in {tool.name for tool in tools}
    result = await provider.call_tool(
        "search_documents", {"query": "checkpoint", "top_k": 1}
    )
    assert result[0]["document_id"] == "doc_agent"
