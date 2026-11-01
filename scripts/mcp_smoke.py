from __future__ import annotations

import asyncio

from mcp import Client

from agentos.mcp.servers.search_server import mcp


async def main() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
        result = await client.call_tool("search_documents", {"query": "checkpoint", "top_k": 1})
        print({"tools": [tool.name for tool in tools.tools], "structured": result.structured_content})


if __name__ == "__main__":
    asyncio.run(main())
