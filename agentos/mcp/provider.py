from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from mcp import Client

from agentos.mcp.registry import ToolRegistry
from agentos.models import ToolMetadata


class ToolProvider(Protocol):
    async def list_tools(self) -> list[ToolMetadata]: ...

    async def call_tool(
        self, name: str, arguments: dict[str, Any], operation_id: str | None = None
    ) -> Any: ...


class LocalToolProvider:
    """Compatibility provider for deterministic offline tests and demos."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def list_tools(self) -> list[ToolMetadata]:
        return self.registry.list_tools()

    async def call_tool(
        self, name: str, arguments: dict[str, Any], operation_id: str | None = None
    ) -> Any:
        return await self.registry.call(name, arguments, operation_id=operation_id)


class MCPToolProvider:
    """Discover and invoke tools through real MCP Client list_tools/call_tool calls."""

    def __init__(self, servers: Mapping[str, Any]):
        self.servers = dict(servers)
        self._routes: dict[str, tuple[str, str]] = {}

    async def list_tools(self) -> list[ToolMetadata]:
        discovered: list[ToolMetadata] = []
        routes: dict[str, tuple[str, str]] = {}
        used: set[str] = set()
        for server_name, server in self.servers.items():
            async with Client(server) as client:
                response = await client.list_tools()
            for tool in response.tools:
                public_name = tool.name
                if public_name in used:
                    public_name = f"{server_name}.{tool.name}"
                used.add(public_name)
                routes[public_name] = (server_name, tool.name)
                output_schema = tool.output_schema or {}
                if (
                    isinstance(output_schema, dict)
                    and output_schema.get("type") == "object"
                    and set(output_schema.get("properties", {})) == {"result"}
                ):
                    definitions = output_schema.get("$defs")
                    output_schema = dict(output_schema["properties"]["result"])
                    if definitions:
                        # Keep "#/$defs/..." references resolvable after unwrapping.
                        output_schema["$defs"] = definitions
                annotations = tool.annotations
                # MCP's idempotentHint defaults to False only for tools that declare
                # side effects; read-only or unannotated tools are treated as replayable.
                idempotent = True
                if annotations is not None and annotations.read_only_hint is False:
                    idempotent = bool(annotations.idempotent_hint)
                discovered.append(
                    ToolMetadata(
                        name=public_name,
                        description=tool.description or tool.title or tool.name,
                        input_schema=tool.input_schema or {},
                        output_schema=output_schema,
                        server_name=server_name,
                        idempotent=idempotent,
                    )
                )
        self._routes = routes
        return discovered

    async def call_tool(
        self, name: str, arguments: dict[str, Any], operation_id: str | None = None
    ) -> Any:
        del operation_id  # MCP has no cross-server transaction/idempotency standard.
        if name not in self._routes:
            await self.list_tools()
        if name not in self._routes:
            raise KeyError(f"unknown MCP tool: {name}")
        server_name, remote_name = self._routes[name]
        async with Client(self.servers[server_name]) as client:
            response = await client.call_tool(remote_name, arguments)
        if getattr(response, "is_error", False):
            texts = [
                item.text
                for item in getattr(response, "content", [])
                if getattr(item, "type", None) == "text"
            ]
            raise RuntimeError(f"MCP tool returned error: {' '.join(texts) or response.content}")
        structured = getattr(response, "structured_content", None)
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
        if structured is not None:
            return structured
        content: Sequence[Any] = getattr(response, "content", [])
        texts = [item.text for item in content if getattr(item, "type", None) == "text"]
        return texts[0] if len(texts) == 1 else texts
