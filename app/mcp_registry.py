from __future__ import annotations
from functools import partial
from typing import Any
from .mcp_client import StdioMCPClient


class MCPRegistry:
    """Discover schemas through tools/list; dispatch functions through tools/call."""
    def __init__(self, client=None):
        self.client = client or StdioMCPClient()
        self.descriptors: dict[str, dict[str, Any]] = {}
        self.callables = {}
        self.schemas = []

    def discover(self):
        descriptors, callables, schemas = {}, {}, []
        for tool in self.client.list_tools():
            if tool.name in descriptors:
                raise ValueError(f"duplicate MCP tool: {tool.name}")
            descriptors[tool.name] = {"name": tool.name, "description": tool.description or "",
                                       "source": "mcp", "input_schema": tool.inputSchema}
            callables[tool.name] = partial(self._invoke, tool.name)
            schemas.append({"type": "function", "function": {
                "name": tool.name, "description": tool.description or "",
                "parameters": tool.inputSchema}})
        self.descriptors, self.callables, self.schemas = descriptors, callables, schemas
        return list(descriptors.values())

    def _invoke(self, name, **arguments):
        return self.client.call_tool(name, arguments)
