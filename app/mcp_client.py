"""Synchronous adapter for a real MCP stdio subprocess.

Each operation opens a session and closes it afterwards. Call from sync workers,
not directly from a running asyncio event loop.
"""
from __future__ import annotations

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


class MCPToolError(ValueError):
    """A server-reported tool error, not a transport retry signal."""


class StdioMCPClient:
    def __init__(self, command=None, args=None, timeout=15):
        self.command = command or sys.executable
        self.args = args if args is not None else ["-m", "app.mcp_server"]
        self.timeout = timeout

    async def _request(self, method, name=None, arguments=None):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        params = StdioServerParameters(command=self.command, args=self.args,
                                       cwd=str(Path(__file__).resolve().parent.parent))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                if method == "list":
                    available = []
                    cursor = None
                    while True:
                        page = await session.list_tools(cursor=cursor)
                        available.extend(page.tools)
                        cursor = page.nextCursor
                        if not cursor:
                            return available
                return await session.call_tool(name, arguments)

    def _run(self, method, name=None, arguments=None):
        def execute():
            return asyncio.run(asyncio.wait_for(self._request(method, name, arguments), self.timeout))
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return execute()
        # Uvicorn may import the application within its running event loop.
        # Run this synchronous adapter's private loop in another thread.
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(execute).result()

    def list_tools(self):
        return self._run("list")

    def call_tool(self, name: str, arguments: dict[str, Any]):
        result = self._run("call", name, arguments)
        if result.isError:
            raise MCPToolError("; ".join(getattr(c, "text", str(c)) for c in result.content))
        if result.structuredContent is not None:
            return result.structuredContent
        return {"content": [block.model_dump(mode="json") for block in result.content]}


def call_mcp_tool(tool_name: str, arguments: dict[str, Any]):
    return StdioMCPClient().call_tool(tool_name, arguments)
