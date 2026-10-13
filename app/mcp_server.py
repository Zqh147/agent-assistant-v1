"""Run with python -m app.mcp_server. Stdout is reserved for MCP protocol."""
from mcp.server.fastmcp import FastMCP
from .tools import calculate as local_calculate, get_current_time as local_time

mcp = FastMCP("agent-assistant-tools")


@mcp.tool()
def calculate(a: float, b: float, operation: str) -> dict:
    """Perform add, subtract, multiply, or divide; zero divisors are rejected."""
    return local_calculate(a, b, operation)


@mcp.tool()
def get_current_time() -> dict:
    """Return the current server local ISO time."""
    return local_time()


if __name__ == "__main__":
    mcp.run(transport="stdio")
