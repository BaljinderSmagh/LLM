from mcp.server.fastmcp import FastMCP
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tools.stock_price import get_stock_price
from tools.search import search_duckduckgo
# Initialize the MCP server
mcp = FastMCP("stock-tools")

# Register the tool using @tool() decorator
@mcp.tool()
def get_stock_price_tool(ticker: str) -> str:
    return get_stock_price(ticker)



@mcp.tool()
def duck_search(query: str) -> str:
    return search_duckduckgo(query)