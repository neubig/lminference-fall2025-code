import argparse
import asyncio
import os
from typing import TYPE_CHECKING, List, Optional

import aiohttp
import dotenv
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse


from .apis.serper_apis import (
    ScholarResponse,
    SearchResponse,
    WebpageContentResponse,
    fetch_webpage_content,
    search_serper,
    search_serper_scholar,
)
from .cache import set_cache_enabled
from .local.crawl4ai_fetcher import Crawl4AiResult

dotenv.load_dotenv()

mcp = FastMCP(
    "11-763 HW2 MCP",
    include_tags=os.environ.get("MCP_INCLUDE_TAGS", "search,browse,rerank").split(","),
)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """
    Check if the MCP server is running.
    curl http://127.0.0.1:8000/health
    """
    return PlainTextResponse("OK")


@mcp.tool(tags={"graph", "necessary"})
def submit_paths(
    paths: List[List[int]], # list of paths, where each path is a list of node indices
    weights: List[int], # list of corresponding path weights
) -> None:
    """
    Submit the top P shortest paths found
    """
    return (paths, weights)

@mcp.tool(tags={"search", "necessary"})
def serper_google_webpage_search(
    query: str,
    num_results: int = 10,
    gl: str = "us", 
    hl: str = "en", 
):
    """
    Search using google search (based on Serper.dev API).

    Args:
        query: Search query string
        num_results: Number of results to return (default: 10)
        gl: Country code to boosts search results whose country of origin matches the parameter value (default: us)
        hl: Host language of user interface (default: en)

    Returns:
        Dictionary containing search results with the following fields:
        - organic: List of organic search results
        - knowledgeGraph: Knowledge graph information (if available)
        - peopleAlsoAsk: List of related questions
        - relatedSearches: List of related searches
    """
    results = search_serper(
        query=query,
        num_results=num_results,
        search_type="search",
        gl=gl,
        hl=hl
    )

    return results


@mcp.tool(tags={"browse", "necessary"})
def serper_fetch_webpage_content(
    webpage_url: str,
    include_markdown: bool = True,
) -> WebpageContentResponse:
    """
    Fetch the content of a webpage using Serper.dev API.

    Args:
        webpage_url: The URL of the webpage to fetch
        include_markdown: Whether to include markdown formatting in the response (default: True)

    Returns:
        Dictionary containing the webpage content with the following fields:
        - text: The webpage content as plain text
        - markdown: The webpage content formatted as markdown (if include_markdown=True)
        - metadata: Additional metadata about the webpage
        - url: The original URL that was fetched
        - success: Boolean indicating if the fetch was successful
    """
    try:
        result = fetch_webpage_content(
            url=webpage_url,
            include_markdown=include_markdown,
        )

        return {
            **result,
            "success": True,
        }
    except Exception as e:
        return {
            "text": "",
            "markdown": "",
            "metadata": {},
            "url": webpage_url,
            "success": False,
            "error": str(e),
        }


@mcp.tool(tags={"search", "necessary"})
def serper_google_scholar_search(
    query: str,
    num_results: int = 10,
) -> ScholarResponse:
    """
    Search for academic papers using google scholar (based on Serper.dev API).

    Args:
        query: Search query string
        num_results: Number of results to return (default: 10)

    Returns:
        Dictionary containing search results with the following fields:
        - organic: List of organic search results
    """
    results = search_serper_scholar(
        query=query,
        num_results=num_results,
    )

    return results


@mcp.tool(tags={"browse", "necessary"})
async def crawl4ai_fetch_webpage_content(
    url: str,
    ignore_links: bool = True,
    use_pruning: bool = False,
    bm25_query: Optional[str] = None,
    bypass_cache: bool = True,
    timeout_ms: int = 80000,
    include_html: bool = False,
) -> Crawl4AiResult:
    """
    Asynchronously fetch webpage content using Crawl4AI.

    Args:
        url: URL to fetch
        ignore_links: If True, remove hyperlinks in markdown
        use_pruning: Apply pruning content filter (used when bm25_query is not provided)
        bm25_query: Optional query to enable BM25-based content filtering
        bypass_cache: If True, bypass Crawl4AI cache
        timeout_ms: Per-page timeout in milliseconds
        include_html: Whether to include raw HTML in the response
    """

    from mcp_agents.mcp_backend.local.crawl4ai_fetcher import fetch_markdown

    result = await fetch_markdown(
        url=url,
        query=bm25_query,
        ignore_links=ignore_links,
        use_pruning=use_pruning,
        bypass_cache=bypass_cache,
        headless=True,
        timeout_ms=timeout_ms,
        include_html=include_html,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MCP server")
    parser.add_argument(
        "--transport",
        type=str,
        default="http",
        choices=["stdio", "http", "sse", "streamable-http"],
        help="Transport protocol to use (default: stdio for local, http for web)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (for HTTP transports)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind to (for HTTP transports)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/mcp",
        help="Path for the HTTP endpoint (default: /mcp)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level for the server",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable API response caching",
    )

    args = parser.parse_args()

    # Set cache enabled/disabled based on argument
    if args.no_cache:
        set_cache_enabled(False)
    else:
        set_cache_enabled(True)

    # Run the server with the provided arguments
    if args.transport == "stdio":
        # stdio transport doesn't accept host/port/path arguments
        # For stdio, we can omit the transport argument since it's the default
        mcp.run(transport="stdio")
    else:
        # HTTP-based transports accept host/port/path/log_level arguments
        mcp.run(
            transport=args.transport,
            host=args.host,
            port=args.port,
            path=args.path,
            log_level=args.log_level,
        )
