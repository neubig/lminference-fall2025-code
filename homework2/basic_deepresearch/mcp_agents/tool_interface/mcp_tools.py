import asyncio
import json
import logging
import os
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import re


from anyio.streams.memory import BrokenResourceError
from fastmcp import Client
from fastmcp.exceptions import FastMCPError, ResourceError
from fastmcp.utilities.exceptions import McpError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseTool
from .data_types import Document, DocumentToolOutput, ToolInput, ToolOutput
from .tool_parsers import LegacyToolCallParser, ToolCallInfo, ToolCallParser
from .utils import extract_snippet_with_context

SERPER_MAX_QUERY_LENGTH = 2048

import mcp_agents

RAG_MCP_PATH = Path(mcp_agents.__file__).parent / "mcp_backend" / "main.py"
DEFAULT_MCP_MAX_CONCURRENT_CALLS = 20


class MCPErrorHandlingMode(Enum):
    """Error handling strategies for MCP calls"""

    RETURN_ERROR = "return_error"  # Always return error dict, never raise exceptions
    RAISE_EXCEPT_TIMEOUT = "raise_except_timeout"  # Raise except timeout errors, return dict for timeout errors
    RAISE_ALL = "raise_all"  # Raise all exceptions


DEFAULT_MCP_ERROR_MODE = MCPErrorHandlingMode.RAISE_EXCEPT_TIMEOUT

logger = logging.getLogger(__name__)


class MCPMixin:
    """Mixin class that provides MCP (Model Context Protocol) functionality to tools"""

    # Global semaphore for controlling concurrent MCP calls
    _global_semaphore = None
    _max_concurrent_calls = None
    _error_handling_mode = None

    def __init__(
        self,
        *args,
        timeout: int = 60,
        name: Optional[str] = None,
        **kwargs,
    ):
        # Initialize global semaphore if not already done
        if MCPMixin._global_semaphore is None:
            MCPMixin._max_concurrent_calls = int(
                os.environ.get(
                    "MCP_MAX_CONCURRENT_CALLS", DEFAULT_MCP_MAX_CONCURRENT_CALLS
                )
            )
            MCPMixin._global_semaphore = asyncio.Semaphore(
                MCPMixin._max_concurrent_calls
            )

        # Initialize global error handling mode if not already done
        if MCPMixin._error_handling_mode is None:
            # Allow setting via kwargs or environment variable
            error_mode_str = kwargs.pop(
                "mcp_error_handling_mode", os.environ.get("MCP_ERROR_HANDLING_MODE")
            )
            if error_mode_str is not None:
                try:
                    MCPMixin._error_handling_mode = MCPErrorHandlingMode(error_mode_str)
                except ValueError:
                    logger.warning(
                        f"Invalid MCP_ERROR_HANDLING_MODE: {error_mode_str}. "
                        f"Using default: {DEFAULT_MCP_ERROR_MODE.value}"
                    )
                    MCPMixin._error_handling_mode = DEFAULT_MCP_ERROR_MODE
            else:
                MCPMixin._error_handling_mode = DEFAULT_MCP_ERROR_MODE

        # Fetch needed MCP arguments before calling super().__init__
        self.transport_type = kwargs.pop("transport_type", None) or os.environ.get(
            "MCP_TRANSPORT", "StreamableHttpTransport"
        )
        self.mcp_executable = kwargs.pop("mcp_executable", None) or os.environ.get(
            "MCP_EXECUTABLE", RAG_MCP_PATH
        )
        self.mcp_port = kwargs.pop("mcp_port", None) or os.environ.get(
            "MCP_TRANSPORT_PORT", 8000
        )
        self.mcp_host = kwargs.pop("mcp_host", None) or os.environ.get(
            "MCP_TRANSPORT_HOST", "localhost"
        )
        # Call super().__init__ to ensure proper MRO handling
        super().__init__(*args, timeout=timeout, name=name, **kwargs)
        self.timeout = timeout
        self.mcp_client_config = kwargs
        self.pinged = False

    def init_mcp_client(self):
        """Initialize MCP client based on environment variables"""
        if not Client:
            raise ImportError(
                "MCP client not available. Please install the MCP client library."
            )

        transport_type = self.transport_type

        if transport_type == "StreamableHttpTransport":
            logger.debug(
                f"Using MCP transport: {transport_type}, port: {self.mcp_port}"
            )
            return Client(f"http://{self.mcp_host}:{self.mcp_port}/mcp", timeout=self.timeout)
        elif transport_type == "FastMCPTransport":
            if not self.mcp_executable:
                raise ValueError(
                    "MCP_EXECUTABLE environment variable not set for FastMCPTransport"
                )
            logger.debug(
                f"Using MCP transport: {transport_type}, executable: {self.mcp_executable}"
            )
            return Client(self.mcp_executable, timeout=self.timeout)
        else:
            raise ValueError(f"Invalid MCP transport: {transport_type}")

    @retry(
        retry=retry_if_exception_type(
            (
                ConnectionError,
                TimeoutError,
                asyncio.TimeoutError,
                McpError,
                FastMCPError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _execute_mcp_call(
        self, tool_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an MCP tool call with proper error handling and global concurrency control"""
        if MCPMixin._global_semaphore is None:
            MCPMixin._max_concurrent_calls = int(
                os.environ.get(
                    "MCP_MAX_CONCURRENT_CALLS", DEFAULT_MCP_MAX_CONCURRENT_CALLS
                )
            )
            MCPMixin._global_semaphore = asyncio.Semaphore(
                MCPMixin._max_concurrent_calls
            )
        async with MCPMixin._global_semaphore:
            try:
                mcp_client = self.init_mcp_client()
                async with mcp_client:
                    if not self.pinged:
                        await mcp_client.ping()
                        self.pinged = True
                    result = await mcp_client.call_tool(tool_name, params)

                    # Handle different response formats
                    if hasattr(result, "content") and result.content:
                        if isinstance(result.content[0], dict):
                            return result.content[0]
                        elif hasattr(result.content[0], "text"):
                            return json.loads(result.content[0].text)
                        else:
                            return {"data": str(result.content[0])}
                    else:
                        return {"error": "No content in response", "data": []}

            except (asyncio.TimeoutError, TimeoutError) as e:
                error_msg = f"MCP call timed out after {self.timeout} seconds"
                print(f"Error: {error_msg}")
                if MCPMixin._error_handling_mode in [
                    MCPErrorHandlingMode.RETURN_ERROR,
                    MCPErrorHandlingMode.RAISE_EXCEPT_TIMEOUT,
                ]:
                    return {"error": error_msg, "data": []}
                else:  # raise_all
                    raise e
            except McpError as e:
                # Check if this is actually a timeout error disguised as McpError
                if "Timed out while waiting for response" in str(e):
                    error_msg = f"MCP call timed out: {str(e)}"
                    print(f"Error: {error_msg}")
                    if MCPMixin._error_handling_mode in [
                        MCPErrorHandlingMode.RETURN_ERROR,
                        MCPErrorHandlingMode.RAISE_EXCEPT_TIMEOUT,
                    ]:
                        return {"error": error_msg, "data": []}
                    else:  # raise_all
                        raise e
                else:
                    error_msg = f"MCP error: {str(e)}"
                    print(f"Error: {error_msg}")
                    if (
                        MCPMixin._error_handling_mode
                        == MCPErrorHandlingMode.RETURN_ERROR
                    ):
                        return {"error": error_msg, "data": []}
                    else:  # raise_timeout or raise_all
                        raise e
            except (ConnectionError, FastMCPError) as e:
                error_msg = f"MCP call failed: {str(e)}"
                print(f"Error: {error_msg}")
                if MCPMixin._error_handling_mode == MCPErrorHandlingMode.RETURN_ERROR:
                    return {"error": error_msg, "data": []}
                else:  # raise_all
                    raise e
            except (BrokenResourceError, ResourceError) as e:
                error_msg = f"MCP call failed: {str(e)}"

                print(f"Error: {error_msg}")
                # For most of this error, it's caused by the MCP server had encountered some bugs
                #
                if MCPMixin._error_handling_mode in [
                    MCPErrorHandlingMode.RETURN_ERROR,
                    MCPErrorHandlingMode.RAISE_EXCEPT_TIMEOUT,
                ]:
                    return {"error": error_msg, "data": []}
                else:  # raise_all
                    raise e
            except Exception as e:
                # Check if this is a timeout-related error
                error_str = str(e).lower()
                is_timeout_error = any(
                    timeout_keyword in error_str
                    for timeout_keyword in ["timeout", "timed out", "time out"]
                )

                if is_timeout_error:
                    error_msg = f"MCP call timed out: {str(e)}"
                    print(f"Error: {error_msg}")
                    if MCPMixin._error_handling_mode in [
                        MCPErrorHandlingMode.RETURN_ERROR,
                        MCPErrorHandlingMode.RAISE_EXCEPT_TIMEOUT,
                    ]:
                        return {"error": error_msg, "data": []}
                    else:  # raise_all
                        raise e
                else:
                    error_msg = f"Unexpected error: {str(e)}"
                    print(f"Error: {error_msg}")
                    if (
                        MCPMixin._error_handling_mode
                        == MCPErrorHandlingMode.RETURN_ERROR
                    ):
                        return {"error": error_msg, "data": []}
                    else:  # raise_timeout or raise_all
                        raise e

    @abstractmethod
    def get_mcp_tool_name(self) -> str:
        """Return the MCP tool name for this browse tool"""
        pass

    @abstractmethod
    def get_mcp_params(self, tool_call_info: ToolCallInfo) -> Dict[str, Any]:
        """
        Build parameters for MCP tool call.

        Args:
            tool_call_info: ToolCallInfo object containing content and parameters

        Returns:
            Dictionary of parameters for MCP tool
        """
        pass


class MCPSearchTool(MCPMixin, BaseTool, ABC):
    """Base class for MCP search tools with shared pipeline logic"""

    def __init__(
        self,
        tool_parser: Optional[ToolCallParser | str] = None,
        number_documents_to_search: int = 10,
        timeout: int = 60,
        name: Optional[str] = None,
        create_string_output: bool = True,
        **kwargs,
    ):
        super().__init__(
            tool_parser=tool_parser,
            timeout=timeout,
            name=name,
            create_string_output=create_string_output,
            **kwargs,
        )
        self.number_documents_to_search = number_documents_to_search

    @abstractmethod
    def extract_documents(self, raw_output: Dict[str, Any]) -> List[Document]:
        """
        Extract documents from raw MCP response.
        This should return a list of Document objects with title, snippet, url, and score.

        Args:
            raw_output: Raw response from MCP tool

        Returns:
            List of Document objects
        """
        pass

    # ===== Optional methods for subclasses to override =====

    def _create_error_output(
        self,
        error_msg: str,
        call_id: str,
        runtime: float,
        output: str = "",
        raw_output: Optional[Dict[str, Any]] = None,
    ) -> DocumentToolOutput:
        """Create a standardized error output for search tools"""
        return DocumentToolOutput(
            output=output,
            error=error_msg,
            called=True,
            timeout=False,
            runtime=runtime,
            call_id=call_id,
            raw_output=raw_output,
            tool_name=self.name,
            documents=[],
        )

    def preprocess_input(
        self, tool_input: Union[str, ToolInput, ToolOutput]
    ) -> Optional[ToolCallInfo]:
        """
        Preprocess and extract input for MCP search execution.
        Uses the tool parser system to extract content and parameters from tool calls.

        Args:
            tool_input: Raw input to the tool

        Returns:
            ToolCallInfo object with content and parameters, or None if invalid
        """
        if isinstance(tool_input, str):
            return self.parse_call(tool_input)
        else:
            raise ValueError(
                f"MCP Search Tool input must be a string, got {type(tool_input)}"
            )

    async def __call__(
        self, tool_input: Union[str, ToolInput, ToolOutput]
    ) -> DocumentToolOutput:
        """Shared pipeline for all search tools"""
        # print(f"{self.__class__.__name__} called with tool_input: {tool_input}")

        call_id = self._generate_call_id()
        start_time = time.time()

        # Step 1: Preprocess input
        tool_call_info = self.preprocess_input(tool_input)
        if not tool_call_info:
            return self._create_error_output(
                "No valid query found in tool call.",
                call_id,
                time.time() - start_time,
            )

        params = self.get_mcp_params(tool_call_info)

        # Step 4: Execute MCP call
        raw_output = await self._execute_mcp_call(self.get_mcp_tool_name(), params)

        # Step 5: Check for execution errors
        if error := raw_output.get("error"):
            return self._create_error_output(
                f"Query failed: {error}",
                call_id,
                time.time() - start_time,
                raw_output=raw_output,
            )

        # Step 6: Extract documents for structured output
        documents = self.extract_documents(raw_output)

        if not documents:
            return self._create_error_output(
                "No results found for the query.",
                call_id,
                time.time() - start_time,
                raw_output=raw_output,
            )

        # Step 7: Create content from documents using stringify
        content_parts = []
        for doc in documents:
            content_parts.append(doc.stringify())
        content = "\n\n".join(content_parts)

        return DocumentToolOutput(
            tool_name=self.name,
            output=content if self.create_string_output else "",
            called=True,
            error="",
            timeout=False,
            runtime=time.time() - start_time,
            call_id=call_id,
            raw_output=raw_output,
            documents=documents,
            query=tool_call_info.content,  # Save the original query
        )

    def _format_output(self, output: Union[ToolOutput, DocumentToolOutput]) -> str:
        """Format the search results into string representation"""
        if output.error:
            return output.error
        else:
            if isinstance(self.tool_parser, LegacyToolCallParser):
                content_parts = []
                for doc in output.documents:
                    content_parts.append(doc.stringify())
                content = "\n\n".join(content_parts)
                return content
            else:
                combined_snippet_text = []
                for index, doc in enumerate(output.documents):
                    combined_snippet_text.append(
                        f"<snippet id={output.call_id}-{index}>\n{doc.stringify()}\n</snippet>"
                    )
                return "\n".join(combined_snippet_text)



class SerperSearchTool(MCPSearchTool):
    """Tool for web search using Serper Google API via MCP"""

    def __init__(
        self,
        tool_parser: Optional[ToolCallParser | str] = None,
        number_documents_to_search: int = 10,
        timeout: int = 60,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            tool_parser=tool_parser,
            number_documents_to_search=number_documents_to_search,
            timeout=timeout,
            name=name,
            **kwargs,
        )

    def get_mcp_tool_name(self) -> str:
        return "serper_google_webpage_search"

    def get_mcp_params(self, tool_call_info: ToolCallInfo) -> Dict[str, Any]:
        """Build parameters for Serper API"""
        # Start with default parameters
        params = {
            "query": tool_call_info.content[:SERPER_MAX_QUERY_LENGTH],
            "num_results": self.number_documents_to_search,
        }

        # Override with validated parameters from tool call
        if "num_results" in tool_call_info.parameters:
            try:
                params["num_results"] = int(tool_call_info.parameters["num_results"])
            except ValueError:
                pass  # Keep default if conversion fails

        return params

    def extract_documents(self, raw_output: Dict[str, Any]) -> List[Document]:
        """Extract documents from Serper response"""
        organic_results = raw_output.get("organic", [])
        documents = []

        for result in organic_results:
            if isinstance(result, dict):
                doc = Document(
                    title=result.get("title", "").strip(),
                    url=result.get("link", "").strip(),
                    snippet=result.get("snippet", "").strip(),
                    text=None,
                    score=None,
                )
                if doc.title or doc.snippet or doc.url:
                    documents.append(doc)

        return documents


class MCPBrowseTool(MCPMixin, BaseTool, ABC):
    """Base class for MCP browse tools that fetch webpage content from URLs in search results"""

    def __init__(
        self,
        tool_parser: Optional[ToolCallParser | str] = None,
        max_pages_to_fetch: int = 5,
        timeout: int = 120,
        use_localized_snippets: bool = True,
        context_chars: int = 2000,
        name: Optional[str] = None,
        create_string_output: bool = True,
        **kwargs,
    ):
        super().__init__(
            tool_parser=tool_parser,
            timeout=timeout,
            name=name,
            create_string_output=create_string_output,
            **kwargs,
        )
        self.max_pages_to_fetch = max_pages_to_fetch
        self.use_localized_snippets = use_localized_snippets
        self.context_chars = context_chars

    def _create_error_output(
        self,
        error_msg: str,
        call_id: str,
        runtime: float,
        output: str = "",
        raw_output: Optional[Dict[str, Any]] = None,
    ) -> DocumentToolOutput:
        """Create a standardized error output for browse tools"""
        return DocumentToolOutput(
            output=output,
            error=error_msg,
            called=True,
            timeout=False,
            runtime=runtime,
            call_id=call_id,
            raw_output=raw_output,
            tool_name=self.name,
            documents=[],
        )

    def extract_urls(self, raw_output: Dict[str, Any]) -> List[str]:
        """Extract URLs from Serper search response"""
        urls = []

        # Handle organic search results
        organic_results = raw_output.get("organic", [])
        for result in organic_results:
            if isinstance(result, dict) and "link" in result:
                urls.append(result["link"])

        return urls

    def extract_urls_and_snippets(
        self, raw_output: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Extract URLs, titles, and snippets from Serper search response"""
        results = []

        # Handle organic search results
        organic_results = raw_output.get("organic", [])
        for result in organic_results:
            if isinstance(result, dict) and "link" in result:
                results.append(
                    {
                        "url": result["link"],
                        "title": result.get("title", "").strip(),
                        "snippet": result.get("snippet", "").strip(),
                    }
                )

        return results

    @abstractmethod
    def _extract_raw_content_from_response(
        self, raw_output: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extract raw text content from webpage fetch response.
        This method handles tool-specific response formats.

        Args:
            raw_output: Raw response from MCP webpage fetch tool

        Returns:
            Raw text content from webpage, or None if extraction failed
        """
        pass

    @abstractmethod
    def _extract_metadata_from_document(
        self, document: Document, raw_output: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract metadata for display purposes from document and raw response.

        Args:
            document: Document object being processed
            raw_output: Raw response from MCP webpage fetch tool

        Returns:
            Tuple of (webpage_title, fallback_message)
            - webpage_title: Title extracted from webpage, if available
            - fallback_message: Error or informational message, if any
        """
        pass

    async def _fetch_single_webpage(
        self,
        url: str,
        **kwargs,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Fetch content from a single webpage.

        Args:
            url: URL to fetch

        Returns:
            Tuple of (url, raw_output_from_mcp)
        """

        # Create a minimal ToolCallInfo for URL fetching
        url_tool_call = ToolCallInfo(
            content=url, parameters=kwargs, start_pos=0, end_pos=len(url)
        )
        params = self.get_mcp_params(url_tool_call)
        raw_output = await self._execute_mcp_call(self.get_mcp_tool_name(), params)
        return url, raw_output

    async def _fetch_webpages_parallel(
        self, documents: List[Document], **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch multiple webpages in parallel using document objects.

        Args:
            documents: List of Document objects with URLs to fetch

        Returns:
            Dictionary mapping document IDs to fetch results
        """
        # Limit the number of documents to fetch
        docs_with_urls = [doc for doc in documents if doc.url]
        docs_to_fetch = docs_with_urls[: self.max_pages_to_fetch]

        async def fetch_document(doc):
            url, raw_output = await self._fetch_single_webpage(doc.url, **kwargs)
            return doc.id, raw_output

        tasks = [fetch_document(doc) for doc in docs_to_fetch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in results
        processed_results = {}
        for result in results:
            if isinstance(result, Exception):
                # For exceptions, we don't have the doc_id, so we need to handle this differently
                # We'll skip the exception case or handle it in the calling code
                continue
            else:
                doc_id, raw_output = result
                processed_results[doc_id] = raw_output

        return processed_results

    async def __call__(
        self, tool_input: Union[str, ToolInput, ToolOutput]
    ) -> DocumentToolOutput:
        """Browse webpages from URLs found in search tool output or from direct URL string"""
        call_id = self._generate_call_id()
        start_time = time.time()

        # Step 1: Handle different input types
        input_documents = []

        original_query = None

        if isinstance(tool_input, str):
            # Direct URL string input - use tool parser to extract URL
            tool_call_info = self.parse_call(tool_input)
            if not tool_call_info:
                return self._create_error_output(
                    "No valid URL found in tool call.",
                    call_id,
                    time.time() - start_time,
                )

            doc = Document(
                title="",
                snippet="",
                url=tool_call_info.content.strip(),
                text=None,
                score=None,
            )
            input_documents = [doc]

        elif isinstance(tool_input, (ToolOutput, DocumentToolOutput)):
            # Original logic for ToolOutput/DocumentToolOutput
            if not tool_input.raw_output:
                return self._create_error_output(
                    "ToolOutput does not contain raw_output to extract URLs from.",
                    call_id,
                    time.time() - start_time,
                )

            # Get documents from input (prioritize DocumentToolOutput, fallback to raw_output)
            if isinstance(tool_input, DocumentToolOutput) and tool_input.documents:
                input_documents = tool_input.documents
                original_query = tool_input.query
            else:
                # Fallback: extract from raw_output for backward compatibility
                url_snippet_pairs = self.extract_urls_and_snippets(
                    tool_input.raw_output
                )
                for pair in url_snippet_pairs:
                    doc = Document(
                        title=pair.get("title", ""),
                        snippet=pair.get("snippet", ""),
                        url=pair["url"],
                    )
                    input_documents.append(doc)
        else:
            # return self._create_error_output(
            #     "MCPBrowseTool expects ToolOutput from search tool or URL string as input.",
            #     call_id,
            #     time.time() - start_time,
            # )
            raise ValueError(
                "MCPBrowseTool expects ToolOutput from search tool or URL string as input."
            )

        if not input_documents:
            raise ValueError("No documents with URLs found to browse.")

        logger.debug(
            f"Found {len(input_documents)} documents to browse, fetching up to {self.max_pages_to_fetch}"
        )

        # Step 2: Create copies of input documents to enrich with fetched content
        enriched_documents = []
        for doc in input_documents[: self.max_pages_to_fetch]:
            # Create a copy of the document that we'll enrich with text content
            enriched_doc = Document(
                id=doc.id,
                title=doc.title,
                snippet=doc.snippet,
                url=doc.url,
                text=None,  # Will be populated with fetched content
                score=doc.score,
            )
            enriched_documents.append(enriched_doc)

        # Create mapping for easy lookup
        docs_map = {doc.id: doc for doc in enriched_documents}

        # Step 3: Fetch webpages in parallel
        additional_params = {"query": original_query} if original_query else {}
        fetch_results = await self._fetch_webpages_parallel(
            enriched_documents, **additional_params
        )

        # Step 4: Process fetch results and update documents directly
        for doc_id, raw_output in fetch_results.items():
            document = docs_map.get(doc_id, None)
            if error := raw_output.get("error"):
                # Find document to get URL for error message
                document.error = error
            else:
                # Extract raw content directly (tool-specific logic)
                raw_content = self._extract_raw_content_from_response(raw_output)

                # Store the raw content directly in the document's text attribute
                document.text = raw_content

        # Step 6: Create output content using stringify on enriched documents
        webpage_contents = []
        for doc in enriched_documents:
            # Extract tool-specific metadata for display
            webpage_title, fallback_message = self._extract_metadata_from_document(
                doc, fetch_results.get(doc.id, {})
            )

            content = doc.stringify(
                webpage_title=webpage_title,
                use_localized_snippets=self.use_localized_snippets,
                context_chars=self.context_chars,
                fallback_message=fallback_message,
            )

            if not doc.title.strip():
                if webpage_title and webpage_title.strip():
                    doc.title = webpage_title.strip()

            if content:
                webpage_contents.append(content)

        # Combine all webpage contents
        final_content = "\n\n".join(webpage_contents)

        return DocumentToolOutput(
            tool_name=self.name,
            output=final_content if self.create_string_output else "",
            called=True,
            error="",
            timeout=False,
            runtime=time.time() - start_time,
            call_id=call_id,
            # raw_output=[
            #     {
            #         "doc_id": doc_id,
            #         "url": next(
            #             (doc.url for doc in docs_to_fetch if doc.id == doc_id),
            #             "unknown",
            #         ),
            #         "raw_output": raw_output,
            #     }
            #     for doc_id, raw_output in fetch_results.items()
            # ],
            raw_output=None,
            documents=enriched_documents,
            query=getattr(tool_input, "query", None),  # Copy query from input
        )

    def _format_output(self, output: Union[ToolOutput, DocumentToolOutput]) -> str:
        """Format the browse results into string representation"""
        if isinstance(self.tool_parser, LegacyToolCallParser):
            return output.output
        else:
            combined_webpage_text = []
            for index, doc in enumerate(output.documents):
                combined_webpage_text.append(
                    f"<webpage id={output.call_id}-{index}>\n{doc.stringify()}\n</webpage>"
                )
            return "\n".join(combined_webpage_text)


class SerperBrowseTool(MCPBrowseTool):
    """Tool for fetching webpage content using Serper API via MCP"""

    def __init__(
        self,
        tool_parser: Optional[ToolCallParser | str] = None,
        max_pages_to_fetch: int = 5,
        timeout: int = 120,
        use_localized_snippets: bool = True,
        context_chars: int = 2000,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            tool_parser=tool_parser,
            max_pages_to_fetch=max_pages_to_fetch,
            timeout=timeout,
            use_localized_snippets=use_localized_snippets,
            context_chars=context_chars,
            name=name,
            **kwargs,
        )

    def get_mcp_tool_name(self) -> str:
        return "serper_fetch_webpage_content"

    def get_mcp_params(self, tool_call_info: ToolCallInfo) -> Dict[str, Any]:
        """Build parameters for Serper webpage fetch API"""
        return {"webpage_url": tool_call_info.content, "include_markdown": True}

    def _extract_raw_content_from_response(
        self, raw_output: Dict[str, Any]
    ) -> Optional[str]:
        """Extract raw text content from Serper response"""
        # Check if webpage fetching failed
        if raw_output.get("success") is False:
            return None

        # Extract content from Serper response
        markdown_content = raw_output.get("markdown", "")
        text_content = raw_output.get("text", "")
        return markdown_content or text_content

    def _extract_metadata_from_document(
        self, document: Document, raw_output: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract metadata for display from Serper response"""
        # Check if webpage fetching failed
        if raw_output.get("success") is False:
            error_message = (
                f"Failed to fetch content: {raw_output.get('error', 'Unknown error')}"
            )
            return None, error_message

        # Extract title from Serper response
        metadata = raw_output.get("metadata", {})
        webpage_title = metadata.get("title", "").strip() if metadata else ""

        return webpage_title, None


class Crawl4AIBrowseTool(MCPBrowseTool):
    """Tool for fetching webpage content using Crawl4AI via MCP"""

    def __init__(
        self,
        tool_parser: Optional[ToolCallParser | str] = None,
        max_pages_to_fetch: int = 5,
        timeout: int = 180,  # Crawl4AI might take longer
        ignore_links: bool = True,
        use_pruning: bool = False,
        bm25_query: Optional[str] = None,
        bypass_cache: bool = True,
        include_html: bool = False,
        use_localized_snippets: bool = True,
        context_chars: int = 2000,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            tool_parser=tool_parser,
            max_pages_to_fetch=max_pages_to_fetch,
            timeout=timeout,
            use_localized_snippets=use_localized_snippets,
            context_chars=context_chars,
            name=name,
            **kwargs,
        )
        self.ignore_links = ignore_links
        self.use_pruning = use_pruning
        self.bm25_query = bm25_query
        self.bypass_cache = bypass_cache
        self.timeout_ms = (
            max(0, (timeout - 1)) * 1000
        )  # make sure crawl4ai timeout is shorter than MCP timeout
        self.include_html = include_html

    def get_mcp_tool_name(self) -> str:
        return "crawl4ai_fetch_webpage_content"

    def get_mcp_params(self, tool_call_info: ToolCallInfo) -> Dict[str, Any]:
        """Build parameters for Crawl4AI API"""
        use_pruning = self.use_pruning
        bm25_query = tool_call_info.parameters.get("query", self.bm25_query)

        # If use_pruning is True and bm25_query is None, set use_pruning to True and bm25_query to None
        if self.use_pruning and bm25_query is None:
            use_pruning = True
            bm25_query = None
        else:
            use_pruning = False

        input_params = {
            "url": tool_call_info.content,
            "ignore_links": self.ignore_links,
            "use_pruning": use_pruning,
            "bm25_query": bm25_query,
            "bypass_cache": self.bypass_cache,
            "timeout_ms": self.timeout_ms,
            "include_html": self.include_html,
        }
        # print(input_params)
        return input_params

    def _extract_raw_content_from_response(
        self, raw_output: Dict[str, Any]
    ) -> Optional[str]:
        """Extract raw text content from Crawl4AI response"""
        # Check if crawling was successful
        if not raw_output.get("success", False):
            return None

        # Extract content from Crawl4AI response
        markdown_content = raw_output.get("markdown", "")
        fit_markdown_content = raw_output.get("fit_markdown", "")
        html_content = raw_output.get("html", "")

        # Prefer fit_markdown if available, then markdown, then html
        return fit_markdown_content or markdown_content or html_content

    def _extract_metadata_from_document(
        self, document: Document, raw_output: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract metadata for display from Crawl4AI response"""
        # Check if crawling was successful
        if not raw_output.get("success", False):
            error_msg = raw_output.get("error", "Unknown error")
            fallback_message = (
                f"Note: Crawl4AI failed ({error_msg}), using search snippet"
            )
            return None, fallback_message

        # Crawl4AI doesn't provide webpage title
        webpage_title = None

        # Handle case where no content was extracted
        markdown_content = raw_output.get("markdown", "")
        fit_markdown_content = raw_output.get("fit_markdown", "")
        html_content = raw_output.get("html", "")
        full_content = fit_markdown_content or markdown_content or html_content

        if not full_content:
            fallback_message = (
                "Note: No content extracted by Crawl4AI, using search snippet"
            )
            return webpage_title, fallback_message

        return webpage_title, None





class GraphPathEvaluationTool:
        
    def __init__(
        self,
        correct_paths: List[Dict[str, Any]],
        expected_count: int,
        tool_start_tag: str = "<predicted_paths>",
        tool_end_tag: str = "</predicted_paths>",
        result_start_tag: str = "<evaluation>",
        result_end_tag: str = "</evaluation>",
        timeout: int = 30
    ):
        """
        Initialize the evaluation tool with correct solution.
        
        Args:
            correct_paths: List of correct path dicts with 'path' and 'weight' keys
            expected_count: Expected number of paths (P)
            tool_start_tag: XML-style tag to mark start of input
            tool_end_tag: XML-style tag to mark end of input
            result_start_tag: XML-style tag to mark start of output
            result_end_tag: XML-style tag to mark end of output
            timeout: Timeout in seconds (not used, for API compatibility)
        """
        self.correct_paths = correct_paths
        self.expected_count = expected_count
        self.tool_start_tag = tool_start_tag
        self.tool_end_tag = tool_end_tag
        self.result_start_tag = result_start_tag
        self.result_end_tag = result_end_tag
        self.timeout = timeout
        
        # Pre-process correct paths into a set for fast comparison
        self._correct_set = {
            (tuple(p["path"]), p["weight"]) 
            for p in correct_paths
        }
    
    def _extract_content(self, text: str) -> Optional[str]:
        """Extract content between tool tags."""
        pattern = re.escape(self.tool_start_tag) + r"(.*?)" + re.escape(self.tool_end_tag)
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        return None
    
    def _parse_predicted_paths(self, json_str: str) -> List[Dict[str, Any]]:
        """Parse JSON string into predicted paths."""
        try:
            data = json.loads(json_str)
            
            # Handle different JSON formats
            if isinstance(data, dict):
                # Format: {"paths": [...], "weights": [...]}
                if "paths" in data and "weights" in data:
                    paths = data["paths"]
                    weights = data["weights"]
                    return [
                        {"path": path, "weight": weight}
                        for path, weight in zip(paths, weights)
                    ]
                # Format: {"paths": [{"path": [...], "weight": ...}]}
                elif "paths" in data and isinstance(data["paths"], list):
                    return data["paths"]
            
            # Format: [{"path": [...], "weight": ...}, ...]
            elif isinstance(data, list):
                return data
            
            return []
        except json.JSONDecodeError:
            return []
    
    def _evaluate(self, predicted_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate predicted paths against correct solution."""
        if not predicted_paths:
            return {
                "score": 0.0,
                "matches": 0,
                "expected": self.expected_count,
                "predicted_count": 0,
                "correct_paths_found": [],
                "incorrect_paths": [],
                "missing_paths": [
                    {"path": list(path), "weight": weight}
                    for path, weight in self._correct_set
                ],
                "message": "No predicted paths provided"
            }
        
        # Create set of predicted paths
        predicted_set = {
            (tuple(p["path"]), p["weight"]) 
            for p in predicted_paths
        }
        
        # Find matches
        matches_set = self._correct_set.intersection(predicted_set)
        matches = len(matches_set)
        score = matches / self.expected_count
        
        # YOUR_TASK_2.1.1: Find correct paths that were found; 2 lines of code
        correct_found = [
            # Your code here
        ]
        
        # YOUR_TASK_2.1.2: Find incorrect predictions; 2 lines of code
        incorrect = [
            # Your code here
        ]
        
        # YOUR_TASK_2.1.3: Find missing correct paths; 2 lines of code
        missing = [
            # Your code here
        ]
        
        return {
            "score": score,
            "matches": matches,
            "expected": self.expected_count,
            "predicted_count": len(predicted_paths),
            "correct_paths_found": correct_found,
            "incorrect_paths": incorrect,
            "missing_paths": missing,
            "message": f"Found {matches}/{self.expected_count} correct paths ({score:.1%})"
        }
    
    async def __call__(self, input_text: str) -> Dict[str, Any]:
        """
        Evaluate predicted paths from tagged input.
        
        Args:
            input_text: String containing predicted paths in JSON format,
                       wrapped in tool tags
        
        Returns:
            Dictionary with evaluation results
        """
        # Extract content between tags
        content = self._extract_content(input_text)
        
        if content is None:
            return {
                "error": f"Could not find content between {self.tool_start_tag} and {self.tool_end_tag}",
                "score": 0.0
            }
        
        # Parse predicted paths
        predicted_paths = self._parse_predicted_paths(content)
        
        # Evaluate
        result = self._evaluate(predicted_paths)
        
        return result
    
    def format_result(self, result: Dict[str, Any]) -> str:
        """
        Format evaluation result as JSON wrapped in result tags.
        
        Args:
            result: Dictionary from evaluation
        
        Returns:
            Formatted string with result tags and JSON content
        """
        json_str = json.dumps(result, indent=2)
        return f"{self.result_start_tag}\n{json_str}\n{self.result_end_tag}"
