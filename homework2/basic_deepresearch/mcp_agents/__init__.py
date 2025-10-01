__version__ = "0.0.1"

# Main library components
from .agent_interface import BaseAgent
from .client import GenerateWithToolsOutput, GenerationConfig, LLMToolClient

# Tool interface components
from .tool_interface import (
    AgentAsTool,
    BaseTool,
    ChainedTool,
    Document,
    DocumentToolOutput,
    MCPMixin,
    SerperBrowseTool,
    SerperSearchTool,
    ToolCallInfo,
    ToolCallParser,
    ToolInput,
    ToolOutput,
)
from .workflow import BaseWorkflow, BaseWorkflowConfiguration

__all__ = [
    # Core components
    "BaseAgent",
    "LLMToolClient",
    "GenerateWithToolsOutput",
    "GenerationConfig",
    "BaseWorkflow",
    "BaseWorkflowConfiguration",
    # Tool interface - Core
    "BaseTool",
    "ToolInput",
    "ToolOutput",
    "AgentAsTool",
    "ChainedTool",
    # Tool interface - Data types
    "Document",
    "DocumentToolOutput",
    # Tool interface - MCP Tools
    "MCPMixin",
    "SerperSearchTool",
    "SerperBrowseTool",
    # Tool interface - Parsing
    "ToolCallInfo",
    "ToolCallParser",
]
