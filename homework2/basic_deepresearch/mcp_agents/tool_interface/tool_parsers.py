import inspect
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .data_types import ToolOutput

# Global parser registry
_PARSER_REGISTRY: Dict[str, type] = {}


def register_parser(name: str):
    """
    Decorator to register a parser class in the global registry.

    Args:
        name: The name to register the parser under

    Returns:
        Decorator function that registers the parser class
    """

    def decorator(parser_class: type) -> type:
        if not issubclass(parser_class, ToolCallParser):
            raise ValueError(
                f"Parser class {parser_class.__name__} must inherit from ToolCallParser"
            )
        _PARSER_REGISTRY[name.lower()] = parser_class
        return parser_class

    return decorator


def get_registered_parsers() -> Dict[str, type]:
    """Get a copy of the current parser registry."""
    return _PARSER_REGISTRY.copy()


class ToolCallInfo(BaseModel):
    """Information about a parsed tool call"""

    content: str
    parameters: Dict[str, str] = Field(default_factory=dict)
    start_pos: int
    end_pos: int


class ToolCallParser(ABC):
    """Abstract base class for tool call parsers"""

    @abstractmethod
    def has_calls(self, text: str, tool_name: str) -> bool:
        """Check if this parser can find calls for the given tool in the text"""
        pass

    @abstractmethod
    def parse_call(self, text: str, tool_name: str) -> Optional[ToolCallInfo]:
        """Parse the first tool call for the given tool in the text"""
        pass

    @abstractmethod
    def format_result(self, formatted_output: str, output: "ToolOutput") -> str:
        """Format the tool output according to this parser's format"""
        pass

    @property
    @abstractmethod
    def stop_sequences(self) -> List[str]:
        """Get stop sequences for this parser format"""
        pass


@register_parser("legacy")
class LegacyToolCallParser(ToolCallParser):
    """Parser for legacy <tag>content</tag> format"""

    def __init__(
        self,
        tool_start_tag: str,
        tool_end_tag: Optional[str] = None,
        result_start_tag: Optional[str] = None,
        result_end_tag: Optional[str] = None,
    ):
        assert result_start_tag is not None
        self.tool_start_tag = tool_start_tag
        self.tool_end_tag = tool_end_tag or self._infer_end_tag(tool_start_tag)
        self.result_start_tag = result_start_tag
        self.result_end_tag = result_end_tag or self._infer_end_tag(
            self.result_start_tag
        )

    @staticmethod
    def _infer_end_tag(start_tag: str) -> str:
        """Infer the end tag from the start tag by replacing < with </"""
        if not start_tag:
            return None

        if start_tag.startswith("<") and start_tag.endswith(">"):
            tag_name = start_tag[1:-1]  # Remove < and >
            return f"</{tag_name}>"
        elif start_tag:
            # For non-XML style tags, just add "_end" suffix
            return f"{start_tag}_end"
        else:
            return None

    @property
    def stop_sequences(self) -> List[str]:
        """Get stop sequences for this parser format"""
        return [self.tool_end_tag] if self.tool_end_tag else []

    def has_calls(self, text: str, tool_name: str) -> bool:
        """Check if this parser can find calls in the given text"""
        if not (self.tool_start_tag and self.tool_end_tag):
            return False

        pattern = re.escape(self.tool_start_tag) + r".*?" + re.escape(self.tool_end_tag)
        return bool(re.search(pattern, text, re.DOTALL))

    def parse_call(self, text: str, tool_name: str) -> Optional[ToolCallInfo]:
        """Parse the first tool call in the text"""
        if not (self.tool_start_tag and self.tool_end_tag):
            return None

        pattern = (
            re.escape(self.tool_start_tag) + r"(.*?)" + re.escape(self.tool_end_tag)
        )
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return ToolCallInfo(
                content=match.group(1).strip(),
                parameters={},  # Legacy tools don't support parameters
                start_pos=match.start(),
                end_pos=match.end(),
            )
        return None

    def format_result(self, formatted_output: str, output: "ToolOutput") -> str:
        """Format the tool output with legacy result tags"""
        result_start_tag = self.result_start_tag
        if output.call_id:
            result_start_tag = (
                f"{self.result_start_tag.replace('>', '')} id={output.call_id}>"
            )
        return f"{result_start_tag}\n{formatted_output}\n{self.result_end_tag}"


@register_parser("null")
class NullToolCallParser(ToolCallParser):
    """Parser for tools that don't need any parsing - returns input as-is"""

    def __init__(self):
        pass

    def has_calls(self, text: str, tool_name: str) -> bool:
        """Null parser never finds calls"""
        return False

    def parse_call(self, text: str, tool_name: str) -> Optional[ToolCallInfo]:
        """Null parser never parses calls"""
        return ToolCallInfo(
            content=text,
            parameters={},
            start_pos=0,
            end_pos=len(text),
        )

    def format_result(self, formatted_output: str, output: "ToolOutput") -> str:
        """Null parser uses simple output format"""
        return output.output

    @property
    def stop_sequences(self) -> List[str]:
        """Null parser has no stop sequences"""
        return []


@register_parser("unified")
class UnifiedToolCallParser(ToolCallParser):
    """Parser for unified <tool name="xxx">content</tool> format"""

    def __init__(self):
        pass

    @property
    def stop_sequences(self) -> List[str]:
        """Get stop sequences for unified format"""
        return ["</tool>"]

    def has_calls(self, text: str, tool_name: str) -> bool:
        """Check if this parser can find calls for the given tool in the text"""
        # Look for <tool name="toolname"> patterns matching this tool
        pattern = r'<tool\s+name="' + re.escape(tool_name) + r'"[^>]*?>.*?</tool>'
        return bool(re.search(pattern, text, re.DOTALL))

    def parse_call(self, text: str, tool_name: str) -> Optional[ToolCallInfo]:
        """Parse the first tool call for the given tool in the text, including parameters and position"""
        # Find <tool name="toolname" ...>content</tool> with position info
        pattern = r"<tool\s+([^>]*?)>(.*?)</tool>"

        for match in re.finditer(pattern, text, re.DOTALL):
            attr_string = match.group(1)
            content = match.group(2).strip()

            # Parse attributes
            attributes = {}
            attr_pattern = r'(\w+)="([^"]*)"'
            attr_matches = re.findall(attr_pattern, attr_string)

            for key, value in attr_matches:
                attributes[key] = value

            # Check if this is for our tool
            if attributes.get("name") == tool_name:
                # Extract parameters (everything except 'name')
                parameters = {k: v for k, v in attributes.items() if k != "name"}

                return ToolCallInfo(
                    content=content,
                    parameters=parameters,
                    start_pos=match.start(),
                    end_pos=match.end(),
                )

        return None


    def format_result(self, formatted_output: str, output: "ToolOutput") -> str:
        """Format the tool output with structured XML format"""
        return f"<tool_output>{formatted_output}</tool_output>"


@register_parser("v20250824")
class UnifiedToolCallParserV20250824(ToolCallParser):
    """Parser for unified <call_tool name="xxx">content</call_tool> format"""

    def __init__(self):
        pass

    @property
    def stop_sequences(self) -> List[str]:
        """Get stop sequences for unified format"""
        return ["</call_tool>", "</call>"]

    def has_calls(self, text: str, tool_name: str) -> bool:
        """Check if this parser can find calls for the given tool in the text"""
        # Look for <tool name="toolname"> patterns matching this tool
        for pattern in [
            r'<call_tool\s+name="' + re.escape(tool_name) + r'"[^>]*?>.*?</call_tool>',
            r'<call_tool\s+name="' + re.escape(tool_name) + r'"[^>]*?>.*?</call>',
        ]:
            if bool(re.search(pattern, text, re.DOTALL)):
                return True
        return False

    def parse_call(self, text: str, tool_name: str) -> Optional[ToolCallInfo]:
        """Parse the first tool call for the given tool in the text, including parameters and position"""
        # Find <tool name="toolname" ...>content</tool> with position info
        for pattern in [
            r"<call_tool\s+([^>]*?)>(.*?)</call_tool>",
            r"<call_tool\s+([^>]*?)>(.*?)</call>",
        ]:

            for match in re.finditer(pattern, text, re.DOTALL):
                attr_string = match.group(1)
                content = match.group(2).strip()

                # Parse attributes
                attributes = {}
                attr_pattern = r'(\w+)="([^"]*)"'
                attr_matches = re.findall(attr_pattern, attr_string)

                for key, value in attr_matches:
                    attributes[key] = value

                # Check if this is for our tool
                if attributes.get("name") == tool_name:
                    # Extract parameters (everything except 'name')
                    parameters = {k: v for k, v in attributes.items() if k != "name"}

                    return ToolCallInfo(
                        content=content,
                        parameters=parameters,
                        start_pos=match.start(),
                        end_pos=match.end(),
                    )

        return None

    def format_result(self, formatted_output: str, output: "ToolOutput") -> str:
        """Format the tool output with structured XML format"""
        return f"<tool_output>{formatted_output}</tool_output>"


# Unified parser factory function
def create_tool_parser(parser_type: str | None, **parser_args) -> ToolCallParser:
    """
    Create a tool parser of the specified type with given arguments.
    Uses signature inspection to automatically pass only the required arguments.

    Args:
        parser_type: Type of parser to create (registered parser name)
        **parser_args: Arguments to pass to the parser constructor

    Returns:
        Configured parser instance

    Raises:
        ValueError: If parser_type is not registered
        TypeError: If required arguments are missing
    """
    parser_type = parser_type.lower()

    if parser_type not in _PARSER_REGISTRY:
        available_parsers = list(_PARSER_REGISTRY.keys())
        raise ValueError(
            f"Unknown parser type: {parser_type}. Available parsers: {available_parsers}"
        )

    parser_class = _PARSER_REGISTRY[parser_type]

    if parser_type is None:
        return NullToolCallParser()

    # Use signature inspection to determine which arguments the parser accepts
    sig = inspect.signature(parser_class.__init__)

    # Filter kwargs to only include parameters that the parser constructor accepts
    filtered_args = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        if param_name in parser_args:
            filtered_args[param_name] = parser_args[param_name]
        elif param.default is inspect.Parameter.empty:
            # Required parameter not provided
            raise TypeError(
                f"Missing required argument '{param_name}' for {parser_type} parser"
            )

    # Warn about unused arguments (optional)
    unused_args = set(parser_args.keys()) - set(filtered_args.keys())
    if unused_args:
        raise ValueError(
            f"Unused arguments provided to {parser_type} parser: {unused_args}"
        )

    return parser_class(**filtered_args)
