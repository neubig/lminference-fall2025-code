import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp_agents.agent_interface import BaseAgent
from mcp_agents.client import LLMToolClient
from mcp_agents.tool_interface.chained_tool import ChainedTool
from mcp_agents.tool_interface.mcp_tools import (
    Crawl4AIBrowseTool,
    SerperBrowseTool,
    SerperSearchTool,
)
from mcp_agents.workflow import BaseWorkflow, BaseWorkflowConfiguration


@dataclass
class ThinkAgent(BaseAgent):
    """Agent that reasons about the current situation and decides what information to search for next"""

    prompt = """You are a research assistant that reasons systematically about questions.

CURRENT TASK:
{question}

CONVERSATION HISTORY:
{history}

INSTRUCTIONS:
Analyze the current situation and decide what specific information you need to search for next.

Your response should be in the following format:
<think>[Your thorough reasoning about what you know so far, what's missing, and what specific information you need to search for next. Consider:
- What you already know from previous searches
- What key information is still missing  
- What would be most helpful to search for next
- How this search will help answer the original question]</think>

Be specific and detailed in your reasoning about what information you need to find.
"""


@dataclass
class SearchAgent(BaseAgent):
    """Agent that generates search queries based on the current reasoning"""

    prompt = """You are a search query generator. Based on the reasoning provided, create an effective search query for using google search.

ORIGINAL QUESTION:
{question}

CONVERSATION HISTORY:
{history}

SEARCHED QUERIES:
{searched_queries}

INSTRUCTIONS:
Based on the reasoning above, generate a specific, effective search query that will find the needed information.

Your response should be in the following format:
<query>[Your specific search query here]</query>

Guidelines for the search query:
- Make queries specific and focused
- Use relevant keywords and terminology
- Avoid queries too similar to previous searches
- Keep queries concise but informative
- Consider what authoritative sources might contain
"""

    def postprocess_output(self, result: Dict[str, Any]) -> str:
        output_string = result.generated_text
        if "</think>" in output_string:
            return "".join(output_string.split("</think>")[1:]).strip()
        return output_string

    def extract_search_query(self, generated_text: str) -> str:
        """
        YOUR_TASK_1.1: check if there is a search query and return if there is a match.
        Hint: remember to use re.DOTALL; three lines of code
        """
        return generated_text



@dataclass
class AnswerAgent(BaseAgent):
    """Agent that provides the final answer based on accumulated research"""

    prompt = """You are a research assistant providing a final answer based on accumulated research.

ORIGINAL QUESTION:
{question}

RESEARCH CONVERSATION HISTORY:
{history}

INSTRUCTIONS:
Based on all the research and reasoning above, provide a comprehensive, well-structured answer to the original question.
"""

    def postprocess_output(self, result: Dict[str, Any]) -> str:
        output_string = result.generated_text
        if "</think>" in output_string:
            return "".join(output_string.split("</think>")[1:]).strip()
        return output_string


class ReActWorkflow(BaseWorkflow):
    """ReAct workflow implementing Thought-Action-Observation pattern"""

    @property
    def _default_configuration_path(self) -> Optional[str]:
        """Default configuration file path for this workflow."""
        return "./react_agent.yaml"

    class Configuration(BaseWorkflowConfiguration):
        # react agent configuration
        react_agent_base_url: str
        react_agent_model_name: str = "Qwen3-8B"
        react_agent_tokenizer_name: str = "Qwen/Qwen3-8B"
        react_agent_api_key: str = "dummy-key"

        # Search configuration
        search_tool_name: str = "serper"
        number_documents_to_search: int = 2
        search_timeout: int = 60

        # Browse configuration
        browse_tool_name: Optional[str] = "serper"
        browse_timeout: int = 60
        browse_max_pages_to_fetch: int = 10


        # ReAct configuration
        num_think_search_cycles: int = 5

    def setup_components(self) -> None:
        cfg = self.configuration
        assert cfg is not None

        search_tool_chain = []
        if cfg.search_tool_name == "serper":
            self.search_tool = SerperSearchTool(
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
            )
        else:
            raise ValueError(f"Invalid search tool name: {cfg.search_tool_name}. Only 'serper' is supported.")
        search_tool_chain.append(self.search_tool)

        if cfg.browse_tool_name is not None:
            if cfg.browse_tool_name == "serper":
                self.browse_tool = SerperBrowseTool(
                    max_pages_to_fetch=cfg.browse_max_pages_to_fetch,
                    timeout=cfg.browse_timeout,
                )
            elif cfg.browse_tool_name == "crawl4ai":
                self.browse_tool = Crawl4AIBrowseTool(
                    max_pages_to_fetch=cfg.browse_max_pages_to_fetch,
                    timeout=cfg.browse_timeout,
                )
            else:
                raise ValueError(f"Invalid browse tool name: {cfg.browse_tool_name}")
            search_tool_chain.append(self.browse_tool)


        # Create search pipeline
        self.search_pipeline = ChainedTool(
            search_tool_chain,
            tool_parser="legacy",
            tool_start_tag="<query>",
            tool_end_tag="</query>",
            result_start_tag="<snippet>",
            result_end_tag="</snippet>",
            output_formatting="last",
        )

        # Initialize agents
        with LLMToolClient(
            model_name=cfg.react_agent_model_name,
            tokenizer_name=cfg.react_agent_tokenizer_name,
            base_url=cfg.react_agent_base_url,
            api_key=cfg.react_agent_api_key,
        ) as client:
            self.think_agent = ThinkAgent(client=client)
            self.search_agent = SearchAgent(client=client, tools=[self.search_pipeline])
            self.answer_agent = AnswerAgent(client=client)

    def _build_history_string(self, history: List[Dict[str, str]]) -> str:
        """Build a formatted history string from the conversation history"""
        if not history:
            return ""

        return "".join([ele["content"] for ele in history])

    async def __call__(
        self,
        question: str,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        cfg = self.configuration
        assert cfg is not None


        conversation_history = []
        tool_call_history = []
        searched_queries = []

        if verbose:
            print(f"🤔 Starting ReAct workflow for: {question}")
            print(f"📊 Will perform {cfg.num_think_search_cycles} think-search cycles")
            print("=" * 80)

        # Perform N cycles of think-search
        for cycle in range(cfg.num_think_search_cycles):
            cycle_num = cycle + 1

            if verbose:
                print(f"\n🔄 CYCLE {cycle_num}/{cfg.num_think_search_cycles}")
                print("-" * 50)

            # Build current history string for both agents
            history_str = self._build_history_string(conversation_history)

            # THOUGHT: Use ThinkAgent to reason about current situation
            if verbose:
                print("💭 THINKING...")

            think_result = await self.think_agent(
                question=question,
                history=history_str,
                max_tokens=max_tokens,
                temperature=temperature,
                generation_prefix="<think>",
                stop=[], # #"YOUR_TASK_1.2: What tag leads to a stop here?"
            )
            tool_call_history.append(think_result.model_dump())

            think_content = think_result.generated_text
            conversation_history.append({"type": "think", "content": think_content})

            if verbose:
                print(
                    f"💡 Reasoning: {think_content[:200]}{'...' if len(think_content) > 200 else ''}"
                )

            # ACTION: Generate search query based on the thinking
            if verbose:
                print("🔍 GENERATING SEARCH...")

            # Update history string to include the thinking we just did
            history_str = self._build_history_string(conversation_history)

            search_result = await self.search_agent(
                question=question,
                history=history_str,
                searched_queries="\n".join(
                    f"- {ele['content']}"
                    for ele in conversation_history
                    if ele["type"] == "query"
                ),
                max_tokens=max_tokens,
                temperature=temperature,
                generation_prefix="<query>",
            )
            tool_call_history.append(search_result.model_dump())

            search_output = self.search_agent.postprocess_output(search_result)
            conversation_history.append({"type": "query", "content": search_output})

            search_query = self.search_agent.extract_search_query(search_output)
            searched_queries.append(search_query)

        # FINAL ANSWER: Use AnswerAgent to synthesize everything
        if verbose:
            print(f"\n✅ GENERATING FINAL ANSWER...")
            print("-" * 50)

        final_history_str = self._build_history_string(conversation_history)

        answer_result = await self.answer_agent(
            question=question,
            history=final_history_str,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        final_answer = self.answer_agent.postprocess_output(answer_result)

        answer_result.tool_calls = tool_call_history

        return final_answer,answer_result, conversation_history, searched_queries

