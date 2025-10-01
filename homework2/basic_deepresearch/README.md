# Basic Deep Research Agent

This task requires you to implement a simple [FastMCP-based](https://github.com/jlowin/fastmcp) Deep Research Agent with [ReAct](https://github.com/ysymyth/ReAct). The evaluation will be based on MMLU.

## Overview

Your task will be elicited in `simple_react.ipynb`

1. Write some code in `react_agent.py` to make the agent work `(YOUR_TASK_1.1, YOUR_TASK_1.2)`
2. Check the graph shared task pipeline in `simple_react.ipynb` `(YOUR_TASK_2.1)`. Write some code of `GraphPathEvaluationTool` in `mcp_agents.tool_interface.mcp_tools` to make the pipeline work  `(YOUR_TASK_2.1.1, YOUR_TASK_2.1.2, YOUR_TASK_2.1.3)`
3. Write some code in `simple_react.ipynb` to make the MMLU pipline work `(YOUR_TASK_2.2)`
4. Write some simple analysis code for `count_tokens_in_results(...)` to report the statistics on a small subset of the dataset `(YOUR_TASK_3.1, YOUR_TASK_3.2)`

## Evaluation of the system

### Pipeline
First, set up the environment variables and the MCP server. 

```bash
conda create -n hw2-dr python=3.10 -y && conda activate hw2-dr
pip install -r requirements.txt

export SERPER_API_KEY=<your_serper_api_key>
export OPENAI_API_KEY=<your_openai_api_key>

python -m mcp_agents.mcp_backend.main --port 8000
```

Please also put the API keys in the `.yaml` files, as well as `simple_react.ipynb`

Then, you can simply follow the instructions in the notebook for the tasks

### Notes
1. In TA's test run, each MMLU example takes 1 minute to run with `gpt-4o-mini`

2. Alternatively, you can use your own VLLM served model to replace `gpt-4o`. In that case, change the base URLs, model names, etc in your `.yaml`, e.g., if you serve an `Qwen3-8B`, you can rewrite the `.yaml` with `"http://localhost:30001/v1"`, `"Qwen/Qwen3-8B"`, etc.

## Acknowledgements
This repository is partially adapted from the infrastructure written by [Shannon Shen](https://www.szj.io/), a great PhD student from MIT.
