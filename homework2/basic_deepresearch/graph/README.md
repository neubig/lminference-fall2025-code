# Graph Path Finding with Dynamic Programming and LLM Integration

This project implements a comprehensive system for generating random graphs, finding shortest paths using dynamic programming, and evaluating language model performance on graph path-finding problems.

## Overview

The system consists of several key components:

1. **Random Graph Generation**: Creates graphs with specified parameters (N nodes, M edges per node, W weight range, P paths to find)
2. **Dynamic Programming Solution**: Finds the top-P shortest paths from node 0 to node N-1
3. **LLM Integration**: Queries language models to solve the same problems using function calling
4. **Evaluation System**: Compares LLM solutions against correct answers and provides accuracy scores

## Features

### Graph Generation
- Creates directed graphs with N nodes (zero-indexed)
- Each node i has edges to nodes i+1 through i+M (with wraparound)
- Edge weights are randomly assigned between 1 and W (inclusive)
- Supports finding top-P shortest paths

### Dynamic Programming Algorithm
- Uses a modified Dijkstra's algorithm to find multiple shortest paths
- Avoids cycles by tracking visited nodes in each path
- Returns paths sorted by total weight (shortest first)

### LLM Integration
- Supports multiple model providers (OpenAI, Anthropic, litellm_proxy, etc.)
- Uses modern function calling with tools parameter
- Handles different API authentication methods
- Robust error handling for API failures

### Evaluation System
- Generates random test cases with specified constraints
- Compares LLM predictions against ground truth solutions
- Provides detailed scoring (number of correct paths / P)
- Saves comprehensive results in JSON format

## Installation

```bash
pip install pydantic litellm
```

## Usage

### Basic Example

```python
from graph_path_finder import *

# Create a random graph
edges, params = create_random_graph(N=5, M=3, W=100, P=1)

# Find correct solution
solution = find_top_p_paths(edges, params["N"], params["P"])

# Generate LLM prompt
prompt = generate_problem_prompt(edges, params["N"], params["P"])

# Query LLM (requires API key)
llm_response = query_llm_with_function_call(prompt, "gpt-4", "your-api-key")
predicted_solution = convert_llm_response_to_solution(llm_response)

# Evaluate
score = evaluate_solution(solution, predicted_solution, params["P"])
```

### Running Full Evaluation

```bash
export LLM_MODEL="gpt-4"  # or "litellm_proxy/claude-sonnet-4-20250514"
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://your-proxy-url"  # if using litellm_proxy

python graph_path_finder.py
```

## Configuration

### Environment Variables

- `LLM_MODEL`: Model name (e.g., "gpt-4", "litellm_proxy/claude-sonnet-4-20250514")
- `LLM_API_KEY`: API key for the model
- `LLM_BASE_URL`: Base URL for litellm_proxy (optional)

### Random Generation Parameters

The `generate_random_examples()` function creates test cases with:
- N: 3 to 20 nodes
- M: 2 to min(10, N-1) edges per node
- W: 10 to 1000 weight range
- P: 1 (70% probability), 2 (20% probability), 3 (10% probability)

## API Reference

### Core Functions

#### `create_random_graph(N, M, W, P)`
Creates a random directed graph.

**Parameters:**
- `N` (int): Number of nodes (> 1)
- `M` (int): Number of edges per node
- `W` (int): Maximum edge weight
- `P` (int): Number of paths to find

**Returns:**
- `Tuple[List[Tuple[int, int, int]], Dict[str, int]]`: (edges, parameters)

#### `find_top_p_paths(edges, N, P)`
Finds the top P shortest paths using dynamic programming.

**Parameters:**
- `edges` (List[Tuple[int, int, int]]): List of (source, target, weight) tuples
- `N` (int): Number of nodes
- `P` (int): Number of paths to find

**Returns:**
- `GraphPathSolution`: Object containing the shortest paths

#### `query_llm_with_function_call(prompt, model, api_key)`
Queries a language model with function calling.

**Parameters:**
- `prompt` (str): Problem description
- `model` (str): Model name
- `api_key` (str): API key

**Returns:**
- `Dict[str, Any]`: Function call response with paths and weights

#### `evaluate_solution(correct_solution, predicted_solution, P)`
Evaluates predicted solution against correct solution.

**Parameters:**
- `correct_solution` (GraphPathSolution): Ground truth
- `predicted_solution` (GraphPathSolution): LLM prediction
- `P` (int): Number of paths expected

**Returns:**
- `float`: Score between 0.0 and 1.0

### Data Models

#### `PathInfo`
```python
class PathInfo(BaseModel):
    path: List[int]      # List of node indices
    weight: int          # Total path weight
```

#### `GraphPathSolution`
```python
class GraphPathSolution(BaseModel):
    paths: List[PathInfo]  # List of paths with weights
```

## Algorithm Details

### Graph Structure
- Nodes are zero-indexed from 0 to N-1
- Each node i connects to nodes (i+1) % N, (i+2) % N, ..., (i+M) % N
- Self-loops are avoided
- Edge weights are uniformly random between 1 and W

### Path Finding Algorithm
1. Uses priority queue with (cost, path) tuples
2. Starts from node 0, targets node N-1
3. Explores neighbors while avoiding cycles
4. Tracks visited (node, path_length) states to prevent redundant exploration
5. Returns first P paths that reach the target node

### LLM Function Calling
The system uses modern tool calling format:
```python
tool_schema = {
    "type": "function",
    "function": {
        "name": "submit_paths",
        "description": "Submit the top P shortest paths found",
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                "weights": {"type": "array", "items": {"type": "integer"}}
            },
            "required": ["paths", "weights"]
        }
    }
}
```

## Example Output

```json
{
  "model": "gpt-4",
  "average_score": 0.80,
  "total_examples": 5,
  "results": [
    {
      "example_id": 1,
      "graph_params": {"N": 5, "M": 3, "W": 100, "P": 1},
      "score": 1.0,
      "correct_solution": {
        "paths": [{"path": [0, 2, 4], "weight": 45}]
      },
      "predicted_solution": {
        "paths": [{"path": [0, 2, 4], "weight": 45}]
      }
    }
  ]
}
```

## Error Handling

The system includes robust error handling for:
- API failures and timeouts
- Invalid LLM responses
- Missing function call results
- Graph connectivity issues
- Invalid parameter combinations

## Supported Models

- OpenAI models (GPT-3.5, GPT-4, etc.)
- Anthropic models (Claude variants)
- Any model accessible through litellm_proxy
- Custom models via environment variable configuration

## Performance Notes

- Graph generation is O(N*M) in time complexity
- Path finding is O(V*E*log(V)) where V=nodes, E=edges
- Memory usage scales with the number of paths explored
- LLM query time depends on model and API latency

## Limitations

- Assumes graphs are connected (paths exist from 0 to N-1)
- Does not handle negative edge weights
- Path finding may be slow for very large graphs or high P values
- LLM performance varies significantly by model and problem complexity