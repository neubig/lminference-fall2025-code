"""
Graph Path Finding with Dynamic Programming and LLM Integration

This module provides functionality to:
1. Generate random graphs with specified parameters
2. Find top-P shortest paths using dynamic programming
3. Query language models to solve the same problem
4. Evaluate LLM performance against correct solutions
"""

import random
import heapq
import json
import os
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel
import litellm


class PathInfo(BaseModel):
    """Information about a single path"""
    path: List[int]
    weight: int


class GraphPathSolution(BaseModel):
    """Solution containing top-P paths with their weights"""
    paths: List[PathInfo]


def create_random_graph(N: int, M: int, W: int, P: int) -> Tuple[List[Tuple[int, int, int]], Dict[str, int]]:
    """
    Create a random graph with N nodes, M edges per node, weights up to W, and P paths to find.
    
    Args:
        N: Number of nodes (positive int > 1, zero-indexed)
        M: Number of edges per node (positive int)
        W: Maximum weight range (positive integer, weights are 1 to W inclusive)
        P: Number of paths to find (positive int)
    
    Returns:
        Tuple of (edges_list, parameters_dict)
        - edges_list: List of (source, target, weight) tuples
        - parameters_dict: Dictionary with N, M, W, P values
    """
    edges = []
    
    # For each node i, create edges to nodes i+1 to i+M (wrapping around if needed)
    for i in range(N):
        targets = [(i + j) % N for j in range(1, M + 1)]
        
        # Create edges to all targets
        for target in targets:
            weight = random.randint(1, W)
            edges.append((i, target, weight))
    
    return edges, {"N": N, "M": M, "W": W, "P": P}


def find_top_p_paths(edges: List[Tuple[int, int, int]], N: int, P: int) -> GraphPathSolution:
    """
    Find the top P shortest paths from node 0 to node N-1 using dynamic programming.
    
    Args:
        edges: List of (source, target, weight) tuples
        N: Number of nodes
        P: Number of paths to find
    
    Returns:
        GraphPathSolution containing the top P shortest paths
    """
    # Build adjacency list
    graph = {i: [] for i in range(N)}
    for src, dst, weight in edges:
        graph[src].append((dst, weight))
    
    # Use modified Dijkstra's algorithm to find top P paths
    # Each state is (cost, path)
    pq = [(0, [0])]  # (cost, path)
    paths_found = []
    visited_states = set()
    
    while pq and len(paths_found) < P:
        cost, path = heapq.heappop(pq)
        current_node = path[-1]
        
        # Create a state key to avoid revisiting the same (node, path_length) combination
        state_key = (current_node, len(path))
        if state_key in visited_states:
            continue
        visited_states.add(state_key)
        
        # If we reached the target node, add this path to results
        if current_node == N - 1:
            paths_found.append(PathInfo(path=path, weight=cost))
            continue
        
        # Explore neighbors
        for neighbor, edge_weight in graph[current_node]:
            if neighbor not in path:  # Avoid cycles
                new_cost = cost + edge_weight
                new_path = path + [neighbor]
                heapq.heappush(pq, (new_cost, new_path))
    
    return GraphPathSolution(paths=paths_found)


def generate_problem_prompt(edges: List[Tuple[int, int, int]], N: int, P: int) -> str:
    """
    Generate a problem description that can be used to prompt a language model.
    
    Args:
        edges: List of (source, target, weight) tuples
        N: Number of nodes
        P: Number of paths to find
    
    Returns:
        String containing the problem description
    """
    prompt = f"""You are given a directed graph with {N} nodes (numbered 0 to {N-1}) and the following edges:

Edges (source -> target, weight):
"""
    
    for src, dst, weight in edges:
        prompt += f"{src} -> {dst}, weight: {weight}\n"
    
    prompt += f"""
Find the top {P} shortest path{'s' if P > 1 else ''} from node 0 to node {N-1}.

Return your answer by calling the submit_paths function with:
- paths: A list of paths, where each path is a list of node indices
- weights: A list of corresponding path weights

For example, if the shortest path is [0, 2, 4] with weight 10, call:
submit_paths(paths=[[0, 2, 4]], weights=[10])
"""
    
    return prompt


def query_llm_with_function_call(prompt: str, model: str, api_key: str) -> Dict[str, Any]:
    """
    Query a language model with the given prompt and function calling capability.
    
    Args:
        prompt: The problem description
        model: Model name (e.g., "gpt-4", "litellm_proxy/claude-sonnet-4-20250514")
        api_key: API key for the model
    
    Returns:
        Dictionary containing the function call result
    """
    # For litellm_proxy, we'll pass api_key and base_url directly to completion
    # For other providers, set environment variables
    if not "litellm_proxy" in model.lower():
        if "openai" in model.lower() or "gpt" in model.lower():
            os.environ["OPENAI_API_KEY"] = api_key
        elif "anthropic" in model.lower() or "claude" in model.lower():
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif "sambanova" in model.lower():
            os.environ["SAMBANOVA_API_KEY"] = api_key
        else:
            # Default to OpenAI
            os.environ["OPENAI_API_KEY"] = api_key
    
    # Define the tool schema (modern litellm format)
    tool_schema = {
        "type": "function",
        "function": {
            "name": "submit_paths",
            "description": "Submit the top P shortest paths found",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer"}
                        },
                        "description": "List of paths, where each path is a list of node indices"
                    },
                    "weights": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of corresponding path weights"
                    }
                },
                "required": ["paths", "weights"]
            }
        }
    }
    
    try:
        # Prepare completion arguments
        completion_args = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [tool_schema],
            "tool_choice": {"type": "function", "function": {"name": "submit_paths"}}
        }
        
        # Add base_url and api_key for litellm_proxy
        if "litellm_proxy" in model.lower():
            base_url = os.getenv("LLM_BASE_URL")
            if base_url:
                completion_args["base_url"] = base_url
            completion_args["api_key"] = api_key
        
        response = litellm.completion(**completion_args)
        
        # Extract tool call arguments
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            return function_args
        else:
            return {"paths": [], "weights": []}
    
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return {"paths": [], "weights": []}


def convert_llm_response_to_solution(llm_response: Dict[str, Any]) -> GraphPathSolution:
    """
    Convert LLM response to GraphPathSolution format.
    
    Args:
        llm_response: Dictionary with 'paths' and 'weights' keys
    
    Returns:
        GraphPathSolution object
    """
    paths = []
    llm_paths = llm_response.get("paths", [])
    llm_weights = llm_response.get("weights", [])
    
    for i, path in enumerate(llm_paths):
        weight = llm_weights[i] if i < len(llm_weights) else 0
        paths.append(PathInfo(path=path, weight=weight))
    
    return GraphPathSolution(paths=paths)


def evaluate_solution(correct_solution: GraphPathSolution, predicted_solution: GraphPathSolution, P: int) -> float:
    """
    Evaluate the predicted solution against the correct solution.
    
    Args:
        correct_solution: The correct GraphPathSolution
        predicted_solution: The predicted GraphPathSolution
        P: Number of paths that should be found
    
    Returns:
        Score between 0.0 and 1.0 (number of correct paths / P)
    """
    if not correct_solution.paths or not predicted_solution.paths:
        return 0.0
    
    # Create sets of (path_tuple, weight) for comparison
    correct_paths = {(tuple(path.path), path.weight) for path in correct_solution.paths}
    predicted_paths = {(tuple(path.path), path.weight) for path in predicted_solution.paths}
    
    # Count how many predicted paths match correct paths
    matches = len(correct_paths.intersection(predicted_paths))
    
    return matches / P


def generate_random_examples(num_examples: int) -> List[Dict[str, Any]]:
    """
    Generate random examples with specified constraints.
    
    Args:
        num_examples: Number of examples to generate
    
    Returns:
        List of example dictionaries
    """
    examples = []
    
    for _ in range(num_examples):
        # N between 3 and 20
        N = random.randint(3, 20)
        
        # M between 2 and min(10, N-1)
        M = random.randint(2, min(10, N-1))
        
        # W between 10 and 1000
        W = random.randint(10, 1000)
        
        # P with specified probabilities
        p_choice = random.random()
        if p_choice < 0.7:
            P = 1
        elif p_choice < 0.9:
            P = 2
        else:
            P = 3
        
        edges, params = create_random_graph(N, M, W, P)
        
        examples.append({
            "edges": edges,
            "params": params
        })
    
    return examples


def run_evaluation(examples: List[Dict[str, Any]], model: str, api_key: str) -> Dict[str, Any]:
    """
    Run evaluation on a list of examples.
    
    Args:
        examples: List of example dictionaries
        model: Model name
        api_key: API key
    
    Returns:
        Dictionary containing evaluation results
    """
    results = []
    total_score = 0.0
    
    for i, example in enumerate(examples, 1):
        edges = example["edges"]
        params = example["params"]
        N, P = params["N"], params["P"]
        
        # Get correct solution
        correct_solution = find_top_p_paths(edges, N, P)
        
        # Generate prompt and query LLM
        prompt = generate_problem_prompt(edges, N, P)
        llm_response = query_llm_with_function_call(prompt, model, api_key)
        predicted_solution = convert_llm_response_to_solution(llm_response)
        
        # Evaluate
        score = evaluate_solution(correct_solution, predicted_solution, P)
        total_score += score
        
        results.append({
            "example_id": i,
            "graph_params": params,
            "edges": edges,
            "correct_solution": correct_solution.dict(),
            "predicted_solution": predicted_solution.dict(),
            "score": score
        })
    
    average_score = total_score / len(examples) if examples else 0.0
    
    return {
        "model": model,
        "average_score": average_score,
        "total_examples": len(examples),
        "results": results
    }


if __name__ == "__main__":
    # Example usage
    model = os.getenv("LLM_MODEL", "gpt-4")
    api_key = os.getenv("LLM_API_KEY", "")
    
    if not api_key:
        print("Please set LLM_API_KEY environment variable")
        exit(1)
    
    # Generate test examples
    examples = generate_random_examples(5)
    
    # Run evaluation
    results = run_evaluation(examples, model, api_key)
    
    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Average score: {results['average_score']:.2f}")
    print("Results saved to evaluation_results.json")