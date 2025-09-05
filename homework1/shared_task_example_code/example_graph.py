#!/usr/bin/env python3
"""
Example usage of the graph path finder system.
"""

from graph_path_finder import *

def main():
    print("=== Graph Path Finding Example ===\n")
    
    # Create a simple example graph
    print("1. Creating a random graph...")
    edges, params = create_random_graph(N=5, M=2, W=50, P=1)
    
    print(f"Graph parameters: N={params['N']}, M={params['M']}, W={params['W']}, P={params['P']}")
    print("Edges:")
    for src, dst, weight in edges:
        print(f"  {src} -> {dst} (weight: {weight})")
    
    # Find the correct solution
    print("\n2. Finding shortest path with dynamic programming...")
    solution = find_top_p_paths(edges, params["N"], params["P"])
    
    print("Shortest path(s):")
    for i, path_info in enumerate(solution.paths, 1):
        print(f"  Path {i}: {path_info.path} (weight: {path_info.weight})")
    
    # Generate prompt for LLM
    print("\n3. Generating LLM prompt...")
    prompt = generate_problem_prompt(edges, params["N"], params["P"])
    print("Prompt preview:")
    print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    
    # Note: Actual LLM querying requires API key
    print("\n4. LLM Integration")
    print("To query an LLM, set environment variables:")
    print("  export LLM_MODEL='gpt-4'")
    print("  export LLM_API_KEY='your-api-key'")
    print("  python graph_path_finder.py")
    
    print("\n=== Example Complete ===")

if __name__ == "__main__":
    main()