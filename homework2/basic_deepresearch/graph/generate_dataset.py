#!/usr/bin/env python3
"""
Generate a dataset of 1000 graph path finding examples.
Each example is saved as a JSON line in a JSONL file.
"""

import json
from graph_path_finder import create_random_graph, find_top_p_paths, generate_problem_prompt
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any


class GraphExample(BaseModel):
    """Complete graph example with problem and solution"""
    id: int
    graph_params: Dict[str, int]  # N, M, W, P
    edges: List[Tuple[int, int, int]]  # (source, target, weight)
    prompt: str
    solution: Dict[str, Any]  # GraphPathSolution as dict


def generate_dataset(num_examples: int = 1000, output_file: str = "graph_dataset.jsonl") -> None:
    """
    Generate a dataset of graph path finding examples.
    
    Args:
        num_examples: Number of examples to generate
        output_file: Output JSONL file path
    """
    print(f"Generating {num_examples} graph path finding examples...")
    
    with open(output_file, 'w') as f:
        for i in range(1, num_examples + 1):
            if i % 100 == 0:
                print(f"Generated {i}/{num_examples} examples...")
            
            # Generate random graph parameters
            import random
            
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
            
            # Generate graph and solution
            edges, params = create_random_graph(N, M, W, P)
            solution = find_top_p_paths(edges, N, P)
            prompt = generate_problem_prompt(edges, N, P)
            
            # Create example
            example = GraphExample(
                id=i,
                graph_params=params,
                edges=edges,
                prompt=prompt,
                solution=solution.model_dump()
            )
            
            # Write to JSONL file
            f.write(example.model_dump_json() + '\n')
    
    print(f"Dataset saved to {output_file}")
    print(f"Generated {num_examples} examples successfully!")


def load_dataset(file_path: str) -> List[GraphExample]:
    """
    Load dataset from JSONL file.
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        List of GraphExample objects
    """
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            example_dict = json.loads(line.strip())
            examples.append(GraphExample(**example_dict))
    return examples


def dataset_statistics(file_path: str) -> None:
    """
    Print statistics about the dataset.
    
    Args:
        file_path: Path to JSONL file
    """
    examples = load_dataset(file_path)
    
    print(f"\n=== Dataset Statistics ===")
    print(f"Total examples: {len(examples)}")
    
    # Graph size distribution
    n_values = [ex.graph_params['N'] for ex in examples]
    print(f"N (nodes): min={min(n_values)}, max={max(n_values)}, avg={sum(n_values)/len(n_values):.1f}")
    
    m_values = [ex.graph_params['M'] for ex in examples]
    print(f"M (edges per node): min={min(m_values)}, max={max(m_values)}, avg={sum(m_values)/len(m_values):.1f}")
    
    w_values = [ex.graph_params['W'] for ex in examples]
    print(f"W (max weight): min={min(w_values)}, max={max(w_values)}, avg={sum(w_values)/len(w_values):.1f}")
    
    # P distribution
    p_counts = {}
    for ex in examples:
        p = ex.graph_params['P']
        p_counts[p] = p_counts.get(p, 0) + 1
    
    print(f"P (paths to find) distribution:")
    for p in sorted(p_counts.keys()):
        percentage = (p_counts[p] / len(examples)) * 100
        print(f"  P={p}: {p_counts[p]} examples ({percentage:.1f}%)")
    
    # Solution statistics
    path_lengths = []
    path_weights = []
    for ex in examples:
        for path_info in ex.solution['paths']:
            path_lengths.append(len(path_info['path']))
            path_weights.append(path_info['weight'])
    
    if path_lengths:
        print(f"Path lengths: min={min(path_lengths)}, max={max(path_lengths)}, avg={sum(path_lengths)/len(path_lengths):.1f}")
        print(f"Path weights: min={min(path_weights)}, max={max(path_weights)}, avg={sum(path_weights)/len(path_weights):.1f}")


if __name__ == "__main__":
    import sys
    
    # Default parameters
    num_examples = 1000
    output_file = "graph_dataset.jsonl"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        num_examples = int(sys.argv[1])
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Generate dataset
    generate_dataset(num_examples, output_file)
    
    # Print statistics
    dataset_statistics(output_file)