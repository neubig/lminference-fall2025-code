from inference import *
from datasets import load_dataset

import numpy as np
from typing import List, Dict, Any, Tuple


# NOTE: Keep in mind the real data will come from a hidden test set
graph = load_dataset('vashistht/11763_datasets', 'graph_dev', split='dev_test')
infobench = load_dataset('vashistht/11763_datasets', 'infobench', split='dev_test')
mmlu_med = load_dataset('vashistht/11763_datasets', 'mmlu_med', split='dev_test')

graph_prompts = list(graph['prompt'])
graph_messages = [[{"role": "user", "content": prompt}] for prompt in graph_prompts]

infobench_prompts = [generate_problem_prompt("InfoBench", example) for example in infobench]
infobench_messages = [[{"role": "user", "content": prompt}] for prompt in infobench_prompts]

mmlu_prompts = [generate_problem_prompt("MMLU", example) for example in mmlu_med]
mmlu_messages = [[{"role": "user", "content": prompt}] for prompt in mmlu_prompts]

all_prompts = graph_prompts + infobench_prompts + mmlu_prompts


def simulate_poisson_batch_arrivals_exhaustive(
    prompts: List[str],
    total_time_seconds: float,
    mean_batch_size: float = 4.0,
    batch_size_distribution: str = "poisson",
    min_batch_size: int = 1,
    max_batch_size: int = None,
    random_seed: int = None
) -> List[Dict[str, Any]]:
    """
    Simulate batch arrivals ensuring ALL prompts are used.
    Adjusts the arrival rate to fit all prompts within the time window.
    
    Args:
        prompts: List of prompts to sample from (without replacement)
        total_time_seconds: Total simulation time in seconds
        mean_batch_size: Average number of requests per batch
        batch_size_distribution: How to determine batch sizes ("poisson" or "uniform")
        min_batch_size: Minimum batch size
        max_batch_size: Maximum batch size
        random_seed: Random seed for reproducibility
        
    Returns:
        List of batch dictionaries (same format as simulate_poisson_batch_arrivals)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if max_batch_size is None:
        max_batch_size = len(prompts)
    
    # Create a shuffled copy
    remaining_idxs = list(range(len(prompts)))#prompts.copy()
    np.random.shuffle(remaining_idxs)
    
    # Estimate number of batches needed
    estimated_batches = int(np.ceil(len(prompts) / mean_batch_size))
    
    # Calculate mean arrival rate to fit all batches in time window
    # Leave some buffer at the end
    mean_arrival_rate = estimated_batches / (total_time_seconds * 0.95)
    
    batches = []
    batch_id = 0
    current_time = 0.0
    
    while remaining_idxs:
        # Generate inter-arrival time
        if len(batches) == 0:
            # First batch arrives immediately or with small delay
            inter_arrival_time = np.random.exponential(0.5 / mean_arrival_rate)
        else:
            inter_arrival_time = np.random.exponential(1.0 / mean_arrival_rate)
        
        current_time += inter_arrival_time
        
        # Determine batch size
        if batch_size_distribution == "poisson":
            batch_size = np.random.poisson(mean_batch_size)
            batch_size = max(min_batch_size, min(batch_size, max_batch_size))
        elif batch_size_distribution == "uniform":
            batch_size = np.random.randint(min_batch_size, max_batch_size + 1)
        else:
            raise ValueError(f"Unknown batch_size_distribution: {batch_size_distribution}")
        
        # Don't exceed remaining prompts
        batch_size = min(batch_size, len(remaining_idxs))
        
        if batch_size == 0:
            continue
        
        # Extract prompts for this batch
        batch_idxs = remaining_idxs[:batch_size]
        remaining_idxs = remaining_idxs[batch_size:]

        batch_prompts = [prompts[idx] for idx in batch_idxs]

        sample_messages = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
        
        # Create batch record
        batch = {
            "batch_id": batch_id,
            "arrival_time": current_time,
            "batch_size": batch_size,
            "prompt_idxs": batch_idxs,
            "prompts": batch_prompts,
            "max_length": 2048,
        }
        batches.append(batch)
        batch_id += 1
    
    print(f"Generated {len(batches)} batches over {current_time:.2f} seconds")
    print(f"Target time was {total_time_seconds:.2f} seconds")
    
    return batches

# each system will be tested with the same simulation. The final simulation is not guaranteed to have the same 
# overall request frequency and batch shape distribution, but it will be within the same ballpark and at least 
# not significantly harder
batches = simulate_poisson_batch_arrivals_exhaustive(prompts=all_prompts,
                                                     total_time_seconds=60.0*20,
                                                     mean_batch_size=3,
                                                     batch_size_distribution="poisson",
                                                     min_batch_size=1,
                                                     max_batch_size=8,
                                                     )
# with open("batch_arrivals.json", "w") as f: # TODO: uncomment to generate your own
#     json.dump(batches, f)

batches_sample = simulate_poisson_batch_arrivals_exhaustive(prompts=all_prompts[5::20],
                                                     total_time_seconds=60.0,
                                                     mean_batch_size=3,
                                                     batch_size_distribution="poisson",
                                                     min_batch_size=1,
                                                     max_batch_size=8,
                                                     )

# with open("batch_arrivals_sample.json", "w") as f: # TODO: uncomment to generate your own
#     json.dump(batches_sample, f)

print(batches)
print(batches_sample)