# %% [markdown]
# I am assuming we have access two jsonls
# - Student Outptuts
#     - `index`: Unique identifier for each question.
#     - `output`: The model's response to the question.
#
# - Hidden Test Set with the following fields:
#     - `index`: Unique identifier for each question.
#     - `task`: The name of the task (e.g., "mmlu_med").
#     - `prompt`: The question prompt presented to the model.
#     - `gold_answer`: The correct answer to the question.
#     - (Not needed)`meta`: Additional metadata about the question, including unique id in the dataset and other fields.
#
# For this grading logic, we assume we can get the task and other info by essentially grouping by `index` from the hidden test set and joining with the student outputs on `index`.
# ""

# %%
import json
import os
from tqdm import tqdm
from typing import List, Dict, Any
from grader import (
    InfoBenchEvaluator,
    GraphEvaluator,
    MMLUEvaluator,
    ResponseParser,
    evaluate_single
)
import nest_asyncio
nest_asyncio.apply()

# %% [markdown]
# ## create test dataset

# %%
# import re
# import math_verify
# test = math_verify.parse("\\boxed{B}")
# # response = "The answer is (C)."
# response = "The answer is B."
# gt = "B"

# def extract_answer(response: str) -> str:
#     answer_match = re.search(r'The answer is\s*\(?([A-Z])\)?', response, re.IGNORECASE)
#     if answer_match:
#         return [answer_match.group(1).upper()]
#     return []

# answer = extract_answer(response)
# print(answer)
# math_verify.verify(gt, test), math_verify.verify(gt, answer)


# %%
student_outputs_data = [
    # Index 1: Graph (1 path) - CORRECT (function call format)
    {
        "index": 1,
        "output": "To find the shortest path from node 0 to node 9, I'll use Dijkstra's algorithm.\n\nLooking at the edges from node 0: 0->8 has weight 3, which is the smallest.\nFrom node 8: 8->9 has weight 22.\nTotal: 3 + 22 = 25\n\nsubmit_paths(paths=[[0, 8, 9]], weights=[25])"
    },

    # Index 2: Graph (3 paths) - PARTIALLY CORRECT (2 of 3, gold-like format)
    {
        "index": 2,
        "output": "Finding top 3 shortest paths from 0 to 15:\n\n1. 0 -> 7 -> 8 -> 15: 77 + 45 + 108 = 230\n2. 0 -> 4 -> 8 -> 15: 125 + 28 + 108 = 261\n\n{\"paths\": [{\"path\": [0, 7, 8, 15], \"weight\": 230}, {\"path\": [0, 4, 8, 15], \"weight\": 261}]}"
    },

    # Index 3: InfoBench (PyTorch NN) - GOOD (has comments, correct structure)
    {
        "index": 3,
        "output": """```python
import torch
import torch.nn as nn

# Define a two-hidden layer feedforward neural network
class TwoHiddenLayerNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(TwoHiddenLayerNN, self).__init__()

        # First hidden layer with 64 neurons
        self.fc1 = nn.Linear(input_size, 64)

        # Second hidden layer with 64 neurons
        self.fc2 = nn.Linear(64, 64)

        # Output layer
        self.fc3 = nn.Linear(64, output_size)

        # ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass through first hidden layer with ReLU
        x = self.relu(self.fc1(x))

        # Pass through second hidden layer with ReLU
        x = self.relu(self.fc2(x))

        # Output layer (no activation for raw logits)
        x = self.fc3(x)
        return x

# Example usage
model = TwoHiddenLayerNN(input_size=10, output_size=2)
```"""
    },

    # Index 4: InfoBench (Email) - PARTIAL (is email, about salary, but too short/informal)
    {
        "index": 4,
        "output": "Subject: Salary\n\nHi,\n\nI want more money.\n\nThanks"
    },

    # Index 5: MMLU - CORRECT
    {
        "index": 5,
        "output": "Let me analyze each option:\n\n- Glucose: ~4 kcal/gram\n- Palmitic acid (fat): ~9 kcal/gram\n- Leucine (amino acid): ~4 kcal/gram\n- Alcohol: ~7 kcal/gram\n\nFats release the most energy when oxidized. Palmitic acid is a fatty acid.\n\nThe answer is \\boxed{B}"
    },

    # Index 6: MMLU - WRONG (chose A instead of C)
    {
        "index": 6,
        "output": "The patient has elevated lymphocytes, which suggests leukemia. Since they're B-cell origin, it's lymphocytic. The answer is \\boxed{A}"
    }
]
# Save to file
with open("student_outputs.jsonl", "w") as f:
    for item in student_outputs_data:
        f.write(json.dumps(item) + "\n")
print("Created student_outputs.jsonl")

# %%
def load_hidden_test(path: str) -> List[Dict[str, Any]]:
    """Load hidden test JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_student_outputs(path: str) -> Dict[int, str]:
    """Load student outputs JSONL, return dict mapping index -> output."""
    outputs = {}
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                outputs[item["index"]] = item.get("output", "")
    return outputs

# %%
def save_jsonl(data: list, path: str):
    """Save list of dicts to JSONL."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def save_json(data: dict, path: str):
    """Save dict to JSON."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

# %%
# NOTe: Remove later helper for printing metrics

def print_metrics(metrics: dict):
    """Print metrics summary."""
    print("\n" + "=" * 50)
    print(f"RESULTS: {metrics['student_id']}")
    print("=" * 50)
    for task, m in metrics["task_metrics"].items():
        print(f"{task:12s}: {m['accuracy']:.4f} ({m['count']} examples)")
    print("-" * 50)
    print(f"{'OVERALL':12s}: {metrics['overall_accuracy']:.4f}")
    print("=" * 50)


# %% [markdown]
# ## Run Evaluation

# %%
# ============================================================================
# METRICS
# ============================================================================
def calculate_metrics(results: list, student_id: str) -> dict:
    """Calculate task-wise and overall metrics."""
    task_scores = {"mmlu_med": [], "graph": [], "infobench": []}

    for r in results:
        task = r["task"]
        if task in task_scores:
            task_scores[task].append(r["score"])

    metrics = {
        "student_id": student_id,
        "total_examples": len(results),
        "task_metrics": {},
        "overall_accuracy": 0.0
    }

    all_scores = []
    for task, scores in task_scores.items():
        if scores:
            metrics["task_metrics"][task] = {
                "count": len(scores),
                "accuracy": sum(scores) / len(scores),
                "total_score": sum(scores)
            }
            all_scores.extend(scores)

    if all_scores:
        metrics["overall_accuracy"] = sum(all_scores) / len(all_scores)

    return metrics

# %%
def run_eval(
    hidden_test: list,
    student_outputs: dict,
    infobench_evaluator: InfoBenchEvaluator
) -> list:
    """Run evaluation on all test items."""
    results = []

    for idx, test_item in enumerate(tqdm(hidden_test, desc="Evaluating")):
        index = test_item["index"]
        student_response = student_outputs.get(index, "")
        result = evaluate_single(idx, test_item, student_response, infobench_evaluator)
        results.append(result)

    return results


# %% [markdown]
# # RUN EVALUATION

# %%
openai_key = os.getenv("OPENAI_API_KEY")

# %%

# === Configuration ===
HIDDEN_TEST_PATH = "combined_dataset.jsonl"
STUDENT_OUTPUT_PATH = "student_outputs.jsonl"
OUTPUT_DIR = "./eval_results"
STUDENT_ID = "test_student"
EVAL_MODEL = "gpt-5-nano-2025-08-07"

if not openai_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
# === Load data ===
print("Loading data...")
hidden_test = load_hidden_test(HIDDEN_TEST_PATH)
student_outputs = load_student_outputs(STUDENT_OUTPUT_PATH)

print(f"Hidden test size: {len(hidden_test)}")
print(f"Student outputs: {len(student_outputs)}")

# %%
# # === Initialize InfoBench Evaluator ===
print("\nInitializing InfoBench evaluator...")
infobench_evaluator = InfoBenchEvaluator(openai_key, EVAL_MODEL)

# print("Verifying OpenAI connection...")
# if not infobench_evaluator.verify_connection():
#     raise RuntimeError("OpenAI connection failed - cannot proceed")
# print("OpenAI connection verified ✓")


# %%
# === Run Evaluation ===
print(f"\nEvaluating: {STUDENT_ID}")
print("-" * 50)
results = run_eval(hidden_test, student_outputs, infobench_evaluator)

# === Calculate Metrics ===\
metrics = calculate_metrics(results, STUDENT_ID)

# %%
# === Save Results ===
results_path = os.path.join(OUTPUT_DIR, f"{STUDENT_ID}_results.jsonl")
metrics_path = os.path.join(OUTPUT_DIR, f"{STUDENT_ID}_metrics.json")

save_jsonl(results, results_path)
save_json(metrics, metrics_path)

# === Print Summary ===
print_metrics(metrics)
print(f"\nResults saved to: {results_path}")
print(f"Metrics saved to: {metrics_path}")

# %%


# %%



