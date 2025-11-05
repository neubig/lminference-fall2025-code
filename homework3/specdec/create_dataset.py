import json
from datasets import load_dataset

# --- 1. Load the datasets ---
print("Loading datasets...")
graph_dev = load_dataset("vashistht/11763_datasets", 'graph_dev')
infobench = load_dataset("vashistht/11763_datasets", 'infobench')
mmlu_med = load_dataset("vashistht/11763_datasets", 'mmlu_med')


graph_sys = (
    "You are a graph algorithm expert. Solve the following graph problem. "
    "Think step by step, show your work, and find the requested path(s) and their total weights. "
    "Enclose your final answer in \\boxed{}."
)

mmlu_med_sys = (
    "You are a medical science expert. Answer the following question. "
    "Analyze the medical facts. For any calculations, use standard values (fats: 9 kcal/g, carbs: 4 kcal/g, protein: 4 kcal/g). "
    "Explain your reasoning step by step. Enclose your final answer in \\boxed{}."
)

infobench_sys = (
    "You are an expert in information retrieval and logical reasoning. "
    "Carefully follow the instruction using the provided context."
)

combined_data = []
N_EXAMPLES = 5 # How many examples to take from each


print(f"\nProcessing {N_EXAMPLES} examples from graph_dev...")
for ex in graph_dev["dev_test"].select(range(N_EXAMPLES)):
    system_prompt = graph_sys
    input_prompt = f"Problem:\n{ex['prompt']}"
    
    combined_data.append({
        "task": "graph",
        # "system": system_prompt,
        # "input": input_prompt,
        "formatted_prompt": f"{system_prompt}\n\n{input_prompt}" # New combined field
    })

print(f"Processing {N_EXAMPLES} examples from infobench...")
for ex in infobench["dev_test"].select(range(N_EXAMPLES)):
    system_prompt = infobench_sys
    input_prompt = f"Instruction:\n{ex['instruction']}"

    combined_data.append({
        "task": "infobench",
        "system": system_prompt,
        # "input": input_prompt,
        "formatted_prompt": f"{system_prompt}\n\n{input_prompt}"
    })

print(f"Processing {N_EXAMPLES} examples from mmlu_med...")
for ex in mmlu_med["dev_test"].select(range(N_EXAMPLES)):
    q = ex.get("question", "")
    choices = ex.get("choices", [])

    formatted_choices = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)]) if choices else ""
    text = f"Question:\n{q}\n\nChoices:\n{formatted_choices}" if formatted_choices else f"Question:\n{q}"
    
    system_prompt = mmlu_med_sys
    input_prompt = text.strip()

    combined_data.append({
        "task": "mmlu_med",
        # "system": system_prompt,
        "formatted_prompt": f"{system_prompt}\n\n{input_prompt}"
    })

##################### RANDOM PROMPTS #####################
source = 'random'
prompts = [
    "1+1 is 2. 2+2 is 4. 3+3 is 6. 4+4 is 8. 5+5 is 10. ",
    "1*1 is 1. 2*2 is 4. 3*3 is 9. 4*4 is 16. 5*5 is 25. ",
    "1+1+1+1+1",
    "__________________________________",
    "The quick brown fox jumps over the lazy dog. The quick brown"
]

for prompt in prompts:
    combined_data.append({
        "task": "random",
        # "system": "",
        "formatted_prompt": prompt
    })

output_filename = "specdec/prompts.jsonl"
with open(output_filename, "w") as f:
    for example in combined_data:
        f.write(json.dumps(example) + "\n")