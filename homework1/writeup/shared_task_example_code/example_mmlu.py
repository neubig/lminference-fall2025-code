#!/usr/bin/env python3
"""
Example usage of the MMLU inference.
"""

from inference import *

def main():

    model = "gpt-4o-mini"
    api_key = os.getenv("OPENAI_API_KEY")
    
    print("=== LLM Inference Example ===\n")
    
    # Create a simple example graph
    print("1. loading dataset...")
    dataset = load_custom_dataset("MMLU-preview")
    
    print(f"Dataset loaded: {len(dataset)} examples")
    
    # Find the correct solution
    print("\n2 Show one example and one solution in the prompt format...")

    example = dataset[0]
    print(format_example(example, include_answer=True))
    
    
    # Generate prompt for LLM
    print("\n3. Generate the prompt and show the evaluation result...")
    prompt = generate_problem_prompt("MMLU", example)
    llm_response = query_llm(prompt, model, api_key)
    print("LLM response: ", llm_response)
    predicted_solution = convert_llm_response_to_solution(llm_response, "MMLU")
    print("Predicted solution: ", predicted_solution)
    score = evaluate_solution(example, predicted_solution, "MMLU", model, api_key)
    print(f"Evaluation result: {score}")
    

if __name__ == "__main__":
    main()