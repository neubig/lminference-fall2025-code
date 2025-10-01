#!/usr/bin/env python3
"""
Example usage of the MMLU inference.
"""

from inference import *

def main():

    model = "gpt-4o-mini"
    api_key = os.getenv("OPENAI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    print("=== LLM Inference Example ===\n")
    
    # Create a simple example graph
    print("1. loading dataset...")
    dataset = load_custom_dataset("InfoBench")
    
    
    print(f"Dataset loaded: {len(dataset)} examples")
    
    # Find the correct solution
    print("\n2. Show one example of the dataset...")

    example = dataset[0]
    print(f"Instruction: {example['instruction']}\nQuestion: {example['input']}\nGeneration:")

    print("\n3. Show the rubrics for the example...")
    print("Rubrics:\n" + "\n".join(example['decomposed_questions']))
    
    
    # Generate prompt for LLM
    print("\n4. Generate the prompt and show the evaluation result...")
    prompt = generate_problem_prompt("InfoBench", example)
    llm_response = query_llm(prompt, model, api_key)
    print("LLM response: ", llm_response)
    predicted_solution = convert_llm_response_to_solution(llm_response, "InfoBench")
    # print("Predicted solution: ", predicted_solution)
    score = evaluate_solution(example, predicted_solution, "InfoBench", model, api_key, openai_api_key)
    print(f"Evaluation result: {score}")
    

if __name__ == "__main__":
    main()