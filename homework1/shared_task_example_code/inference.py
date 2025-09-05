import json
import re
import os
import time
from typing import Dict, Any, List
from tqdm import tqdm

import litellm
from openai import OpenAI
from datasets import load_dataset


choices = ["A", "B", "C", "D"]

SYS_MSG ="Based on the provided Input (if any) and Generated Text, answer the ensuing Questions with either a YES or NO choice. Your selection should be based on your judgment as well as the following rules:\n\n- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. However, note that even minor inaccuracies exclude the text from receiving a 'YES' rating. As an illustration. consider a question that asks. \"Does each sentence in the generated text use a second person?” If even one sentence does not use the second person, the answer should NOT be 'YES'. To qualify for a 'YES' rating, the generated text must be entirely accurate and relevant to the question\n\n- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question. For instance, if the question asks. \"Is the second sentence in the generated text a compound sentence?\" and the generated text only has one sentence. it offers no relevant information to answer the question. Consequently, the answer should be 'NO'.'''"


def load_custom_dataset(dataset_name: str):
    # load MMLU and InfoBench from HugginFace
    if dataset_name == "MMLU-preview":
        raw_dataset = load_dataset("cais/mmlu","college_medicine")
        dataset = list(raw_dataset['test'])
    elif dataset_name == "MMLU":
        dataset = []
        college_med = load_dataset("cais/mmlu", "college_medicine")
        professional_med = load_dataset("cais/mmlu", "professional_medicine")
        dataset.extend(college_med['test'])
        dataset.extend(professional_med['test'])
    elif dataset_name == "InfoBench":
        raw_dataset = load_dataset("kqsong/InFoBench")
        dataset = list(raw_dataset['train'])
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    return dataset


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(example, include_answer=False):
    prompt = f"Question: {example['question']}\n Options:"
    these_choices = example["choices"]

    for i in range(len(these_choices)):
        prompt += f"\n{choices[i]}. {these_choices[i]}"

    prompt += "\nAnswer:"   
    if include_answer:
        # for in-context learning
        prompt += f" {choices[example['answer']]}\n\n"
    return prompt


def extract_answer(text):
    # remove the latex box, common for AIME
    text = re.sub(r'\$\\boxed\{([A-Za-z])\}\$', r'\1', text)

    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        pattern = r"option \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return None


def generate_problem_prompt(dataset_name: str, example: str) -> str:
    if dataset_name == "MMLU":
        # https://github.com/hendrycks/test/blob/master/evaluate.py
        prompt = f"The following is a multiple choice question (with answers) about {format_subject(example['subject'])}.  Output the answer in the format of \"The answer is (X)\" at the end.\n\n"
        return prompt + format_example(example, include_answer=False)
    
    elif dataset_name == "InfoBench":
        # https://arxiv.org/pdf/2401.03601
        return f"Instruction: {example['instruction']}\nQuestion: {example['input']}\nGeneration:"
    else:
        raise ValueError(f"Dataset {dataset_name} not found")


def query_llm(prompt: str, model: str, api_key: str, prompt_mode=True) -> Dict[str, Any]:
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
    
    try:
        # Prepare completion arguments
        if prompt_mode:
            completion_args = {
                "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                }
        else:
                # prompts are formatted messages already
                completion_args = {
                    "model": model,
                    "messages": prompt,
                }
        
        # Add base_url and api_key for litellm_proxy
        if "litellm_proxy" in model.lower():
            base_url = os.getenv("LLM_BASE_URL")
            if base_url:
                completion_args["base_url"] = base_url
            completion_args["api_key"] = api_key
        
        response = litellm.completion(**completion_args)
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return None


def convert_llm_response_to_solution(llm_response: str, dataset_name: str) -> str:
    if dataset_name == "MMLU":
        # adapted from https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py
        return extract_answer(llm_response.replace('**', ''))
    elif dataset_name == "InfoBench":
        return llm_response
    else:
        raise ValueError(f"Dataset {dataset_name} not found")


def bool_ratio(bool_results: List[bool]) -> float:
    "Calculate true false ratio for eval results"
    count = {"true":0, "false":0}
    for entry in bool_results:
        if entry:
            count["true"] += 1
        else:
            count["false"] += 1
        
    return count['true']/sum(count.values())


def info_bench_eval(example: str, predicted_solution: str, model: str, api_key: str, openai_api_key: str) -> float:
    # https://github.com/qinyiwei/InfoBench/blob/main/evaluation.py
    message = []
    answer = ""
    input_task = example['input']
    output = predicted_solution
    client = OpenAI(api_key=openai_api_key)

    for question in example["decomposed_questions"]:
        if len(message) == 0:
            if input_task:
                content =  f"{SYS_MSG}\n\nInput:\n\"{input_task}\"\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
            else:
                content =  f"{SYS_MSG}\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
        else:
            content = f"{question}\n"
        message.append({"role": "user", "content": content})
        # create a chat completion
        success = False
        early_stop = True
        while not success:
            try:
                # default config
                temperature = 1.0
                eval_model = "gpt-5-nano-2025-08-07"

                completion = client.chat.completions.create(
                        model=eval_model,
                        messages=message,
                        temperature=temperature,
                    )
                generation = completion.choices[0].message.content
                message.append(
                        {"role": "assistant", "content": generation})
                # check if generation is yes or no
                if generation.lower().startswith("yes") or generation.lower().startswith("no"):
                    if generation.lower().startswith("yes"):
                        answer += "Yes\n"
                    else:
                        answer += "No\n"
                else:
                    if "YES" in generation and "NO" not in generation:
                        answer += "Yes\n"
                    elif "YES" not in generation and "NO" in generation:
                        answer += "No\n"
                    else:
                        for msg in message:
                            print(msg['content'])
                        print("NO YES or NO answer!" + generation)
                        answer += "None\n"
                        early_stop = True
                        break
                success = True
            except Exception as e:
                print("ERROR!")
                print(e)
                print("Retry!")
                time.sleep(5)

            # when no answer occurs, break the loop and continue to next instance
            if early_stop:
                break

    answer = answer[:-1]
    # save eval results as List[bool]
    bool_results = []
    for i in answer.split('\n'):
        if i == "Yes":
            bool_results.append(True)
        elif i == "No":
            bool_results.append(False)
        else:
            bool_results.append(None)

    return bool_ratio(bool_results)


def evaluate_solution(example: str, predicted_solution: str, dataset_name: str, model: str, api_key: str, openai_api_key: str="") -> float:
    if dataset_name == "MMLU":
        return choices[example["answer"]] == predicted_solution
    elif dataset_name == "InfoBench":
        # https://github.com/qinyiwei/InfoBench/blob/main/evaluation.py
        return info_bench_eval(example, predicted_solution, model, api_key, openai_api_key)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")


def run_evaluation(examples: List[Dict[str, Any]], model: str, api_key: str, task: str, openai_api_key: str="<your-openai-api-key>") -> Dict[str, Any]:
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
    
    for i, example in tqdm(enumerate(examples, 1), total=len(examples), desc="Evaluating examples"):
        prompt = generate_problem_prompt(task, example)
        llm_response = query_llm(prompt, model, api_key)
        predicted_solution = convert_llm_response_to_solution(llm_response, task)
        
        # Evaluate
        score = evaluate_solution(example, predicted_solution, task, model, api_key, openai_api_key)

        total_score += score
        
        results.append({
            "example_id": i,
            "example": example,
            "predicted_solution": predicted_solution,
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
    model = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash")
    # model = "gpt-4o-mini"
    
    # api_key = os.getenv("LLM_API_KEY", "")
    openai_api_key =  os.getenv("OPENAI_API_KEY")
    api_key = os.getenv("GOOGLE_API_KEY")

    task = os.getenv("TASK", "MMLU") # "MMLU" or "InfoBench"
    debug = True # True or False
    
    if not api_key:
        print("Please set LLM_API_KEY environment variable")
        exit(1)
    
    # Generate test examples
    examples = load_custom_dataset(task)
    
    # Run evaluation
    if debug:
        examples = examples[:5]
    results = run_evaluation(examples, model, api_key, task, openai_api_key)
    
    # Save results
    debug_str = "debug" if debug else "release"

    model_display = model.split("/")[-1]

    with open(f"evaluation_results_{model_display}_{task}_{debug_str}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Average score: {results['average_score']:.2f}")
    print(f"Results saved to evaluation_results_{model_display}_{task}_{debug_str}.json")