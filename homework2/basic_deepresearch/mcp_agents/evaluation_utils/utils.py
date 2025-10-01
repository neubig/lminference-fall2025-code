import os
from typing import Dict, Any
import litellm


# from: https://github.com/centerforaisafety/hle/blob/7b6be5aad6f9b43af3857de7867f3b52f6e4acb3/hle_eval/run_model_predictions.py#L11
QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

# from: https://github.com/centerforaisafety/hle/blob/7b6be5aad6f9b43af3857de7867f3b52f6e4acb3/hle_eval/run_judge_results.py#L16-L33
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()

CHOICE_STRINGS = ["yes", "no"]

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
