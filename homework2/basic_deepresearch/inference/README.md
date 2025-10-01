# Inference with LLMs on various tasks

This project implements a comprehensive evaluation system for [MMLU](https://arxiv.org/abs/2009.03300) and [InfoBench](https://arxiv.org/abs/2401.03601).

## Overview

The system consists of several key components:

1. **Data Loading**: Loads the data from HugginFace: [MMLU](https://huggingface.co/datasets/cais/mmlu) and [InfoBench](https://huggingface.co/datasets/kqsong/InFoBench)
2. **LLM Integration**: Queries language models to solve the problems following their original formats. Reference code: [MMLU](https://github.com/hendrycks/test/tree/master), [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro/tree/main) and [InfoBench](https://github.com/qinyiwei/InfoBench/blob/main)
3. **Evaluation System**: Evaluate LLM solutions by comparing with the correct answers (MMLU) or given rubrics (InfoBench)


## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```bash
export OPENAI_API_KEY="your-api-key"

python example_mmlu.py
python example_infobench.py
```

```python
from inference import *

# load the data
dataset = load_custom_dataset("MMLU-preview")

# show an example
example = dataset[0]
print(format_example(example, include_answer=True))

# Generate LLM prompt
prompt = generate_problem_prompt("MMLU", example)
    

# Query LLM (requires API key)
llm_response = query_llm(prompt, "gpt-4o-mini", "you-api-key")
predicted_solution = convert_llm_response_to_solution(llm_response, "MMLU")

# Evaluate
score = evaluate_solution(example, predicted_solution, "MMLU", model, api_key)
```

### Running Full Evaluation

```bash
export LLM_MODEL="gemini/gemini-2.5-flash"  # or "litellm_proxy/claude-sonnet-4-20250514"
export LLM_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-api-key" # For consistent evaluation of InfoBench rubrics, we use gpt-5-nano-2025-08-07
export LLM_BASE_URL="https://your-proxy-url"  # if using litellm_proxy
export TASK="MMLU" # MMLU or InfoBench

python inference.py
```

## Configuration

### Environment Variables

- `LLM_MODEL`: Model name (e.g., "gemini/gemini-2.5-flash", "litellm_proxy/claude-sonnet-4-20250514")
- `LLM_API_KEY`: API key for the model
- `OPENAI_API_KEY`: OpenAI API key for the rubric evaluator
- `LLM_BASE_URL`: Base URL for litellm_proxy (optional)

## Example Output

```json
{
  "model": "gemini/gemini-2.5-flash",
  "average_score": 1.0,
  "total_examples": 5,
  "results": [
    {
      "example_id": 1,
      "example": {
        "question": "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
        "subject": "abstract_algebra",
        "choices": [
          "0",
          "4",
          "2",
          "6"
        ],
        "answer": 1
      },
      "predicted_solution": "B",
      "score": true
    }]
}
```

## Error Handling

The system includes robust error handling for:
- API failures and timeouts
- Invalid LLM responses
- Invalid parameter combinations

## Supported Models

- OpenAI models (GPT-3.5, GPT-4, etc.)
- Google models (Gemini variants)
- Any model accessible through litellm_proxy
- Custom models via environment variable configuration

## Limitations

- Do not handle API rate limit
- LLM performance varies significantly by model and problem complexity
