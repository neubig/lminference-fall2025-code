# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Temperature Sampling with GPT-2
#
# By Graham Neubig for [11-664/763 Inference Algorithms for Language Modeling](https://phontron.com/class/lminference-fall2025/)
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neubig/lminference-fall2025-code/blob/main/02-generation-basics/generation.ipynb)
#
# This notebook explores temperature sampling, the most fundamental technique for controlling randomness in text generation. Temperature sampling modifies the probability distribution over next tokens by scaling the logits before applying softmax.
#
# **Temperature Values and Their Effects:**
# - **T = 0.0**: Greedy sampling (deterministic, always picks most likely token)
# - **T = 0.5**: Conservative sampling (focused, coherent, less creative)
# - **T = 1.0**: Standard sampling (balanced creativity and coherence)
# - **T = 1.5**: Creative sampling (diverse, more surprising outputs)
#
# **Mathematical Foundation:**
# Temperature sampling scales logits by 1/T before applying softmax: `P(token) = softmax(logits / T)`
# - Lower T makes the distribution more peaked (focused on likely tokens)
# - Higher T makes the distribution more uniform (considers more options)
# - T = 0 reduces to greedy decoding (argmax)

# %%
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    from litellm import completion  # type: ignore
except ImportError:
    completion = None

from nanogpt import GPT2, GPT2Tokenizer  # type: ignore[attr-defined]

# %% [markdown]
# ## 1. Text Generation with Temperature Sampling
#
# The core implementation involves two key functions:
# 1. `temperature_sample()`: Applies temperature scaling to logits and samples a token
# 2. `generate_with_temperature()`: Generates a complete text sequence using temperature sampling
#
# Temperature scaling works by dividing logits by the temperature value before applying softmax.
# This changes the "sharpness" of the probability distribution without changing the relative ordering of tokens.


# %%
def temperature_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Sample from logits using temperature scaling.

    Args:
        logits: Raw model outputs [vocab_size]
        temperature: Sampling temperature (0.0 = greedy, higher = more random)

    Returns:
        Sampled token index
    """
    if temperature == 0.0:
        # Greedy sampling: always pick the most likely token
        return torch.argmax(logits, dim=-1)

    # Scale logits by temperature and sample
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze()


def generate_with_temperature(
    model: GPT2,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    temperature: float,
    max_length: int = 30,
) -> str:
    """
    Generate text with specified temperature.

    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        prompt: Input text to continue
        temperature: Sampling temperature
        max_length: Maximum number of tokens to generate

    Returns:
        Generated text (including original prompt)
    """
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)])

    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(input_ids)
            next_token_logits = logits[0, -1, :]
            next_token = temperature_sample(next_token_logits, temperature)
            input_ids = torch.cat(
                [input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1
            )

    return tokenizer.decode(input_ids[0].tolist())


def main() -> None:
    """Run the temperature demonstration and evaluation."""
    # %%
    # Demonstrate temperature effects with real GPT-2 model
    print("Loading GPT-2 model...")
    model = GPT2.from_pretrained("gpt2")
    model.to("cpu")
    model.eval()
    tokenizer = GPT2Tokenizer()

    prompt = "The future of artificial intelligence is"
    temperatures = [0.0, 0.5, 1.0, 1.5]

    print(f"Prompt: '{prompt}'\n")

    for temp in temperatures:
        generated = generate_with_temperature(
            model, tokenizer, prompt, temp, max_length=25
        )
        generated_part = generated[len(prompt) :].strip()
        temp_label = "Greedy" if temp == 0.0 else f"T={temp}"
        print(f"{temp_label:8}: {generated_part}")

    # Run evaluation and show results
    try:
        results = run_evaluation()

        # Group by temperature and calculate averages
        temp_groups = {}
        for result in results:
            if result.temperature not in temp_groups:
                temp_groups[result.temperature] = []
            temp_groups[result.temperature].append(result)

        print("\nAverage metrics by temperature:")
        for temp in sorted(temp_groups.keys()):
            group = temp_groups[temp]
            avg_diversity = sum(r.diversity for r in group) / len(group)
            avg_time = sum(r.generation_time for r in group) / len(group)
            avg_fluency = sum(r.fluency_score for r in group) / len(group)

            temp_label = "Greedy" if temp == 0.0 else f"T={temp}"
            print(
                f"{temp_label:8}: Diversity={avg_diversity:.3f}, Time={avg_time:.3f}s, Fluency={avg_fluency:.1f}/10"
            )

        print("\nSample outputs:")
        for temp in sorted(temp_groups.keys()):
            example = temp_groups[temp][0]
            temp_label = "Greedy" if temp == 0.0 else f"T={temp}"
            print(
                f"{temp_label}: '{example.generated_text[:50]}...' (Fluency: {example.fluency_score:.1f})"
            )

        print("\nKey Findings:")
        print("- Diversity generally increases with temperature")
        print("- Generation speed is relatively consistent across temperatures")
        print("- Fluency scores help identify the optimal temperature range")

        # Find optimal temperature based on fluency
        best_temp = max(
            temp_groups.keys(),
            key=lambda t: sum(r.fluency_score for r in temp_groups[t])
            / len(temp_groups[t]),
        )
        best_fluency = sum(r.fluency_score for r in temp_groups[best_temp]) / len(
            temp_groups[best_temp]
        )
        print(f"- Highest average fluency: T={best_temp} ({best_fluency:.1f}/10)")

    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nTo run fluency evaluation, set these environment variables:")
        print("- LLM_API_KEY: Your API key")
        print("- LLM_MODEL: Model name (required)")
        print("- LLM_BASE_URL: Base URL (optional, for custom endpoints)")
        print("\nExample:")
        print("export LLM_API_KEY='your-api-key-here'")
        print("export LLM_MODEL='gpt-3.5-turbo'")
    except Exception as e:
        print(f"Evaluation Error: {e}")
        print("Check your API configuration and try again.")


if __name__ == "__main__":
    main()

# %% [markdown]
# ## 2. Evaluation and Analysis
#
# To systematically compare temperature settings, we need quantitative metrics:
# 1. **Diversity**: Ratio of unique words to total words (higher = more diverse)
# 2. **Generation Speed**: Time taken to generate text
# 3. **Fluency**: Quality assessment using an external API (optional)
#
# We'll run experiments across multiple prompts and temperature values to understand the trade-offs.


# %%
@dataclass
class GenerationResult:
    """Results from a text generation experiment."""

    temperature: float
    prompt: str
    generated_text: str
    diversity: float
    generation_time: float
    fluency_score: float


def calculate_diversity(text: str) -> float:
    """
    Calculate text diversity as unique word ratio.

    Args:
        text: Generated text to analyze

    Returns:
        Ratio of unique words to total words (0.0 to 1.0)
    """
    words = text.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def evaluate_fluency(text: str) -> float:
    """
    Evaluate text fluency using an external API.

    Args:
        text: Text to evaluate

    Returns:
        Fluency score (0-10)

    Raises:
        ValueError: If required environment variables are not set
        Exception: If API call fails
    """
    # Get configuration from environment variables
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    base_url = os.getenv("LLM_BASE_URL")

    if not api_key:
        raise ValueError(
            "LLM_API_KEY environment variable is required for fluency evaluation"
        )

    if not model:
        raise ValueError(
            "LLM_MODEL environment variable is required for fluency evaluation"
        )

    if completion is None:
        raise ValueError("litellm package is required for fluency evaluation")

    try:
        # Prepare completion arguments
        completion_args = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Rate the fluency and coherence of this text on a scale of 0-10 (10 = perfect). Only respond with a number: '{text}'",
                }
            ],
            "api_key": api_key,
            "max_tokens": 5,
        }

        # Add base URL if provided
        if base_url:
            completion_args["base_url"] = base_url

        response = completion(**completion_args)
        score_text = response.choices[0].message.content.strip()  # type: ignore

        # Extract numeric score
        try:
            score = float(score_text)
            # Clamp score to valid range
            return max(0.0, min(10.0, score))
        except ValueError:
            # If response isn't a number, try to extract it
            import re

            numbers = re.findall(r"\d+\.?\d*", score_text)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(10.0, score))
            else:
                raise ValueError(
                    f"Could not parse fluency score from response: {score_text}"
                ) from None

    except Exception as e:
        raise Exception(f"Fluency evaluation failed: {str(e)}") from e


def run_evaluation() -> list[GenerationResult]:
    """
    Run systematic evaluation across temperatures and prompts.

    Returns:
        List of GenerationResult objects

    Raises:
        ValueError: If LLM configuration is missing
    """
    # Check for required LLM configuration
    api_key = os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL")
    base_url = os.getenv("LLM_BASE_URL")

    if not api_key:
        raise ValueError("LLM_API_KEY environment variable is required for evaluation")

    if not model_name:
        raise ValueError("LLM_MODEL environment variable is required for evaluation")

    print(f"Using LLM: {model_name}")
    if base_url:
        print(f"Base URL: {base_url}")

    # Load GPT-2 model and tokenizer
    model = GPT2.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer()

    # Test prompts representing different use cases
    prompts = [
        "The future of artificial intelligence",
        "Once upon a time in a magical forest",
        "Machine learning algorithms work by",
        "The most important factor in climate change is",
    ]

    temperatures = [0.0, 0.5, 1.0, 1.5]
    results = []

    print("Running evaluation across prompts and temperatures...")

    for prompt in prompts:
        for temp in temperatures:
            print(f"  Evaluating T={temp} with prompt: '{prompt[:30]}...'")

            start_time = time.time()
            generated = generate_with_temperature(
                model, tokenizer, prompt, temp, max_length=20
            )
            generation_time = time.time() - start_time

            generated_part = generated[len(prompt) :].strip()
            diversity = calculate_diversity(generated_part)
            fluency_score = evaluate_fluency(generated_part)

            results.append(
                GenerationResult(
                    temperature=temp,
                    prompt=prompt,
                    generated_text=generated_part,
                    diversity=diversity,
                    generation_time=generation_time,
                    fluency_score=fluency_score,
                )
            )

    return results


# %% [markdown]
# ## Summary and Best Practices
#
# **Temperature Selection Guide:**
# - **T = 0.0 (Greedy)**: Deterministic, focused, potentially repetitive
# - **T = 0.5**: Conservative, coherent, good for factual content
# - **T = 1.0**: Balanced creativity and coherence, good default
# - **T = 1.5**: Creative, diverse, good for creative writing
#
# **Key Trade-offs:**
# - Lower temperature → Higher coherence, lower diversity
# - Higher temperature → Lower coherence, higher diversity
# - The optimal temperature depends on your specific application
#
# **Practical Recommendations:**
# 1. Start with T = 1.0 as a baseline for most applications
# 2. Use T = 0.0 (greedy) when you need deterministic, reproducible outputs
# 3. Lower temperature (0.3-0.7) for factual content, documentation, or technical writing
# 4. Higher temperature (1.0-1.5) for creative writing, brainstorming, or content generation
# 5. Very high temperatures (>2.0) often produce incoherent text and should be used sparingly
# 6. Always evaluate on your specific use case - optimal temperature varies by domain and application
