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
# # Meta-Generation: Cross-Model Likelihood Analysis
#
# By Graham Neubig for [11-664/763 Inference Algorithms for Language Modeling](https://phontron.com/class/lminference-fall2025/)
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neubig/lminference-fall2025-code/blob/main/02-generation-basics/meta_generation.ipynb)
#
# This notebook explores meta-generation: the fascinating process of using one language model to generate text and then evaluating that text with different models. This approach reveals insights about model preferences, quality assessment, and the relationship between model size and text evaluation.
#
# ## Learning Objectives
# - Understand cross-model evaluation and its applications
# - Implement log probability calculation for text sequences
# - Compare how different model sizes evaluate the same generated text
# - Analyze model agreement and disagreement patterns
# - Explore the relationship between generation and evaluation models
#
# ## Key Concepts
# **Meta-Generation**: Using one model to generate text and another to evaluate it
# **Log Probability**: The logarithm of the probability a model assigns to a text sequence
# **Cross-Model Analysis**: Comparing how different models evaluate the same text

# %% [markdown]
# ## Setting Up Our Environment
#
# First, let's import the necessary libraries and define our data structures. We'll use PyTorch for tensor operations, our custom GPT-2 implementation, and temperature sampling from the generation module.

# %%
from __future__ import annotations

from dataclasses import dataclass

import tiktoken
import torch
import torch.nn.functional as F
from nanogpt import GPT2, GPT2Tokenizer
from generation import temperature_sample


@dataclass
class GenerationResult:
    """Result of a single generation with cross-model evaluation"""

    prompt: str
    generated_text: str
    small_log_prob: float
    medium_log_prob: float
    small_per_token_log_prob: float
    medium_per_token_log_prob: float
    token_count: int


# %% [markdown]
# ## 1. Model Setup and Log Probability Calculation
#
# The foundation of meta-generation is the ability to calculate how likely a piece of text is according to different models. We'll set up two GPT-2 models of different sizes and implement log probability calculation.
#
# **Log Probability Calculation**: For a sequence of tokens $w_1, w_2, \ldots, w_n$, the log probability is:
# $$\log P(w_1, w_2, \ldots, w_n) = \sum_{i=1}^{n} \log P(w_i | w_1, \ldots, w_{i-1})$$
#
# We'll compare:
# - **Small Model**: GPT-2 base (117M parameters, 12 layers, 12 heads, 768 dimensions)
# - **Medium Model**: GPT-2 medium (345M parameters, 24 layers, 16 heads, 1024 dimensions)


# %%
def load_models(device: str = "auto") -> tuple[GPT2, GPT2, GPT2Tokenizer]:
    """Load small and medium GPT-2 models for cross-model evaluation."""
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print(f"Loading models on {device}...")

    # Create tokenizer
    tokenizer = GPT2Tokenizer()

    # Load small model (GPT-2 base model)
    print("Loading GPT-2 small model...")
    small_model = GPT2.from_pretrained("gpt2")
    small_model.to(device)
    small_model.eval()

    # Load medium model (GPT-2 medium model)
    print("Loading GPT-2 medium model...")
    medium_model = GPT2.from_pretrained("gpt2-medium")
    medium_model.to(device)
    medium_model.eval()

    print("Models loaded successfully!")
    return small_model, medium_model, tokenizer


def calculate_log_probability(
    model: GPT2, tokenizer: GPT2Tokenizer, text: str, device: str
) -> tuple[float, float, int]:
    """
    Calculate log probability of text under a model.

    Args:
        model: GPT-2 model to evaluate with
        tokenizer: Tokenizer for text processing
        text: Text to evaluate
        device: Device to run computation on

    Returns:
        Tuple of (total_log_prob, per_token_log_prob, token_count)
    """
    # Tokenize (handle special tokens like <|endoftext|>)
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
    inputs = torch.tensor([tokens]).to(device)

    if len(inputs[0]) <= 1:
        return float("-inf"), float("-inf"), 0

    with torch.no_grad():
        # Pass targets to get logits for all positions
        logits, _ = model(inputs, targets=inputs)

        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()

        # Calculate log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Get log probabilities for actual tokens
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Total and per-token log probability
        total_log_prob = token_log_probs.sum().item()
        per_token_log_prob = token_log_probs.mean().item()
        token_count = len(shift_labels[0])

    return total_log_prob, per_token_log_prob, token_count


def demonstrate_log_probability_calculation() -> None:
    """Demonstrate log probability calculation with example texts."""
    print("Loading models for log probability demonstration...")
    small_model, medium_model, tokenizer = load_models()
    device = next(small_model.parameters()).device

    # Example texts with different expected qualities
    example_texts = [
        "The future of artificial intelligence is bright and promising.",
        "The future of artificial intelligence is banana purple elephant.",
        "Artificial intelligence will transform many industries.",
    ]

    print("\nLog Probability Comparison:")
    print("-" * 80)

    for i, text in enumerate(example_texts, 1):
        small_total, small_per_token, token_count = calculate_log_probability(small_model, tokenizer, text, str(device))
        medium_total, medium_per_token, _ = calculate_log_probability(medium_model, tokenizer, text, str(device))

        print(f"\n{i}. Text: '{text}'")
        print(f"   Tokens: {token_count}")
        print(f"   Small model  - Total: {small_total:.3f}, Per-token: {small_per_token:.3f}")
        print(f"   Medium model - Total: {medium_total:.3f}, Per-token: {medium_per_token:.3f}")

        # Show which model prefers this text
        if small_per_token > medium_per_token:
            print(f"   → Small model prefers this text (+{small_per_token - medium_per_token:.3f})")
        elif medium_per_token > small_per_token:
            print(f"   → Medium model prefers this text (+{medium_per_token - small_per_token:.3f})")
        else:
            print("   → Models agree on this text")


# %%
# Run the log probability demonstration
demonstrate_log_probability_calculation()


# %% [markdown]
# ## 2. Meta-Generation Experiment
#
# Now we'll run the core meta-generation experiment: generate text with one model and evaluate it with multiple models. This reveals how different models assess the same generated content.
#
# **Experimental Setup**:
# 1. Generate text using the small model with temperature sampling
# 2. Evaluate each generated text with both small and medium models
# 3. Compare log probabilities to understand model preferences
# 4. Analyze patterns in cross-model evaluation


# %%
def run_meta_generation_experiment(
    prompt: str = "The future of artificial intelligence is",
    num_generations: int = 100,
    max_length: int = 30,
    temperature: float = 1.0,
) -> list[GenerationResult]:
    """
    Run the meta-generation experiment: generate with small model, evaluate with both models.

    Args:
        prompt: Text prompt to start generation
        num_generations: Number of texts to generate
        max_length: Maximum tokens to generate per text
        temperature: Sampling temperature for generation

    Returns:
        List of GenerationResult objects with cross-model evaluations
    """

    # Load models
    small_model, medium_model, tokenizer = load_models()
    device = next(small_model.parameters()).device

    print(f"Generating {num_generations} outputs from small model...")
    print(f"Prompt: '{prompt}'")
    print(f"Temperature: {temperature}, Max length: {max_length}")
    print("-" * 60)

    results = []

    for i in range(num_generations):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{num_generations}")

        # Generate text using small model
        device = next(small_model.parameters()).device
        enc = tiktoken.get_encoding("gpt2")
        input_ids = torch.tensor([enc.encode(prompt, allowed_special={"<|endoftext|>"})]).to(device)
        eos_token_id = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

        small_model.eval()
        with torch.no_grad():
            for _ in range(max_length):
                logits, _ = small_model(input_ids)
                next_token_logits = logits[0, -1, :]
                next_token = temperature_sample(next_token_logits, temperature)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

                # Stop if we generate end-of-text token
                if next_token.item() == eos_token_id:
                    break

        generated_text = tokenizer.decode(input_ids[0].tolist())

        # Remove prompt from generated text
        generated_only = generated_text[len(prompt) :].strip()

        if not generated_only:
            continue

        # Calculate probabilities under both models
        full_text = prompt + " " + generated_only

        # Small model evaluation
        small_total_log_prob, small_per_token_log_prob, token_count = calculate_log_probability(
            small_model, tokenizer, full_text, str(device)
        )

        # Medium model evaluation
        medium_total_log_prob, medium_per_token_log_prob, _ = calculate_log_probability(
            medium_model, tokenizer, full_text, str(device)
        )

        result = GenerationResult(
            prompt=prompt,
            generated_text=generated_only,
            small_log_prob=small_total_log_prob,
            medium_log_prob=medium_total_log_prob,
            small_per_token_log_prob=small_per_token_log_prob,
            medium_per_token_log_prob=medium_per_token_log_prob,
            token_count=token_count,
        )

        results.append(result)

    print(f"\nGenerated {len(results)} valid outputs")
    return results


def demonstrate_meta_generation() -> list[GenerationResult]:
    """Run a small-scale meta-generation demonstration."""
    print("Running meta-generation demonstration with 20 samples...")

    # Run a smaller experiment for demonstration
    results = run_meta_generation_experiment(
        prompt="The future of artificial intelligence is", num_generations=20, max_length=25, temperature=1.0
    )

    # Show a few example results
    print("\nSample Results:")
    print("-" * 80)

    for i, result in enumerate(results[:5]):
        print(f"\n{i+1}. Generated: '{result.generated_text[:60]}...'")
        print(f"   Small model log prob: {result.small_per_token_log_prob:.3f}")
        print(f"   Medium model log prob: {result.medium_per_token_log_prob:.3f}")

        if result.small_per_token_log_prob > result.medium_per_token_log_prob:
            print("   → Small model prefers this text")
        elif result.medium_per_token_log_prob > result.small_per_token_log_prob:
            print("   → Medium model prefers this text")
        else:
            print("   → Models have similar preferences")

    return results


# %%
# Run the meta-generation demonstration
demo_results = demonstrate_meta_generation()


# %% [markdown]
# ## 3. Cross-Model Analysis and Results
#
# After generating text with one model and evaluating with multiple models, we can analyze the results to understand model preferences and agreement patterns. This section provides comprehensive analysis tools.
#
# **Analysis Dimensions**:
# - **Combined Preferences**: Texts that both models rate highly
# - **Model Agreement**: How often models agree on text quality
# - **Disagreement Patterns**: Cases where models have opposing preferences
# - **Quality Indicators**: What makes text preferred by larger vs smaller models


# %%


def show_medium_preferred_examples(results: list[GenerationResult]) -> None:
    """Show examples where the medium model assigns higher probability than the small model."""

    # Filter and sort by cases where medium model has higher score
    medium_preferred = [r for r in results if r.medium_per_token_log_prob > r.small_per_token_log_prob]
    medium_preferred.sort(key=lambda r: r.medium_per_token_log_prob - r.small_per_token_log_prob, reverse=True)

    print("\n" + "=" * 60)
    print("EXAMPLES WHERE MEDIUM MODEL ASSIGNS HIGHER PROBABILITY")
    print("=" * 60)

    if not medium_preferred:
        print("\nNo examples found where medium model assigns higher probability.")
        return

    print(f"\nFound {len(medium_preferred)} examples where medium model prefers the text:")
    print(f"Showing top {min(10, len(medium_preferred))} examples sorted by preference difference:\n")

    for i, result in enumerate(medium_preferred[:10]):
        difference = result.medium_per_token_log_prob - result.small_per_token_log_prob
        print(f"{i+1}. Prompt: '{result.prompt}'")
        print(f"   Generated: '{result.generated_text}'")
        print(
            f"   Small: {result.small_per_token_log_prob:.3f}, Medium: {result.medium_per_token_log_prob:.3f}, Diff: +{difference:.3f}\n"
        )


# %%
# Show examples where medium model assigns higher probability
if "demo_results" in locals():
    show_medium_preferred_examples(demo_results)
