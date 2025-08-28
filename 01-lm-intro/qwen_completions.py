# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Qwen Model Completion Analysis
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neubig/lminference-fall2025-code/blob/main/01-lm-intro/qwen_completions.ipynb)
#
# This notebook demonstrates text generation using the Qwen 3-1.7B-Base model and analyzes
# the probability distributions of generated completions.
#
# ## Overview
#
# We'll explore:
# - Unconditional text generation (from BOS token)
# - Conditional text generation (from a specific prompt)
# - Analysis of completion probabilities
# - Comparison of high, low, and medium probability completions

# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import logging
import warnings
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# %% [markdown]
# ## Model Setup and Configuration
#
# First, let's set up the Qwen model and tokenizer. We'll handle device selection automatically
# and include robust error handling for different hardware configurations.


# %%
class QwenCompletionGenerator:
    """
    A class for generating text completions using Qwen models with probability analysis.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B-Base") -> None:
        """Initialize the Qwen model and tokenizer."""
        # Determine the best available device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")

        logger.info(f"Loading model: {model_name}")
        self.model_name = model_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with appropriate settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,  # Manual device placement for better control
        ).to(self.device)

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Model loaded successfully!")

        # Display model info
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")


# Initialize the model
generator = QwenCompletionGenerator()

# %% [markdown]
# ## Text Generation with Probability Tracking
#
# Now let's implement the core generation function that tracks token probabilities
# and stops at natural sentence boundaries.


# %%
def generate_completion(self: "QwenCompletionGenerator", prompt: str, max_length: int = 50) -> tuple[str, float]:
    """
    Generate a completion and compute its total log-probability.
    Stops at first period or newline for natural sentence boundaries.

    Args:
        prompt: Input prompt (empty string for unconditional generation)
        max_length: Maximum number of tokens to generate

    Returns:
        Tuple of (generated_text, total_log_probability)
    """
    # Handle empty prompt case - use a space for unconditional generation
    if not prompt.strip():
        prompt = " "

    # Tokenize the prompt
    inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
    input_length = inputs.input_ids.shape[1]
    eos_id = self.tokenizer.eos_token_id

    # Generate with sampling and probability tracking
    with torch.no_grad():
        output = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=1.0,
            pad_token_id=eos_id,
            eos_token_id=eos_id,
            return_dict_in_generate=True,
            output_scores=True,
            top_k=50,
            top_p=0.95,
        )

        # Extract generated tokens (excluding input)
        gen_ids = output.sequences[0][input_length:]

        total_log_prob = 0.0
        tokens_used = 0
        displayed_text = ""

        # Calculate probabilities and find stopping point
        for i, (logits, tok_id) in enumerate(zip(output.scores, gen_ids)):
            # Calculate log probabilities
            log_probs = torch.log_softmax(logits[0], dim=-1)
            total_log_prob += log_probs[tok_id.item()].item()
            tokens_used = i + 1

            # Decode incrementally to detect stopping conditions
            partial_text = self.tokenizer.decode(gen_ids[:tokens_used], skip_special_tokens=True)
            if "." in partial_text or "\n" in partial_text or (eos_id is not None and tok_id.item() == eos_id):
                displayed_text = partial_text
                break

        if not displayed_text:
            displayed_text = self.tokenizer.decode(gen_ids[:tokens_used], skip_special_tokens=True)

        # Clean up the text
        if "." in displayed_text:
            displayed_text = displayed_text.split(".")[0] + "."
        elif "\n" in displayed_text:
            displayed_text = displayed_text.split("\n")[0]
        else:
            displayed_text = displayed_text.strip()

    return displayed_text, total_log_prob


# Add the method to our generator instance
QwenCompletionGenerator.generate_completion = generate_completion

# Test the generation function
test_completion, test_prob = generator.generate_completion("The weather today is")
print(f"Test completion: '{test_completion}' (log_prob: {test_prob:.4f})")

# %% [markdown]
# ## Batch Generation and Analysis
#
# Let's implement functions to generate multiple completions and analyze their probability distributions.


# %%
def generate_multiple_completions(
    self: "QwenCompletionGenerator", prompt: str, num_completions: int = 100
) -> list[tuple[str, float]]:
    """Generate multiple completions from the given prompt."""
    completions = []

    logger.info(f"Generating {num_completions} completions for prompt: '{prompt[:50]}...'")

    for i in range(num_completions):
        if (i + 1) % 20 == 0:
            logger.info(f"Progress: {i + 1}/{num_completions} completions")

        completion, log_prob = self.generate_completion(prompt)
        completions.append((completion, log_prob))

    return completions


def select_completions_by_probability(
    self: "QwenCompletionGenerator", completions: list[tuple[str, float]]
) -> tuple[list[tuple[str, float]], list[tuple[str, float]], list[tuple[str, float]]]:
    """
    Select highest, lowest, and central probability completions.
    """
    # Sort by log probability (higher is better)
    sorted_completions = sorted(completions, key=lambda x: x[1], reverse=True)
    n = len(sorted_completions)

    # Determine selection size based on total completions
    select_size = 5 if n >= 100 else min(3, n // 3) if n >= 9 else min(2, n // 2) if n >= 6 else 1

    # Get highest probability completions
    highest = sorted_completions[:select_size]

    # Get lowest probability completions
    lowest = sorted_completions[-select_size:]

    # Get central probability completions (around median)
    if n >= 3:
        mid_start = max(0, (n - select_size) // 2)
        central = sorted_completions[mid_start : mid_start + select_size]
    else:
        central = sorted_completions[:1]

    return highest, lowest, central


# Add methods to our generator
QwenCompletionGenerator.generate_multiple_completions = generate_multiple_completions
QwenCompletionGenerator.select_completions_by_probability = select_completions_by_probability

# %% [markdown]
# ## Experiment 1: Unconditional Generation (BOS Token)
#
# Let's start by generating completions from the beginning-of-sequence token to see
# what the model generates unconditionally.

# %%
print("=" * 60)
print("EXPERIMENT 1: Unconditional Generation from BOS Token")
print("=" * 60)

# Generate completions from BOS token (empty prompt)
bos_prompt = ""
bos_completions = generator.generate_multiple_completions(bos_prompt, 100)
bos_highest, bos_lowest, bos_central = generator.select_completions_by_probability(bos_completions)

print(f"\nGenerated {len(bos_completions)} completions from BOS token")
print(f"Unique completions: {len(set(comp[0] for comp in bos_completions))}")

print("\nHIGHEST Probability Completions:")
for i, (text, prob) in enumerate(bos_highest, 1):
    print(f"{i:2d}. (log_prob: {prob:7.4f}) '{text}'")

print("\nLOWEST Probability Completions:")
for i, (text, prob) in enumerate(bos_lowest, 1):
    print(f"{i:2d}. (log_prob: {prob:7.4f}) '{text}'")

print("\nMEDIAN Probability Completions:")
for i, (text, prob) in enumerate(bos_central, 1):
    print(f"{i:2d}. (log_prob: {prob:7.4f}) '{text}'")

# %% [markdown]
# ## Experiment 2: Conditional Generation (CMU Prompt)
#
# Now let's generate completions conditioned on a specific prompt about Carnegie Mellon University.

# %%
print("\n" + "=" * 60)
print("EXPERIMENT 2: Conditional Generation from CMU Prompt")
print("=" * 60)

cmu_prompt = "The best thing about Carnegie Mellon University is"
cmu_completions = generator.generate_multiple_completions(cmu_prompt, 100)
cmu_highest, cmu_lowest, cmu_central = generator.select_completions_by_probability(cmu_completions)

print(f"\nGenerated {len(cmu_completions)} completions for prompt: '{cmu_prompt}'")
print(f"Unique completions: {len(set(comp[0] for comp in cmu_completions))}")

print("\nHIGHEST Probability Completions:")
for i, (text, prob) in enumerate(cmu_highest, 1):
    print(f"{i:2d}. (log_prob: {prob:7.4f}) '{cmu_prompt} {text}'")

print("\nLOWEST Probability Completions:")
for i, (text, prob) in enumerate(cmu_lowest, 1):
    print(f"{i:2d}. (log_prob: {prob:7.4f}) '{cmu_prompt} {text}'")

print("\nMEDIAN Probability Completions:")
for i, (text, prob) in enumerate(cmu_central, 1):
    print(f"{i:2d}. (log_prob: {prob:7.4f}) '{cmu_prompt} {text}'")

# %% [markdown]
# ## Statistical Analysis and Comparison
#
# Let's analyze the statistical properties of the generated completions and compare
# the two experimental conditions.

# %%
print("\n" + "=" * 60)
print("STATISTICAL ANALYSIS")
print("=" * 60)

# Extract probabilities for analysis
bos_probs = [prob for _, prob in bos_completions]
cmu_probs = [prob for _, prob in cmu_completions]

# Calculate statistics
bos_stats = {
    "mean": np.mean(bos_probs),
    "std": np.std(bos_probs),
    "min": np.min(bos_probs),
    "max": np.max(bos_probs),
    "median": np.median(bos_probs),
}

cmu_stats = {
    "mean": np.mean(cmu_probs),
    "std": np.std(cmu_probs),
    "min": np.min(cmu_probs),
    "max": np.max(cmu_probs),
    "median": np.median(cmu_probs),
}

print("\nBOS Token Completions Statistics:")
print(f"   Mean log probability: {bos_stats['mean']:8.4f}")
print(f"   Standard deviation:   {bos_stats['std']:8.4f}")
print(f"   Minimum:             {bos_stats['min']:8.4f}")
print(f"   Maximum:             {bos_stats['max']:8.4f}")
print(f"   Median:              {bos_stats['median']:8.4f}")

print("\nCMU Prompt Completions Statistics:")
print(f"   Mean log probability: {cmu_stats['mean']:8.4f}")
print(f"   Standard deviation:   {cmu_stats['std']:8.4f}")
print(f"   Minimum:             {cmu_stats['min']:8.4f}")
print(f"   Maximum:             {cmu_stats['max']:8.4f}")
print(f"   Median:              {cmu_stats['median']:8.4f}")

print(f"\nComparison:")
print(f"   Mean difference (CMU - BOS): {cmu_stats['mean'] - bos_stats['mean']:8.4f}")
print(f"   Std difference (CMU - BOS):  {cmu_stats['std'] - bos_stats['std']:8.4f}")

print(f"\nSystem Information:")
print(f"   Device used: {generator.device}")
print(f"   Model: {generator.model_name}")
print(f"   Torch version: {torch.__version__}")

# %% [markdown]
# ## Visualization of Probability Distributions
#
# Let's create visualizations to better understand the probability distributions.

# %%
# Create probability distribution plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Histogram comparison
ax1.hist(bos_probs, bins=20, alpha=0.7, label="BOS Token", density=True)
ax1.hist(cmu_probs, bins=20, alpha=0.7, label="CMU Prompt", density=True)
ax1.set_xlabel("Log Probability")
ax1.set_ylabel("Density")
ax1.set_title("Probability Distribution Comparison")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Box plot comparison
data_to_plot = [bos_probs, cmu_probs]
labels = ["BOS Token", "CMU Prompt"]
ax2.boxplot(data_to_plot, labels=labels)
ax2.set_ylabel("Log Probability")
ax2.set_title("Probability Distribution Box Plot")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Observations and Insights
#
# Based on our experiments, we can observe several interesting patterns:
#
# 1. **Probability Ranges**: The model assigns different probability ranges to unconditional vs conditional generation
# 2. **Diversity**: Both conditions produce diverse completions, but with different characteristics
# 3. **Context Effect**: The CMU prompt constrains the generation space, affecting the probability distribution
# 4. **Model Behavior**: High-probability completions tend to be more generic, while low-probability ones are more specific or creative
#
# This analysis demonstrates how language models balance between probable (common) and improbable (creative) text generation,
# and how conditioning context affects this balance.

# %%
