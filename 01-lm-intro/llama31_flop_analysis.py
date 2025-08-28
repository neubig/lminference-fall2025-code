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
# # Llama 3.1 FLOP Analysis and Visualization
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neubig/lminference-fall2025-code/blob/main/01-lm-intro/llama31_flop_analysis.ipynb)
#
# This notebook analyzes the computational requirements (FLOPs) of different Llama 3.1 model variants
# and visualizes how these requirements scale with context length and model size.
#
# ## Overview
#
# We'll examine:
# - FLOP scaling with context length for different model sizes
# - Distribution of FLOPs across different components (attention, MLP, etc.)
# - Computational efficiency metrics
# - Architecture comparisons

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List
import pandas as pd
import os

# Set style for better-looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# Configure matplotlib for better readability
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 14,
        "figure.figsize": (12, 8),
    }
)

# %% [markdown]
# ## Model Configurations
#
# First, let's define the configurations for the three Llama 3.1 model variants:

# %%
# Llama 3.1 model configurations
LLAMA31_CONFIGS = {
    "Llama-3.1-8B": {
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "head_dim": 128,
        "parameters": 8.0e9,
    },
    "Llama-3.1-70B": {
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "head_dim": 128,
        "parameters": 70.6e9,
    },
    "Llama-3.1-405B": {
        "hidden_size": 16384,
        "intermediate_size": 53248,
        "num_hidden_layers": 126,
        "num_attention_heads": 128,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "head_dim": 128,
        "parameters": 405.9e9,
    },
}

# Display the configurations
for model_name, config in LLAMA31_CONFIGS.items():
    print(f"\n{model_name}:")
    print(f"  Parameters: {config['parameters'] / 1e9:.1f}B")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Layers: {config['num_hidden_layers']}")
    print(f"  Attention heads: {config['num_attention_heads']}")
    print(f"  KV heads: {config['num_key_value_heads']}")

# %% [markdown]
# ## FLOP Calculation Function
#
# Now let's implement the function to calculate FLOPs for a forward pass through the model.
# This includes attention (both linear and quadratic components), MLP layers, and the language modeling head.


# %%
def calculate_flops_per_token(config: Dict, sequence_length: int) -> Dict[str, float]:
    """
    Calculate FLOPs per forward pass for a given model configuration and sequence length.

    Args:
        config: Model configuration dictionary
        sequence_length: Input sequence length

    Returns:
        Dictionary with FLOP breakdown by component
    """
    h = config["hidden_size"]
    i = config["intermediate_size"]
    v = config["vocab_size"]
    n_layers = config["num_hidden_layers"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    seq_len = sequence_length

    # === ATTENTION LAYER FLOPS ===
    # Q, K, V projections (linear in sequence length)
    q_proj_flops = 2 * seq_len * h * (n_heads * head_dim)
    k_proj_flops = 2 * seq_len * h * (n_kv_heads * head_dim)
    v_proj_flops = 2 * seq_len * h * (n_kv_heads * head_dim)
    qkv_proj_flops = q_proj_flops + k_proj_flops + v_proj_flops

    # Q @ K^T (quadratic in sequence length)
    qk_flops = 2 * n_heads * seq_len * seq_len * head_dim

    # Attention weights @ V (quadratic in sequence length)
    attn_v_flops = 2 * n_heads * seq_len * seq_len * head_dim

    # Output projection
    o_proj_flops = 2 * seq_len * (n_heads * head_dim) * h

    attention_flops_per_layer = qkv_proj_flops + qk_flops + attn_v_flops + o_proj_flops

    # === MLP LAYER FLOPS ===
    # SwiGLU MLP: gate, up, down projections
    gate_proj_flops = 2 * seq_len * h * i
    up_proj_flops = 2 * seq_len * h * i
    down_proj_flops = 2 * seq_len * i * h
    mlp_flops_per_layer = gate_proj_flops + up_proj_flops + down_proj_flops

    # Layer norm FLOPs (minimal)
    layernorm_flops_per_layer = 2 * seq_len * h

    # Total per layer
    flops_per_layer = attention_flops_per_layer + mlp_flops_per_layer + layernorm_flops_per_layer

    # Final layer norm and language modeling head
    final_layernorm_flops = seq_len * h
    lm_head_flops = 2 * seq_len * h * v

    # Total FLOPs
    total_flops = (n_layers * flops_per_layer) + final_layernorm_flops + lm_head_flops

    # Separate linear and quadratic attention components for analysis
    attention_linear = qkv_proj_flops + o_proj_flops
    attention_quadratic = qk_flops + attn_v_flops

    return {
        "attention_linear": attention_linear,
        "attention_quadratic": attention_quadratic,
        "attention_per_layer": attention_flops_per_layer,
        "mlp_per_layer": mlp_flops_per_layer,
        "layernorm_per_layer": layernorm_flops_per_layer,
        "flops_per_layer": flops_per_layer,
        "total_layers_flops": n_layers * flops_per_layer,
        "final_layernorm": final_layernorm_flops,
        "lm_head": lm_head_flops,
        "total": total_flops,
    }


# Test the function with a simple example
test_flops = calculate_flops_per_token(LLAMA31_CONFIGS["Llama-3.1-8B"], 2048)
print(f"Example: Llama-3.1-8B with 2K context requires {test_flops['total'] / 1e12:.2f} TFLOPs")

# %% [markdown]
# ## FLOP Scaling Analysis
#
# Let's analyze how FLOPs scale with context length for different model sizes:


# %%
def create_flop_scaling_figure():
    """Create figure showing FLOP scaling with context length."""
    context_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(bottom=0.28, top=0.90)

    # Plot 1: Total FLOPs vs Context Length
    for model_name, config in LLAMA31_CONFIGS.items():
        flops_data = []
        for ctx_len in context_lengths:
            flops = calculate_flops_per_token(config, ctx_len)
            flops_data.append(flops["total"] / 1e12)  # Convert to TFLOPs

        ax1.scatter(context_lengths, flops_data, label=model_name, s=80, alpha=0.75)

    ax1.set_xlabel("Context Length (tokens)", fontsize=16)
    ax1.set_ylabel("Total FLOPs (TFLOPs)", fontsize=16)
    ax1.set_title("Llama 3.1 FLOP Scaling with Context Length", fontsize=20, fontweight="bold")
    ax1.set_xscale("log", base=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=14, frameon=True, framealpha=0.95)

    # Add context length labels
    ax1.set_xticks(context_lengths)
    ax1.set_xticklabels([f"{x // 1024}K" if x >= 1024 else str(x) for x in context_lengths])

    # Plot 2: FLOP Distribution for 70B model
    model_70b = LLAMA31_CONFIGS["Llama-3.1-70B"]
    context_lengths_breakdown = [1024, 4096, 16384, 65536, 131072]

    attention_linear_ratios = []
    attention_quadratic_ratios = []
    mlp_ratios = []
    lm_head_ratios = []

    print("\n=== FLOP Distribution Analysis: Llama-3.1-70B ===")

    for ctx_len in context_lengths_breakdown:
        flops = calculate_flops_per_token(model_70b, ctx_len)
        n_layers = model_70b["num_hidden_layers"]

        att_linear = flops["attention_linear"] * n_layers
        att_quad = flops["attention_quadratic"] * n_layers
        mlp = flops["mlp_per_layer"] * n_layers
        lm_head = flops["lm_head"]
        total = att_linear + att_quad + mlp + lm_head

        # Calculate ratios (percentages)
        att_linear_ratio = (att_linear / total) * 100
        att_quad_ratio = (att_quad / total) * 100
        mlp_ratio = (mlp / total) * 100
        lm_head_ratio = (lm_head / total) * 100

        attention_linear_ratios.append(att_linear_ratio)
        attention_quadratic_ratios.append(att_quad_ratio)
        mlp_ratios.append(mlp_ratio)
        lm_head_ratios.append(lm_head_ratio)

        print(
            f"Context {ctx_len:>6}: Att(Lin)={att_linear_ratio:>5.1f}%, "
            f"Att(Quad)={att_quad_ratio:>5.1f}%, MLP={mlp_ratio:>5.1f}%, LM={lm_head_ratio:>4.1f}%"
        )

    x = np.arange(len(context_lengths_breakdown))
    width = 0.6

    p1 = ax2.bar(x, attention_linear_ratios, width, label="Attention (Linear)", alpha=0.8)
    p2 = ax2.bar(
        x, attention_quadratic_ratios, width, bottom=attention_linear_ratios, label="Attention (Quadratic)", alpha=0.8
    )
    p3 = ax2.bar(
        x,
        mlp_ratios,
        width,
        bottom=np.array(attention_linear_ratios) + np.array(attention_quadratic_ratios),
        label="MLP Layers",
        alpha=0.8,
    )
    p4 = ax2.bar(
        x,
        lm_head_ratios,
        width,
        bottom=np.array(attention_linear_ratios) + np.array(attention_quadratic_ratios) + np.array(mlp_ratios),
        label="LM Head",
        alpha=0.8,
    )

    ax2.set_xlabel("Context Length", fontsize=16)
    ax2.set_ylabel("FLOP Percentage (%)", fontsize=16)
    ax2.set_title("Llama-3.1-70B FLOP Distribution by Context Length", fontsize=20, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{ctx // 1024}K" for ctx in context_lengths_breakdown])
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=14, frameon=True, framealpha=0.95)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()


# Generate the scaling analysis
create_flop_scaling_figure()

# %% [markdown]
# ## Computational Efficiency Analysis
#
# Let's examine the computational efficiency in terms of FLOPs per parameter and memory vs compute trade-offs:


# %%
def create_efficiency_comparison():
    """Create figure comparing computational efficiency metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: FLOPs per Parameter vs Model Size
    models = []
    params = []
    flops_per_param_8k = []
    flops_per_param_32k = []
    flops_per_param_131k = []

    for model_name, config in LLAMA31_CONFIGS.items():
        models.append(model_name.replace("Llama-3.1-", ""))
        params.append(config["parameters"] / 1e9)  # Convert to billions

        # Calculate FLOPs per parameter for different context lengths
        flops_8k = calculate_flops_per_token(config, 8192)
        flops_32k = calculate_flops_per_token(config, 32768)
        flops_131k = calculate_flops_per_token(config, 131072)

        flops_per_param_8k.append(flops_8k["total"] / config["parameters"])
        flops_per_param_32k.append(flops_32k["total"] / config["parameters"])
        flops_per_param_131k.append(flops_131k["total"] / config["parameters"])

    x = np.arange(len(models))
    width = 0.25

    ax1.bar(x - width, flops_per_param_8k, width, label="8K Context", alpha=0.8)
    ax1.bar(x, flops_per_param_32k, width, label="32K Context", alpha=0.8)
    ax1.bar(x + width, flops_per_param_131k, width, label="131K Context", alpha=0.8)

    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("FLOPs per Parameter", fontsize=12)
    ax1.set_title("Computational Efficiency: FLOPs per Parameter", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Memory vs Compute Trade-off
    context_lengths = [8192, 32768, 131072]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, ctx_len in enumerate(context_lengths):
        memory_usage = []  # KV cache memory (simplified)
        compute_flops = []

        for model_name, config in LLAMA31_CONFIGS.items():
            # Simplified KV cache memory calculation (assuming fp16)
            kv_cache_size = (
                2 * config["num_key_value_heads"] * config["head_dim"] * config["num_hidden_layers"] * ctx_len * 2
            ) / 1e9  # GB
            memory_usage.append(kv_cache_size)

            flops = calculate_flops_per_token(config, ctx_len)
            compute_flops.append(flops["total"] / 1e12)  # TFLOPs

        ax2.scatter(memory_usage, compute_flops, s=100, alpha=0.7, label=f"{ctx_len // 1024}K Context", color=colors[i])

        # Add model labels
        for j, model in enumerate(models):
            ax2.annotate(
                model, (memory_usage[j], compute_flops[j]), xytext=(5, 5), textcoords="offset points", fontsize=9
            )

    ax2.set_xlabel("KV Cache Memory (GB)", fontsize=12)
    ax2.set_ylabel("Compute FLOPs (TFLOPs)", fontsize=12)
    ax2.set_title("Memory vs Compute Trade-off", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.show()


# Generate the efficiency comparison
create_efficiency_comparison()

# %% [markdown]
# ## Architecture Comparison
#
# Finally, let's compare the architectural components across different model sizes:


# %%
def create_architecture_comparison():
    """Create figure comparing architectural components."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    models = list(LLAMA31_CONFIGS.keys())
    model_labels = [name.replace("Llama-3.1-", "") for name in models]

    # Plot 1: Parameter Distribution
    embedding_params = []
    attention_params = []
    mlp_params = []

    for model_name, config in LLAMA31_CONFIGS.items():
        h = config["hidden_size"]
        i = config["intermediate_size"]
        v = config["vocab_size"]
        n_layers = config["num_hidden_layers"]
        n_heads = config["num_attention_heads"]
        n_kv_heads = config["num_key_value_heads"]
        head_dim = config["head_dim"]

        # Calculate parameters (simplified)
        emb = v * h / 1e9
        att_per_layer = (
            (h * (n_heads * head_dim + 2 * n_kv_heads * head_dim) + (n_heads * head_dim) * h) * n_layers / 1e9
        )
        mlp_per_layer = (h * i * 3) * n_layers / 1e9  # gate, up, down projections

        embedding_params.append(emb)
        attention_params.append(att_per_layer)
        mlp_params.append(mlp_per_layer)

    x = np.arange(len(models))
    width = 0.6

    p1 = ax1.bar(x, embedding_params, width, label="Embedding", alpha=0.8)
    p2 = ax1.bar(x, attention_params, width, bottom=embedding_params, label="Attention", alpha=0.8)
    p3 = ax1.bar(
        x, mlp_params, width, bottom=np.array(embedding_params) + np.array(attention_params), label="MLP", alpha=0.8
    )

    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("Parameters (Billions)", fontsize=12)
    ax1.set_title("Parameter Distribution by Component", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Layer Scaling
    layers = [config["num_hidden_layers"] for config in LLAMA31_CONFIGS.values()]
    hidden_sizes = [config["hidden_size"] for config in LLAMA31_CONFIGS.values()]

    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    scatter = ax2.scatter(layers, hidden_sizes, s=200, c=colors, alpha=0.7)

    for i, model in enumerate(model_labels):
        ax2.annotate(model, (layers[i], hidden_sizes[i]), xytext=(5, 5), textcoords="offset points", fontsize=10)

    ax2.set_xlabel("Number of Layers", fontsize=12)
    ax2.set_ylabel("Hidden Size", fontsize=12)
    ax2.set_title("Architecture Scaling: Layers vs Hidden Size", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Attention Head Configuration
    attention_heads = [config["num_attention_heads"] for config in LLAMA31_CONFIGS.values()]
    kv_heads = [config["num_key_value_heads"] for config in LLAMA31_CONFIGS.values()]

    x = np.arange(len(models))
    width = 0.35

    ax3.bar(x - width / 2, attention_heads, width, label="Query/Key/Value Heads", alpha=0.8)
    ax3.bar(x + width / 2, kv_heads, width, label="Key/Value Heads (GQA)", alpha=0.8)

    ax3.set_xlabel("Model", fontsize=12)
    ax3.set_ylabel("Number of Heads", fontsize=12)
    ax3.set_title("Attention Head Configuration", fontsize=14, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_labels)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: MLP Expansion Ratio
    expansion_ratios = []
    for config in LLAMA31_CONFIGS.values():
        ratio = config["intermediate_size"] / config["hidden_size"]
        expansion_ratios.append(ratio)

    bars = ax4.bar(model_labels, expansion_ratios, alpha=0.8, color=colors)
    ax4.set_xlabel("Model", fontsize=12)
    ax4.set_ylabel("MLP Expansion Ratio", fontsize=12)
    ax4.set_title("MLP Expansion Ratio (Intermediate/Hidden)", fontsize=14, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, ratio in zip(bars, expansion_ratios):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0, height + 0.05, f"{ratio:.1f}", ha="center", va="bottom", fontsize=10
        )

    plt.tight_layout()
    plt.show()


# Generate the architecture comparison
create_architecture_comparison()

# %% [markdown]
# ## Summary
#
# This analysis reveals several key insights about Llama 3.1 models:
#
# 1. **Quadratic Scaling**: Attention computation scales quadratically with context length, becoming dominant at very long contexts
# 2. **Model Efficiency**: Larger models are more parameter-efficient in terms of FLOPs per parameter
# 3. **Memory Trade-offs**: KV cache memory grows linearly with context length and can become a bottleneck
# 4. **Architecture Patterns**: All models use the same expansion ratio (3.5x) and GQA configuration (8 KV heads)
#
# These insights are crucial for understanding the computational requirements and trade-offs when deploying Llama 3.1 models in production environments.

# %%
