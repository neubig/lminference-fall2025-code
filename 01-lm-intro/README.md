# Class 01: Language Model Introduction

This directory contains scripts that introduce fundamental concepts of language models, focusing on computational requirements and text generation.

## Scripts

### `llama31_flop_analysis.py`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neubig/lminference-fall2025-code/blob/main/01-lm-intro/llama31_flop_analysis.ipynb)

Analyzes the computational requirements (FLOPs) of Llama 3.1 model variants.

**Key concepts:**
- FLOP scaling with context length and model size
- Distribution of computation across model components (attention, MLP, embeddings)
- Computational efficiency metrics and architecture comparisons
- Memory vs. computation trade-offs

**Output:** Visualizations showing how computational requirements scale with different parameters.

### `qwen_completions.py`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neubig/lminference-fall2025-code/blob/main/01-lm-intro/qwen_completions.ipynb)

Demonstrates text generation and probability analysis using Qwen models.

**Key concepts:**
- Unconditional vs. conditional text generation
- Token probability distributions and sampling
- Comparison of high/low probability completions
- Statistical analysis of generated text

**Output:** Generated text samples with probability analysis and distribution comparisons.

## Running the Scripts

```bash
# Install dependencies (if not already installed)
pip install torch transformers matplotlib seaborn numpy pandas

# Run the scripts
python llama31_flop_analysis.py
python qwen_completions.py

# Or use Jupyter notebooks
jupyter notebook llama31_flop_analysis.ipynb
jupyter notebook qwen_completions.ipynb
```

## Learning Objectives

After running these scripts, you should understand:
1. How computational requirements scale with model size and context length
2. The relationship between model architecture and FLOP distribution
3. How language models generate text through probability distributions
4. The difference between high and low probability text completions
