# CMU LM Inference Course - Code Repository

Code examples and tutorials for the **CMU Language Model Inference** course (Fall 2025).

## Structure

Each class directory contains:

- **Python scripts** (`.py`) - Canonical implementation
- **Jupyter notebooks** (`.ipynb`) - Interactive versions with Colab links

## Quick Start

```bash
# Install dependencies
pip install torch transformers matplotlib seaborn numpy pandas jupytext

# Run examples
cd 01-lm-intro/
python llama31_flop_analysis.py
python qwen_completions.py
```

## Development

1. Write Python scripts with jupytext headers
2. Generate notebooks: `jupytext --to notebook script.py`
3. Test both versions

## License

MIT License
