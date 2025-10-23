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

## Schedule

See [SCHEDULE.md](SCHEDULE.md) for the full course schedule.

### November 2025

| Date | Topic | Instructor |
|------|-------|------------|
| 11-Nov | Prefix Sharing and KV Cache Optimizations | Amanda |
| 13-Nov | Draft Models and Speculative Decoding | Beidi Chen (Guest) |
| 18-Nov | Linearizing Attention and Sparse Models | Amanda |
| 20-Nov | Building MLC-LLM, a Universal LLM Deployment Engine | Tianqi Chen (Guest) |
| 25-Nov | Library Implementation and Optimizations | Graham |

## License

MIT License
