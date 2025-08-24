# CMU LM Inference Course - Code Repository

This repository contains code examples, tutorials, and exercises for the **CMU Language Model Inference** course (Fall 2025). The course covers the theory and practice of efficient inference for large language models.

## Course Overview

The CMU LM Inference course explores:
- Language model architectures and computational requirements
- Efficient inference techniques and optimizations
- Memory management and hardware considerations
- Practical implementation of LM inference systems
- Analysis of computational complexity and scaling

## Repository Structure

The repository is organized by class/topic, with each directory containing:
- **Python scripts** (`.py`) - The canonical implementation
- **Jupyter notebooks** (`.ipynb`) - Interactive versions for exploration and learning

### Current Content

#### `lm-intro/` - Language Model Introduction
- **`llama31_flop_analysis.py/.ipynb`** - Analysis of computational requirements (FLOPs) for Llama 3.1 models
  - FLOP scaling with context length and model size
  - Distribution of computation across model components
  - Efficiency metrics and architecture comparisons
  
- **`qwen_completions.py/.ipynb`** - Text generation and probability analysis using Qwen models
  - Unconditional and conditional text generation
  - Probability distribution analysis
  - Comparison of high/low probability completions

## Getting Started

### Prerequisites

```bash
pip install torch transformers matplotlib seaborn numpy pandas jupytext
```

### Running the Code

Each topic directory contains both Python scripts and Jupyter notebooks:

**Python Scripts:**
```bash
cd lm-intro/
python llama31_flop_analysis.py
python qwen_completions.py
```

**Jupyter Notebooks:**
```bash
cd lm-intro/
jupyter notebook llama31_flop_analysis.ipynb
jupyter notebook qwen_completions.ipynb
```

### Code Organization

- **Python files** are the canonical source and contain the complete implementation
- **Jupyter notebooks** are automatically generated from Python files using `jupytext`
- All code is written to be both executable as scripts and readable as notebooks
- Markdown cells in Python files (using `# %% [markdown]`) become markdown cells in notebooks

## Development Workflow

When contributing new code:

1. Write Python scripts with proper jupytext headers and markdown cells
2. Generate notebooks using: `jupytext --to notebook script.py`
3. Test both Python and notebook versions
4. Ensure code is well-documented and educational

## Course Information

- **Institution:** Carnegie Mellon University
- **Course:** Language Model Inference
- **Semester:** Fall 2025
- **Focus:** Efficient inference techniques for large language models

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure that:
- Code follows the established patterns (Python + Jupyter)
- Examples are educational and well-documented
- All dependencies are clearly specified
- Code is tested and functional

For questions about the course content, please refer to the course materials or contact the instructors.
