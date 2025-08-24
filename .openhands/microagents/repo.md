This repository contains educational code examples and tutorials for the CMU Language Model Inference course (Fall 2025).

## Repository Structure
- Each class/topic has its own directory with class numbering (e.g., `01-lm-intro/`)
- Python scripts (`.py`) are the canonical source code
- Jupyter notebooks (`.ipynb`) are generated from Python scripts using `jupytext`
- All code should be both executable as scripts and readable as educational notebooks

## Development Guidelines
- Python files should include jupytext headers and markdown cells for notebook generation
- Use `jupytext --to notebook script.py` to generate notebooks from Python scripts
- Ensure all code is well-documented and educational in nature
- Include proper imports and dependencies
- Test both Python script and notebook versions before committing

## Dependencies
The main dependencies are: torch, transformers, matplotlib, seaborn, numpy, pandas, jupytext

## Code Style
- Write clean, educational code with clear explanations
- Use markdown cells in Python files (# %% [markdown]) for documentation
- Focus on demonstrating concepts rather than production optimization
- Include examples and visualizations where appropriate

## Testing
- Verify that Python scripts run correctly
- Ensure generated notebooks display properly
- Test that all required dependencies are available