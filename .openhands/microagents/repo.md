This repository contains educational code examples and tutorials for the CMU Language Model Inference course (Fall 2025).

## Repository Structure

- Each class/topic has its own directory with class numbering (e.g., `01-lm-intro/`)
- Python scripts (`.py`) are the canonical source code
- Jupyter notebooks (`.ipynb`) are generated from Python scripts using `jupytext`
- All code should be both executable as scripts and readable as educational notebooks

## Development Guidelines

- This project is managed through `uv` so use that for dependency management
- Python files should include jupytext headers and markdown cells for notebook generation
- Use `jupytext --to notebook script.py` to generate notebooks from Python scripts
- All notebooks must include a "Open in Colab" badge linking to the GitHub notebook file
- Ensure all code is well-documented and educational in nature
- Include proper imports and dependencies
- Test both Python script and notebook versions before committing

## Code Style

- Write clean, educational code with clear explanations
- Use markdown cells in Python files (# %% [markdown]) for documentation
- Focus on demonstrating concepts rather than production optimization
- Include examples and visualizations where appropriate
- Colab badge format: `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neubig/lminference-fall2025-code/blob/main/XX-topic/filename.ipynb)`

## Testing

- Verify that Python scripts run correctly
- Ensure generated notebooks display properly
- Test that all required dependencies are available
