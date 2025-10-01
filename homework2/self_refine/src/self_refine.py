# self_refine.py
import os, json, time, argparse
from dataclasses import dataclass
from pathlib import Path
import torch
import dataset
from typing import List, Dict, Any, Tuple
torch.manual_seed(42)
GraphHandler = dataset.GraphHandler 
MMLUMedHandler = dataset.MMLUMedHandler

os.environ["HF_HOME"] = ""

"""NOTE: The repo includes a bare-bones scaffolds. 
It exists to help you start quickly. 
Please feel free to change your structure. 
Any clean, reproducible solution is acceptable.
"""

@dataclass
class RefineConfig:
    """Configuration for self-refine process."""
    # Adjust as needed
    model_path: str = None
    dtype: str = "bfloat16"
    # chat_template_kwargs: Dict[str, Any] = None


def chat(role: str, content: str) -> str:
    """Format chat messages - adjust for your model's chat template"""
    raise NotImplementedError

# These are for abstractions you can make them dataset specific or agnostic based on your design
def draft_prompt(question: str, handler_type: str) -> str:
    # NOTE: This model is the solver 
    # You might wanna experiment with prompts and instruction for each dataset
    raise NotImplementedError

def feedback_prompt(question: str, attempt: str, handler_type: str) -> str:
    # Give the feedback on the attempt
    raise NotImplementedError

def refine_prompt(question: str, attempt: str, feedback: str, handler_type: str) -> str:
    # Refine the attempt based on feedback 
    raise NotImplementedError


class Generator:
    "LLM Engine for generation, feedback, and refinement"
    # You can use transformers, hf piepeline, vllm, etc.
    def __init__(self, cfg: RefineConfig):
        self.cfg = cfg
        self.model = None  # Initialize your model here


    def _gen(self, prompts: List[str]) -> List[str]:
        """generic generate  function to do inference over a list of prompts"""

    def draft(self, qs: List[str]) -> List[str]:
        """Generate initial drafts for questions"""
       

    def feedback(self, qs_attempts: List[Tuple[str, str]]) -> List[str]:
        """Generate feedback for question-attempt pairs"""

    def refine(self, qs_attempts_feedback: List[Tuple[str, str, str]]) -> List[str]:
        """Generate refinements for the attempts based on feedback"""


def run_self_refine(
    examples: List[Dict[str, Any]],
    handler: dataset.DatasetHandler,
    generator: Generator,
    config: RefineConfig,
) -> List[Dict[str, Any]]:
    """
    Implement the self-refinement algorithm.
    
    Args:
        examples: List of dataset examples
        handler: Dataset handler
        generator: Generator instance for model inference
        config: Your configuration
    
    Returns:
        - You might want to keep track of outputs at different stages so you can do interesting analysis later
    """
    raise NotImplementedError("Implement self-refinement")


def main():
    parser = argparse.ArgumentParser()
    HANDLERS = {
    "graph": GraphHandler,
    "mmlu_med": MMLUMedHandler,
    }
    # TODO: Initialize your generator and config
    # TODO: Load dataset
    # TODO: Run self-refine
    # TODO: Analyze results
    # TODO: Save outputs
    # TODO: Analyse the results

if __name__ == "__main__":
    main()