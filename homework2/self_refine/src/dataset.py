# dataset.py
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import ast 

# Dataset abstract class
class DatasetHandler(ABC):
    @abstractmethod
    def format_question(self, example: Dict[str, Any]) -> str: ...
    # TODO: format the question here based on the example structure

    @abstractmethod
    def parse_answer(self, response: str) -> Any: ...
    # TODO: parse the model response to extract the answer
    
    @abstractmethod
    def verify_answer(self, predicted: Any, ground_truth: Any) -> bool: ...
    # TODO: implement answer verification logic
    
    @abstractmethod
    def get_ground_truth(self, example: Dict[str, Any]) -> Any: ...
    # TODO: extract ground truth from the example


# You need to implement these handlers based on your datasets
# Each dataset can have its own parsing, verification logic

class GraphHandler(DatasetHandler):
    """Handler for graph pathfinding dataset."""
    

class InfobenchHandler(DatasetHandler):
    """Handler for Infobench dataset."""
    

class MMLUMedHandler(DatasetHandler):
    """Handler for MMLU medical dataset."""
    

