# grader.py
"""
Grading logic for MMLU, InfoBench, and Graph tasks.
"""

import json
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from openai import AsyncOpenAI, OpenAI
import math_verify
import time

# ============================================================================
# CONSTANTS
# ============================================================================

SYS_MSG = ("Based on the provided Input (if any) and Generated Text, answer the ensuing Questions with either a YES or NO choice. "
           "Your selection should be based on your judgment as well as the following rules:\n\n"
           "- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. "
           "However, note that even minor inaccuracies exclude the text from receiving a 'YES' rating. "
           "As an illustration. consider a question that asks. \"Does each sentence in the generated text use a second person?\" "
           "If even one sentence does not use the second person, the answer should NOT be 'YES'. "
           "To qualify for a 'YES' rating, the generated text must be entirely accurate and relevant to the question\n\n"
           "- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no information "
           "that could be utilized to answer the question. For instance, if the question asks. "
           "\"Is the second sentence in the generated text a compound sentence?\" and the generated text only has one sentence. "
           "it offers no relevant information to answer the question. Consequently, the answer should be 'NO'.''")


# ============================================================================
# RESPONSE PARSERS
# ============================================================================

class ResponseParser:
    """Parse student responses for different task types"""

    @staticmethod
    def parse_mmlu(response: str) -> Optional[List[str]]:
        """
        Extract multiple choice answer from response using math_verify.
        Returns parsed result or None if parsing fails.
        """
        if not response:
            return None
        response = response.strip()
        # parse the answer after "The answer is (X)"
        answer_match = re.search(r'The answer is\s*\(?([A-Z])\)?', response, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).upper()
            return [answer]
        else:
            # Fallback to math_verify parsing
            parsed = math_verify.parse(response)
        return parsed if parsed else None

    @staticmethod
    def parse_graph(response: str) -> Optional[Dict[str, Any]]:
        """
        Parse graph response. Supports two formats:
        1. Gold-like format: {"paths": [{"path": [0,2,4], "weight": 25}]}
        2. Function call: submit_paths(paths=[[0,2,4]], weights=[25])

        Returns dict with 'paths' (list of lists) and 'weights' (list of ints) or None.
        """
        if not response:
            return None

        response = response.strip()

        # === Format 1: Gold-like format {"paths": [{"path": [...], "weight": ...}]} ===
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if "paths" in parsed and isinstance(parsed["paths"], list):
                    # Check if it's gold-like format (list of dicts with path/weight)
                    if parsed["paths"] and isinstance(parsed["paths"][0], dict) and "path" in parsed["paths"][0]:
                        paths = [p["path"] for p in parsed["paths"]]
                        weights = [p["weight"] for p in parsed["paths"]]
                        return {"paths": paths, "weights": weights}
        except (json.JSONDecodeError, KeyError, TypeError, IndexError):
            pass

        # === Format 2: Function call submit_paths(paths=[[...]], weights=[...]) ===
        func_patterns = [
            r'submit_paths\s*\(\s*paths\s*=\s*(\[.*?\])\s*,\s*weights\s*=\s*(\[.*?\])\s*\)',
            r'submit_paths\s*\(\s*weights\s*=\s*(\[.*?\])\s*,\s*paths\s*=\s*(\[.*?\])\s*\)',
        ]

        for i, pattern in enumerate(func_patterns):
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    if i == 0:  # paths first
                        paths = eval(match.group(1))
                        weights = eval(match.group(2))
                    else:  # weights first
                        weights = eval(match.group(1))
                        paths = eval(match.group(2))
                    return {"paths": paths, "weights": weights}
                except:
                    continue

        return None


# ============================================================================
# EVALUATORS
# ============================================================================

class MMLUEvaluator:
    """Evaluator for MMLU multiple choice questions"""

    @staticmethod
    def evaluate(response: str, gold_answer: str) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate MMLU response using math_verify.
        Returns (score, details_dict)
        """
        parsed_answer = ResponseParser.parse_mmlu(response)
        # gold_parsed = math_verify.parse(gold_answer)
        gold_parsed = gold_answer
        is_correct = math_verify.verify(gold_parsed, parsed_answer)

        return (
            1.0 if is_correct else 0.0,
            {
                "parsed_answer": str(parsed_answer),
                "gold_answer": gold_answer,
                "gold_parsed": str(gold_parsed),
                "correct": is_correct
            }
        )


class GraphEvaluator:
    """Evaluator for graph shortest path problems"""

    @staticmethod
    def evaluate(response: str, gold_answer: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate graph response.
        Returns (score, details_dict)

        Gold answer format: {"paths": [{"path": [...], "weight": ...}, ...]}
        """
        parsed = ResponseParser.parse_graph(response)

        if parsed is None:
            return (
                0.0,
                {
                    "parsed_paths": None,
                    "gold_paths": gold_answer,
                    "parse_error": True,
                    "matches": 0,
                    "total": len(gold_answer.get("paths", []))
                }
            )

        # Convert gold answer to set of (path_tuple, weight)
        gold_paths_set = set()
        for path_info in gold_answer.get("paths", []):
            path_tuple = tuple(path_info["path"])
            weight = path_info["weight"]
            gold_paths_set.add((path_tuple, weight))

        # Convert parsed answer to same format
        parsed_paths_set = set()
        parsed_paths = parsed.get("paths", [])
        parsed_weights = parsed.get("weights", [])

        for i, path in enumerate(parsed_paths):
            if i < len(parsed_weights):
                path_tuple = tuple(path)
                weight = parsed_weights[i]
                parsed_paths_set.add((path_tuple, weight))

        # Calculate matches
        matches = len(gold_paths_set.intersection(parsed_paths_set))
        total = len(gold_paths_set)
        score = matches / total if total > 0 else 0.0

        return (
            score,
            {
                "parsed_paths": parsed,
                "gold_paths": gold_answer,
                "matches": matches,
                "total": total,
                "parse_error": False
            }
        )


class InfoBenchEvaluator:
    """Async InfoBench evaluator - processes decomposed questions  sequentially"""

    def __init__(self, openai_api_key: str, eval_model: str = "gpt-5-nano-2025-08-07"):
        self.openai_api_key = openai_api_key
        self.eval_model = eval_model
        self.client = OpenAI(api_key=openai_api_key)

    def verify_connection(self) -> bool:
        """Verify that we can connect to OpenAI API."""
        try:
            _ = self.client.chat.completions.create(
                model=self.eval_model,
                messages=[{"role": "user", "content": "Say YES"}],
            )
            return True
        except Exception as e:
            print(f"OpenAI connection failed: {e}")
            return False

    def _parse_yes_no(self, generation: str) -> Optional[bool]:
        """Parse a YES/NO response from the evaluator model."""
        generation = generation.strip()
        if generation.lower().startswith("yes"):
            return True
        elif generation.lower().startswith("no"):
            return False
        elif "YES" in generation and "NO" not in generation:
            return True
        elif "YES" not in generation and "NO" in generation:
            return False
        else:
            print(f"Ambiguous answer: {generation}")
            return None

    def _evaluate_sequential(
        self,
        request_i: int,
        meta: Dict[str, Any],
        predicted_solution: str
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        Sequential evaluation - processes decomposed questions while maintaining
        conversation context (message history) across all questions.

        This is the CORRECT implementation that keeps the generated output R
        in context for all questions, not just the first one.

        Returns (request_i, score, details)
        """
        input_task = meta.get('input', '')
        decomposed_questions = meta.get("decomposed_questions", [])

        if not decomposed_questions:
            return request_i, 0.0, {"error": "No decomposed questions found"}

        # Initialize message list ONCE - this is the key fix!
        # The message list accumulates the full conversation history
        message = []
        bool_results = []

        for i, question in enumerate(decomposed_questions):
            # Build the content for this question
            if len(message) == 0:
                # First question: include system message, input, and generated output
                if input_task:
                    content = f"{SYS_MSG}\n\nInput:\n\"{input_task}\"\n\nGenerated Text:\n\"{predicted_solution}\"\n\nQuestion:\n{question}\n"
                else:
                    content = f"{SYS_MSG}\n\nGenerated Text:\n\"{predicted_solution}\"\n\nQuestion:\n{question}\n"
            else:
                # Subsequent questions: just the question (context is already in message history)
                content = f"{question}\n"

            # Append user message to conversation history
            message.append({"role": "user", "content": content})

            # Try to get evaluation with retries
            max_retries = 3
            success = False
            result = None

            for attempt in range(max_retries):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.eval_model,
                        messages=message,
                        temperature=1.0, # only default 1.0 is supported (0. didnt work)
                    )
                    generation = completion.choices[0].message.content
                    # Append assistant response to conversation history
                    message.append({"role": "assistant", "content": generation})
                    print(f"Q{i+1}: {question} => {generation.strip()}")

                    # Parse the YES/NO response
                    result = self._parse_yes_no(generation)
                    success = True
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        delay = (2 ** attempt) + (0.1 * request_i)
                        print(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                        time.sleep(delay)
                    else:
                        print(f"Failed after {max_retries} attempts: {e}")
                        result = None

            bool_results.append(result)

            # Early stop if we get an ambiguous answer
            if result is None:
                # Fill remaining with None and break
                remaining = len(decomposed_questions) - len(bool_results)
                bool_results.extend([None] * remaining)
                break

        # Calculate score
        valid_results = [r for r in bool_results if r is not None]
        if not valid_results:
            return request_i, 0.0, {
                "question_results": bool_results,
                "valid_count": 0,
                "total_questions": len(decomposed_questions)
            }

        num_yes = sum(r is True for r in bool_results)
        ratio = num_yes / len(decomposed_questions)

        return request_i, ratio, {
            "question_results": bool_results,
            "valid_count": len(valid_results),
            "total_questions": len(decomposed_questions)
        }

    def evaluate(self, request_i: int, meta: Dict[str, Any], predicted_solution: str) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate an InfoBench example.

        Args:
            request_i: Index of the request (for logging/tracking)
            meta: Dictionary containing 'input' and 'decomposed_questions'
            predicted_solution: The model's generated output to evaluate

        Returns:
            Tuple of (score, details_dict)
        """
        _, score, details = self._evaluate_sequential(request_i, meta, predicted_solution)
        return score, details


# ============================================================================
# SINGLE ITEM EVALUATION
# ============================================================================

def evaluate_single(
    idx: int,
    test_item: Dict[str, Any],
    student_response: str,
    infobench_evaluator: InfoBenchEvaluator
) -> Dict[str, Any]:
    """
    Evaluate a single test item.
    Returns dict with input, output, score, and details.
    """
    index = test_item["index"]
    task = test_item["task"]
    prompt = test_item["prompt"]
    gold_answer = test_item["gold_answer"]
    meta = test_item.get("meta", {})

    # Evaluate based on task
    if not student_response:
        score = 0.0
        details = {"error": "No response"}
    elif task == "mmlu_med":
        score, details = MMLUEvaluator.evaluate(student_response, gold_answer)
    elif task == "graph":
        score, details = GraphEvaluator.evaluate(student_response, gold_answer)
    elif task == "infobench":
        score, details = infobench_evaluator.evaluate(idx, meta, student_response)
    else:
        score = 0.0
        details = {"error": f"Unknown task {task}"}

    return {
        "index": index,
        "task": task,
        "prompt": prompt,  # Input that student saw
        "student_output": student_response,  # Raw output from student
        "gold_answer": gold_answer,
        "score": score,
        "eval_details": details
    }