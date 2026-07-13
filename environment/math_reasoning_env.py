"""Custom math reasoning environment built on Prime Intellect's verifiers library.

Supports loading math problems from datasets (AIME, GSM8K, custom JSON),
with configurable system prompts and reward functions.
"""

import json
import logging
import os
from typing import Callable, Optional

from datasets import load_dataset

from .reward import MathReward, extract_final_answer


DEFAULT_SYSTEM_PROMPT = """You are a mathematical reasoning assistant with expert-level problem-solving skills.

For each problem, follow this structure:
<reasoning>
Think step by step. Show all work clearly.
Use proper mathematical notation.
Break the problem into steps, numbering each one.
Verify your work at each step.
</reasoning>
\\boxed{Your Final Answer}
"""

SAMPLE_PROBLEMS = [
    {
        "id": "sample_1",
        "problem": "If $3^{2x} = 81$, what is the value of $x$?",
        "answer": "2",
        "source": "sample",
    },
    {
        "id": "sample_2",
        "problem": "A train travels at 60 miles per hour for 2.5 hours. How many miles does it travel?",
        "answer": "150",
        "source": "sample",
    },
    {
        "id": "sample_3",
        "problem": "What is the sum of the first 50 positive integers?",
        "answer": "1275",
        "source": "sample",
    },
    {
        "id": "sample_4",
        "problem": "Solve for $y$: $5y - 3 = 2y + 9$",
        "answer": "4",
        "source": "sample",
    },
    {
        "id": "sample_5",
        "problem": "A bag contains 3 red marbles and 5 blue marbles. What is the probability of drawing a red marble?",
        "answer": "3/8",
        "source": "sample",
    },
]


def load_from_datasets(
    dataset_name: str = "competition_math",
    split: str = "train",
    num_examples: int = -1,
) -> list[dict]:
    """Load math problems from Hugging Face datasets.

    Supports: competition_math, gsm8k, aime_2024, etc.
    """
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception:
        return []

    problems = []
    for i, item in enumerate(dataset):
        if num_examples > 0 and i >= num_examples:
            break

        if "problem" in item:
            problem = item["problem"]
        elif "question" in item:
            problem = item["question"]
        else:
            continue

        # Extract answer from various formats
        answer = None
        for key in ["answer", "solution", "final_answer"]:
            if key in item:
                answer = item[key]
                break

        if answer is None:
            continue

        problems.append({
            "id": f"{dataset_name}_{i}",
            "problem": problem,
            "answer": str(answer).strip(),
            "source": dataset_name,
        })

    return problems


def build_dataset(
    sources: Optional[list[str]] = None,
    num_train: int = -1,
    num_eval: int = -1,
) -> tuple[list[dict], list[dict]]:
    """Build training and evaluation datasets from available sources."""
    if sources is None:
        sources = ["sample", "competition_math", "gsm8k"]

    train_problems = []
    eval_problems = []

    # Always include sample problems
    train_problems.extend(SAMPLE_PROBLEMS)

    # Try loading from each dataset source
    for source in sources:
        if source == "sample":
            continue
        try:
            problems = load_from_datasets(source, split="train", num_examples=num_train)
            train_problems.extend(problems)
        except Exception:
            pass
        try:
            problems = load_from_datasets(source, split="test", num_examples=num_eval)
            eval_problems.extend(problems)
        except Exception:
            pass

    # If eval is empty, use some train problems
    if not eval_problems and train_problems:
        split_idx = max(1, len(train_problems) // 5)
        eval_problems = train_problems[:split_idx]
        train_problems = train_problems[split_idx:]

    return train_problems, eval_problems


def load_environment(
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    sources: Optional[list[str]] = None,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    use_python_tool: bool = False,
    env_name: str = "math-reasoning-agent",
):
    """Load the math reasoning environment.

    This is the standard entrypoint expected by the verifiers ecosystem.
    Returns a verifiers Environment ready for use with prime-rl training.

    Note: verifiers is imported lazily here since it has heavy dependencies
    not needed for the demo UI. Install with: pip install verifiers
    """
    import verifiers as vf
    logger = logging.getLogger(__name__)

    train_problems, eval_problems = build_dataset(
        sources=sources,
        num_train=num_train_examples,
        num_eval=num_eval_examples,
    )

    logger.info(
        "Loaded %d train and %d eval problems for '%s'",
        len(train_problems),
        len(eval_problems),
        env_name,
    )

    parser = vf.Parser(extract_fn=extract_final_answer)
    rubric = MathReward.create_rubric(parser=parser)

    if use_python_tool:
        env = vf.PythonEnv(
            dataset=train_problems,
            eval_dataset=eval_problems if eval_problems else None,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            max_turns=10,
            pip_install_packages="numpy sympy scipy",
            sandbox_cpu_cores=1,
            sandbox_memory_gb=2,
            sandbox_timeout_seconds=120,
        )
    else:
        env = vf.SingleTurnEnv(
            dataset=train_problems,
            eval_dataset=eval_problems if eval_problems else None,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
        )

    return env


class MathReasoningEnv:
    """High-level wrapper for the math reasoning environment.

    Provides a simpler API for use in demos and evaluation scripts,
    without the full verifiers event loop.
    """

    def __init__(
        self,
        problems: Optional[list[dict]] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.system_prompt = system_prompt
        self.reward_fn = MathReward()
        self.problems = problems or SAMPLE_PROBLEMS

    def get_prompt(self, problem: dict) -> str:
        return f"{self.system_prompt}\n\n## Problem\n\n{problem['problem']}\n\nSolve the problem step by step. Show your reasoning inside <reasoning> tags, and provide your final answer in \\boxed{{}}."

    def evaluate_response(self, problem: dict, response: str) -> dict:
        reward = self.reward_fn(response, problem["answer"])
        return {
            "problem": problem["problem"],
            "expected_answer": problem["answer"],
            "model_response": response,
            "extracted_answer": extract_final_answer(response),
            "reward": reward["total"],
            "format_score": reward["format"],
            "answer_score": reward["answer"],
            "reasoning_score": reward["reasoning"],
        }

    def grade(self, predicted: str, expected: str) -> bool:
        """Simple exact/numeric match grader."""
        from .reward import normalize_answer
        try:
            return normalize_answer(predicted) == normalize_answer(expected)
        except Exception:
            return predicted.strip().lower() == expected.strip().lower()
