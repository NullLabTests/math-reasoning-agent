import re
import ast
import math
from typing import Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the content inside the final \\boxed{} command."""
    if "\\boxed{" not in text:
        return None
    # Find the LAST \\boxed{...} in the text
    idx = text.rfind("\\boxed{")
    brace_count = 0
    i = idx + len("\\boxed{")
    start = i
    while i < len(text):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            if brace_count == 0:
                return text[start:i]
            brace_count -= 1
        i += 1
    return text[start:i]


def extract_final_answer(text: str) -> Optional[str]:
    """Try multiple answer extraction strategies."""
    # Try boxed first
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed.strip()
    # Try "final answer is X" pattern
    patterns = [
        r"(?:final|answer|result)(?:\s+is|\s*:|=)\s*(\d+(?:\.\d+)?)",
        r"(?:the\s+)?answer\s+(?:is|:)\s*(.+?)(?:\.|$)",
        r"\[/?box\](.+?)\[/?box\]",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def normalize_answer(ans: str) -> str:
    """Normalize an answer string for comparison."""
    ans = ans.strip()
    # Remove commas
    ans = ans.replace(",", "")
    # Try to parse as a number
    try:
        num = ast.literal_eval(ans)
        if isinstance(num, (int, float)):
            if num == int(num):
                return str(int(num))
            return f"{num:.10f}".rstrip("0").rstrip(".")
    except (ValueError, SyntaxError, MemoryError):
        pass
    return ans.lower().strip()


def format_reward(completion: str, **kwargs) -> float:
    """Reward for proper reasoning format."""
    score = 0.0
    # Has reasoning tags
    if re.search(r"<reasoning>.*?</reasoning>", completion, re.DOTALL):
        score += 0.3
    # Has boxed answer
    if "\\boxed{" in completion and "}" in completion:
        score += 0.3
    # Has proper structure
    if re.search(r"(?:step|reason|therefore|since|because)", completion, re.IGNORECASE):
        score += 0.2
    # Not too short (at least 50 chars for a reasoning trace)
    if len(completion) >= 50:
        score += 0.2
    return min(score, 1.0)


def answer_reward(completion: str, answer: str, **kwargs) -> float:
    """Reward for correct final answer."""
    predicted = extract_final_answer(completion)
    if predicted is None:
        return 0.0
    try:
        return 1.0 if normalize_answer(predicted) == normalize_answer(answer) else 0.0
    except Exception:
        return 0.0


def reasoning_quality_reward(completion: str, **kwargs) -> float:
    """Reward for reasoning quality (chain-of-thought)."""
    score = 0.0
    lines = completion.split("\n")
    # Reward for step-by-step reasoning
    steps = [l for l in lines if re.match(r"^\d+[\.\)]|^-\s|^\\bullet|^\*", l.strip())]
    score += min(len(steps) * 0.1, 0.3)
    # Reward for mathematical notation
    math_expressions = re.findall(r"\\[a-zA-Z]+|\$.*?\$|\(.*?\)", completion)
    score += min(len(math_expressions) * 0.02, 0.2)
    # Penalize for very short completions (no reasoning)
    if len(completion.strip()) < 30:
        score -= 0.5
    # Reward for references to quantities and calculations
    if re.search(r"\d+\.?\d*\s*[+\-*/]\s*\d+\.?\d*", completion):
        score += 0.2
    # Reward for clear conclusion indicators
    if re.search(r"(therefore|thus|hence|so|conclusion|answer)", completion, re.IGNORECASE):
        score += 0.1
    return max(min(score, 1.0), -0.5)


class MathReward:
    """Multi-component reward function for math reasoning.

    Combines:
    - Format reward (0.2 weight)
    - Correct answer reward (0.5 weight)
    - Reasoning quality reward (0.3 weight)
    """

    def __init__(self, weights: Optional[dict] = None):
        self.weights = weights or {
            "format": 0.2,
            "answer": 0.5,
            "reasoning": 0.3,
        }

    def __call__(self, completion: str, answer: str, **kwargs) -> dict:
        format_score = format_reward(completion, **kwargs)
        answer_score = answer_reward(completion, answer, **kwargs)
        reasoning_score = reasoning_quality_reward(completion, **kwargs)

        total = (
            self.weights["format"] * format_score
            + self.weights["answer"] * answer_score
            + self.weights["reasoning"] * reasoning_score
        )

        return {
            "total": total,
            "format": format_score,
            "answer": answer_score,
            "reasoning": reasoning_score,
        }

    @staticmethod
    def create_rubric(parser=None):
        """Create a verifiers Rubric from this reward.
        
        Requires verifiers to be installed: pip install verifiers
        """
        import verifiers as vf
        if parser is None:
            parser = vf.Parser(extract_fn=extract_final_answer)
        rubric = vf.Rubric(parser=parser)

        async def format_rf(completion, answer, **kw):
            return format_reward(completion, **kw)

        async def answer_rf(completion, answer, **kw):
            return answer_reward(completion, answer, **kw)

        async def reasoning_rf(completion, answer, **kw):
            return reasoning_quality_reward(completion, **kw)

        rubric.add_reward_func(format_rf, weight=0.2)
        rubric.add_reward_func(answer_rf, weight=0.5)
        rubric.add_reward_func(reasoning_rf, weight=0.3)

        return rubric
