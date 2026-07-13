#!/usr/bin/env python3
"""Baseline evaluation: score a model on math reasoning before RL training.

Usage:
    python scripts/baseline_eval.py --model PrimeIntellect/Qwen3-0.6B-Base
    python scripts/baseline_eval.py --model PrimeIntellect/Qwen3-0.6B-Base --problems 20
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment import MathReasoningEnv, MathReward, extract_final_answer


def load_model(model_name: str):
    """Load a HuggingFace model for inference."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Error: transformers not installed. Run: pip install transformers")
        sys.exit(1)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def evaluate(
    model_name: str,
    num_problems: int = 10,
    output_file: Optional[str] = None,
    show_traces: bool = False,
):
    model, tokenizer = load_model(model_name)
    env = MathReasoningEnv()
    problems = env.problems[:num_problems]

    results = []
    correct = 0
    total_reward = 0.0

    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on {len(problems)} problems")
    print(f"{'='*60}\n")

    for i, problem in enumerate(problems, 1):
        prompt = env.get_prompt(problem)
        print(f"Problem {i}/{len(problems)}: {problem['problem'][:80]}...")

        start = time.time()
        response = generate_response(model, tokenizer, prompt)
        elapsed = time.time() - start

        eval_result = env.evaluate_response(problem, response)
        is_correct = eval_result["answer_score"] > 0.5

        results.append({
            "id": problem.get("id", i),
            "problem": problem["problem"],
            "expected_answer": problem["answer"],
            "response": response,
            "extracted_answer": eval_result["extracted_answer"],
            "correct": is_correct,
            "reward": eval_result["reward"],
            "time_seconds": round(elapsed, 2),
        })

        if is_correct:
            correct += 1
        total_reward += eval_result["reward"]

        status = "✓" if is_correct else "✗"
        print(f"  [{status}] Expected: {problem['answer']} | Got: {eval_result['extracted_answer']} | "
              f"Reward: {eval_result['reward']:.3f} | Time: {elapsed:.1f}s")

        if show_traces:
            print(f"\n  Response:\n{response[:500]}\n")

    accuracy = correct / len(problems)
    avg_reward = total_reward / len(problems)

    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy:  {accuracy:.1%} ({correct}/{len(problems)})")
    print(f"  Avg Reward: {avg_reward:.4f}")
    print(f"{'='*60}")

    summary = {
        "model": model_name,
        "num_problems": len(problems),
        "accuracy": accuracy,
        "correct": correct,
        "avg_reward": avg_reward,
        "results": results,
    }

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate a baseline model on math reasoning")
    parser.add_argument("--model", default="PrimeIntellect/Qwen3-0.6B-Base",
                        help="HuggingFace model name")
    parser.add_argument("--problems", type=int, default=10,
                        help="Number of problems to evaluate")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")
    parser.add_argument("--traces", action="store_true",
                        help="Show full response traces")
    args = parser.parse_args()

    evaluate(args.model, args.problems, args.output, args.traces)


if __name__ == "__main__":
    main()
