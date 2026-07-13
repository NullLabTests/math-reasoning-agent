#!/usr/bin/env python3
"""Post-training evaluation: compare baseline vs RL-trained model.

Usage:
    python scripts/post_training_eval.py \
        --baseline PrimeIntellect/Qwen3-0.6B-Base \
        --trained ./checkpoints/step-100 \
        --problems 20
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment import MathReasoningEnv, extract_final_answer


def load_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
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


def generate(model, tokenizer, prompt: str, max_new: int = 1024) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def evaluate_model(model, tokenizer, model_name: str, env: MathReasoningEnv,
                   problems: list[dict]) -> dict:
    correct = 0
    total_reward = 0.0
    results = []

    for i, problem in enumerate(problems, 1):
        prompt = env.get_prompt(problem)
        start = time.time()
        response = generate(model, tokenizer, prompt)
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
            "format_score": eval_result["format_score"],
            "answer_score": eval_result["answer_score"],
            "reasoning_score": eval_result["reasoning_score"],
            "time_seconds": round(elapsed, 2),
        })

        if is_correct:
            correct += 1
        total_reward += eval_result["reward"]
        status = "✓" if is_correct else "✗"
        print(f"  [{status}] {problem['problem'][:60]}... -> {eval_result['extracted_answer']} "
              f"(expected: {problem['answer']}) [{elapsed:.1f}s]")

    accuracy = correct / len(problems)
    avg_reward = total_reward / len(problems)

    return {
        "model": model_name,
        "num_problems": len(problems),
        "accuracy": accuracy,
        "correct": correct,
        "avg_reward": avg_reward,
        "results": results,
    }


def compare(baseline_model: str, trained_model: str, num_problems: int = 10,
            output_file: str = None, env_sources: list[str] = None):
    env = MathReasoningEnv()
    problems = env.problems[:num_problems]

    # Load baseline model
    print(f"\n{'='*60}")
    print("Loading BASELINE model...")
    base_model, base_tokenizer = load_model(baseline_model)
    baseline_results = evaluate_model(base_model, base_tokenizer, baseline_model, env, problems)
    del base_model

    # Load trained model
    print(f"\n{'='*60}")
    print("Loading TRAINED model...")
    trained_model_obj, trained_tokenizer = load_model(trained_model)
    trained_results = evaluate_model(trained_model_obj, trained_tokenizer, trained_model, env, problems)
    del trained_model_obj

    # Compare
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Baseline':<15} {'Trained':<15} {'Delta':<10}")
    print(f"{'-'*65}")
    print(f"{'Accuracy':<25} {baseline_results['accuracy']:.1%}          "
          f"{trained_results['accuracy']:.1%}          "
          f"{trained_results['accuracy'] - baseline_results['accuracy']:+.1%}")
    print(f"{'Avg Reward':<25} {baseline_results['avg_reward']:.4f}         "
          f"{trained_results['avg_reward']:.4f}         "
          f"{trained_results['avg_reward'] - baseline_results['avg_reward']:+.4f}")
    print(f"{'Correct':<25} {baseline_results['correct']}/{baseline_results['num_problems']}         "
          f"{trained_results['correct']}/{trained_results['num_problems']}")

    # Detailed per-problem comparison
    print(f"\n{'='*60}")
    print("PER-PROBLEM COMPARISON")
    print(f"{'='*60}")
    for base_r, trained_r in zip(baseline_results["results"], trained_results["results"]):
        base_ext = base_r["extracted_answer"] or "(none)"
        trained_ext = trained_r["extracted_answer"] or "(none)"
        improved = trained_r["correct"] and not base_r["correct"]
        regressed = base_r["correct"] and not trained_r["correct"]
        arrow = "↑ IMPROVED" if improved else ("↓ REGRESSED" if regressed else "  unchanged")
        print(f"  {arrow} | Base: {base_ext:<15} | Trained: {trained_ext:<15} | "
              f"Expected: {base_r['expected_answer']}")

    comparison = {
        "baseline": baseline_results,
        "trained": trained_results,
        "improvement": {
            "accuracy_delta": trained_results["accuracy"] - baseline_results["accuracy"],
            "reward_delta": trained_results["avg_reward"] - baseline_results["avg_reward"],
            "newly_correct": sum(1 for b, t in zip(baseline_results["results"],
                                                     trained_results["results"])
                                 if not b["correct"] and t["correct"]),
            "regressed": sum(1 for b, t in zip(baseline_results["results"],
                                                trained_results["results"])
                              if b["correct"] and not t["correct"]),
        },
    }

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs RL-trained model on math reasoning")
    parser.add_argument("--baseline", default="PrimeIntellect/Qwen3-0.6B-Base",
                        help="Baseline HuggingFace model")
    parser.add_argument("--trained", required=True,
                        help="Trained model path or HF name")
    parser.add_argument("--problems", type=int, default=10,
                        help="Number of problems")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON")
    args = parser.parse_args()

    compare(args.baseline, args.trained, args.problems, args.output)


if __name__ == "__main__":
    main()
