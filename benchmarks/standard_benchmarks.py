#!/usr/bin/env python3
"""
Standard Benchmark Evaluation: Arbitrium Framework on BBH & GPQA

Uses well-established academic benchmarks for instant credibility:
- BBH (Big-Bench Hard): 23 challenging reasoning tasks
- GPQA (Graduate-Level Questions): Expert-level science questions

These benchmarks are respected everywhere and provide objective comparison.

Prerequisites:
    pip install datasets  # Hugging Face datasets library

Usage:
    python -m arbitrium.benchmarks.standard_benchmarks --config config.benchmark.yml
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not installed")
    print("Install with: pip install datasets")
    sys.exit(1)

from arbitrium.config.loader import load_config
from arbitrium.logging import get_contextual_logger
from arbitrium.models.base import LiteLLMModel
from benchmarks.utils import initialize_benchmark

logger = get_contextual_logger("benchmarks.standard_benchmarks")


# BBH task subset (most relevant for multi-model reasoning)
BBH_TASKS = [
    "causal_judgement",  # Cause-effect reasoning
    "formal_fallacies",  # Logical reasoning
    "navigate",  # Spatial reasoning
    "disambiguation_qa",  # Ambiguity resolution
    "logical_deduction_three_objects",  # Deductive reasoning
    "tracking_shuffled_objects_three_objects",  # Working memory
    "web_of_lies",  # Complex inference
    "movie_recommendation",  # Multi-constraint optimization
]


def extract_answer(response: str, choices: list[str]) -> str:
    """Extract answer from model response."""
    response_lower = response.lower().strip()

    # Try to find exact choice match
    for choice in choices:
        if choice.lower() in response_lower:
            return choice

    # Try to find answer markers
    for marker in ["answer:", "answer is:", "the answer is", "therefore,"]:
        if marker in response_lower:
            after_marker = response_lower.split(marker, 1)[1].strip()
            # Get first word/letter after marker
            first_word = after_marker.split()[0] if after_marker.split() else ""
            for choice in choices:
                if choice.lower().startswith(first_word[:1]):
                    return choice

    # Default: return first choice mentioned
    for choice in choices:
        if choice.lower() in response_lower:
            return choice

    return choices[0]  # Fallback


async def run_single_model_on_benchmark(
    questions: list[dict[str, Any]],
    model_name: str,
    config_path: str,
) -> dict[str, Any]:
    """Run single model on benchmark questions."""
    config = load_config(config_path)
    model_config = config["models"][model_name]
    model = LiteLLMModel.from_config(model_name, model_config)

    results = []
    correct = 0

    for i, question_item in enumerate(questions, 1):
        prompt = f"""
Answer the following question. Think step-by-step, then provide your final answer.

Question: {question_item["question"]}

Choices:
{chr(10).join(f"- {choice}" for choice in question_item["choices"])}

Your answer:
""".strip()

        try:
            model_response = await model.generate(prompt)

            if model_response.is_error():
                logger.error(f"Error on question {i}: {model_response.error}")
                results.append(
                    {
                        "question": question_item["question"],
                        "predicted": None,
                        "actual": question_item["answer"],
                        "correct": False,
                        "error": model_response.error,
                    }
                )
                continue

            response_text = model_response.content
            predicted_answer = extract_answer(response_text, question_item["choices"])
            is_correct = predicted_answer == question_item["answer"]

            if is_correct:
                correct += 1

            results.append(
                {
                    "question": question_item["question"],
                    "predicted": predicted_answer,
                    "actual": question_item["answer"],
                    "correct": is_correct,
                    "response": response_text,
                }
            )

            logger.info(f"[{i}/{len(questions)}] {'✓' if is_correct else '✗'} {question_item['task']}")

        except Exception as e:
            logger.error(f"Error on question {i}: {e}")
            results.append(
                {
                    "question": question_item["question"],
                    "predicted": None,
                    "actual": question_item["answer"],
                    "correct": False,
                    "error": str(e),
                }
            )

    accuracy = (correct / len(questions)) * 100 if questions else 0

    return {
        "model": model_name,
        "approach": "single_model",
        "results": results,
        "correct": correct,
        "total": len(questions),
        "accuracy": accuracy,
    }


async def run_arbitrium_on_benchmark(
    questions: list[dict[str, Any]],
    config_path: str,
) -> dict[str, Any]:
    """Run Arbitrium Framework tournament on benchmark questions."""
    # Initialize benchmark components
    _config, _models, comparison = initialize_benchmark(config_path)

    results = []
    correct = 0

    for i, question_item in enumerate(questions, 1):
        prompt = f"""
Answer the following question. Provide your final answer clearly.

Question: {question_item["question"]}

Choices:
{chr(10).join(f"- {choice}" for choice in question_item["choices"])}
""".strip()

        try:
            # Run tournament
            response = await comparison.run(prompt)
            predicted_answer = extract_answer(response, question_item["choices"])
            is_correct = predicted_answer == question_item["answer"]

            if is_correct:
                correct += 1

            results.append(
                {
                    "question": question_item["question"],
                    "predicted": predicted_answer,
                    "actual": question_item["answer"],
                    "correct": is_correct,
                    "champion": comparison.active_model_keys[0] if comparison.active_model_keys else "Unknown",
                    "response": response,
                }
            )

            logger.info(f"[{i}/{len(questions)}] {'✓' if is_correct else '✗'} {question_item['task']} (Champion: {results[-1]['champion']})")

        except Exception as e:
            logger.error(f"Error on question {i}: {e}")
            results.append(
                {
                    "question": question_item["question"],
                    "predicted": None,
                    "actual": question_item["answer"],
                    "correct": False,
                    "error": str(e),
                }
            )

    accuracy = (correct / len(questions)) * 100 if questions else 0

    return {
        "approach": "arbitrium",
        "results": results,
        "correct": correct,
        "total": len(questions),
        "accuracy": accuracy,
    }


def load_bbh_questions(num_per_task: int = 5) -> list[dict[str, Any]]:
    """Load questions from Big-Bench Hard."""
    logger.info(f"Loading BBH tasks: {', '.join(BBH_TASKS)}")

    questions = []

    for task in BBH_TASKS:
        try:
            dataset = load_dataset("lukaemon/bbh", task)  # nosec B615

            # Take subset
            for i, example in enumerate(dataset["test"]):
                if i >= num_per_task:
                    break

                # BBH format: input text, target answer, multiple choice options
                questions.append(
                    {
                        "task": task,
                        "question": example["input"],
                        "answer": example["target"],
                        "choices": example.get("choices", [example["target"]]),  # Some tasks have choices
                    }
                )

        except Exception as e:
            logger.warning(f"Could not load task '{task}': {e}")

    logger.info(f"Loaded {len(questions)} questions from {len(BBH_TASKS)} BBH tasks")
    return questions


def load_gpqa_questions(num_questions: int = 20) -> list[dict[str, Any]]:
    """Load questions from GPQA (Graduate-Level Questions)."""
    logger.info("Loading GPQA dataset...")

    try:
        # GPQA Diamond subset (highest quality)
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")  # nosec B615

        questions = []
        for i, example in enumerate(dataset["train"]):
            if i >= num_questions:
                break

            questions.append(
                {
                    "task": "gpqa",
                    "question": example["Question"],
                    "answer": example["Correct Answer"],
                    "choices": [
                        example["Correct Answer"],
                        example["Incorrect Answer 1"],
                        example["Incorrect Answer 2"],
                        example["Incorrect Answer 3"],
                    ],
                }
            )

        logger.info(f"Loaded {len(questions)} GPQA questions")
        return questions

    except Exception as e:
        logger.error(f"Could not load GPQA: {e}")
        return []


async def run_benchmark_suite(
    benchmark: str,
    config_path: str,
    num_questions: int = 20,
):
    """Run complete benchmark comparison."""
    print("=" * 80)
    print(f"STANDARD BENCHMARK: {benchmark.upper()}")
    print("=" * 80)
    print()

    # Load questions
    if benchmark == "bbh":
        questions = load_bbh_questions(num_per_task=5)
    elif benchmark == "gpqa":
        questions = load_gpqa_questions(num_questions=num_questions)
    else:
        print(f"Unknown benchmark: {benchmark}")
        return

    if not questions:
        print("No questions loaded. Exiting.")
        return

    print(f"\nLoaded {len(questions)} questions")
    print("Running baseline (single model) and Arbitrium Framework tournament...\n")

    # Load config
    config = load_config(config_path)
    models = config["models"]
    baseline_results_list = []

    # Run baselines
    for model_name in models:
        print("=" * 80)
        print(f"BASELINE: Single Model ({model_name})")
        print("=" * 80 + "\n")

        baseline_results = await run_single_model_on_benchmark(questions, model_name, config_path)
        baseline_results_list.append(baseline_results)

        print(f"\n✅ Baseline Complete: {baseline_results['accuracy']:.1f}% accuracy")

    # Run Arbitrium Framework
    print("\n" + "=" * 80)
    print("ARBITRIUM TOURNAMENT")
    print("=" * 80 + "\n")

    tournament_results = await run_arbitrium_on_benchmark(questions, config_path)

    print(f"\n✅ Tournament Complete: {tournament_results['accuracy']:.1f}% accuracy")

    # Save results
    output_path = Path(__file__).parent / f"{benchmark}_benchmark_results.json"

    with open(output_path, "w") as f:
        json.dump(
            {
                "benchmark": benchmark,
                "date": datetime.now().isoformat(),
                "config": config_path,
                "num_questions": len(questions),
                "baselines": baseline_results_list,
                "tournament": tournament_results,
            },
            f,
            indent=2,
        )

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nBenchmark: {benchmark.upper()}")
    print(f"Questions: {len(questions)}")
    print()
    for baseline_results in baseline_results_list:
        print(f"Baseline Accuracy ({baseline_results['model']}):    {baseline_results['accuracy']:.1f}%")
    print(f"Tournament Accuracy:  {tournament_results['accuracy']:.1f}%")
    print()

    if baseline_results_list:
        best_baseline = max(baseline_results_list, key=lambda x: x["accuracy"])
        if tournament_results["accuracy"] > best_baseline["accuracy"]:
            print("✅ Arbitrium Framework OUTPERFORMED best single model!")
        elif tournament_results["accuracy"] < best_baseline["accuracy"]:
            print("⚠️  Best single model outperformed Arbitrium Framework")
        else:
            print("- No significant difference")

    print(f"\n📄 Detailed results saved to: {output_path}")
    print()
    print("NEXT STEPS:")
    print("1. Analyze which question types benefited from tournament")
    print("2. Run statistical significance test (chi-square)")
    print("3. Document results in README with benchmark name")
    print("4. Share: 'Validated on [BBH/GPQA] benchmark'")


def main():
    parser = argparse.ArgumentParser(description="Run standard benchmarks")
    parser.add_argument(
        "--benchmark",
        choices=["bbh", "gpqa", "both"],
        default="bbh",
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--config",
        default="config.benchmark.small.yml",
        help="Config file to use",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=20,
        help="Number of questions (for GPQA)",
    )

    args = parser.parse_args()

    if args.benchmark == "both":
        asyncio.run(run_benchmark_suite("bbh", args.config, args.num_questions))
        asyncio.run(run_benchmark_suite("gpqa", args.config, args.num_questions))
    else:
        asyncio.run(
            run_benchmark_suite(
                args.benchmark,
                args.config,
                args.num_questions,
            )
        )


if __name__ == "__main__":
    main()
