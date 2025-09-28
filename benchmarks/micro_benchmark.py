#!/usr/bin/env python3
"""
Micro-Benchmark: Single Model vs Arbitrium Framework

Purpose: Get first empirical data point showing Arbitrium Framework value.
Method: Run 1 high-stakes question through single model + CoT vs Arbitrium Framework.

This is NOT rigorous scientific validation - it's a quick proof point.

Usage:
    python -m arbitrium.benchmarks.micro_benchmark --config config.benchmark.yml
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from arbitrium.config.loader import load_config
from arbitrium.logging import get_contextual_logger
from arbitrium.models.base import LiteLLMModel
from arbitrium.utils.benchmark import initialize_benchmark
from benchmarks.reporting import generate_manual_evaluation_template
from tests.shared_data import TECHNICAL_ARCHITECTURE_QUESTION

logger = get_contextual_logger("benchmarks.micro_benchmark")

# High-stakes test question
TEST_QUESTION = TECHNICAL_ARCHITECTURE_QUESTION

# Chain-of-thought prompt for single model
COT_PROMPT = """
Think step-by-step. Consider multiple perspectives:
1. Technical architecture considerations
2. Team capacity and skill requirements
3. Business constraints and timeline
4. Risk factors and failure modes
5. Alternative approaches

Identify potential flaws in your reasoning. Then provide your recommendation.

Question: {question}
""".strip()


async def run_single_model_with_cot(question: str, model_name: str, config_path: str) -> dict:
    """Run a single model with chain-of-thought prompting."""
    config = load_config(config_path)
    model_config = config["models"][model_name]
    model = LiteLLMModel.from_config(model_name, model_config)

    logger.info(f"🤖 Running single model ({model.full_display_name}) with CoT prompting...")

    # Format CoT prompt
    prompt = COT_PROMPT.format(question=question)

    # Get response
    start_time = datetime.now()
    response_obj = await model.generate(prompt)
    end_time = datetime.now()

    duration = (end_time - start_time).total_seconds()

    logger.info(f"✅ Single model response complete in {duration:.1f}s")

    return {
        "approach": "Single Model + Chain-of-Thought",
        "model": model.full_display_name,
        "response": response_obj.content,
        "duration_seconds": duration,
        "cost_estimate": 0.0,  # Local models are free
    }


async def run_arbitrium_tournament(question: str, config_path: str) -> dict:
    """Run full Arbitrium Framework tournament."""
    logger.info("🏆 Running Arbitrium Framework tournament...")

    # Initialize benchmark components
    _config, _models, comparison = initialize_benchmark(config_path)

    # Run tournament
    start_time = datetime.now()
    result = await comparison.run(question)
    end_time = datetime.now()

    duration = (end_time - start_time).total_seconds()

    logger.info(f"✅ Tournament complete in {duration:.1f}s")

    eliminated_models = getattr(comparison, "eliminated_models", [])
    return {
        "approach": "Arbitrium Framework Tournament",
        "champion_model": comparison.active_model_keys[0] if comparison.active_model_keys else "Unknown",
        "response": result,
        "duration_seconds": duration,
        "cost_actual": comparison.total_cost,
        "eliminated_models": eliminated_models,
    }


async def main(args=None):
    """Run micro-benchmark."""
    if args is None:
        parser = argparse.ArgumentParser(description="Micro-Benchmark: Single Model vs Arbitrium Framework")
        parser.add_argument(
            "--config",
            default="config.benchmark.small.yml",
            help="Config file to use for the benchmark.",
        )
        args = parser.parse_args()
        args = vars(args)  # Convert to dict

    print("=" * 80)
    print("MICRO-BENCHMARK: Single Model vs Arbitrium Framework")
    print("=" * 80)
    print()
    print("This benchmark provides a first data point for Arbitrium Framework value.")
    print("NOT scientifically rigorous - use for documentation and examples.")
    print()
    print(f"Config: {args['config']}")
    print(f"Test Question: {TEST_QUESTION[:100]}...")
    print()

    # Load config
    config = load_config(args["config"])
    models = config["models"]
    single_model_results = []

    # Run single models
    for model_name in models:
        print("\n" + "=" * 80)
        print(f"APPROACH 1: Single Model ({model_name}) + Chain-of-Thought")
        print("=" * 80 + "\n")

        single_result = await run_single_model_with_cot(TEST_QUESTION, model_name, args["config"])
        single_model_results.append(single_result)

        print("\n✅ Complete!")
        print(f"   Duration: {single_result['duration_seconds']:.1f}s")
        print(f"   Cost: ${single_result['cost_estimate']:.4f}")

    # Run tournament
    print("\n" + "=" * 80)
    print("APPROACH 2: Arbitrium Framework Tournament")
    print("=" * 80 + "\n")

    tournament_result = await run_arbitrium_tournament(TEST_QUESTION, args["config"])

    print("\n✅ Complete!")
    print(f"   Duration: {tournament_result['duration_seconds']:.1f}s")
    print(f"   Cost: ${tournament_result['cost_actual']:.4f}")

    # Save results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    output_path = Path(__file__).parent / "micro_benchmark_results.md"

    with open(output_path, "w") as f:
        f.write("# Micro-Benchmark Results: Single Model vs Arbitrium Framework\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Config: `{args['config']}`\n\n")

        f.write("## Test Question\n\n")
        f.write(f"{TEST_QUESTION}\n\n")

        f.write("---\n\n")

        for i, single_result in enumerate(single_model_results):
            f.write(f"## Approach 1.{i + 1}: Single Model + Chain-of-Thought\n\n")
            f.write(f"Model: {single_result['model']}\n\n")
            f.write(f"Duration: {single_result['duration_seconds']:.1f} seconds\n\n")
            f.write(f"Cost: ${single_result['cost_estimate']:.4f}\n\n")
            f.write("### Response\n\n")
            f.write(f"{single_result['response']}\n\n")

        f.write("---\n\n")

        f.write("## Approach 2: Arbitrium Framework Tournament\n\n")
        f.write(f"Champion: {tournament_result['champion_model']}\n\n")
        f.write(f"Duration: {tournament_result['duration_seconds']:.1f} seconds ({tournament_result['duration_seconds'] / 60:.1f} minutes)\n\n")
        f.write(f"Cost: ${tournament_result['cost_actual']:.4f}\n\n")

        if single_model_results:
            cost_estimate = single_model_results[0]["cost_estimate"]
            cost_actual = tournament_result["cost_actual"]
            if cost_estimate > 0:
                cost_multiple = f"{cost_actual / cost_estimate:.1f}x"
            else:
                cost_multiple = "N/A (free baseline)"
            f.write(f"Cost Multiple: {cost_multiple}\n\n")

            f.write(f"Time Multiple: {tournament_result['duration_seconds'] / single_model_results[0]['duration_seconds']:.1f}x\n\n")

        if tournament_result["eliminated_models"]:
            f.write(f"Eliminated Models: {', '.join(str(m) for m in tournament_result['eliminated_models'])}\n\n")

        f.write("### Champion Response\n\n")
        f.write(f"{tournament_result['response']}\n\n")

        f.write("---\n\n")

        model_names = [res["model"] for res in single_model_results] + ["Arbitrium Framework"]
        f.write(generate_manual_evaluation_template(model_names))

    print(f"\n📄 Results saved to: {output_path}")
    print()
    print("NEXT STEPS:")
    print("1. Share both responses with 3-5 knowledgeable people (anonymized)")
    print("2. Ask them to evaluate using the rubric")
    print("3. Compile feedback and calculate average scores")
    print("4. Document in README as first case study")
    print()
    print("This provides your first empirical evidence for Arbitrium Framework value!")


if __name__ == "__main__":
    asyncio.run(main())
