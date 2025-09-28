#!/usr/bin/env python3
"""Quick integration test for Arbitrium Framework tournament."""

import argparse
import asyncio
import sys

from arbitrium.config.loader import Config
from arbitrium.core.comparison import ModelComparison
from arbitrium.logging import get_contextual_logger, setup_logging
from arbitrium.models.factory import create_models_from_config
from arbitrium.utils.display import Display


async def main() -> None:
    """Run a quick integration test of the tournament system."""
    parser = argparse.ArgumentParser(description="Quick integration test of Arbitrium Framework tournament")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging (DEBUG level)")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="benchmarks/config.benchmark.yml",
        help="Path to config file (default: benchmarks/config.benchmark.yml)",
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default="What is the meaning of life?",
        help="Question for the tournament (default: 'What is the meaning of life?')",
    )
    args = parser.parse_args()

    # Setup logging (console only, no log files)
    setup_logging(verbose=args.verbose)
    logger = get_contextual_logger("test_quick")

    logger.info(f"Loading config from: {args.config}")
    config = Config(args.config)
    if not config.load():
        logger.error(f"Failed to load config from {args.config}")
        sys.exit(1)

    logger.info("Initializing models...")
    models = create_models_from_config(config.config_data.get("models", {}))
    for key, model in models.items():
        logger.info(f"  - {key}: {model.display_name}")

    logger.info("Starting tournament...")
    display = Display()
    comparison = ModelComparison(
        config=config.config_data,
        models=models,  # type: ignore[arg-type]
        display=display,
    )

    question = args.question
    logger.info(f"Question: {question}")

    result = await comparison.run(question)

    print(f"\n{'=' * 80}")
    print("TOURNAMENT RESULT:")
    print("=" * 80)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
