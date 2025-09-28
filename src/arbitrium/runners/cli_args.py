"""Command-line argument handling for Arbitrium Framework."""

import argparse
from typing import Any

# Constants for default file paths
DEFAULT_QUESTION_FILE = "question.txt"
DEFAULT_CONFIG_FILE = "config.yml"
DEFAULT_REPORTS_DIR = "reports"


def parse_arguments() -> dict[str, Any]:
    """Parse command line arguments.

    Returns:
        Dictionary of parsed arguments
    """
    parser = argparse.ArgumentParser(description="Arbitrium Framework - LLM Comparison and Evaluation Tool")

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("-d", "--debug", action="store_true", help="Enable detailed debug logging")
    common_parser.add_argument("-v", "--verbose", help="Enable verbose output", action="store_true")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Tournament command (default)
    tournament_parser = subparsers.add_parser("tournament", help="Run model comparison tournament (default)", parents=[common_parser])
    tournament_parser.add_argument("-m", "--models", type=str, help="Comma-separated list of model keys to run")
    tournament_parser.add_argument("-c", "--config", help="Path to config file", default=DEFAULT_CONFIG_FILE)
    tournament_parser.add_argument("-q", "--question", help="Path to question file", default=DEFAULT_QUESTION_FILE)
    tournament_parser.add_argument("-r", "--reports-dir", help="Directory to save reports", default=DEFAULT_REPORTS_DIR)
    tournament_parser.add_argument("-i", "--interactive", help="Run in interactive mode", action="store_true")
    tournament_parser.add_argument("--no-color", help="Disable colored output", action="store_true")
    tournament_parser.add_argument("--no-secrets", help="Skip loading secrets", action="store_true")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks", parents=[common_parser])
    benchmark_parser.add_argument("benchmark_type", choices=["micro", "standard", "kb-validation", "config-test"], help="Type of benchmark to run")
    benchmark_parser.add_argument("-c", "--config", help="Path to config file", default="config.benchmark.small.yml")
    benchmark_parser.add_argument("--model", help="Model to use for baseline (micro benchmark)", default="llama3")
    benchmark_parser.add_argument("--num-questions", type=int, default=20, help="Number of questions (standard benchmark)")
    benchmark_parser.add_argument("--benchmark-name", choices=["bbh", "gpqa", "both"], default="bbh", help="Benchmark name (standard)")

    # Parse the arguments
    args = parser.parse_args()

    # Convert to dictionary
    args_dict = vars(args)

    # Default to tournament if no command specified
    if not args_dict.get("command"):
        args_dict["command"] = "tournament"

    # Verbose is implied if debug is set
    if args_dict.get("debug"):
        args_dict["verbose"] = True

    return args_dict
