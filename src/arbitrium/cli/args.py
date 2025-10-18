"""Command-line argument handling for Arbitrium Framework."""

import argparse
from typing import Any

from arbitrium.__about__ import __version__

# CLI-only default (framework itself has no defaults)
DEFAULT_CONFIG_FILE = "config.example.yml"


def parse_arguments() -> dict[str, Any]:
    """Parse command line arguments.

    Returns:
        Dictionary of parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Arbitrium Framework - LLM Comparison and Evaluation Tool"
    )

    # Version flag
    parser.add_argument(
        "--version",
        action="version",
        version=f"Arbitrium Framework {__version__}",
    )

    # Main arguments (tournament is the default behavior)
    parser.add_argument(
        "-m",
        "--models",
        type=str,
        help="Comma-separated list of model keys to run",
    )
    parser.add_argument(
        "-c",
        "--config",
        help=f"Path to config file (CLI default: {DEFAULT_CONFIG_FILE})",
        default=DEFAULT_CONFIG_FILE,
    )
    parser.add_argument("-q", "--question", help="Path to question file")
    parser.add_argument(
        "-o",
        "--outputs-dir",
        default=None,
        help="Output directory for all files (default: current directory)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        help="Run in interactive mode",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable detailed debug logging",
    )
    parser.add_argument(
        "-v", "--verbose", help="Enable verbose output", action="store_true"
    )
    parser.add_argument(
        "--no-color", help="Disable colored output", action="store_true"
    )
    parser.add_argument(
        "--no-secrets", help="Skip loading secrets", action="store_true"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Convert to dictionary
    args_dict = vars(args)

    # Always tournament mode (no subcommands)
    args_dict["command"] = "tournament"

    # Verbose is implied if debug is set
    if args_dict.get("debug"):
        args_dict["verbose"] = True

    return args_dict
