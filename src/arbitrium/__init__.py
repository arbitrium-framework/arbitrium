"""
Arbitrium Framework - Multi-LLM Tournament System.

This package provides a framework for comparing and evaluating multiple LLMs
through a tournament-style competition system.

Example:
    >>> from arbitrium import Arbitrium
    >>>
    >>> # Initialize from config file
    >>> arbitrium = await Arbitrium.from_config("config.yml")
    >>>
    >>> # Run tournament
    >>> result, metrics = await arbitrium.run_tournament("Your question here")
    >>> print(f"Winner: {metrics['champion_model']}, Cost: ${metrics['total_cost']:.4f}")
    >>>
    >>> # Or run single model
    >>> response = await arbitrium.run_single_model("gpt-4", "Hello!")
"""

from .__about__ import __version__
from .arbitrium import Arbitrium

__all__ = [
    "Arbitrium",
    "__version__",
]
