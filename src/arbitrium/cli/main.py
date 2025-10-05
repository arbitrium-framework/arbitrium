#!/usr/bin/env python3
"""
Arbitrium Framework - LLM Comparison and Evaluation Tool

This is the main entry point for the Arbitrium Framework CLI application.
"""

import asyncio
import sys

import colorama

# Public API imports - CLI uses only exported interface
from arbitrium import Arbitrium

# CLI-specific components
from arbitrium.cli.args import parse_arguments
from arbitrium.logging import get_contextual_logger
from arbitrium.utils.async_ import async_input
from arbitrium.utils.exceptions import FatalError


class App:
    """
    Main application class for Arbitrium Framework CLI.

    This class is a thin wrapper around Arbitrium,
    handling CLI-specific concerns like argument parsing and question input.
    """

    def __init__(self) -> None:
        """
        Initialize the application.
        """
        self.args = parse_arguments()
        self.logger = get_contextual_logger("arbitrium.cli")
        self.outputs_dir = self._get_outputs_dir()
        self.arbitrium: Arbitrium | None = None

    def _fatal_error(self, message: str) -> None:
        """
        Log a fatal error and raise FatalError exception.
        """
        self.logger.error(message)
        raise FatalError(message)

    def _get_outputs_dir(self) -> str:
        """
        Get outputs directory from CLI arguments.

        CLI provides default value "." (current directory).
        Directory will be created automatically when files are saved.
        """
        outputs_dir_arg = self.args.get("outputs_dir", ".")
        outputs_dir: str = str(outputs_dir_arg)
        return outputs_dir

    async def _initialize_arbitrium(self) -> None:
        """
        Initialize Arbitrium from config file.

        CLI injects outputs_dir into config before passing to framework.
        """
        config_path = self.args.get("config", "config.example.yml")
        skip_secrets = self.args.get("no_secrets", False)

        self.logger.info(f"Loading configuration from {config_path}")

        # Load config and inject outputs_dir from CLI
        from arbitrium.config.loader import Config

        config_obj = Config(config_path)
        if not config_obj.load():
            if config_path != "config.example.yml":
                self.logger.warning(f"Config file '{config_path}' not found, falling back to config.example.yml")
                config_obj = Config("config.example.yml")
                if not config_obj.load():
                    self._fatal_error("Failed to load configuration from config.example.yml")
            else:
                self._fatal_error(f"Failed to load configuration from {config_path}")

        # CLI injects outputs_dir into settings
        config_obj.config_data["outputs_dir"] = self.outputs_dir

        try:
            # Create Arbitrium from modified settings
            self.arbitrium = await Arbitrium.from_settings(
                settings=config_obj.config_data,
                skip_secrets=skip_secrets,
            )
        except Exception as e:
            if config_path != "config.example.yml":
                self.logger.warning(f"Failed to initialize with {config_path}, trying config.example.yml")
                try:
                    config_obj = Config("config.example.yml")
                    if not config_obj.load():
                        self._fatal_error("Failed to load configuration from config.example.yml")
                    config_obj.config_data["outputs_dir"] = self.outputs_dir
                    self.arbitrium = await Arbitrium.from_settings(
                        settings=config_obj.config_data,
                        skip_secrets=skip_secrets,
                    )
                except Exception:
                    self._fatal_error("Failed to load configuration from config.example.yml")
            else:
                self._fatal_error(f"Failed to load configuration: {e}")

        # Filter models if specific models were requested
        if self.args.get("models"):
            assert self.arbitrium is not None
            models_arg: str = self.args.get("models")  # type: ignore[assignment]
            requested_models = [m.strip() for m in models_arg.split(",")]

            # Filter healthy models
            filtered_models = {key: model for key, model in self.arbitrium.healthy_models.items() if key in requested_models}

            if not filtered_models:
                self._fatal_error(f"None of the requested models ({', '.join(requested_models)}) are available or healthy")

            self.logger.info(f"Filtering to requested models: {', '.join(filtered_models.keys())}")

            # Update arbitrium with filtered models
            self.arbitrium._healthy_models = filtered_models

        # Check if we have healthy models
        assert self.arbitrium is not None
        if not self.arbitrium.is_ready:
            self._fatal_error("❌ No models passed health check")

    async def _get_app_question(self) -> str:
        """
        Gets the question from config, file, or interactive input.
        Priority: 1) CLI argument, 2) Config file, 3) Interactive mode
        """
        if self.arbitrium is None:
            self._fatal_error("Arbitrium not initialized")

        question = ""

        # Check if interactive mode is requested
        if self.args.get("interactive", False):
            self.logger.info("Enter your question:", extra={"display_type": "header"})
            question = await async_input("> ")
            return question.strip()

        assert self.arbitrium is not None
        config_question = self.arbitrium.config_data.get("question")
        if config_question:
            self.logger.info("Using question from config file")
            return str(config_question).strip()

        # No question in config, fall back to interactive mode
        self.logger.info("No question file or config question found, entering interactive mode")
        self.logger.info("Enter your question:", extra={"display_type": "header"})
        question = await async_input("> ")

        return question.strip()

    async def run(self) -> None:
        """
        Main function to run the Arbitrium Framework CLI application.
        """
        self.logger.info("Starting Arbitrium Framework")

        # Initialize arbitrium
        await self._initialize_arbitrium()

        if self.arbitrium is None:
            self._fatal_error("Arbitrium not initialized")

        # Get the question
        question = await self._get_app_question()

        # Run tournament
        assert self.arbitrium is not None
        try:
            _result, _metrics = await self.arbitrium.run_tournament(question)
            # Result is displayed via logging during tournament execution
            # Metrics are also logged by the tournament itself
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
            print("\nInterrupted by user. Exiting...")
        except Exception as err:
            self._fatal_error(f"Error during tournament: {err!s}")

        self.logger.info("Arbitrium Framework completed successfully")


def run_from_cli() -> None:
    """
    Entry point for the command-line script.
    """
    # Initialize logging FIRST to ensure consistent formatting from the start
    from arbitrium.logging import setup_logging

    setup_logging()

    colorama.init(autoreset=True)

    try:
        app = App()
        asyncio.run(app.run())
    except FatalError:
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        sys.exit(130)


if __name__ == "__main__":
    run_from_cli()
