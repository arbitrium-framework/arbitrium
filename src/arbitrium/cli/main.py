#!/usr/bin/env python3
"""
Arbitrium Framework - LLM Comparison and Evaluation Tool

This is the main entry point for the Arbitrium Framework CLI application.
"""

import asyncio
import sys
from typing import TYPE_CHECKING

import colorama

# Public API imports - CLI uses only exported interface
from arbitrium import Arbitrium

# CLI-specific components
from arbitrium.cli.args import parse_arguments
from arbitrium.logging import get_contextual_logger
from arbitrium.utils.async_ import async_input
from arbitrium.utils.exceptions import FatalError

if TYPE_CHECKING:
    from arbitrium.config.loader import Config

# Constants for configuration
DEFAULT_CONFIG_FILE = "config.yml"
CONFIG_LOAD_ERROR_MSG = "Failed to load configuration from config.yml"
ARBITRIUM_NOT_INITIALIZED_MSG = "Arbitrium not initialized"


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

    def _load_config_with_fallback(self, config_path: str) -> "Config":
        """
        Load config from path with fallback to default config.

        Returns loaded Config object or raises FatalError.
        """
        from arbitrium.config.loader import Config

        config_obj = Config(config_path)
        if config_obj.load():
            return config_obj

        # Try fallback if not using default config
        if config_path != DEFAULT_CONFIG_FILE:
            self.logger.warning(
                f"Config file '{config_path}' not found, falling back to {DEFAULT_CONFIG_FILE}"
            )
            config_obj = Config(DEFAULT_CONFIG_FILE)
            if config_obj.load():
                return config_obj
            self._fatal_error(CONFIG_LOAD_ERROR_MSG)
        else:
            self._fatal_error(
                f"Failed to load configuration from {config_path}"
            )

        # This line is never reached due to _fatal_error raising exception
        return config_obj

    async def _try_create_arbitrium_from_config_obj(
        self, config_obj: "Config", skip_secrets: bool
    ) -> Arbitrium:
        """Try to create Arbitrium instance from a config object."""
        config_obj.config_data["outputs_dir"] = self.outputs_dir
        return await Arbitrium.from_settings(
            settings=config_obj.config_data,
            skip_secrets=skip_secrets,
        )

    async def _try_create_with_fallback(
        self, config_path: str, skip_secrets: bool
    ) -> Arbitrium:
        """Try to create Arbitrium with default config as fallback."""
        self.logger.warning(
            f"Failed to initialize with {config_path}, trying {DEFAULT_CONFIG_FILE}"
        )
        try:
            config_obj = self._load_config_with_fallback(DEFAULT_CONFIG_FILE)
            return await self._try_create_arbitrium_from_config_obj(
                config_obj, skip_secrets
            )
        except Exception:
            self._fatal_error(CONFIG_LOAD_ERROR_MSG)
        # This line is never reached due to _fatal_error raising exception
        raise RuntimeError("Unreachable code")

    async def _create_arbitrium_from_config(
        self, config_path: str, skip_secrets: bool
    ) -> Arbitrium:
        """
        Create Arbitrium instance from config with fallback.

        Returns Arbitrium instance or raises FatalError.
        """
        config_obj = self._load_config_with_fallback(config_path)

        try:
            return await self._try_create_arbitrium_from_config_obj(
                config_obj, skip_secrets
            )
        except Exception as e:
            # Try fallback if not using default config
            if config_path == DEFAULT_CONFIG_FILE:
                self._fatal_error(f"Failed to load configuration: {e}")

            return await self._try_create_with_fallback(
                config_path, skip_secrets
            )

    def _filter_requested_models(self) -> None:
        """
        Filter Arbitrium models based on CLI arguments.

        Updates self.arbitrium._healthy_models with filtered models.
        """
        if not self.args.get("models") or self.arbitrium is None:
            return

        models_arg: str = self.args.get("models")  # type: ignore[assignment]
        requested_models = [m.strip() for m in models_arg.split(",")]

        filtered_models = {
            key: model
            for key, model in self.arbitrium.healthy_models.items()
            if key in requested_models
        }

        if not filtered_models:
            self._fatal_error(
                f"None of the requested models ({', '.join(requested_models)}) are available or healthy"
            )

        self.logger.info(
            f"Filtering to requested models: {', '.join(filtered_models.keys())}"
        )
        self.arbitrium._healthy_models = filtered_models

    def _validate_arbitrium_ready(self) -> None:
        """Validate that Arbitrium has healthy models."""
        if self.arbitrium is None:
            self._fatal_error(ARBITRIUM_NOT_INITIALIZED_MSG)
        assert self.arbitrium is not None
        if not self.arbitrium.is_ready:
            self._fatal_error("❌ No models passed health check")

    async def _initialize_arbitrium(self) -> None:
        """
        Initialize Arbitrium from config file.

        CLI injects outputs_dir into config before passing to framework.
        """
        config_path = self.args.get("config", DEFAULT_CONFIG_FILE)
        skip_secrets = self.args.get("no_secrets", False)

        self.logger.info(f"Loading configuration from {config_path}")

        # Create Arbitrium from config with fallback support
        self.arbitrium = await self._create_arbitrium_from_config(
            config_path, skip_secrets
        )

        # Filter models if specific models were requested
        self._filter_requested_models()

        # Validate that we have healthy models
        self._validate_arbitrium_ready()

    async def _get_app_question(self) -> str:
        """
        Gets the question from config, file, or interactive input.
        Priority: 1) CLI argument, 2) Config file, 3) Interactive mode
        """
        if self.arbitrium is None:
            self._fatal_error(ARBITRIUM_NOT_INITIALIZED_MSG)

        question = ""

        # Check if interactive mode is requested
        if self.args.get("interactive", False):
            self.logger.info(
                "Enter your question:", extra={"display_type": "header"}
            )
            question = await async_input("> ")
            return question.strip()

        assert self.arbitrium is not None
        config_question = self.arbitrium.config_data.get("question")
        if config_question:
            self.logger.info("Using question from config file")
            return str(config_question).strip()

        # No question in config, fall back to interactive mode
        self.logger.info(
            "No question file or config question found, entering interactive mode"
        )
        self.logger.info(
            "Enter your question:", extra={"display_type": "header"}
        )
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
            self._fatal_error(ARBITRIUM_NOT_INITIALIZED_MSG)

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
