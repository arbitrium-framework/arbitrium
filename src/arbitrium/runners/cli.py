#!/usr/bin/env python3
"""
Arbitrium Framework - LLM Comparison and Evaluation Tool

This is the main entry point for the Arbitrium Framework CLI application.
"""

import asyncio
import sys

import colorama

from arbitrium.config.loader import Config
from arbitrium.core.comparison import ModelComparison
from arbitrium.logging import get_contextual_logger
from arbitrium.models.base import LiteLLMModel
from arbitrium.models.factory import create_models_from_config
from arbitrium.runners.cli_args import parse_arguments
from arbitrium.runners.cli_handler import ConsoleEventHandler
from arbitrium.runners.cli_host import CliHost
from arbitrium.utils.async_ import async_input
from arbitrium.utils.constants import HEALTH_CHECK_PROMPT
from arbitrium.utils.exceptions import ConfigurationError, FatalError
from arbitrium.utils.secrets import load_secrets


class App:
    """
    Main application class for Arbitrium Framework CLI.
    """

    def __init__(self) -> None:
        """
        Initialize the application.
        """
        self.args = parse_arguments()
        self.logger = get_contextual_logger("arbitrium.cli")
        self.event_handler = ConsoleEventHandler()
        self.host = CliHost()
        self.config = self._load_app_config()
        self._load_app_secrets()
        self.reports_dir = self._setup_app_directories()

    def _fatal_error(self, message: str) -> None:
        """
        Log a fatal error and raise FatalError exception.
        """
        self.logger.error(message)
        raise FatalError(message)

    def _load_app_config(self) -> Config:
        """
        Loads the application configuration.
        """
        config_path = self.args.get("config", "configs/default.yaml")
        config = Config(config_path)
        if not config.load():
            self._fatal_error(f"Failed to load configuration from {config_path}")
        return config

    def _load_app_secrets(self) -> None:
        """
        Loads secrets from environment variables or 1Password.
        """
        if self.args.get("no_secrets", False):
            self.logger.info("Skipping secret loading due to --no-secrets flag")
            return

        try:
            active_providers = {
                model_cfg.get("provider", "").lower() for model_cfg in self.config.config_data.get("models", {}).values() if model_cfg.get("provider")
            }
            load_secrets(self.config.config_data, list(active_providers))
        except ConfigurationError as e:
            self._fatal_error(str(e))

    def _setup_app_directories(self) -> str:
        """
        Returns the reports directory path without creating it.
        Directory will be created automatically if/when reports are saved.
        """
        reports_dir_arg = self.args.get("reports_dir", "reports")
        reports_dir: str = str(reports_dir_arg)
        return reports_dir

    async def _get_app_question(self) -> str:
        """
        Gets the question from file or interactive input.
        """
        question_path: str = self.args.get("question", "question.txt")
        question = ""

        if self.args.get("interactive", False):
            self.logger.info("Enter your question:", extra={"display_type": "header"})
            question = await async_input("> ")
        else:
            try:
                question = await self.host.read_file(question_path)
                self.logger.info(f"Loaded question from {question_path}")
            except FileNotFoundError:
                self._fatal_error(f"Question file not found: {question_path}")
            except OSError as file_error:
                self._fatal_error(f"File error with question file: {file_error!s}")

        if not question.strip():
            self._fatal_error("No question provided")

        return question.strip()

    def _initialize_models(self) -> dict[str, LiteLLMModel]:
        """
        Initializes the models for the comparison.
        """

        models = create_models_from_config(self.config.config_data)

        if self.args.get("models"):
            models_arg: str = self.args.get("models")  # type: ignore[assignment]
            requested_models = [m.strip() for m in models_arg.split(",")]
            models = {key: models[key] for key in requested_models if key in models}

            if not models:
                self._fatal_error(f"None of the requested models ({', '.join(requested_models)}) are available in config")

            self.logger.info(f"Filtering to requested models: {', '.join(models.keys())}")

        if not models:
            self._fatal_error("No valid models configured")

        for model in models.values():
            self.logger.info(f"Initialized {model.full_display_name}")

        return models

    async def _health_check_models(self, models: dict[str, LiteLLMModel]) -> dict[str, LiteLLMModel]:
        """
        Perform health check on all models to verify they're accessible.
        """
        self.logger.info("🔍 Performing model health check...")
        self.logger.info("🔍 Checking model availability...", extra={"display_type": "colored_text", "color": "info"})

        healthy_models = {}
        failed_models = []

        for model_key, model in models.items():
            try:
                response = await model.generate(HEALTH_CHECK_PROMPT)

                if response.is_error():
                    self.logger.warning(f"❌ {model.full_display_name}: {response.error}")
                    self.logger.warning(
                        f"❌ {model.full_display_name}: Failed",
                        extra={"display_type": "colored_text", "color": "warning"},
                    )
                    failed_models.append(model_key)
                else:
                    self.logger.info(f"✅ {model.full_display_name}: Healthy")
                    self.logger.info(
                        f"✅ {model.full_display_name}: Available",
                        extra={"display_type": "colored_text", "color": "success"},
                    )
                    healthy_models[model_key] = model

            except Exception as e:
                self.logger.warning(f"❌ {model.full_display_name}: {e!s}")
                self.logger.error(
                    f"❌ {model.full_display_name}: Error - {e!s}",
                    extra={"display_type": "colored_text", "color": "error"},
                )
                failed_models.append(model_key)

        if failed_models:
            self.logger.warning(
                f"\n⚠️  {len(failed_models)} model(s) failed health check: {', '.join(failed_models)}",
                extra={"display_type": "colored_text", "color": "warning"},
            )
            self.logger.info("Continuing with available models...", extra={"display_type": "colored_text", "color": "info"})

        if not healthy_models:
            self._fatal_error("❌ No models passed health check")

        return healthy_models

    async def run(self) -> None:
        """
        Main function to run the Arbitrium Framework CLI application.
        """
        self.logger.info("Starting Arbitrium Framework")

        question = await self._get_app_question()
        models = self._initialize_models()

        healthy_models = await self._health_check_models(models)

        comparison = ModelComparison(
            config=self.config.config_data,
            models=healthy_models,  # type: ignore[arg-type]
            event_handler=self.event_handler,
            host=self.host,
        )

        try:
            await comparison.run(question)
        except KeyboardInterrupt:
            self.event_handler.publish("log", {"level": "info", "message": "Interrupted by user"})
            self.event_handler.publish(
                "display",
                {"type": "print", "text": "\nInterrupted by user. Exiting..."},
            )
        except Exception as err:
            self._fatal_error(f"Error during model comparison: {err!s}")

        self.event_handler.publish("log", {"level": "info", "message": "Arbitrium Framework completed successfully"})


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
