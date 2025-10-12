"""
Unit tests for CLI main application.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbitrium.cli.main import App
from arbitrium.utils.exceptions import FatalError


class TestApp:
    """Tests for App class."""

    @patch("arbitrium.cli.main.parse_arguments")
    def test_app_initialization(self, mock_parse: MagicMock) -> None:
        """Test App initialization."""
        mock_parse.return_value = {"outputs_dir": ".", "config": "config.yml"}

        app = App()

        assert app.args is not None
        assert app.logger is not None
        assert app.outputs_dir == "."
        assert app.arbitrium is None

    @patch("arbitrium.cli.main.parse_arguments")
    def test_get_outputs_dir_default(self, mock_parse: MagicMock) -> None:
        """Test getting outputs directory with default value."""
        mock_parse.return_value = {}

        app = App()

        assert app.outputs_dir == "."

    @patch("arbitrium.cli.main.parse_arguments")
    def test_get_outputs_dir_custom(self, mock_parse: MagicMock) -> None:
        """Test getting outputs directory with custom value."""
        mock_parse.return_value = {"outputs_dir": "/custom/path"}

        app = App()

        assert app.outputs_dir == "/custom/path"

    @patch("arbitrium.cli.main.parse_arguments")
    def test_fatal_error_raises(self, mock_parse: MagicMock) -> None:
        """Test that _fatal_error raises FatalError."""
        mock_parse.return_value = {}

        app = App()

        with pytest.raises(FatalError, match="Test error"):
            app._fatal_error("Test error")

    @patch("arbitrium.cli.main.parse_arguments")
    def test_load_config_with_fallback_success(
        self, mock_parse: MagicMock
    ) -> None:
        """Test loading config successfully."""
        mock_parse.return_value = {}
        app = App()

        mock_config = MagicMock()
        mock_config.load.return_value = True

        with patch("arbitrium.config.loader.Config", return_value=mock_config):
            result = app._load_config_with_fallback("test_config.yml")

        assert result == mock_config

    @patch("arbitrium.cli.main.parse_arguments")
    def test_load_config_with_fallback_to_default(
        self, mock_parse: MagicMock
    ) -> None:
        """Test loading config with fallback to default."""
        mock_parse.return_value = {}
        app = App()

        mock_config_fail = MagicMock()
        mock_config_fail.load.return_value = False

        mock_config_success = MagicMock()
        mock_config_success.load.return_value = True

        with patch(
            "arbitrium.config.loader.Config",
            side_effect=[mock_config_fail, mock_config_success],
        ):
            result = app._load_config_with_fallback("custom_config.yml")

        assert result == mock_config_success

    @patch("arbitrium.cli.main.parse_arguments")
    def test_load_config_with_fallback_fails(
        self, mock_parse: MagicMock
    ) -> None:
        """Test loading config fails with both paths."""
        mock_parse.return_value = {}
        app = App()

        mock_config = MagicMock()
        mock_config.load.return_value = False

        with patch("arbitrium.config.loader.Config", return_value=mock_config):
            with pytest.raises(FatalError):
                app._load_config_with_fallback("custom_config.yml")

    @patch("arbitrium.cli.main.parse_arguments")
    def test_load_config_default_fails(self, mock_parse: MagicMock) -> None:
        """Test loading default config fails."""
        mock_parse.return_value = {}
        app = App()

        mock_config = MagicMock()
        mock_config.load.return_value = False

        with patch("arbitrium.config.loader.Config", return_value=mock_config):
            with pytest.raises(FatalError):
                app._load_config_with_fallback("config.yml")

    @pytest.mark.asyncio
    @patch("arbitrium.cli.main.parse_arguments")
    async def test_try_create_arbitrium_from_config_obj(
        self, mock_parse: MagicMock
    ) -> None:
        """Test creating Arbitrium from config object."""
        mock_parse.return_value = {"outputs_dir": "./output"}
        app = App()

        mock_config = MagicMock()
        mock_config.config_data = {"test": "data"}

        mock_arbitrium = MagicMock()

        with patch(
            "arbitrium.cli.main.Arbitrium.from_settings",
            new_callable=AsyncMock,
            return_value=mock_arbitrium,
        ):
            result = await app._try_create_arbitrium_from_config_obj(
                mock_config, skip_secrets=False
            )

        assert result == mock_arbitrium
        assert mock_config.config_data["outputs_dir"] == "./output"

    @pytest.mark.asyncio
    @patch("arbitrium.cli.main.parse_arguments")
    async def test_try_create_with_fallback_success(
        self, mock_parse: MagicMock
    ) -> None:
        """Test creating Arbitrium with fallback succeeds."""
        mock_parse.return_value = {}
        app = App()

        mock_config = MagicMock()
        mock_config.load.return_value = True
        mock_config.config_data = {}

        mock_arbitrium = MagicMock()

        with patch("arbitrium.config.loader.Config", return_value=mock_config):
            with patch(
                "arbitrium.cli.main.Arbitrium.from_settings",
                new_callable=AsyncMock,
                return_value=mock_arbitrium,
            ):
                result = await app._try_create_with_fallback(
                    "test_config.yml", skip_secrets=False
                )

        assert result == mock_arbitrium

    @pytest.mark.asyncio
    @patch("arbitrium.cli.main.parse_arguments")
    async def test_try_create_with_fallback_fails(
        self, mock_parse: MagicMock
    ) -> None:
        """Test creating Arbitrium with fallback fails."""
        mock_parse.return_value = {}
        app = App()

        mock_config = MagicMock()
        mock_config.load.return_value = False

        with patch("arbitrium.config.loader.Config", return_value=mock_config):
            with pytest.raises(FatalError):
                await app._try_create_with_fallback(
                    "test_config.yml", skip_secrets=False
                )

    @pytest.mark.asyncio
    @patch("arbitrium.cli.main.parse_arguments")
    async def test_create_arbitrium_from_config_success(
        self, mock_parse: MagicMock
    ) -> None:
        """Test creating Arbitrium from config succeeds."""
        mock_parse.return_value = {}
        app = App()

        mock_config = MagicMock()
        mock_config.load.return_value = True
        mock_config.config_data = {}

        mock_arbitrium = MagicMock()

        with patch("arbitrium.config.loader.Config", return_value=mock_config):
            with patch(
                "arbitrium.cli.main.Arbitrium.from_settings",
                new_callable=AsyncMock,
                return_value=mock_arbitrium,
            ):
                result = await app._create_arbitrium_from_config(
                    "test_config.yml", skip_secrets=False
                )

        assert result == mock_arbitrium

    @pytest.mark.asyncio
    @patch("arbitrium.cli.main.parse_arguments")
    async def test_create_arbitrium_from_config_with_exception(
        self, mock_parse: MagicMock
    ) -> None:
        """Test creating Arbitrium from config with exception."""
        mock_parse.return_value = {}
        app = App()

        mock_config = MagicMock()
        mock_config.load.return_value = True
        mock_config.config_data = {}

        with patch("arbitrium.config.loader.Config", return_value=mock_config):
            with patch(
                "arbitrium.cli.main.Arbitrium.from_settings",
                new_callable=AsyncMock,
                side_effect=Exception("Test error"),
            ):
                with pytest.raises(FatalError):
                    await app._create_arbitrium_from_config(
                        "config.yml", skip_secrets=False
                    )

    @patch("arbitrium.cli.main.parse_arguments")
    def test_filter_requested_models_no_filter(
        self, mock_parse: MagicMock
    ) -> None:
        """Test filtering models when no filter is requested."""
        mock_parse.return_value = {}
        app = App()
        app.arbitrium = MagicMock()
        app.arbitrium.healthy_models = {
            "model1": MagicMock(),
            "model2": MagicMock(),
        }

        app._filter_requested_models()

        # Should not change models
        assert len(app.arbitrium.healthy_models) == 2

    @patch("arbitrium.cli.main.parse_arguments")
    def test_filter_requested_models_with_filter(
        self, mock_parse: MagicMock
    ) -> None:
        """Test filtering models with requested models."""
        mock_parse.return_value = {"models": "model1,model3"}
        app = App()
        app.arbitrium = MagicMock()
        app.arbitrium.healthy_models = {
            "model1": MagicMock(),
            "model2": MagicMock(),
            "model3": MagicMock(),
        }

        app._filter_requested_models()

        assert len(app.arbitrium._healthy_models) == 2
        assert "model1" in app.arbitrium._healthy_models
        assert "model3" in app.arbitrium._healthy_models

    @patch("arbitrium.cli.main.parse_arguments")
    def test_filter_requested_models_none_available(
        self, mock_parse: MagicMock
    ) -> None:
        """Test filtering models when none of requested are available."""
        mock_parse.return_value = {"models": "model4,model5"}
        app = App()
        app.arbitrium = MagicMock()
        app.arbitrium.healthy_models = {
            "model1": MagicMock(),
            "model2": MagicMock(),
        }

        with pytest.raises(FatalError, match="None of the requested models"):
            app._filter_requested_models()

    @patch("arbitrium.cli.main.parse_arguments")
    def test_validate_arbitrium_ready_not_initialized(
        self, mock_parse: MagicMock
    ) -> None:
        """Test validating Arbitrium when not initialized."""
        mock_parse.return_value = {}
        app = App()
        app.arbitrium = None

        with pytest.raises(FatalError, match="not initialized"):
            app._validate_arbitrium_ready()

    @patch("arbitrium.cli.main.parse_arguments")
    def test_validate_arbitrium_ready_not_ready(
        self, mock_parse: MagicMock
    ) -> None:
        """Test validating Arbitrium when not ready."""
        mock_parse.return_value = {}
        app = App()
        app.arbitrium = MagicMock()
        app.arbitrium.is_ready = False

        with pytest.raises(FatalError, match="No models passed health check"):
            app._validate_arbitrium_ready()

    @patch("arbitrium.cli.main.parse_arguments")
    def test_validate_arbitrium_ready_success(
        self, mock_parse: MagicMock
    ) -> None:
        """Test validating Arbitrium when ready."""
        mock_parse.return_value = {}
        app = App()
        app.arbitrium = MagicMock()
        app.arbitrium.is_ready = True

        # Should not raise
        app._validate_arbitrium_ready()

    @pytest.mark.asyncio
    @patch("arbitrium.cli.main.parse_arguments")
    async def test_initialize_arbitrium_success(
        self, mock_parse: MagicMock
    ) -> None:
        """Test initializing Arbitrium successfully."""
        mock_parse.return_value = {"config": "test.yml", "no_secrets": False}
        app = App()

        mock_config = MagicMock()
        mock_config.load.return_value = True
        mock_config.config_data = {}

        mock_arbitrium = MagicMock()
        mock_arbitrium.is_ready = True

        with patch("arbitrium.config.loader.Config", return_value=mock_config):
            with patch(
                "arbitrium.cli.main.Arbitrium.from_settings",
                new_callable=AsyncMock,
                return_value=mock_arbitrium,
            ):
                await app._initialize_arbitrium()

        assert app.arbitrium == mock_arbitrium

    @pytest.mark.asyncio
    @patch("arbitrium.cli.main.parse_arguments")
    async def test_get_app_question_not_initialized(
        self, mock_parse: MagicMock
    ) -> None:
        """Test getting question when Arbitrium not initialized."""
        mock_parse.return_value = {}
        app = App()
        app.arbitrium = None

        with pytest.raises(FatalError, match="not initialized"):
            await app._get_app_question()

    @pytest.mark.asyncio
    @patch("arbitrium.cli.main.parse_arguments")
    @patch("arbitrium.cli.main.async_input")
    async def test_get_app_question_interactive_mode(
        self, mock_input: AsyncMock, mock_parse: MagicMock
    ) -> None:
        """Test getting question in interactive mode."""
        mock_parse.return_value = {"interactive": True}
        mock_input.return_value = "Test question?"
        app = App()
        app.arbitrium = MagicMock()
        app.arbitrium.config_data = {}

        result = await app._get_app_question()

        assert result == "Test question?"

    @pytest.mark.asyncio
    @patch("arbitrium.cli.main.parse_arguments")
    async def test_get_app_question_from_config(
        self, mock_parse: MagicMock
    ) -> None:
        """Test getting question from config."""
        mock_parse.return_value = {"interactive": False}
        app = App()
        app.arbitrium = MagicMock()
        app.arbitrium.config_data = {"question": "Config question?"}

        result = await app._get_app_question()

        assert result == "Config question?"

    @pytest.mark.asyncio
    @patch("arbitrium.cli.main.parse_arguments")
    @patch("arbitrium.cli.main.async_input")
    async def test_get_app_question_fallback_interactive(
        self, mock_input: AsyncMock, mock_parse: MagicMock
    ) -> None:
        """Test getting question falls back to interactive."""
        mock_parse.return_value = {"interactive": False}
        mock_input.return_value = "Fallback question?"
        app = App()
        app.arbitrium = MagicMock()
        app.arbitrium.config_data = {}

        result = await app._get_app_question()

        assert result == "Fallback question?"

    @pytest.mark.asyncio
    @patch("arbitrium.cli.main.parse_arguments")
    async def test_run_success(self, mock_parse: MagicMock) -> None:
        """Test running app successfully."""
        mock_parse.return_value = {
            "config": "test.yml",
            "no_secrets": False,
            "interactive": False,
        }
        app = App()

        mock_config = MagicMock()
        mock_config.load.return_value = True
        mock_config.config_data = {"question": "Test?"}

        mock_arbitrium = MagicMock()
        mock_arbitrium.is_ready = True
        mock_arbitrium.config_data = {"question": "Test?"}
        mock_arbitrium.run_tournament = AsyncMock(
            return_value=("result", {"metrics": "data"})
        )

        with patch("arbitrium.config.loader.Config", return_value=mock_config):
            with patch(
                "arbitrium.cli.main.Arbitrium.from_settings",
                new_callable=AsyncMock,
                return_value=mock_arbitrium,
            ):
                await app.run()

        mock_arbitrium.run_tournament.assert_called_once()

    @pytest.mark.asyncio
    @patch("arbitrium.cli.main.parse_arguments")
    async def test_run_keyboard_interrupt(self, mock_parse: MagicMock) -> None:
        """Test running app with keyboard interrupt."""
        mock_parse.return_value = {
            "config": "test.yml",
            "no_secrets": False,
            "interactive": False,
        }
        app = App()

        mock_config = MagicMock()
        mock_config.load.return_value = True
        mock_config.config_data = {"question": "Test?"}

        mock_arbitrium = MagicMock()
        mock_arbitrium.is_ready = True
        mock_arbitrium.config_data = {"question": "Test?"}
        mock_arbitrium.run_tournament = AsyncMock(
            side_effect=KeyboardInterrupt()
        )

        with patch("arbitrium.config.loader.Config", return_value=mock_config):
            with patch(
                "arbitrium.cli.main.Arbitrium.from_settings",
                new_callable=AsyncMock,
                return_value=mock_arbitrium,
            ):
                # Should not raise
                await app.run()

    @pytest.mark.asyncio
    @patch("arbitrium.cli.main.parse_arguments")
    async def test_run_tournament_error(self, mock_parse: MagicMock) -> None:
        """Test running app with tournament error."""
        mock_parse.return_value = {
            "config": "test.yml",
            "no_secrets": False,
            "interactive": False,
        }
        app = App()

        mock_config = MagicMock()
        mock_config.load.return_value = True
        mock_config.config_data = {"question": "Test?"}

        mock_arbitrium = MagicMock()
        mock_arbitrium.is_ready = True
        mock_arbitrium.config_data = {"question": "Test?"}
        mock_arbitrium.run_tournament = AsyncMock(
            side_effect=Exception("Tournament error")
        )

        with patch("arbitrium.config.loader.Config", return_value=mock_config):
            with patch(
                "arbitrium.cli.main.Arbitrium.from_settings",
                new_callable=AsyncMock,
                return_value=mock_arbitrium,
            ):
                with pytest.raises(
                    FatalError, match="Error during tournament"
                ):
                    await app.run()

    @pytest.mark.asyncio
    @patch("arbitrium.cli.main.parse_arguments")
    async def test_run_not_initialized_error(
        self, mock_parse: MagicMock
    ) -> None:
        """Test running app when not initialized."""
        mock_parse.return_value = {
            "config": "test.yml",
            "no_secrets": False,
            "interactive": False,
        }
        app = App()

        mock_config = MagicMock()
        mock_config.load.return_value = True
        mock_config.config_data = {}

        mock_arbitrium = MagicMock()
        mock_arbitrium.is_ready = True

        with patch("arbitrium.config.loader.Config", return_value=mock_config):
            with patch(
                "arbitrium.cli.main.Arbitrium.from_settings",
                new_callable=AsyncMock,
                return_value=None,
            ):
                with pytest.raises(FatalError):
                    await app.run()


class TestRunFromCli:
    """Tests for run_from_cli function."""

    @patch("arbitrium.cli.main.colorama.init")
    @patch("arbitrium.cli.main.App")
    @patch("asyncio.run")
    def test_run_from_cli_success(
        self,
        mock_asyncio_run: MagicMock,
        mock_app_class: MagicMock,
        mock_colorama: MagicMock,
    ) -> None:
        """Test run_from_cli succeeds."""
        from arbitrium.cli.main import run_from_cli

        mock_app = MagicMock()
        mock_app_class.return_value = mock_app

        with patch("arbitrium.logging.setup.setup_logging"):
            run_from_cli()

        mock_colorama.assert_called_once()
        mock_asyncio_run.assert_called_once()

    @patch("arbitrium.cli.main.colorama.init")
    @patch("arbitrium.cli.main.App")
    @patch("sys.exit")
    def test_run_from_cli_fatal_error(
        self,
        mock_exit: MagicMock,
        mock_app_class: MagicMock,
        mock_colorama: MagicMock,
    ) -> None:
        """Test run_from_cli with FatalError."""
        from arbitrium.cli.main import run_from_cli

        mock_app = MagicMock()
        mock_app_class.return_value = mock_app

        with patch("arbitrium.logging.setup.setup_logging"):
            with patch("asyncio.run", side_effect=FatalError("Fatal error")):
                run_from_cli()

        mock_exit.assert_called_once_with(1)

    @patch("arbitrium.cli.main.colorama.init")
    @patch("arbitrium.cli.main.App")
    @patch("sys.exit")
    def test_run_from_cli_keyboard_interrupt(
        self,
        mock_exit: MagicMock,
        mock_app_class: MagicMock,
        mock_colorama: MagicMock,
    ) -> None:
        """Test run_from_cli with KeyboardInterrupt."""
        from arbitrium.cli.main import run_from_cli

        mock_app = MagicMock()
        mock_app_class.return_value = mock_app

        with patch("arbitrium.logging.setup.setup_logging"):
            with patch("asyncio.run", side_effect=KeyboardInterrupt()):
                run_from_cli()

        mock_exit.assert_called_once_with(130)
