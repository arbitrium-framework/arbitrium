"""
Unit tests for logging setup and formatting.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from arbitrium.logging.setup import (
    ColorFormatter,
    DuplicateFilter,
    _create_file_handler,
    _validate_log_file_path,
    setup_logging,
)


class TestDuplicateFilter:
    """Tests for DuplicateFilter class."""

    def test_filter_initialization(self) -> None:
        """Test filter initializes correctly."""
        filter = DuplicateFilter()
        assert filter.seen_messages == set()
        assert filter.max_cache_size > 0


class TestColorFormatter:
    """Tests for ColorFormatter class."""

    def test_formatter_initialization(self) -> None:
        """Test formatter initializes correctly."""
        formatter = ColorFormatter()
        assert formatter.include_module is True

    def test_formatter_without_module(self) -> None:
        """Test formatter can be initialized without module info."""
        formatter = ColorFormatter(include_module=False)
        assert formatter.include_module is False

    @patch("arbitrium.logging.setup.ColorFormatter._should_use_color")
    def test_format_basic_message(self, mock_color: MagicMock) -> None:
        """Test formatting a basic log message."""
        mock_color.return_value = False
        formatter = ColorFormatter(
            "[%(levelname)s] %(message)s", include_module=False
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        assert "Test message" in result

    @patch("arbitrium.logging.setup.ColorFormatter._should_use_color")
    def test_format_section_header(self, mock_color: MagicMock) -> None:
        """Test formatting section header display type."""
        mock_color.return_value = False
        formatter = ColorFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Section Header",
            args=(),
            exc_info=None,
        )
        record.display_type = "section_header"  # type: ignore

        result = formatter.format(record)
        assert "Section Header" in result
        assert "---" in result

    @patch("arbitrium.logging.setup.ColorFormatter._should_use_color")
    def test_format_header(self, mock_color: MagicMock) -> None:
        """Test formatting header display type."""
        mock_color.return_value = False
        formatter = ColorFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Header",
            args=(),
            exc_info=None,
        )
        record.display_type = "header"  # type: ignore

        result = formatter.format(record)
        assert "Header" in result
        assert "=" in result

    @patch("arbitrium.logging.setup.ColorFormatter._should_use_color")
    def test_format_model_response(self, mock_color: MagicMock) -> None:
        """Test formatting model response display type."""
        mock_color.return_value = False
        formatter = ColorFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Model response text",
            args=(),
            exc_info=None,
        )
        record.display_type = "model_response"  # type: ignore
        record.model_name = "test_model"  # type: ignore

        result = formatter.format(record)
        assert "Model: test_model" in result
        assert "Response:" in result

    @patch("arbitrium.logging.setup.ColorFormatter._should_use_color")
    def test_format_colored_text(self, mock_color: MagicMock) -> None:
        """Test formatting colored text display type."""
        mock_color.return_value = False
        formatter = ColorFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Colored text",
            args=(),
            exc_info=None,
        )
        record.display_type = "colored_text"  # type: ignore
        record.color = "red"  # type: ignore

        result = formatter.format(record)
        assert "Colored text" in result

    @patch("arbitrium.utils.terminal.should_use_color")
    def test_should_use_color_calls_utility(
        self, mock_should_use: MagicMock
    ) -> None:
        """Test that _should_use_color calls terminal utility."""
        mock_should_use.return_value = True
        formatter = ColorFormatter()

        result = formatter._should_use_color()

        assert result is True
        mock_should_use.assert_called_once()


class TestValidateLogFilePath:
    """Tests for _validate_log_file_path function."""

    def test_none_log_file(self) -> None:
        """Test that None log file returns None."""
        result = _validate_log_file_path(None)
        assert result is None

    def test_empty_log_file(self) -> None:
        """Test that empty string returns None."""
        result = _validate_log_file_path("")
        assert result is None

    def test_valid_log_file_in_temp(self) -> None:
        """Test valid log file path in temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "test.log")
            result = _validate_log_file_path(log_path)
            assert result == log_path

    def test_log_file_creates_directory(self) -> None:
        """Test that log file directory is created if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "subdir" / "test.log")
            result = _validate_log_file_path(log_path)
            assert result == log_path
            assert Path(tmpdir, "subdir").exists()

    @patch("os.makedirs")
    def test_os_error_returns_none(self, mock_makedirs: MagicMock) -> None:
        """Test that OSError returns None."""
        mock_makedirs.side_effect = OSError("Permission denied")

        result = _validate_log_file_path("/invalid/path/test.log")

        assert result is None

    @patch("os.access")
    def test_non_writable_directory(self, mock_access: MagicMock) -> None:
        """Test that non-writable directory returns None."""
        mock_access.return_value = False

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "test.log")
            result = _validate_log_file_path(log_path)

            assert result is None


class TestCreateFileHandler:
    """Tests for _create_file_handler function."""

    def test_create_handler_success(self) -> None:
        """Test successfully creating file handler."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as tmp:
            log_file = tmp.name

        try:
            from arbitrium.logging.structured import ContextFilter

            handler = _create_file_handler(
                log_file,
                "%(message)s",
                DuplicateFilter(),
                ContextFilter(),
                include_module=True,
            )

            assert handler is not None
            assert isinstance(handler, logging.FileHandler)
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_create_handler_with_json_format(self) -> None:
        """Test creating file handler (always uses JSON format)."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as tmp:
            log_file = tmp.name

        try:
            from arbitrium.logging.structured import ContextFilter

            handler = _create_file_handler(
                log_file,
                "%(message)s",
                DuplicateFilter(),
                ContextFilter(),
                include_module=True,
            )

            assert handler is not None
            assert isinstance(handler, logging.FileHandler)
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_create_handler_os_error(self) -> None:
        """Test that OS error returns None."""
        handler = _create_file_handler(
            "/invalid/path/test.log",
            "%(message)s",
            DuplicateFilter(),
            MagicMock(),
            include_module=True,
        )

        assert handler is None

    @patch("logging.FileHandler")
    def test_create_handler_unexpected_error(
        self, mock_handler: MagicMock
    ) -> None:
        """Test that unexpected error returns None."""
        mock_handler.side_effect = Exception("Unexpected error")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as tmp:
            log_file = tmp.name

        try:
            handler = _create_file_handler(
                log_file,
                "%(message)s",
                DuplicateFilter(),
                MagicMock(),
                include_module=True,
            )

            assert handler is None
        finally:
            Path(log_file).unlink(missing_ok=True)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default(self) -> None:
        """Test setup_logging with default parameters."""
        logger = setup_logging(enable_file_logging=False)

        assert logger is not None
        assert logger.name == "arbitrium"
        assert logger.level == logging.DEBUG

    def test_setup_logging_with_log_file(self) -> None:
        """Test setup_logging with log file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as tmp:
            log_file = tmp.name

        try:
            logger = setup_logging(log_file=log_file)

            assert logger is not None
            assert Path(log_file).exists()
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_setup_logging_debug_mode(self) -> None:
        """Test setup_logging in debug mode."""
        logger = setup_logging(debug=True, enable_file_logging=False)

        assert logger is not None
        # Root logger should be at DEBUG level
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_logging_verbose_mode(self) -> None:
        """Test setup_logging in verbose mode."""
        logger = setup_logging(verbose=True, enable_file_logging=False)

        assert logger is not None

    def test_setup_logging_with_log_file_always_json(self) -> None:
        """Test setup_logging with log file (always uses JSON format)."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as tmp:
            log_file = tmp.name

        try:
            logger = setup_logging(log_file=log_file)

            assert logger is not None
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_setup_logging_without_module_info(self) -> None:
        """Test setup_logging without module information."""
        logger = setup_logging(include_module=False, enable_file_logging=False)

        assert logger is not None

    def test_setup_logging_creates_timestamped_file(self) -> None:
        """Test that setup_logging creates timestamped log file by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = Path.cwd()
            try:
                import os

                os.chdir(tmpdir)

                logger = setup_logging(enable_file_logging=True)

                assert logger is not None

                # Check that a log file was created with timestamp pattern
                log_files = list(Path(tmpdir).glob("arbitrium_*_logs.log"))
                assert len(log_files) > 0
            finally:
                os.chdir(original_dir)
                # Clean up log files
                for log_file in Path(tmpdir).glob("arbitrium_*_logs.log"):
                    log_file.unlink()

    @patch("arbitrium.logging.setup._validate_log_file_path")
    def test_setup_logging_invalid_log_path(
        self, mock_validate: MagicMock
    ) -> None:
        """Test setup_logging with invalid log file path."""
        mock_validate.return_value = None

        logger = setup_logging(log_file="/invalid/path/test.log")

        assert logger is not None
        # Should continue without file logging

    def test_litellm_configuration(self) -> None:
        """Test that LiteLLM loggers are configured correctly."""
        logger = setup_logging(enable_file_logging=False)

        assert logger is not None

        # Check that LiteLLM logger is configured
        litellm_logger = logging.getLogger("litellm")
        assert litellm_logger.level == logging.ERROR
        assert litellm_logger.propagate is False

    def test_third_party_logger_levels(self) -> None:
        """Test that third-party loggers have correct levels."""
        logger = setup_logging(enable_file_logging=False)

        assert logger is not None

        # Check various third-party loggers
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
        assert logging.getLogger("openai").level == logging.WARNING

    def test_setup_logging_removes_existing_handlers(self) -> None:
        """Test that setup_logging removes existing handlers."""
        # Add a test handler
        root_logger = logging.getLogger()
        test_handler = logging.StreamHandler()
        root_logger.addHandler(test_handler)
        initial_handler_count = len(root_logger.handlers)

        logger = setup_logging(enable_file_logging=False)

        assert logger is not None
        # Handler count should be different after setup
        assert len(root_logger.handlers) != initial_handler_count
