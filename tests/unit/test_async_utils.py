"""
Unit tests for async utility functions.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbitrium.utils.async_ import (
    _check_input_validation,
    _check_non_interactive_environment,
    _try_set_future_exception,
    _try_set_future_result,
    _validate_default_value,
    async_input,
)


class TestValidateDefaultValue:
    """Tests for _validate_default_value function."""

    def test_empty_default_returns_empty(self) -> None:
        """Test that empty default is returned as-is."""
        logger = MagicMock()
        result = _validate_default_value("", None, 0, None, logger)
        assert result == ""
        logger.warning.assert_not_called()

    def test_valid_default_with_no_constraints(self) -> None:
        """Test valid default with no constraints."""
        logger = MagicMock()
        result = _validate_default_value("test", None, 0, None, logger)
        assert result == "test"
        logger.warning.assert_not_called()

    def test_default_fails_validation_function(self) -> None:
        """Test default value that fails custom validation."""
        logger = MagicMock()
        validation_func = lambda x: x.startswith("valid")
        result = _validate_default_value(
            "invalid", validation_func, 0, None, logger
        )
        assert result == ""
        logger.warning.assert_called()
        logger.error.assert_called()

    def test_default_fails_min_length(self) -> None:
        """Test default value that fails min length check."""
        logger = MagicMock()
        result = _validate_default_value("ab", None, 5, None, logger)
        assert result == ""
        logger.warning.assert_called()

    def test_default_fails_max_length(self) -> None:
        """Test default value that fails max length check."""
        logger = MagicMock()
        result = _validate_default_value("too long text", None, 0, 5, logger)
        assert result == ""
        logger.warning.assert_called()

    def test_default_passes_all_validations(self) -> None:
        """Test default value that passes all validations."""
        logger = MagicMock()
        validation_func = lambda x: len(x) > 0
        result = _validate_default_value(
            "valid", validation_func, 3, 10, logger
        )
        assert result == "valid"
        logger.warning.assert_not_called()


class TestTrySetFutureResult:
    """Tests for _try_set_future_result function."""

    @pytest.mark.asyncio
    async def test_set_result_success(self) -> None:
        """Test successfully setting future result."""
        logger = MagicMock()
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        result = _try_set_future_result(future, "test_value", logger)

        assert result is True
        assert future.result() == "test_value"

    @pytest.mark.asyncio
    async def test_set_result_already_done(self) -> None:
        """Test setting result when future is already done."""
        logger = MagicMock()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        future.set_result("already_set")

        result = _try_set_future_result(future, "new_value", logger)

        assert result is False
        assert future.result() == "already_set"


class TestTrySetFutureException:
    """Tests for _try_set_future_exception function."""

    @pytest.mark.asyncio
    async def test_set_exception_success(self) -> None:
        """Test successfully setting future exception."""
        logger = MagicMock()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        exception = ValueError("test error")

        _try_set_future_exception(future, exception, logger)

        with pytest.raises(ValueError, match="test error"):
            future.result()

    @pytest.mark.asyncio
    async def test_set_exception_already_done(self) -> None:
        """Test setting exception when future is already done."""
        logger = MagicMock()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        future.set_result("already_set")

        exception = ValueError("test error")
        _try_set_future_exception(future, exception, logger)

        # Future should still have original result
        assert future.result() == "already_set"


class TestCheckInputValidation:
    """Tests for _check_input_validation function."""

    def test_input_passes_all_validations(self) -> None:
        """Test input that passes all validation checks."""
        result = _check_input_validation("valid input", 3, 20, None, "")
        assert result is True

    def test_input_fails_min_length(self) -> None:
        """Test input that fails minimum length check."""
        result = _check_input_validation("ab", 5, None, None, "")
        assert result is False

    def test_input_fails_max_length(self) -> None:
        """Test input that fails maximum length check."""
        result = _check_input_validation("too long", 0, 5, None, "")
        assert result is False

    def test_input_fails_custom_validation(self) -> None:
        """Test input that fails custom validation function."""
        validation_func = lambda x: x.isdigit()
        result = _check_input_validation(
            "abc123", 0, None, validation_func, "Must be digits"
        )
        assert result is False

    def test_input_passes_custom_validation(self) -> None:
        """Test input that passes custom validation function."""
        validation_func = lambda x: x.isdigit()
        result = _check_input_validation(
            "123", 0, None, validation_func, "Must be digits"
        )
        assert result is True


class TestCheckNonInteractiveEnvironment:
    """Tests for _check_non_interactive_environment function."""

    @patch("sys.stdin.isatty")
    def test_non_interactive_environment(self, mock_isatty: MagicMock) -> None:
        """Test detection of non-interactive environment."""
        mock_isatty.return_value = False
        logger = MagicMock()

        result = _check_non_interactive_environment(
            "Enter input: ", "default", logger
        )

        assert result == "default"
        logger.warning.assert_called()

    @patch("sys.stdin.isatty")
    def test_interactive_environment(self, mock_isatty: MagicMock) -> None:
        """Test detection of interactive environment."""
        mock_isatty.return_value = True
        logger = MagicMock()

        result = _check_non_interactive_environment(
            "Enter input: ", "default", logger
        )

        assert result is None
        logger.warning.assert_not_called()


class TestAsyncInput:
    """Tests for async_input function."""

    @pytest.mark.asyncio
    @patch("sys.stdin.isatty")
    async def test_non_interactive_returns_default(
        self, mock_isatty: MagicMock
    ) -> None:
        """Test that non-interactive environment returns default immediately."""
        mock_isatty.return_value = False

        result = await async_input("Enter: ", default="default_value")

        assert result == "default_value"

    @pytest.mark.asyncio
    @patch("sys.stdin.isatty")
    @patch("builtins.input")
    async def test_valid_user_input(
        self, mock_input: MagicMock, mock_isatty: MagicMock
    ) -> None:
        """Test receiving valid user input."""
        mock_isatty.return_value = True
        mock_input.return_value = "user input"

        result = await async_input("Enter: ", timeout=1)

        assert result == "user input"

    @pytest.mark.asyncio
    @patch("sys.stdin.isatty")
    @patch("builtins.input")
    async def test_input_timeout(
        self, mock_input: MagicMock, mock_isatty: MagicMock
    ) -> None:
        """Test that input times out and returns default."""
        mock_isatty.return_value = True

        # Simulate a blocking input that never returns
        async def slow_input(*args):
            await asyncio.sleep(10)
            return "too slow"

        with patch("asyncio.get_running_loop") as mock_loop:
            loop = asyncio.get_running_loop()
            mock_loop.return_value = loop
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=slow_input
            )

            result = await async_input(
                "Enter: ", default="timeout_default", timeout=0.1
            )

            assert result == "timeout_default"

    @pytest.mark.asyncio
    @patch("sys.stdin.isatty")
    @patch("builtins.input")
    async def test_input_validation_retry(
        self, mock_input: MagicMock, mock_isatty: MagicMock
    ) -> None:
        """Test that invalid input triggers retry."""
        mock_isatty.return_value = True
        # First return invalid, then valid
        mock_input.side_effect = ["ab", "valid"]

        result = await async_input("Enter: ", min_length=5, timeout=2)

        assert result == "valid"
        assert mock_input.call_count == 2

    @pytest.mark.asyncio
    @patch("sys.stdin.isatty")
    async def test_invalid_default_value(self, mock_isatty: MagicMock) -> None:
        """Test that invalid default value is replaced with empty string."""
        mock_isatty.return_value = True

        # Create a validation function that rejects the default
        validation_func = lambda x: x.startswith("valid")

        with patch("builtins.input", return_value="valid_input"):
            result = await async_input(
                "Enter: ",
                default="invalid_default",
                validation_func=validation_func,
                timeout=1,
            )

            # Should get user input since default was invalid
            assert result == "valid_input"

    @pytest.mark.asyncio
    @patch("sys.stdin.isatty")
    @patch("builtins.input")
    async def test_max_length_validation(
        self, mock_input: MagicMock, mock_isatty: MagicMock
    ) -> None:
        """Test max length validation."""
        mock_isatty.return_value = True
        mock_input.side_effect = ["toolong", "ok"]

        result = await async_input("Enter: ", max_length=5, timeout=2)

        assert result == "ok"
        assert mock_input.call_count == 2

    @pytest.mark.asyncio
    @patch("sys.stdin.isatty")
    @patch("builtins.input")
    async def test_custom_validation_function(
        self, mock_input: MagicMock, mock_isatty: MagicMock
    ) -> None:
        """Test custom validation function."""
        mock_isatty.return_value = True
        mock_input.side_effect = ["abc", "123"]

        validation_func = lambda x: x.isdigit()
        result = await async_input(
            "Enter number: ",
            validation_func=validation_func,
            validation_message="Must be a number",
            timeout=2,
        )

        assert result == "123"

    @pytest.mark.asyncio
    @patch("sys.stdin.isatty")
    async def test_exception_during_input(
        self, mock_isatty: MagicMock
    ) -> None:
        """Test handling of exception during input."""
        mock_isatty.return_value = True

        with patch("asyncio.get_running_loop") as mock_loop:
            loop = asyncio.get_running_loop()
            mock_loop.return_value = loop
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=RuntimeError("Input error")
            )

            result = await async_input(
                "Enter: ", default="error_default", timeout=1
            )

            # Should return default on error
            assert result == "error_default"
