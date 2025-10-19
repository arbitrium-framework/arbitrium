"""
Exception classes for Arbitrium Framework.

This module provides a structured hierarchy of custom exceptions
to enable more granular error handling throughout the application.
"""


class ArbitriumError(Exception):
    """Base exception class for all Arbitrium Framework-specific errors."""

    def __init__(self, message: str, *args: object, **kwargs: object) -> None:
        self.message = message
        super().__init__(message, *args)


class ConfigurationError(ArbitriumError):
    """Raised when there are issues with the configuration."""


class FatalError(ArbitriumError):
    """Raised for fatal errors that should terminate the application."""


class APIError(ArbitriumError):
    """Base class for API-related errors."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        status_code: int | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.provider = provider
        self.status_code = status_code

        # Enhance the message with provider and status code if available
        enhanced_message = message
        if provider:
            enhanced_message = f"[{provider}] {enhanced_message}"
        if status_code:
            enhanced_message = f"{enhanced_message} (Status: {status_code})"

        super().__init__(enhanced_message, *args)


class RateLimitError(APIError):
    """Raised when hitting rate limits on API calls."""


class AuthenticationError(APIError):
    """Raised when API authentication fails."""


class ModelError(ArbitriumError):
    """Base class for model-related errors."""

    def __init__(
        self,
        message: str,
        model_key: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.model_key = model_key
        enhanced_message = message

        if model_key:
            enhanced_message = f"[{model_key}] {enhanced_message}"

        super().__init__(enhanced_message, *args)


class ModelResponseError(ModelError):
    """Raised when a model's response is invalid or problematic."""


class FileSystemError(ArbitriumError):
    """Base class for filesystem-related errors."""

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.file_path = file_path
        enhanced_message = message

        if file_path:
            enhanced_message = f"[{file_path}] {enhanced_message}"

        super().__init__(enhanced_message, *args)


class InputError(ArbitriumError):
    """Raised when there's an issue with user input."""


class BudgetExceededError(ArbitriumError):
    """Raised when tournament exceeds the configured budget."""

    def __init__(
        self,
        message: str,
        spent: float,
        budget: float,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.spent = spent
        self.budget = budget
        enhanced_message = f"Budget exceeded: spent ${spent:.4f} >= limit ${budget:.4f}. {message}"
        super().__init__(enhanced_message, *args)


class TournamentTimeoutError(ArbitriumError):
    """Raised when tournament exceeds the configured time limit."""

    def __init__(
        self,
        message: str,
        elapsed: float,
        timeout: float,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.elapsed = elapsed
        self.timeout = timeout
        enhanced_message = f"Tournament timeout: {elapsed:.1f}s >= limit {timeout:.1f}s. {message}"
        super().__init__(enhanced_message, *args)
