"""
Unit tests for secrets management.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from arbitrium.utils.exceptions import ConfigurationError
from arbitrium.utils.secrets import (
    _ensure_op_cli_is_available,
    _fetch_secret_from_1password,
    _get_missing_providers,
    _handle_missing_op_cli,
    get_secret_config,
    load_secrets,
)


class TestGetSecretConfig:
    """Tests for get_secret_config function."""

    def test_valid_secret_config(self) -> None:
        """Test extracting valid secret configuration."""
        config = {
            "secrets": {
                "providers": {
                    "openai": {
                        "env_var": "OPENAI_API_KEY",
                        "op_path": "op://vault/openai/key",
                    },
                    "anthropic": {
                        "env_var": "ANTHROPIC_API_KEY",
                        "op_path": "op://vault/anthropic/key",
                    },
                }
            }
        }

        result = get_secret_config(config)

        assert result == {
            "openai": ("OPENAI_API_KEY", "op://vault/openai/key"),
            "anthropic": ("ANTHROPIC_API_KEY", "op://vault/anthropic/key"),
        }

    def test_config_none_raises_error(self) -> None:
        """Test that None config raises ConfigurationError."""
        with pytest.raises(
            ConfigurationError, match="Secret configuration not provided"
        ):
            get_secret_config(None)  # type: ignore

    def test_config_not_dict_raises_error(self) -> None:
        """Test that non-dict config raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Invalid config type"):
            get_secret_config("not a dict")  # type: ignore

    def test_missing_secrets_section_raises_error(self) -> None:
        """Test that missing secrets section raises ConfigurationError."""
        config = {"other": "data"}

        with pytest.raises(
            ConfigurationError, match="No 'secrets' section found"
        ):
            get_secret_config(config)

    def test_empty_secrets_section_raises_error(self) -> None:
        """Test that empty secrets section raises ConfigurationError."""
        config = {"secrets": {}}

        with pytest.raises(
            ConfigurationError, match="No 'secrets' section found"
        ):
            get_secret_config(config)

    def test_missing_providers_raises_error(self) -> None:
        """Test that missing providers raises ConfigurationError."""
        config = {"secrets": {"other": "data"}}

        with pytest.raises(ConfigurationError, match="No providers found"):
            get_secret_config(config)

    def test_invalid_provider_config_raises_error(self) -> None:
        """Test that invalid provider config raises ConfigurationError."""
        config = {"secrets": {"providers": {"openai": "invalid"}}}

        with pytest.raises(ConfigurationError, match="must be a dictionary"):
            get_secret_config(config)

    def test_missing_env_var_raises_error(self) -> None:
        """Test that missing env_var raises ConfigurationError."""
        config = {
            "secrets": {
                "providers": {"openai": {"op_path": "op://vault/openai/key"}}
            }
        }

        with pytest.raises(ConfigurationError, match="Missing 'env_var'"):
            get_secret_config(config)

    def test_missing_op_path_raises_error(self) -> None:
        """Test that missing op_path raises ConfigurationError."""
        config = {
            "secrets": {"providers": {"openai": {"env_var": "OPENAI_API_KEY"}}}
        }

        with pytest.raises(ConfigurationError, match="Missing 'op_path'"):
            get_secret_config(config)


class TestEnsureOpCliIsAvailable:
    """Tests for _ensure_op_cli_is_available function."""

    @patch("subprocess.run")
    def test_op_cli_available(self, mock_run: MagicMock) -> None:
        """Test when 1Password CLI is available and authenticated."""
        mock_run.return_value = MagicMock(returncode=0)

        # Should not raise
        _ensure_op_cli_is_available()

        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_op_cli_not_installed(self, mock_run: MagicMock) -> None:
        """Test when 1Password CLI is not installed."""
        mock_run.side_effect = FileNotFoundError()

        with pytest.raises(ConfigurationError, match="is not installed"):
            _ensure_op_cli_is_available()

    @patch("subprocess.run")
    def test_op_cli_not_signed_in(self, mock_run: MagicMock) -> None:
        """Test when not signed in to 1Password CLI."""
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "op")

        with pytest.raises(ConfigurationError, match="not signed in"):
            _ensure_op_cli_is_available()

    @patch("subprocess.run")
    def test_op_cli_timeout(self, mock_run: MagicMock) -> None:
        """Test when 1Password CLI times out."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("op", 5)

        with pytest.raises(ConfigurationError, match="timed out"):
            _ensure_op_cli_is_available()


class TestGetMissingProviders:
    """Tests for _get_missing_providers function."""

    def test_all_secrets_present(self) -> None:
        """Test when all secrets are present in environment."""
        secret_config = {
            "openai": ("OPENAI_API_KEY", "op://vault/openai/key"),
            "anthropic": ("ANTHROPIC_API_KEY", "op://vault/anthropic/key"),
        }

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "key1",  # pragma: allowlist secret
                "ANTHROPIC_API_KEY": "key2",  # pragma: allowlist secret
            },
        ):
            result = _get_missing_providers(
                secret_config, ["openai", "anthropic"]
            )

        assert result == []

    def test_some_secrets_missing(self) -> None:
        """Test when some secrets are missing."""
        secret_config = {
            "openai": ("OPENAI_API_KEY", "op://vault/openai/key"),
            "anthropic": ("ANTHROPIC_API_KEY", "op://vault/anthropic/key"),
        }

        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "key1"},  # pragma: allowlist secret
            clear=True,
        ):
            result = _get_missing_providers(
                secret_config, ["openai", "anthropic"]
            )

        assert result == ["anthropic"]

    def test_all_secrets_missing(self) -> None:
        """Test when all secrets are missing."""
        secret_config = {
            "openai": ("OPENAI_API_KEY", "op://vault/openai/key"),
            "anthropic": ("ANTHROPIC_API_KEY", "op://vault/anthropic/key"),
        }

        with patch.dict(os.environ, {}, clear=True):
            result = _get_missing_providers(
                secret_config, ["openai", "anthropic"]
            )

        assert "openai" in result
        assert "anthropic" in result

    def test_case_insensitive_provider_names(self) -> None:
        """Test that provider names are case-insensitive."""
        secret_config = {
            "openai": ("OPENAI_API_KEY", "op://vault/openai/key"),
        }

        with patch.dict(os.environ, {}, clear=True):
            result = _get_missing_providers(
                secret_config, ["OpenAI", "OPENAI"]
            )

        assert "openai" in result


class TestHandleMissingOpCli:
    """Tests for _handle_missing_op_cli function."""

    def test_raises_configuration_error(self) -> None:
        """Test that function raises ConfigurationError with proper message."""
        secret_config = {
            "openai": ("OPENAI_API_KEY", "op://vault/openai/key"),
            "anthropic": ("ANTHROPIC_API_KEY", "op://vault/anthropic/key"),
        }
        providers_to_fetch = ["openai", "anthropic"]
        original_error = ConfigurationError("1Password CLI not available")

        with pytest.raises(
            ConfigurationError,
            match="Secrets not found in environment variables",
        ):
            _handle_missing_op_cli(
                secret_config, providers_to_fetch, original_error
            )


class TestFetchSecretFrom1Password:
    """Tests for _fetch_secret_from_1password function."""

    @patch("subprocess.run")
    def test_successful_fetch(self, mock_run: MagicMock) -> None:
        """Test successfully fetching secret from 1Password."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="secret_value\n"
        )

        # Start with empty environment
        test_env = {}
        with patch.dict(os.environ, test_env, clear=True):
            _fetch_secret_from_1password("TEST_KEY", "op://vault/test/key")
            # Check within the context where the environment was modified
            assert "TEST_KEY" in os.environ
            assert os.environ["TEST_KEY"] == "secret_value"

        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_empty_secret_raises_error(self, mock_run: MagicMock) -> None:
        """Test that empty secret raises ConfigurationError."""
        mock_run.return_value = MagicMock(returncode=0, stdout="   \n")

        with pytest.raises(ConfigurationError, match="is empty"):
            _fetch_secret_from_1password("TEST_KEY", "op://vault/test/key")

    @patch("subprocess.run")
    def test_subprocess_error_raises_config_error(
        self, mock_run: MagicMock
    ) -> None:
        """Test that subprocess error raises ConfigurationError."""
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(
            1, "op", stderr="Permission denied"
        )

        with pytest.raises(ConfigurationError, match="Failed to read secret"):
            _fetch_secret_from_1password("TEST_KEY", "op://vault/test/key")

    @patch("subprocess.run")
    def test_timeout_raises_config_error(self, mock_run: MagicMock) -> None:
        """Test that timeout raises ConfigurationError."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("op", 5)

        with pytest.raises(ConfigurationError, match="Timeout"):
            _fetch_secret_from_1password("TEST_KEY", "op://vault/test/key")


class TestLoadSecrets:
    """Tests for load_secrets function."""

    @patch("arbitrium.utils.secrets.get_secret_config")
    def test_all_secrets_present(self, mock_get_config: MagicMock) -> None:
        """Test when all required secrets are present."""
        mock_get_config.return_value = {
            "openai": ("OPENAI_API_KEY", "op://vault/openai/key"),
        }

        config = {"secrets": {"providers": {}}}

        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "key1"}  # pragma: allowlist secret
        ):
            load_secrets(config, ["openai"])

        # Should not raise, and should not try to fetch from 1Password

    @patch("arbitrium.utils.secrets.get_secret_config")
    @patch("arbitrium.utils.secrets._ensure_op_cli_is_available")
    @patch("arbitrium.utils.secrets._fetch_secret_from_1password")
    def test_fetch_missing_secrets(
        self,
        mock_fetch: MagicMock,
        mock_ensure: MagicMock,
        mock_get_config: MagicMock,
    ) -> None:
        """Test fetching missing secrets from 1Password."""
        mock_get_config.return_value = {
            "openai": ("OPENAI_API_KEY", "op://vault/openai/key"),
            "anthropic": ("ANTHROPIC_API_KEY", "op://vault/anthropic/key"),
        }

        config = {"secrets": {"providers": {}}}

        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "key1"},  # pragma: allowlist secret
            clear=True,
        ):
            load_secrets(config, ["openai", "anthropic"])

        # Should check for 1Password CLI
        mock_ensure.assert_called_once()
        # Should fetch only missing secret
        mock_fetch.assert_called_once_with(
            "ANTHROPIC_API_KEY", "op://vault/anthropic/key"
        )

    @patch("arbitrium.utils.secrets.get_secret_config")
    @patch("arbitrium.utils.secrets._ensure_op_cli_is_available")
    def test_op_cli_unavailable_raises_error(
        self, mock_ensure: MagicMock, mock_get_config: MagicMock
    ) -> None:
        """Test error when 1Password CLI is unavailable and secrets missing."""
        mock_get_config.return_value = {
            "openai": ("OPENAI_API_KEY", "op://vault/openai/key"),
        }
        mock_ensure.side_effect = ConfigurationError(
            "1Password CLI not available"
        )

        config = {"secrets": {"providers": {}}}

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ConfigurationError, match="Secrets not found in environment"
            ):
                load_secrets(config, ["openai"])
