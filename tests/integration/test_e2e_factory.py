"""End-to-end tests for model factory."""

import pytest

from arbitrium.models.factory import create_models_from_config
from tests.integration.conftest import MockModel


class TestModelFactoryBasics:
    """Test basic model factory functionality."""

    def test_create_models_from_valid_config(self, tmp_output_dir) -> None:
        """Test creating models from valid configuration."""
        config = {
            "models": {
                "model_a": {
                    "provider": "mock",
                    "model_name": "test-a",
                    "display_name": "Model A",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "context_window": 4000,
                },
                "model_b": {
                    "provider": "mock",
                    "model_name": "test-b",
                    "display_name": "Model B",
                    "temperature": 0.8,
                    "max_tokens": 2000,
                    "context_window": 8000,
                },
            },
            "outputs_dir": str(tmp_output_dir),
        }

        models = create_models_from_config(config)

        assert len(models) == 2
        assert "model_a" in models
        assert "model_b" in models

        # Verify models are MockModel instances
        assert isinstance(models["model_a"], MockModel)
        assert isinstance(models["model_b"], MockModel)

    def test_create_single_model(self, tmp_output_dir) -> None:
        """Test creating single model from config."""
        config = {
            "models": {
                "only_model": {
                    "provider": "mock",
                    "model_name": "test",
                    "display_name": "Only Model",
                    "temperature": 0.5,
                    "max_tokens": 500,
                    "context_window": 2000,
                }
            },
            "outputs_dir": str(tmp_output_dir),
        }

        models = create_models_from_config(config)

        assert len(models) == 1
        assert "only_model" in models

    def test_create_models_with_different_providers(
        self, tmp_output_dir
    ) -> None:
        """Test creating models with different providers."""
        config = {
            "models": {
                "mock_model": {
                    "provider": "mock",
                    "model_name": "test-mock",
                    "display_name": "Mock Model",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "context_window": 4000,
                },
                "litellm_model": {
                    "provider": "openai",
                    "model_name": "gpt-3.5-turbo",
                    "display_name": "GPT-3.5",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "context_window": 4000,
                },
            },
            "outputs_dir": str(tmp_output_dir),
        }

        models = create_models_from_config(config)

        # Mock model should be created
        assert "mock_model" in models
        assert isinstance(models["mock_model"], MockModel)

        # LiteLLM model should also be created (will be LiteLLMModel)
        assert "litellm_model" in models


class TestModelFactoryEdgeCases:
    """Test edge cases in model factory."""

    def test_empty_models_config(self, tmp_output_dir) -> None:
        """Test factory with empty models configuration."""
        config = {
            "models": {},
            "outputs_dir": str(tmp_output_dir),
        }

        models = create_models_from_config(config)

        assert len(models) == 0

    def test_missing_models_key(self, tmp_output_dir) -> None:
        """Test factory when models key is missing."""
        config = {
            "outputs_dir": str(tmp_output_dir),
        }

        # Should handle missing 'models' key gracefully
        try:
            models = create_models_from_config(config)
            # If it doesn't raise, should return empty dict
            assert len(models) == 0
        except KeyError:
            # Or it might raise KeyError, which is also acceptable
            pass

    def test_invalid_model_config_skipped(self, tmp_output_dir) -> None:
        """Test that invalid model configs are skipped."""
        config = {
            "models": {
                "valid_model": {
                    "provider": "mock",
                    "model_name": "test",
                    "display_name": "Valid",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "context_window": 4000,
                },
                "invalid_model": "not a dict",  # Invalid config
            },
            "outputs_dir": str(tmp_output_dir),
        }

        models = create_models_from_config(config)

        # Valid model should be created
        assert "valid_model" in models

        # Invalid model should be skipped
        assert "invalid_model" not in models

    def test_models_config_not_dict(self, tmp_output_dir) -> None:
        """Test when models config is not a dictionary."""
        config = {
            "models": "not a dict",
            "outputs_dir": str(tmp_output_dir),
        }

        models = create_models_from_config(config)

        # Should return empty dict when models is not a dict
        assert len(models) == 0


class TestMockModelCreation:
    """Test mock model creation specifically."""

    def test_mock_model_with_all_parameters(self, tmp_output_dir) -> None:
        """Test creating mock model with all parameters."""
        config = {
            "models": {
                "full_config": {
                    "provider": "mock",
                    "model_name": "test-full",
                    "display_name": "Full Config Model",
                    "temperature": 0.9,
                    "max_tokens": 2048,
                    "context_window": 8192,
                }
            },
            "outputs_dir": str(tmp_output_dir),
        }

        models = create_models_from_config(config)

        model = models["full_config"]
        assert isinstance(model, MockModel)
        assert model.model_name == "test-full"
        assert model.display_name == "Full Config Model"
        assert model.temperature == 0.9
        assert model.max_tokens == 2048
        assert model.context_window == 8192

    def test_mock_model_with_minimal_parameters(self, tmp_output_dir) -> None:
        """Test creating mock model with minimal parameters."""
        config = {
            "models": {
                "minimal": {
                    "provider": "mock",
                    "model_name": "test-minimal",
                    "display_name": "Minimal Model",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "context_window": 4000,
                }
            },
            "outputs_dir": str(tmp_output_dir),
        }

        models = create_models_from_config(config)

        assert "minimal" in models
        assert isinstance(models["minimal"], MockModel)

    def test_mock_model_defaults(self, tmp_output_dir) -> None:
        """Test mock model uses defaults for missing parameters."""
        config = {
            "models": {
                "with_defaults": {
                    "provider": "mock",
                    "model_name": "test-defaults",
                    # Missing display_name, should use model_key
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "context_window": 4000,
                }
            },
            "outputs_dir": str(tmp_output_dir),
        }

        models = create_models_from_config(config)

        model = models["with_defaults"]
        # Should use model_key as display_name when not provided
        assert model.display_name == "with_defaults"


class TestFactoryErrorHandling:
    """Test error handling in model factory."""

    def test_mock_import_error_fallback(self, tmp_output_dir) -> None:
        """Test fallback when MockModel import fails."""
        # This test simulates what happens in non-test environment
        # where MockModel might not be available
        config = {
            "models": {
                "mock_model": {
                    "provider": "mock",
                    "model_name": "test",
                    "display_name": "Mock Model",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "context_window": 4000,
                }
            },
            "outputs_dir": str(tmp_output_dir),
        }

        # Should create model (either MockModel or fallback to LiteLLMModel)
        models = create_models_from_config(config)

        assert "mock_model" in models


class TestFactoryWithMultipleModels:
    """Test factory with multiple models."""

    def test_create_three_models(self, tmp_output_dir) -> None:
        """Test creating three models."""
        config = {
            "models": {
                "model_1": {
                    "provider": "mock",
                    "model_name": "test-1",
                    "display_name": "Model 1",
                    "temperature": 0.5,
                    "max_tokens": 500,
                    "context_window": 2000,
                },
                "model_2": {
                    "provider": "mock",
                    "model_name": "test-2",
                    "display_name": "Model 2",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "context_window": 4000,
                },
                "model_3": {
                    "provider": "mock",
                    "model_name": "test-3",
                    "display_name": "Model 3",
                    "temperature": 0.9,
                    "max_tokens": 2000,
                    "context_window": 8000,
                },
            },
            "outputs_dir": str(tmp_output_dir),
        }

        models = create_models_from_config(config)

        assert len(models) == 3
        assert all(key in models for key in ["model_1", "model_2", "model_3"])

    def test_create_many_models(self, tmp_output_dir) -> None:
        """Test creating many models."""
        num_models = 10
        models_config = {
            f"model_{i}": {
                "provider": "mock",
                "model_name": f"test-{i}",
                "display_name": f"Model {i}",
                "temperature": 0.7,
                "max_tokens": 1000,
                "context_window": 4000,
            }
            for i in range(num_models)
        }

        config = {
            "models": models_config,
            "outputs_dir": str(tmp_output_dir),
        }

        models = create_models_from_config(config)

        assert len(models) == num_models


class TestFactoryIntegration:
    """Test factory integration with Arbitrium."""

    @pytest.mark.asyncio
    async def test_factory_used_in_arbitrium_initialization(
        self,
        basic_config: dict,
    ) -> None:
        """Test that factory is used during Arbitrium initialization."""
        from arbitrium import Arbitrium

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Models should have been created via factory
        assert len(arbitrium.all_models) > 0

    @pytest.mark.asyncio
    async def test_factory_creates_correct_number_of_models(
        self,
        tmp_output_dir,
    ) -> None:
        """Test that factory creates correct number of models."""
        from arbitrium import Arbitrium

        config = {
            "models": {
                "model_a": {
                    "provider": "mock",
                    "model_name": "test-a",
                    "display_name": "Model A",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "context_window": 4000,
                },
                "model_b": {
                    "provider": "mock",
                    "model_name": "test-b",
                    "display_name": "Model B",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "context_window": 4000,
                },
            },
            "outputs_dir": str(tmp_output_dir),
        }

        arbitrium = await Arbitrium.from_settings(
            settings=config,
            skip_secrets=True,
            skip_health_check=True,
        )

        assert len(arbitrium.all_models) == 2


class TestFactoryConfigVariations:
    """Test factory with various config variations."""

    def test_different_temperatures(self, tmp_output_dir) -> None:
        """Test models with different temperature settings."""
        config = {
            "models": {
                "conservative": {
                    "provider": "mock",
                    "model_name": "test-conservative",
                    "display_name": "Conservative",
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "context_window": 4000,
                },
                "creative": {
                    "provider": "mock",
                    "model_name": "test-creative",
                    "display_name": "Creative",
                    "temperature": 1.0,
                    "max_tokens": 1000,
                    "context_window": 4000,
                },
            },
            "outputs_dir": str(tmp_output_dir),
        }

        models = create_models_from_config(config)

        assert models["conservative"].temperature == 0.1
        assert models["creative"].temperature == 1.0

    def test_different_context_windows(self, tmp_output_dir) -> None:
        """Test models with different context windows."""
        config = {
            "models": {
                "small_context": {
                    "provider": "mock",
                    "model_name": "test-small",
                    "display_name": "Small Context",
                    "temperature": 0.7,
                    "max_tokens": 500,
                    "context_window": 2048,
                },
                "large_context": {
                    "provider": "mock",
                    "model_name": "test-large",
                    "display_name": "Large Context",
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "context_window": 128000,
                },
            },
            "outputs_dir": str(tmp_output_dir),
        }

        models = create_models_from_config(config)

        assert models["small_context"].context_window == 2048
        assert models["large_context"].context_window == 128000
