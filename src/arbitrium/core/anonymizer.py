"""Model anonymization for unbiased tournament evaluation."""

import random

from arbitrium.logging import get_contextual_logger


class ModelAnonymizer:
    """Handles model anonymization to prevent bias in evaluations."""

    def __init__(self, deterministic_mode: bool = False):
        """
        Initialize the anonymizer.

        Args:
            deterministic_mode: If True, use fixed random seed for reproducibility
        """
        self.logger = get_contextual_logger("arbitrium.anonymizer")
        self.rng = random.Random(42) if deterministic_mode else random.Random()

    @staticmethod
    def anonymize_model_keys(model_keys: list[str]) -> dict[str, str]:
        """
        Create anonymized mapping of model keys to LLM1, LLM2, etc.

        Args:
            model_keys: List of model keys to anonymize

        Returns:
            Dictionary mapping real keys to anonymized names
        """
        return {key: f"LLM{i + 1}" for i, key in enumerate(model_keys)}

    def anonymize_responses(
        self, responses: dict[str, str]
    ) -> tuple[dict[str, str], dict[str, str]]:
        """
        Create anonymized labels for model responses.

        Anonymization is deterministic (alphabetical order) to ensure consistency.

        Args:
            responses: Dictionary of model names to their responses

        Returns:
            Tuple of (anonymized_responses, reverse_mapping)
            - anonymized_responses: Responses with anonymized keys (LLM1, LLM2, etc.)
            - reverse_mapping: Maps anonymized names back to real names
        """
        model_names = sorted(responses.keys())
        code_names = [f"LLM{i + 1}" for i in range(len(model_names))]

        forward_mapping = {
            name: code_names[i] for i, name in enumerate(model_names)
        }
        reverse_mapping = {v: k for k, v in forward_mapping.items()}

        anonymized_responses = {
            forward_mapping[name]: responses[name] for name in model_names
        }

        self.logger.debug("Anonymization mapping (alphabetical order):")
        for real_name, anon_name in forward_mapping.items():
            self.logger.debug(f"  {real_name} → {anon_name}")

        return anonymized_responses, reverse_mapping
