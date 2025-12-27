from abc import ABC, abstractmethod
from typing import Any


class SecretsProvider(ABC):
    @abstractmethod
    def get_secret(self, key: str) -> str | None:
        pass

    @abstractmethod
    def load_secrets(
        self, config: dict[str, Any], required_providers: list[str]
    ) -> None:
        pass
