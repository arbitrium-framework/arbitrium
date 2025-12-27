from abc import ABC, abstractmethod
from typing import Any


class SecretsProvider(ABC):
    @abstractmethod
    def get_secret(self, provider: str) -> str | None:
        pass

    @abstractmethod
    def load_all(self, config: dict[str, Any]) -> dict[str, str]:
        pass
