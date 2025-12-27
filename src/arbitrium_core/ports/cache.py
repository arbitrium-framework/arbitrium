from abc import ABC, abstractmethod


class CacheProtocol(ABC):
    @abstractmethod
    def get(
        self, model_name: str, prompt: str, temperature: float, max_tokens: int
    ) -> tuple[str, float] | None:
        pass

    @abstractmethod
    def set(
        self,
        model_name: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        response: str,
        cost: float,
    ) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
