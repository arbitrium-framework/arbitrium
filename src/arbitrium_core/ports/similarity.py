from abc import ABC, abstractmethod
from typing import Any


class SimilarityEngine(ABC):
    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        pass

    @abstractmethod
    def transform(self, texts: list[str]) -> Any:
        pass

    @abstractmethod
    def is_fitted(self) -> bool:
        pass

    @abstractmethod
    def compute_similarity(
        self, query_vector: Any, corpus_vectors: Any
    ) -> list[float]:
        pass
