from abc import ABC, abstractmethod
from typing import Any


class WorkflowSerializer(ABC):
    @abstractmethod
    async def load_from_file(self, path: str) -> dict[str, Any]:
        pass

    @abstractmethod
    async def load_from_string(self, content: str) -> dict[str, Any]:
        pass
