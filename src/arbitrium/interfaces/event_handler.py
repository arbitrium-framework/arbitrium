from abc import ABC, abstractmethod
from typing import Any


class EventHandler(ABC):
    @abstractmethod
    def publish(self, event_name: str, data: dict[str, Any]) -> None: ...
