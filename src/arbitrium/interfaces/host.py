from abc import ABC, abstractmethod


class HostEnvironment(ABC):
    @abstractmethod
    async def read_file(self, path: str) -> str: ...

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None: ...

    @abstractmethod
    def get_secret(self, key: str) -> str | None: ...
