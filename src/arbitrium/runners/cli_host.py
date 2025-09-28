import os

import aiofiles

from arbitrium.interfaces.host import HostEnvironment


class CliHost(HostEnvironment):
    async def read_file(self, path: str) -> str:
        async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
            content: str = await f.read()
            return content

    async def write_file(self, path: str, content: str) -> None:
        async with aiofiles.open(path, mode="w", encoding="utf-8") as f:
            await f.write(content)

    def get_secret(self, key: str) -> str | None:
        return os.getenv(key)
