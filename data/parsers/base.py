import asyncio
import logging
from abc import ABC, abstractmethod
from collections import deque

import aiohttp
from bs4 import BeautifulSoup

logger = logging.Logger(__name__)


MAX_TRIES = 5


class BaseParser(ABC):
    BASE_URL: str
    FILE_NAME: str

    async def _get_page_soup(self, url: str, tried_amount: int = 0) -> BeautifulSoup:
        await asyncio.sleep(0.1)

        async def fetch():
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200 and tried_amount < MAX_TRIES:
                        logger.error(
                            f"Could not fetch {url} - {response.status}. Trying again..."
                        )

                        await asyncio.sleep(40)

                        return await self._get_page_soup(url, tried_amount + 1)
                    else:
                        content = await response.text()

            return content

        try:
            content = await fetch()
        except aiohttp.ClientError as e:
            logger.error(repr(e))
            content = ""

        if isinstance(content, BeautifulSoup):
            return content

        soup = BeautifulSoup(content or "", features="html.parser")

        return soup

    @abstractmethod
    async def parse(self) -> None:
        """
        Should save file with FILE_NAME name on the data folder.
        :return:
        """
        pass
