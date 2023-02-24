import asyncio
import os.path
import re
import uuid
from collections import defaultdict
from typing import List, Tuple, Union

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from data.parsers import surrogate_remover
from data.parsers.base import BaseParser


class MedUniverParser(BaseParser):
    BASE_URL = "https://meduniver.com/"
    FILE_NAME = "meduniver.csv"

    async def _get_sections_urls(self) -> List[Tuple[str, str]]:
        url = os.path.join(self.BASE_URL, "forum/")
        soup = await self._get_page_soup(url)

        forum_descriptions = soup.find_all("span", {"class": "forumdesc"})
        href_tags = [
            list(element.previous_elements)[4] for element in forum_descriptions
        ]
        sections_info = [
            (element.text.split(".")[0], element["href"]) for element in href_tags
        ]

        return sections_info

    async def _filter_sections(
        self, sections: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        stopper_section = "Книга жалоб на медицинскую помощь"
        k = 0

        for title, _ in sections:
            if title == stopper_section:
                break
            k += 1

        return sections[:k]

    async def _parse_section_comments(
        self, soup: BeautifulSoup
    ) -> List[Tuple[str, str]]:
        href_tags = soup.find_all("a")
        case_href = [
            tag["href"] for tag in href_tags if tag.get("id", "").startswith("tid-link")
        ]

        # skipping common href for all sections
        case_href = case_href[1:]

        comments = []

        for case_url in case_href:
            comments.append(self._get_case_description(case_url))

        comments = await asyncio.gather(*comments)

        return list(zip(case_href, comments))

    async def _get_case_description(self, url: str) -> str:
        soup = await self._get_page_soup(url)
        text_div_tag = soup.find("div", {"class": "postcolor"})

        if not text_div_tag:
            return ""

        return str(text_div_tag)

    async def _get_section_next_url(self, url: str) -> Union[str, None]:
        if "st=" not in url:
            url += "&prune_day=100&sort_by=Z-A&sort_key=last_post&topicfilter=all&st=30"
        else:
            page_number = int(re.findall("st=(\d+)", url)[0])
            url = re.sub(r"st=\d+", f"st={page_number + 30}", url)

        return url

    async def _parse_section(self, url: str) -> List[Tuple[str, str]]:
        soup = await self._get_page_soup(url)
        page_number_tag = soup.find("span", {"class": "pagelink"})

        if page_number_tag:
            page_number = int(re.findall("(\d+) стран", page_number_tag.text)[0])
        else:
            page_number = 0

        page_urls = [url]

        for _ in range(page_number):
            page_urls.append(await self._get_section_next_url(page_urls[-1]))

        page_urls.pop(0)

        page_soups = [soup] + await asyncio.gather(
            *[self._get_page_soup(url) for url in page_urls]
        )
        pages_comments = await asyncio.gather(
            *[self._parse_section_comments(page_soup) for page_soup in page_soups]
        )
        comments = []

        for page_comments in pages_comments:
            comments.extend(page_comments)

        return comments

    async def parse(self) -> None:
        sections = await self._get_sections_urls()
        sections = await self._filter_sections(sections)

        data = defaultdict(list)

        tasks = []

        for section_name, url in sections:
            tasks.append(self._parse_section(url))

        file_name = self.FILE_NAME

        try:
            for (section_name, _), comment_task in tqdm(
                zip(sections, tasks), total=len(tasks)
            ):
                section_comments = await comment_task

                for url, comment in section_comments:
                    data["name"].append(section_name)
                    data["case"].append(comment)
                    data["url"].append(url)

        except Exception as e:
            print(repr(e))
            file_name = uuid.uuid4().hex

        df = pd.DataFrame(data)
        df = surrogate_remover(df)
        df.to_csv(f"../{file_name}", index=False)


async def run():
    parser = MedUniverParser()
    await parser.parse()


if __name__ == "__main__":
    asyncio.run(run())
