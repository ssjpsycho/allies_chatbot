import argparse
import asyncio
from html.parser import HTMLParser
from typing import Any

import httpx
from ebooklib import ITEM_DOCUMENT, epub

from allies_bot.config import Settings
from allies_bot.knowledge import DocumentChunk, KnowledgeBase

REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 6


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return " ".join(self.parts)


def html_to_text(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    return parser.text()


async def get_bookstack(client: httpx.AsyncClient, path: str, **kwargs: Any) -> httpx.Response:
    for attempt in range(MAX_RETRIES):
        response = await client.get(path, **kwargs)
        if response.status_code not in {429, 502, 503, 504}:
            response.raise_for_status()
            await asyncio.sleep(REQUEST_DELAY_SECONDS)
            return response
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
        await asyncio.sleep(delay)
    response.raise_for_status()
    raise AssertionError("Unreachable")


async def fetch_wiki(settings: Settings) -> list[DocumentChunk]:
    if not settings.bookstack_token:
        raise ValueError("BOOKSTACK_TOKEN is required for API wiki ingestion.")
    headers = {"Authorization": f"Token {settings.bookstack_token}"}
    documents: list[DocumentChunk] = []
    book_slugs: dict[int, str] = {}
    async with httpx.AsyncClient(base_url=settings.bookstack_base_url, headers=headers, timeout=30) as client:
        offset = 0
        while True:
            response = await get_bookstack(client, "/api/pages", params={"count": 100, "offset": offset})
            data = response.json()
            pages = data["data"]
            for summary in pages:
                page = await get_bookstack(client, f"/api/pages/{summary['id']}")
                content = page.json()
                book_id = content["book_id"]
                if book_id not in book_slugs:
                    book = await get_bookstack(client, f"/api/books/{book_id}")
                    book_slugs[book_id] = book.json()["slug"]
                documents.append(
                    DocumentChunk(
                        source_id=f"bookstack:{content['id']}",
                        source_label=content["name"],
                        source_url=(
                            f"{settings.bookstack_base_url}/books/{book_slugs[book_id]}"
                            f"/page/{content['slug']}"
                        ),
                        text=html_to_text(content.get("html", "")),
                    )
                )
            offset += len(pages)
            if offset >= data["total"] or not pages:
                break
    return documents


def fetch_epub(settings: Settings) -> list[DocumentChunk]:
    if not settings.epub_path.exists():
        raise FileNotFoundError(f"EPUB not found: {settings.epub_path}")
    book = epub.read_epub(str(settings.epub_path))
    documents = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        text = html_to_text(item.get_content().decode("utf-8", errors="replace"))
        if text.strip():
            documents.append(
                DocumentChunk(
                    source_id=f"epub:{item.get_name()}",
                    source_label=f"EPUB: {item.get_name()}",
                    source_url=None,
                    text=text,
                )
            )
    return documents


async def main() -> None:
    parser = argparse.ArgumentParser(description="Index Allies of Majesty sources.")
    parser.add_argument("--wiki", action="store_true", help="Index BookStack pages through its API.")
    parser.add_argument("--epub", action="store_true", help="Index the configured EPUB file.")
    args = parser.parse_args()
    if not args.wiki and not args.epub:
        parser.error("Choose --wiki, --epub, or both.")

    settings = Settings()
    documents: list[DocumentChunk] = []
    if args.wiki:
        documents.extend(await fetch_wiki(settings))
    if args.epub:
        documents.extend(fetch_epub(settings))
    indexed = KnowledgeBase(settings).index(documents)
    print(f"Indexed {indexed} chunks from {len(documents)} source documents.")


if __name__ == "__main__":
    asyncio.run(main())