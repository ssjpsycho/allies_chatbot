import asyncio

import httpx

from allies_bot.ingest import get_bookstack
from allies_bot.knowledge import QDRANT_UPSERT_BATCH_SIZE, batched, chunk_text
from allies_bot.messages import split_for_discord


def test_chunk_text_preserves_content_and_overlap() -> None:
    text = "alpha " * 500
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert chunks[0].startswith("alpha")
    assert chunks[-1].endswith("alpha")


def test_batched_preserves_order_and_limit() -> None:
    result = list(batched([1, 2, 3, 4, 5], 2))

    assert result == [[1, 2], [3, 4], [5]]


def test_qdrant_batch_size_is_smaller_than_embedding_batch_size() -> None:
    assert QDRANT_UPSERT_BATCH_SIZE < 128


def test_split_for_discord_respects_message_limit() -> None:
    content = ("word " * 600).strip()
    messages = split_for_discord(content)

    assert len(messages) == 2
    assert all(len(message) <= 2000 for message in messages)
    assert " ".join(messages) == content


def test_get_bookstack_retries_rate_limit(monkeypatch) -> None:
    request = httpx.Request("GET", "https://wiki.example.test/api/pages")
    responses = [
        httpx.Response(429, headers={"Retry-After": "0"}, request=request),
        httpx.Response(200, json={"data": []}, request=request),
    ]

    async def get(*_args, **_kwargs) -> httpx.Response:
        return responses.pop(0)

    async def no_sleep(*_args) -> None:
        return None

    client = type("Client", (), {"get": get})()
    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    response = asyncio.run(get_bookstack(client, "/api/pages"))

    assert response.status_code == 200