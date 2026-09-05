import hashlib
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeVar

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from allies_bot.config import Settings

EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536
EMBEDDING_BATCH_SIZE = 128
QDRANT_UPSERT_BATCH_SIZE = 64

Item = TypeVar("Item")


@dataclass(frozen=True)
class DocumentChunk:
    source_id: str
    source_label: str
    source_url: str | None
    text: str


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = end - overlap
    return chunks


def batched(items: list[Item], batch_size: int) -> Iterator[list[Item]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class KnowledgeBase:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.openai = OpenAI(api_key=settings.openai_api_key)
        self.qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    def ensure_collection(self) -> None:
        if not self.qdrant.collection_exists(self.settings.qdrant_collection):
            self.qdrant.create_collection(
                collection_name=self.settings.qdrant_collection,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    def index(self, documents: list[DocumentChunk]) -> int:
        self.ensure_collection()
        expanded = [
            (document, part)
            for document in documents
            for part in chunk_text(document.text)
        ]
        if not expanded:
            return 0
        vectors = []
        for batch in batched(expanded, EMBEDDING_BATCH_SIZE):
            vectors.extend(
                self.openai.embeddings.create(
                    model=EMBEDDING_MODEL, input=[part for _, part in batch]
                ).data
            )
        points = []
        for (document, part), vector in zip(expanded, vectors, strict=True):
            digest = hashlib.sha256(part.encode("utf-8")).hexdigest()
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document.source_id}:{digest}"))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.embedding,
                    payload={
                        "source_label": document.source_label,
                        "source_url": document.source_url,
                        "text": part,
                    },
                )
            )
        for batch in batched(points, QDRANT_UPSERT_BATCH_SIZE):
            self.qdrant.upsert(collection_name=self.settings.qdrant_collection, points=batch)
        return len(points)

    def search(self, question: str, limit: int = 5) -> list[dict[str, str | None]]:
        self.ensure_collection()
        vector = self.openai.embeddings.create(model=EMBEDDING_MODEL, input=question).data[0].embedding
        result = self.qdrant.query_points(
            collection_name=self.settings.qdrant_collection,
            query=vector,
            limit=limit,
        )
        return [point.payload for point in result.points]

    def answer(self, question: str) -> tuple[str, list[dict[str, str | None]]]:
        sources = self.search(question)
        if not sources:
            return "I do not have indexed source material to answer that yet.", []
        context = "\n\n".join(
            f"Source: {source['source_label']}\n{source['text']}" for source in sources
        )
        completion = self.openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied source excerpts. If they do not answer the "
                        "question, say so plainly. Do not invent rules, lore, or citations."
                    ),
                },
                {"role": "user", "content": f"Question: {question}\n\nSource excerpts:\n{context}"},
            ],
        )
        return completion.choices[0].message.content or "No answer was generated.", sources