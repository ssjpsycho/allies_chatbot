import hashlib
import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeVar

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchText,
    MatchValue,
    PointStruct,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)

from allies_bot.config import Settings

EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536
EMBEDDING_BATCH_SIZE = 128
QDRANT_UPSERT_BATCH_SIZE = 64
CONVERSATION_MEMORY_LIMIT = 8
CONVERSATION_VECTOR_SIZE = 1
SEARCH_RESULT_LIMIT = 5
KEYWORD_RESULT_LIMIT = 8
SEARCH_STOPWORDS = frozenset([
    "about", "after", "asked", "asks", "before", "being", "does", "doing", "from",
    "have", "how", "into", "lower", "lowers", "made", "make", "more", "most", "that",
    "than", "them", "there", "these", "they", "things", "what", "when", "which", "with",
    "would", "your",
])

Item = TypeVar("Item")


@dataclass(frozen=True)
class DocumentChunk:
    source_id: str
    source_label: str
    source_url: str | None
    text: str


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    content: str
    created_at: str


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
        self.qdrant.create_payload_index(
            collection_name=self.settings.qdrant_collection,
            field_name="text",
            field_schema=TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=TokenizerType.WORD,
                lowercase=True,
                phrase_matching=True,
            ),
        )

    @property
    def memory_collection(self) -> str:
        return f"{self.settings.qdrant_collection}_conversation_memory"

    def ensure_memory_collection(self) -> None:
        if not self.qdrant.collection_exists(self.memory_collection):
            self.qdrant.create_collection(
                collection_name=self.memory_collection,
                vectors_config=VectorParams(size=CONVERSATION_VECTOR_SIZE, distance=Distance.DOT),
            )

    def load_conversation(self, conversation_key: str) -> list[ConversationMessage]:
        self.ensure_memory_collection()
        records, _ = self.qdrant.scroll(
            collection_name=self.memory_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="conversation_key", match=MatchValue(value=conversation_key))]
            ),
            limit=CONVERSATION_MEMORY_LIMIT * 2,
            with_payload=True,
            with_vectors=False,
        )
        messages = [
            ConversationMessage(
                role=str(record.payload["role"]),
                content=str(record.payload["content"]),
                created_at=str(record.payload["created_at"]),
            )
            for record in records
        ]
        return sorted(messages, key=lambda message: message.created_at)[-CONVERSATION_MEMORY_LIMIT * 2 :]

    def save_conversation_message(
        self, conversation_key: str, message: ConversationMessage
    ) -> None:
        self.ensure_memory_collection()
        point_id = str(uuid.uuid4())
        self.qdrant.upsert(
            collection_name=self.memory_collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=[1.0],
                    payload={
                        "conversation_key": conversation_key,
                        "role": message.role,
                        "content": message.content,
                        "created_at": message.created_at,
                    },
                )
            ],
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

    @staticmethod
    def search_terms(question: str) -> list[str]:
        terms = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", question.lower())
        return [term for term in dict.fromkeys(terms) if term not in SEARCH_STOPWORDS]

    def search(self, question: str, limit: int = SEARCH_RESULT_LIMIT) -> list[dict[str, str | None]]:
        self.ensure_collection()
        vector = self.openai.embeddings.create(model=EMBEDDING_MODEL, input=question).data[0].embedding
        result = self.qdrant.query_points(
            collection_name=self.settings.qdrant_collection,
            query=vector,
            limit=limit,
        )
        semantic = [point.payload for point in result.points]
        seen = {self._source_key(source) for source in semantic}
        keyword_scores: dict[str, tuple[int, dict[str, str | None]]] = {}
        for term in self.search_terms(question):
            matches, _ = self.qdrant.scroll(
                collection_name=self.settings.qdrant_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="text", match=MatchText(text=term))]
                ),
                limit=KEYWORD_RESULT_LIMIT,
                with_payload=True,
                with_vectors=False,
            )
            for point in matches:
                source = point.payload
                key = self._source_key(source)
                score = sum(
                    str(source.get(field) or "").lower().count(term)
                    for field in ("source_label", "text")
                )
                keyword_scores[key] = (keyword_scores.get(key, (0, source))[0] + score, source)
        keyword_matches = [
            source
            for _, source in sorted(
                keyword_scores.values(), key=lambda item: item[0], reverse=True
            )
            if self._source_key(source) not in seen
        ]
        for source in keyword_matches:
            seen.add(self._source_key(source))
        return (keyword_matches + semantic)[: limit + KEYWORD_RESULT_LIMIT]

    @staticmethod
    def _source_key(source: dict[str, str | None]) -> str:
        return f"{source.get('source_label')}:{source.get('text')}"

    def answer(
        self, question: str, history: list[ConversationMessage] | None = None
    ) -> tuple[str, list[dict[str, str | None]]]:
        history = history or []
        retrieval_query = "\n".join([message.content for message in history[-4:]] + [question])
        sources = self.search(retrieval_query)
        if not sources:
            return "I do not have indexed source material to answer that yet.", []
        context = "\n\n".join(
            f"Source: {source['source_label']}\n{source['text']}" for source in sources
        )
        conversation = "\n".join(
            f"{message.role.title()}: {message.content}" for message in history
        )
        completion = self.openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied source excerpts and use conversation history "
                        "to resolve references such as 'that' or 'it'. List direct rules separately "
                        "from clearly labeled inferences. Do not turn implications into rules, and "
                        "do not claim a complete list unless the excerpts establish completeness. "
                        "If the sources do not answer the question, say so plainly. Do not invent "
                        "rules, lore, or citations."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation history:\n{conversation or '(none)'}\n\n"
                        f"Current question: {question}\n\nSource excerpts:\n{context}"
                    ),
                },
            ],
        )
        return completion.choices[0].message.content or "No answer was generated.", sources