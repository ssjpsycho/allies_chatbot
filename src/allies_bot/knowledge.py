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

VECTOR_SIZE = 1536
EMBEDDING_BATCH_SIZE = 128
QDRANT_UPSERT_BATCH_SIZE = 64
CONVERSATION_MEMORY_LIMIT = 8
CONVERSATION_VECTOR_SIZE = 1
SEARCH_RESULT_LIMIT = 5
KEYWORD_RESULT_LIMIT = 32
DIRECT_RULE_RESULT_LIMIT = 64
SEARCH_STOPWORDS = frozenset([
    "about", "after", "asked", "asks", "before", "being", "does", "doing", "from",
    "have", "how", "into", "made", "make", "more", "most", "that",
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
        self.collection_ready = False

    def ensure_collection(self) -> None:
        if self.collection_ready:
            return
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
        self.collection_ready = True

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
                    model=self.settings.embedding_model, input=[part for _, part in batch]
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
        terms = [term for term in dict.fromkeys(terms) if term not in SEARCH_STOPWORDS]
        lowering_terms = {"lower", "lowers", "lowering", "reduce", "reduces", "reduced"}
        if lowering_terms.intersection(terms):
            terms.extend(term for term in ("damage", "reduce", "decrease") if term not in terms)
        return terms

    @staticmethod
    def keyword_score(source: dict[str, str | None], terms: list[str]) -> tuple[int, int, int]:
        text = str(source.get("text") or "").lower()
        distinct_terms = sum(term in text for term in terms)
        occurrences = sum(text.count(term) for term in terms)
        proximity = 0
        for first_term in terms:
            for second_term in terms:
                if first_term == second_term:
                    continue
                if re.search(rf"{re.escape(first_term)}.{{0,100}}{re.escape(second_term)}", text):
                    proximity += 1
        return distinct_terms, proximity, occurrences

    def search(self, question: str, limit: int = SEARCH_RESULT_LIMIT) -> list[dict[str, str | None]]:
        self.ensure_collection()
        vector = self.openai.embeddings.create(
            model=self.settings.embedding_model, input=question
        ).data[0].embedding
        result = self.qdrant.query_points(
            collection_name=self.settings.qdrant_collection,
            query=vector,
            limit=limit,
        )
        semantic: list[dict[str, str | None]] = []
        seen: set[str] = set()
        for point in result.points:
            source = point.payload
            key = self._source_key(source)
            if key not in seen:
                semantic.append(source)
                seen.add(key)
        terms = self.search_terms(question)
        keyword_result = self.qdrant.query_points(
            collection_name=self.settings.qdrant_collection,
            query=vector,
            query_filter=Filter(
                should=[FieldCondition(key="text", match=MatchText(text=term)) for term in terms]
            ),
            limit=KEYWORD_RESULT_LIMIT,
        )
        keyword_sources: dict[str, dict[str, str | None]] = {}
        for point in keyword_result.points:
            source = point.payload
            key = self._source_key(source)
            existing = keyword_sources.get(key)
            if existing is None or self.keyword_score(source, terms) > self.keyword_score(
                existing, terms
            ):
                keyword_sources[key] = source
        attribute_terms = [term for term in terms if term in {"endurance", "resolve", "passion"}]
        for attribute in attribute_terms:
            phrases = (
                f"damage {attribute}",
                f"damages {attribute}",
                f"damage to {attribute}",
                f"damaging {attribute}",
                f"lower {attribute}",
                f"lowers {attribute}",
                f"puts {attribute}",
            )
            direct_result = self.qdrant.query_points(
                collection_name=self.settings.qdrant_collection,
                query=vector,
                query_filter=Filter(
                    should=[
                        FieldCondition(key="text", match=MatchText(text=phrase))
                        for phrase in phrases
                    ]
                ),
                limit=DIRECT_RULE_RESULT_LIMIT,
            )
            for point in direct_result.points:
                source = point.payload
                key = self._source_key(source)
                existing = keyword_sources.get(key)
                if existing is None or self.keyword_score(source, terms) > self.keyword_score(
                    existing, terms
                ):
                    keyword_sources[key] = source
        keyword_matches = [
            source
            for _, source in sorted(
                keyword_sources.items(),
                key=lambda item: self.keyword_score(item[1], terms),
                reverse=True,
            )
            if self._source_key(source) not in seen
        ]
        for source in keyword_matches:
            seen.add(self._source_key(source))
        return (keyword_matches + semantic)[: limit + KEYWORD_RESULT_LIMIT]

    @staticmethod
    def _source_key(source: dict[str, str | None]) -> str:
        return str(source.get("source_url") or source.get("source_label") or source.get("text"))

    @staticmethod
    def is_character_creation_question(question: str) -> bool:
        question = question.lower()
        return any(
            phrase in question
            for phrase in ("brand new", "character creation", "pick", "select", "choose", "learn")
        )

    @staticmethod
    def is_enemy_source(source: dict[str, str | None]) -> bool:
        url = str(source.get("source_url") or "").lower()
        label = str(source.get("source_label") or "").lower()
        return "catalogue-of-evil" in url or "corruption" in label

    @staticmethod
    def is_flame_song_source(source: dict[str, str | None]) -> bool:
        label = str(source.get("source_label") or "").lower()
        url = str(source.get("source_url") or "").lower()
        return label == "flame family" or url.endswith("/flame-family")

    @staticmethod
    def is_minstrel_source(source: dict[str, str | None]) -> bool:
        label = str(source.get("source_label") or "").lower()
        url = str(source.get("source_url") or "").lower()
        return "minstrel" in label or "minstrel" in url

    def answer(
        self, question: str, history: list[ConversationMessage] | None = None
    ) -> tuple[str, list[dict[str, str | None]]]:
        history = history or []
        retrieval_query = "\n".join([message.content for message in history[-4:]] + [question])
        character_creation = self.is_character_creation_question(question)
        explicit_enemy_request = any(
            term in question.lower() for term in ("unholy", "corrupt", "corruption", "enemy", "evil")
        )
        if character_creation:
            retrieval_query += (
                "\ncharacter creation select 3 Spiritual Effects first effects different Families "
                "Order eligibility Spiritual Effect versus Song learn"
            )
        sources = self.search(retrieval_query)
        if character_creation and not explicit_enemy_request:
            sources = [
                source
                for source in sources
                if not self.is_enemy_source(source)
                and not self.is_flame_song_source(source)
                and not self.is_minstrel_source(source)
            ]
        if not sources:
            return "I do not have indexed source material to answer that yet.", []
        context = "\n\n".join(
            f"Source: {source['source_label']}\n{source['text']}" for source in sources
        )
        conversation = "\n".join(
            f"{message.role.title()}: {message.content}" for message in history
        )
        completion = self.openai.chat.completions.create(
            model=self.settings.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied source excerpts and use conversation history "
                        "to resolve references such as 'that' or 'it'. For questions asking what "
                        "lowers or damages an attribute, list only explicit damage, reduction, or "
                        "status rules for that attribute. Do not treat a resistance statistic, a "
                        "stat used to resist an effect, or a related stat as something that lowers "
                        "the attribute. Inspect every supplied source excerpt and include each "
                        "distinct source page that contains an explicit matching rule; do not omit "
                        "a page merely because another page describes a similar effect. Label any "
                        "remaining inference separately, and do not claim a complete list unless "
                        "the excerpts establish completeness. For character-building questions, "
                        "separate Spiritual Effects from Songs and verify that the character's "
                        "Order or discipline can learn each recommendation. Do not recommend a "
                        "Song to a non-Minstrel, and do not treat a Minstrel-only option as a "
                        "general Ministering Spirit option. If eligibility is not explicit in the "
                        "excerpts, say that it is unverified instead of assuming it. The rules state "
                        "that a new Ministering Spirit selects 3 Spiritual Effects, which are the "
                        "first Spiritual Effect in 3 different Families; do not say they select only "
                        "one. Recommend three eligible Effects when the supplied rules support them, "
                        "and identify their Families. Do not use Markdown tables because Discord does "
                        "not render them reliably; use plain headings and bullet lists instead. "
                        "Treat the source's classification as authoritative: if a Family entry is a "
                        "Song, do not recommend it as a Spiritual Effect. Flame Family entries are "
                        "Songs and are not valid Spiritual Effect picks for a standard Ministering "
                        "Spirit; do not relabel them as Effects. "
                        "When the user asks what to pick, list only eligible picks and their "
                        "Families. Do not list what not to pick or explain excluded alternatives "
                        "unless the user explicitly asks for restrictions or comparisons. Do not "
                        "mention Songs, Minstrels, or excluded options in the final answer unless "
                        "the user explicitly asks about them. "
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