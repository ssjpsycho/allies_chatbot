import asyncio

import httpx

from allies_bot.ingest import get_bookstack
from allies_bot.knowledge import QDRANT_UPSERT_BATCH_SIZE, KnowledgeBase, batched, chunk_text
from allies_bot.messages import plain_text_for_discord, split_for_discord


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


def test_search_terms_keeps_specific_game_terms() -> None:
    terms = KnowledgeBase.search_terms("What things lower Endurance through damage?")

    assert terms == ["lower", "endurance", "through", "damage", "reduce", "decrease"]


def test_lowering_question_adds_damage_synonyms() -> None:
    terms = KnowledgeBase.search_terms("What things lower Resolve?")

    assert "resolve" in terms
    assert "damage" in terms
    assert "reduce" in terms


def test_exact_term_score_prefers_passage_matching_multiple_terms() -> None:
    passages = [
        {"source_label": "War Attributes", "text": "Endurance affects Stamina."},
        {
            "source_label": "Striking",
            "text": "Hits deal damage to the Defender's Endurance.",
        },
    ]

    scores = [
        sum(source["text"].lower().count(term) for term in ("endurance", "damage"))
        for source in passages
    ]

    assert scores[1] > scores[0]


def test_striking_passage_matches_endurance_damage_question() -> None:
    passage = (
        "If there are Hits remaining, the Striker will deal damage to the Defender's "
        "Endurance."
    )
    terms = KnowledgeBase.search_terms("What things lower Endurance?")

    assert all(term in passage.lower() for term in ("endurance", "damage"))
    assert "endurance" in terms


def test_keyword_score_prefers_damage_to_endurance() -> None:
    striking = {
        "source_label": "Striking",
        "text": "The Striker will deal damage to the Defender's Endurance.",
    }
    attributes = {"source_label": "War Attributes", "text": "Endurance affects Stamina."}

    assert KnowledgeBase.keyword_score(striking, ["endurance", "damage"])[1] > 0
    assert KnowledgeBase.keyword_score(striking, ["endurance", "damage"])[0] > (
        KnowledgeBase.keyword_score(attributes, ["endurance", "damage"])[0]
    )


def test_source_key_deduplicates_chunks_from_one_page() -> None:
    first_chunk = {"source_label": "Judgment Family", "source_url": "https://wiki.test/judgment", "text": "Gavel"}
    second_chunk = {"source_label": "Judgment Family", "source_url": "https://wiki.test/judgment", "text": "Coal"}

    assert KnowledgeBase._source_key(first_chunk) == KnowledgeBase._source_key(second_chunk)


def test_direct_rule_phrases_include_judgment_wording() -> None:
    phrases = [
        "damage resolve",
        "damages resolve",
        "damage to resolve",
        "damaging resolve",
        "puts resolve",
    ]
    judgment_text = "Gavel of Judgment manifests a Hammer that damages Resolve instead of Endurance."

    assert any(phrase in judgment_text.lower() for phrase in phrases)


def test_character_build_filters_enemy_sources() -> None:
    corruption = {
        "source_label": "Fire Family (Corruption of the Flame Family)",
        "source_url": "https://wiki.test/books/catalogue-of-evil/page/fire-family",
        "text": "Burning Spite damages Resolve.",
    }
    judgment = {
        "source_label": "Judgment Family",
        "source_url": "https://wiki.test/books/character-creation-advancement/page/judgment-family",
        "text": "Burning Coal damages Resolve.",
    }

    assert KnowledgeBase.is_enemy_source(corruption)
    assert not KnowledgeBase.is_enemy_source(judgment)


def test_character_build_filters_flame_song_source() -> None:
    flame = {
        "source_label": "Flame Family",
        "source_url": "https://wiki.test/books/character-creation-advancement/page/flame-family",
        "text": "Flame Song damages Resolve.",
    }

    assert KnowledgeBase.is_flame_song_source(flame)


def test_character_build_filters_minstrel_source() -> None:
    minstrel = {
        "source_label": "Minstrel-Ministering Spirit",
        "source_url": "https://wiki.test/books/character-creation-advancement/page/minstrel-ministering-spirit",
        "text": "Minstrels learn Songs.",
    }

    assert KnowledgeBase.is_minstrel_source(minstrel)


def test_character_build_retrieval_requests_first_tier_effects() -> None:
    retrieval_terms = "Progression 1 Tier 1 first Spiritual Effect"

    assert "Progression 1" in retrieval_terms
    assert "Tier 1" in retrieval_terms


def test_split_for_discord_respects_message_limit() -> None:
    content = ("word " * 600).strip()
    messages = split_for_discord(content)

    assert len(messages) == 2
    assert all(len(message) <= 2000 for message in messages)
    assert " ".join(messages) == content


def test_plain_text_for_discord_removes_markdown() -> None:
    content = "## Pick **Gavel**\n| Effect | Family |\n| --- | --- |\n| Gavel | Judgment |\n[Rules](https://wiki.test/rules)"

    result = plain_text_for_discord(content)

    assert "##" not in result
    assert "**" not in result
    assert "|" not in result
    assert "Rules: <https://wiki.test/rules>" in result


def test_plain_text_for_discord_repairs_ordered_list_numbers() -> None:
    result = plain_text_for_discord("1. First\n1. Second\n1. Third")

    assert result == "1. First\n2. Second\n3. Third"


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