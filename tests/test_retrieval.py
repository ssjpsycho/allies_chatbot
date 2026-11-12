from types import SimpleNamespace

import pytest

from allies_bot.knowledge import DocumentChunk, KnowledgeBase, chunk_document


def source(label, text, heading=""):
    return {
        "source_id": f"bookstack:{label}", "source_label": label,
        "source_url": f"https://wiki.test/{label}", "text": text, "heading": heading,
    }


def knowledge():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.settings = SimpleNamespace(
        qdrant_collection="rules", embedding_model="embedding", chat_model="chat",
    )
    kb.ensure_collection = lambda: None
    kb.openai = SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **_: SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1])],
        )),
    )
    return kb


def test_direct_damage_scan_paginates_and_does_not_drop_semantic_matches():
    kb = knowledge()
    direct = [source(f"Page {i}", f"Effect {i} damages Resolve.") for i in range(60)]
    calls = []

    def scroll(**kwargs):
        calls.append(kwargs)
        if kwargs.get("offset") is None:
            return [SimpleNamespace(payload=s) for s in direct[:30]], "next"
        return [SimpleNamespace(payload=s) for s in direct[30:]], None

    kb.qdrant = SimpleNamespace(
        query_points=lambda **_: SimpleNamespace(points=[
            SimpleNamespace(payload=direct[-1]),
        ]),
        scroll=scroll,
    )
    results = kb.search("all ways to cause Resolve damage")
    assert {s["source_label"] for s in results} == {s["source_label"] for s in direct}
    assert len(calls) == 2
    assert calls[-1]["offset"] == "next"


def test_source_page_expansion_paginates_and_orders_by_chunk_index():
    kb = knowledge()
    first = source("Minion Level: 12", "part zero")
    first["chunk_index"] = 0
    last = source("Minion Level: 12", "part one")
    last["chunk_index"] = 1
    calls = []

    def scroll(**kwargs):
        calls.append(kwargs)
        return ([SimpleNamespace(payload=last)], "next") if len(calls) == 1 else (
            [SimpleNamespace(payload=first)], None,
        )

    kb.qdrant = SimpleNamespace(scroll=scroll)
    assert kb.source_chunks("Minion Level: 12") == [first, last]
    assert calls[1]["offset"] == "next"


def test_order_scope_keeps_explicit_discussion_but_excludes_unrelated_discipline():
    watchman = source("Watchman (Messenger Discipline)", "Watchmen can Excite.")
    assert not KnowledgeBase.order_source_allowed(
        watchman, "all ways a Ministering Spirit can damage Resolve",
    )
    assert KnowledgeBase.order_source_allowed(
        watchman, "Can a Ministering Spirit become a Watchman?",
    )
    assert KnowledgeBase.order_source_allowed(
        watchman, "What can a Messenger do?",
    )
    assert KnowledgeBase.order_source_allowed(
        source("Striking", "A Messenger may..."), "Ministering Spirit damage",
    )


def test_enemy_intent_recognizes_minions_but_not_substrings():
    assert KnowledgeBase.requests_enemy_sources("What are Minion Level 12 attributes?")
    assert KnowledgeBase.requests_enemy_sources("What can enemies do?")
    assert not KnowledgeBase.requests_enemy_sources("How does Divine Revelation work?")


def test_answer_expands_named_page_and_retains_order_eligibility_rules():
    kb = knowledge()
    captured = {}
    chunks = [source("Minion Level: 12", f"stat {i}") for i in range(4)]
    kb.search = lambda query, **kwargs: (
        captured.update(search_kwargs=kwargs) or [chunks[2]]
    )
    kb.source_chunks = lambda label: chunks if label == "Minion Level: 12" else []

    def complete(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Answer"))])

    kb.openai.chat = SimpleNamespace(completions=SimpleNamespace(create=complete))
    answer, sources = kb.answer("What are the attributes of a Minion Level 12?")
    assert answer == "Answer"
    assert sources == chunks
    assert captured["search_kwargs"]["exclude_enemy"] is False
    assert all(f"stat {i}" in str(captured["messages"]) for i in range(4))


def test_order_scoping_applies_outside_character_creation():
    kb = knowledge()
    eligible = source("Judgment Family", "Gavel damages Resolve.")
    other = source("Watchman (Messenger Discipline)", "Excite damages Resolve.")
    rules = source("Ministering Spirits", "Ministering Spirits learn Spiritual Effects.")
    kb.search = lambda *args, **kwargs: [other, eligible]
    kb.source_chunks = lambda label: [rules] if label == "Ministering Spirits" else []
    kb.openai.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: (
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Answer"))])
    )))
    _, sources = kb.answer("All ways a Ministering Spirit can damage Resolve?")
    assert other not in sources
    assert eligible in sources
    assert rules in sources


def test_songs_and_hybrid_orders_are_not_standard_ministering_options():
    question = "All ways a Ministering Spirit can damage Resolve?"
    assert not KnowledgeBase.order_source_allowed(source("Flame Family", "Song"), question)
    assert not KnowledgeBase.order_source_allowed(
        source("Warrior-Ministering Spirit", "Hybrid"), question,
    )
    assert KnowledgeBase.order_source_allowed(
        source("Flame Family", "Song"), "Can a Ministering Spirit learn Flame Family Songs?",
    )
    assert KnowledgeBase.order_source_allowed(
        source("Flame Family", "Song"), "What can a Minstrel-Ministering Spirit learn?",
    )


def test_healing_attribute_question_does_not_trigger_global_damage_scan():
    kb = knowledge()
    kb.qdrant = SimpleNamespace(query_points=lambda **_: SimpleNamespace(points=[]))
    assert kb.search("How does Manna restore Endurance?") == []


def test_named_table_coverage_is_supplied_to_answer_generator():
    kb = knowledge()
    chunks = [source("Minion Level: 12", (
        "Table: Warrior (ML:12); Row: Force; Column 2\t7\n"
        "Table: W-Me (ML:12); Row: Force; Column 2\t6\n"
        "Table: Wolf (ML:12); Row: Force; Column 2\t5"
    ))]
    captured = {}
    kb.search = lambda *args, **kwargs: chunks
    kb.source_chunks = lambda label: chunks

    def complete(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Answer"))])

    kb.openai.chat = SimpleNamespace(completions=SimpleNamespace(create=complete))
    kb.answer("Describe a Minion Level 12.")
    context = str(captured["messages"])
    assert "Named-source table coverage checklist" in context
    assert all(f"- {role} (ML:12)" in context for role in ("Warrior", "W-Me", "Wolf"))


def test_weapon_effects_expand_permission_rules_for_order_questions():
    kb = knowledge()
    weapon = source(
        "Weapons, Armor, & Clothing",
        "Table; Row: Bow; Column: Passion Effect\t7 Resolve damage\n"
        "Table; Row: Bow; Column: Usable By\tW, Me",
    )
    restriction = source(
        "Weapons, Armor, & Clothing",
        "The Ministering Spirit may choose either a Sword or a Staff.",
        "Character Creation Notes",
    )
    captured = {}
    kb.search = lambda *args, **kwargs: [weapon]
    kb.source_chunks = lambda label: (
        [weapon, restriction] if label == "Weapons, Armor, & Clothing" else []
    )

    def complete(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Answer"))])

    kb.openai.chat = SimpleNamespace(completions=SimpleNamespace(create=complete))
    _, sources = kb.answer("How can a Ministering Spirit damage Resolve?")
    assert restriction in sources
    assert "Me is NOT Ministering Spirit" in captured["messages"][0]["content"]


def test_index_persists_per_source_chunk_order():
    kb = knowledge()
    documents = [
        DocumentChunk(f"bookstack:{i}", f"Page {i}", "https://wiki.test", "word " * 400)
        for i in range(2)
    ]
    points = []
    kb.openai.embeddings.create = lambda **kwargs: SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1]) for _ in kwargs["input"]],
    )
    kb.qdrant = SimpleNamespace(upsert=lambda **kwargs: points.extend(kwargs["points"]))
    kb.index(documents)
    for document in documents:
        expected = chunk_document(document.source_label, document.text)
        actual = [p.payload for p in points if p.payload["source_id"] == document.source_id]
        assert [p["text"] for p in actual] == [text for text, _ in expected]
        assert [p["chunk_index"] for p in actual] == [str(i) for i in range(len(expected))]


def test_broad_stat_cards_render_all_numeric_fields_without_model():
    kb = knowledge()
    chunks = [source("Minion Level: 12", (
        "Table: Warrior (ML:12); Row: Force; Column 2\t7\n"
        "Table: Warrior (ML:12); Row: Guard; Column 2\t3\n"
        "Table: W-Me (ML:12); Row: Force; Column 2\t6\n"
        "Table: Wolf (ML:12); Row: Force; Column 2\t5\n"
        "Table: Wolf (ML:12); Row: Ability; Column 2\t(empty)"
    ))]
    kb.search = lambda *args, **kwargs: chunks
    kb.source_chunks = lambda label: chunks
    answer, _ = kb.answer("What are the attributes of a Minion Level 12?")
    assert "all 3 indexed stat cards" in answer
    assert "Warrior (ML:12)\nForce: 7; Guard: 3" in answer
    assert "W-Me (ML:12)\nForce: 6" in answer
    assert "Wolf (ML:12)\nForce: 5" in answer
    assert "Ability:" not in answer
    assert KnowledgeBase.stat_card_answer(
        "Minion Level: 12", chunks, "What are Warrior attributes for Minion Level 12?",
    ) is None


def test_conflicting_stat_card_values_are_reported():
    with pytest.raises(ValueError, match="Conflicting"):
        KnowledgeBase.stat_card_answer("Minion Level: 12", [
            source("Minion Level: 12", (
                "Table: Warrior (ML:12); Row: Force; Column 2\t7\n"
                "Table: Warrior (ML:12); Row: Force; Column 2\t8"
            )),
        ], "Minion Level 12 attributes?")
