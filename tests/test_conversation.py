from types import SimpleNamespace

import pytest

from allies_bot.knowledge import ConversationMessage, KnowledgeBase


def memory():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.settings = SimpleNamespace(
        qdrant_collection="rules", chat_model="chat", followup_model="rewrite",
        embedding_model="embedding",
    )
    kb.ensure_memory_collection = lambda: None
    return kb


def test_loads_latest_messages_across_all_pages_not_random_first_page():
    kb = memory()
    records = [
        SimpleNamespace(payload={
            "role": role, "content": f"{turn}-{role}",
            "created_at": f"2026-11-12T10:{turn:02d}:00+00:00",
        })
        for turn in range(12) for role in ("assistant", "user")
    ]
    calls = []

    def scroll(**kwargs):
        calls.append(kwargs)
        if kwargs.get("offset") is None:
            return records[:16], "next"
        return records[16:], None

    kb.qdrant = SimpleNamespace(scroll=scroll)
    history = kb.load_conversation("server:channel:user")
    assert len(history) == 16
    assert [m.content for m in history] == [
        f"{turn}-{role}" for turn in range(4, 12) for role in ("user", "assistant")
    ]
    assert len(calls) == 2
    assert calls[0]["scroll_filter"].must[0].match.value == "server:channel:user"


def test_save_turn_is_one_acknowledged_batch_in_user_assistant_order():
    kb = memory()
    captured = []
    kb.qdrant = SimpleNamespace(upsert=lambda **kwargs: captured.append(kwargs))
    kb.save_conversation_turn("server:channel:user", "What is Manna?", "A healing effect.")
    assert len(captured) == 1
    assert captured[0]["wait"] is True
    points = captured[0]["points"]
    assert [p.payload["role"] for p in points] == ["user", "assistant"]
    assert all(p.payload["conversation_key"] == "server:channel:user" for p in points)
    assert points[0].payload["created_at"] < points[1].payload["created_at"]


def test_followup_resolves_reference_before_search_and_scope_detection():
    kb = memory()
    history = [
        ConversationMessage("user", "What can a Ministering Spirit use?", "2026-11-12"),
        ConversationMessage("assistant", "Gavel of Judgment.", "2026-11-12"),
    ]
    resolved = "How much Allotment does Gavel of Judgment cost for a Ministering Spirit?"
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content=resolved if len(calls) == 1 else "It costs 3.",
        ))])

    kb.openai = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    searches = []
    kb.search = lambda query, **kwargs: (
        searches.append(query) or [{
            "source_label": "Judgment Family", "text": "Gavel of Judgment costs 3.",
            "source_url": "https://wiki.test",
        }]
    )
    kb.source_chunks = lambda label: []
    answer, _ = kb.answer("How much does that cost?", history)
    assert answer == "It costs 3."
    assert searches == [resolved]
    assert any(m["role"] == "assistant" and m["content"] == "Gavel of Judgment."
               for m in calls[-1]["messages"])


def test_new_topic_does_not_inherit_stale_minion_or_order_intent():
    kb = memory()
    kb.openai = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
            message=SimpleNamespace(content="What does Holy Manna cost?"),
        )]),
    )))
    history = [ConversationMessage("user", "Minion Level 12 stats?", "2026-11-12")]
    assert kb.standalone_question("What does Holy Manna cost?", history) == (
        "What does Holy Manna cost?"
    )


def test_empty_followup_rewrite_fails_explicitly():
    kb = memory()
    kb.openai = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
            message=SimpleNamespace(content=None),
        )]),
    )))
    with pytest.raises(ValueError, match="resolve"):
        kb.standalone_question("What about it?", [
            ConversationMessage("user", "Manna?", "2026-11-12"),
        ])
