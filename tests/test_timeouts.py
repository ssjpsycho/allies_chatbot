import time
from types import SimpleNamespace

import pytest

from allies_bot.knowledge import (
    ANSWER_BUDGET_SECONDS,
    ConversationMessage,
    KnowledgeBase,
    answer_deadline,
    request_timeout,
)


def test_request_timeout_respects_remaining_budget_and_expiry():
    token = answer_deadline.set(time.monotonic() + 2)
    try:
        assert 0 < request_timeout("test", 75) <= 2
        answer_deadline.set(time.monotonic() - 1)
        with pytest.raises(TimeoutError, match="before test"):
            request_timeout("test", 75)
    finally:
        answer_deadline.reset(token)


def test_answer_establishes_and_cleans_up_request_deadline_on_error():
    kb = KnowledgeBase.__new__(KnowledgeBase)

    def fail(question, history):
        assert 0 < request_timeout("test", 1000) <= ANSWER_BUDGET_SECONDS
        raise RuntimeError("Failure")

    kb._answer = fail
    with pytest.raises(RuntimeError, match="Failure"):
        kb.answer("Question")
    assert answer_deadline.get() is None


def test_followup_rewrite_is_time_and_output_bounded():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.settings = SimpleNamespace(chat_model="chat")
    captured = {}

    def complete(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content="Standalone question"),
        )])

    kb.openai = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    assert kb.standalone_question("And that?", [
        ConversationMessage("user", "Manna?", "2026-11-12"),
    ]) == "Standalone question"
    assert captured["timeout"] <= 15
    assert captured["max_completion_tokens"] == 512


def test_incomplete_rewrite_is_not_used_for_retrieval():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.settings = SimpleNamespace(chat_model="chat")
    kb.openai = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="length", message=SimpleNamespace(content="Incomplete"),
        )]),
    )))
    with pytest.raises(ValueError, match="output limit"):
        kb.standalone_question("And that?", [
            ConversationMessage("user", "Manna?", "2026-11-12"),
        ])


def test_provider_defaults_do_not_retry_past_discord_deadline(monkeypatch):
    from allies_bot import knowledge

    captured = {}
    monkeypatch.setattr(knowledge, "OpenAI", lambda **kwargs: captured.update(openai=kwargs))
    monkeypatch.setattr(knowledge, "QdrantClient", lambda **kwargs: captured.update(qdrant=kwargs))
    KnowledgeBase(SimpleNamespace(
        openai_api_key="test", qdrant_url="https://wiki.test", qdrant_api_key="test",
    ))
    assert captured["openai"]["max_retries"] == 0
    assert captured["openai"]["timeout"] == 60
    assert captured["qdrant"]["timeout"] == 10
