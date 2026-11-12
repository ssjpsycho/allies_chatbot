import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from allies_bot import config, knowledge


@pytest.fixture
def bot_module(monkeypatch):
    settings = SimpleNamespace(channel_ids={20}, discord_guild_id=None)
    monkeypatch.setattr(config, "Settings", lambda: settings)
    backend = SimpleNamespace()
    monkeypatch.setattr(knowledge, "KnowledgeBase", lambda settings: backend)
    path = Path(__file__).parents[1] / "src/allies_bot/bot.py"
    spec = importlib.util.spec_from_file_location("bot_conversation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def interaction(user=30, channel=20):
    return SimpleNamespace(
        guild_id=10, channel_id=channel, user=SimpleNamespace(id=user),
        response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )


def test_discord_followup_loads_previous_turn_with_user_channel_scope(bot_module):
    module = bot_module
    stored = {}
    seen = []
    module.bot.knowledge.load_conversation = lambda key: list(stored.get(key, []))

    def answer(question, history):
        seen.append((question, list(history)))
        return "Manna restores Endurance.", []

    def save(key, question, answer):
        stored.setdefault(key, []).extend([
            knowledge.ConversationMessage("user", question, "2026-11-12"),
            knowledge.ConversationMessage("assistant", answer, "2026-11-12"),
        ])

    module.bot.knowledge.answer = answer
    module.bot.knowledge.save_conversation_turn = save

    async def run():
        await module.ask.callback(interaction(), "What is Manna?")
        await module.ask.callback(interaction(), "How much does it restore?")
        await module.ask.callback(interaction(user=31), "What is Manna?")

    asyncio.run(run())
    assert seen[0][1] == []
    assert [m.content for m in seen[1][1]] == ["What is Manna?", "Manna restores Endurance."]
    assert seen[2][1] == []
    assert set(stored) == {"10:20:30", "10:20:31"}


def test_discord_reports_load_failure_instead_of_answering_without_context(bot_module):
    module = bot_module

    def fail(key):
        raise RuntimeError("Memory unavailable")

    module.bot.knowledge.load_conversation = fail
    request = interaction()
    asyncio.run(module.ask.callback(request, "What about that?"))
    assert "could not load your conversation history" in request.followup.send.call_args.args[0]


def test_discord_warns_when_exchange_could_not_be_saved(bot_module):
    module = bot_module
    module.bot.knowledge.load_conversation = lambda key: []
    module.bot.knowledge.answer = lambda *args: ("Answer", [])

    def fail(*args):
        raise RuntimeError("Write failed")

    module.bot.knowledge.save_conversation_turn = fail
    request = interaction()
    asyncio.run(module.ask.callback(request, "Question"))
    assert request.followup.send.call_count == 2
    assert "could not save this exchange" in request.followup.send.call_args.args[0]


def test_simultaneous_questions_from_same_user_are_serialized(bot_module, monkeypatch):
    module = bot_module
    active = 0
    peak = 0
    completed = []

    async def answer(request, question, key):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        completed.append(question)
        active -= 1

    monkeypatch.setattr(module, "answer_question", answer)

    async def run():
        await asyncio.gather(
            module.ask.callback(interaction(), "First"),
            module.ask.callback(interaction(), "Second"),
        )

    asyncio.run(run())
    assert peak == 1
    assert completed == ["First", "Second"]


def test_slow_answer_shows_progress_then_delivers_and_saves(bot_module, monkeypatch):
    import time

    module = bot_module
    monkeypatch.setattr(module, "ANSWER_PROGRESS_SECONDS", 0.001)
    module.bot.knowledge.load_conversation = lambda key: []
    saved = []

    def answer(*args):
        time.sleep(0.02)
        return "Completed answer", []

    module.bot.knowledge.answer = answer
    module.bot.knowledge.save_conversation_turn = lambda *args: saved.append(args)
    request = interaction()
    asyncio.run(module.ask.callback(request, "Question"))
    messages = [call.args[0] for call in request.followup.send.call_args_list]
    assert "still checking" in messages[0]
    assert "Completed answer" in messages[1]
    assert saved == [("10:20:30", "Question", "Completed answer")]


def test_timed_out_answer_reports_failure_without_saving(bot_module):
    module = bot_module
    module.bot.knowledge.load_conversation = lambda key: []
    saved = []
    module.bot.knowledge.save_conversation_turn = lambda *args: saved.append(args)

    def answer(*args):
        raise TimeoutError("Answer deadline exceeded")

    module.bot.knowledge.answer = answer
    request = interaction()
    asyncio.run(module.ask.callback(request, "Question"))
    assert "timed out" in request.followup.send.call_args.args[0]
    assert not saved


def test_resolution_error_is_not_reported_as_connection_failure(bot_module):
    module = bot_module
    module.bot.knowledge.load_conversation = lambda key: []

    def answer(*args):
        raise knowledge.FollowupResolutionError("Follow-up resolution exceeded its output limit.")

    module.bot.knowledge.answer = answer
    request = interaction()
    asyncio.run(module.ask.callback(request, "What about it?"))
    message = request.followup.send.call_args.args[0]
    assert "specific effect or topic name" in message
    assert "could not reach" not in message
