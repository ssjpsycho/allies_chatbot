import asyncio
import logging
import time
from weakref import WeakValueDictionary

import discord
from discord import app_commands
from openai import APITimeoutError

from allies_bot.config import Settings
from allies_bot.knowledge import ANSWER_BUDGET_SECONDS, KnowledgeBase
from allies_bot.messages import plain_text_for_discord, split_for_discord

logger = logging.getLogger(__name__)
BACKEND_TIMEOUT_SECONDS = 90
MEMORY_WRITE_TIMEOUT_SECONDS = 15
ANSWER_PROGRESS_SECONDS = 20


class AlliesBot(discord.Client):
    def __init__(self, settings: Settings) -> None:
        super().__init__(intents=discord.Intents.default())
        self.settings = settings
        self.knowledge = KnowledgeBase(settings)
        self.tree = app_commands.CommandTree(self)
        self.conversation_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    async def setup_hook(self) -> None:
        if self.settings.discord_guild_id:
            guild = discord.Object(id=self.settings.discord_guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()

    def channel_is_allowed(self, channel_id: int) -> bool:
        return channel_id in self.settings.channel_ids


settings = Settings()
bot = AlliesBot(settings)


@bot.tree.command(name="ask", description="Ask about the indexed Allies of Majesty sources.")
@app_commands.describe(question="Your question")
async def ask(interaction: discord.Interaction, question: str) -> None:
    if not interaction.channel_id or not bot.channel_is_allowed(interaction.channel_id):
        await interaction.response.send_message("This bot is not enabled in this channel.", ephemeral=True)
        return
    await interaction.response.defer(thinking=True)
    conversation_key = f"{interaction.guild_id or 0}:{interaction.channel_id}:{interaction.user.id}"
    lock = bot.conversation_locks.setdefault(conversation_key, asyncio.Lock())
    async with lock:
        await answer_question(interaction, question, conversation_key)


async def answer_question(
    interaction: discord.Interaction, question: str, conversation_key: str,
) -> None:
    try:
        history = await asyncio.wait_for(
            asyncio.to_thread(bot.knowledge.load_conversation, conversation_key),
            timeout=BACKEND_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.exception("Timed out while loading conversation memory")
        await interaction.followup.send(
            "I could not load your conversation history in time. Please try again shortly."
        )
        return
    except Exception:
        logger.exception("Failed to load conversation memory")
        await interaction.followup.send(
            "I could not load your conversation history. Please try again shortly."
        )
        return

    try:
        started = time.monotonic()
        worker = asyncio.create_task(asyncio.to_thread(bot.knowledge.answer, question, history))
        done, _ = await asyncio.wait({worker}, timeout=ANSWER_PROGRESS_SECONDS)
        if not done:
            await interaction.followup.send(
                "I am still checking the rules and preparing your answer. "
                "You do not need to submit the question again."
            )
        # Keep the conversation lock until the worker ends. wait_for(to_thread(...))
        # only cancels the waiter, leaving a timed-out worker running in the background.
        answer, sources = await worker
        logger.info("Discord answer elapsed_seconds=%.3f budget_seconds=%d",
                    time.monotonic() - started, ANSWER_BUDGET_SECONDS)
    except (TimeoutError, APITimeoutError):
        logger.exception("Timed out while answering Discord question")
        await interaction.followup.send(
            "The knowledge service timed out while preparing your answer. "
            "This exchange was not saved; please try again shortly."
        )
        return
    except Exception:
        logger.exception("Failed to answer Discord question")
        await interaction.followup.send(
            "I could not reach the knowledge service. Please try again shortly."
        )
        return

    source_lines = []
    for source in sources[:12]:
        label = str(source["source_label"])
        url = source["source_url"]
        source_lines.append(f"- {label}: <{url}>" if url else f"- {label}")
    source_text = "\n".join(source_lines)
    content = plain_text_for_discord(
        f"Question: {question}\n\n{answer}\n\nSources\n{source_text}"
    )
    for message in split_for_discord(content):
        await interaction.followup.send(message)

    try:
        await asyncio.wait_for(
            asyncio.to_thread(
                bot.knowledge.save_conversation_turn, conversation_key, question, answer,
            ),
            timeout=MEMORY_WRITE_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.exception("Could not save Discord conversation memory")
        await interaction.followup.send(
            "I could not save this exchange to conversation history. Your next question "
            "may need to repeat the context."
        )


@bot.tree.command(name="sources", description="Show the configured knowledge sources.")
async def sources(interaction: discord.Interaction) -> None:
    if not interaction.channel_id or not bot.channel_is_allowed(interaction.channel_id):
        await interaction.response.send_message("This bot is not enabled in this channel.", ephemeral=True)
        return
    await interaction.response.send_message(
        f"Wiki: {settings.bookstack_base_url}\nEPUB: {settings.epub_path.name}"
    )


if __name__ == "__main__":
    bot.run(settings.discord_token)