import asyncio
import logging
from datetime import UTC, datetime

import discord
from discord import app_commands

from allies_bot.config import Settings
from allies_bot.knowledge import ConversationMessage, KnowledgeBase
from allies_bot.messages import split_for_discord

logger = logging.getLogger(__name__)
BACKEND_TIMEOUT_SECONDS = 90
MEMORY_WRITE_TIMEOUT_SECONDS = 15


class AlliesBot(discord.Client):
    def __init__(self, settings: Settings) -> None:
        super().__init__(intents=discord.Intents.default())
        self.settings = settings
        self.knowledge = KnowledgeBase(settings)
        self.tree = app_commands.CommandTree(self)

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
    history = []
    try:
        history = await asyncio.wait_for(
            asyncio.to_thread(bot.knowledge.load_conversation, conversation_key),
            timeout=BACKEND_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.exception("Timed out while loading conversation memory; continuing without history")
    except Exception:
        logger.exception("Failed to load conversation memory; continuing without history")

    try:
        answer, sources = await asyncio.wait_for(
            asyncio.to_thread(bot.knowledge.answer, question, history),
            timeout=BACKEND_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.exception("Timed out while answering Discord question")
        await interaction.followup.send(
            "I am taking too long to reach the knowledge service. Please try again shortly."
        )
        return
    except Exception:
        logger.exception("Failed to answer Discord question")
        await interaction.followup.send(
            "I could not reach the knowledge service. Please try again shortly."
        )
        return

    source_lines = []
    for source in sources[:3]:
        label = str(source["source_label"])
        url = source["source_url"]
        source_lines.append(f"- {label}: <{url}>" if url else f"- {label}")
    source_text = "\n".join(source_lines)
    content = f"**Question:** {question}\n\n{answer}\n\n**Sources**\n{source_text}"
    for message in split_for_discord(content):
        await interaction.followup.send(message)

    now = datetime.now(UTC).isoformat()
    try:
        await asyncio.wait_for(
            asyncio.gather(
                asyncio.to_thread(
                    bot.knowledge.save_conversation_message,
                    conversation_key,
                    ConversationMessage(role="user", content=question, created_at=now),
                ),
                asyncio.to_thread(
                    bot.knowledge.save_conversation_message,
                    conversation_key,
                    ConversationMessage(role="assistant", content=answer, created_at=now),
                ),
            ),
            timeout=MEMORY_WRITE_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.exception("Could not save Discord conversation memory")


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