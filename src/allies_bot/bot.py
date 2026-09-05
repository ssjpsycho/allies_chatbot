import asyncio

import discord
from discord import app_commands

from allies_bot.config import Settings
from allies_bot.knowledge import KnowledgeBase
from allies_bot.messages import split_for_discord


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
    answer, sources = await asyncio.to_thread(bot.knowledge.answer, question)
    source_lines = []
    for source in sources[:3]:
        label = str(source["source_label"])
        url = source["source_url"]
        source_lines.append(f"- {label}: <{url}>" if url else f"- {label}")
    source_text = "\n".join(source_lines)
    content = f"{answer}\n\n**Sources**\n{source_text}"
    for message in split_for_discord(content):
        await interaction.followup.send(message)


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