from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    discord_token: str
    discord_guild_id: int | None = None
    allowed_channel_ids: str
    openai_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "allies_of_majesty"
    bookstack_base_url: str = "https://wiki.alliesofmajesty.com"
    bookstack_token: str | None = None
    epub_path: Path = Field(default=Path("../Allies of Majesty Chronicles vol 1.epub"))

    @property
    def channel_ids(self) -> frozenset[int]:
        values = [item.strip() for item in self.allowed_channel_ids.split(",")]
        return frozenset(int(value) for value in values if value)