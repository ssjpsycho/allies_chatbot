DISCORD_MESSAGE_LIMIT = 2000


def split_for_discord(content: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    if not content:
        return []

    messages: list[str] = []
    remaining = content
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at <= 0:
            split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at <= 0:
            split_at = limit
        messages.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    messages.append(remaining)
    return messages