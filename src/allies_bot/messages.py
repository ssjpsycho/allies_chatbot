import re

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


def plain_text_for_discord(content: str) -> str:
    content = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1: <\2>", content)
    content = re.sub(r"^#{1,6}\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", content)
    content = re.sub(r"(?m)^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$", "", content)
    content = content.replace("|", " - ")
    content = content.replace("```", "")
    counters: dict[int, int] = {}
    normalized_lines: list[str] = []
    for line in content.splitlines():
        match = re.match(r"^(\s*)\d+[.)]\s+(.*)$", line)
        if match:
            indent = len(match.group(1))
            counters[indent] = counters.get(indent, 0) + 1
            counters = {level: count for level, count in counters.items() if level <= indent}
            normalized_lines.append(f"{match.group(1)}{counters[indent]}. {match.group(2)}")
        else:
            counters.clear()
            normalized_lines.append(line)
    return "\n".join(normalized_lines).strip()