"""Apply explicitly approved heading or table edits from a formatting preview."""

import argparse
import asyncio
import hashlib
import html
import json
import os
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx

from allies_bot.audit import AuditSettings, WikiSnapshot
from allies_bot.ingest import get_bookstack
from allies_bot.wiki_format import (
    HEADINGS,
    FormattingPlan,
    PageProposal,
    apply_edits,
    parse_positioned,
    preservation_checks,
    propose_page,
)
from allies_bot.wiki_layout import propose_layout
from allies_bot.wiki_triage import propose_reviewed_page

HEADING_REASONS = {
    "Normalize heading hierarchy below the BookStack page title (H1).",
    "Replace empty heading with an anchor-preserving span.",
    "Restore progression-linked bold label as a semantic heading.",
}
TABLE_REASONS = {
    "Mark existing column labels as table headers.",
    "Mark existing row label as a table row header.",
}


def table_edits_html(source: str, proposal: PageProposal) -> str:
    after = apply_edits(source, [e for e in proposal.edits if e.reason in TABLE_REASONS])
    if not all(preservation_checks(source, after).values()):
        raise ValueError(f"Table preservation checks failed for page {proposal.page_id}.")
    return after


def heading_html(source: str, proposal: PageProposal) -> str:
    edits = [edit for edit in proposal.edits if edit.reason in HEADING_REASONS]
    after = apply_edits(source, edits)
    if not all(preservation_checks(source, after).values()):
        raise ValueError(f"Heading preservation checks failed for page {proposal.page_id}.")
    before_tables = table_html(source)
    if before_tables != table_html(after):
        raise ValueError(f"Heading changes touched a table on page {proposal.page_id}.")
    return after


def table_html(source: str) -> list[str]:
    parser = parse_positioned(source)
    return [
        source[span.start:span.end]
        for node in parser.root.descendants({"table"})
        if (span := parser.spans[id(node)]).end is not None
    ]


def protect_heading_links(source: str) -> tuple[str, set[str]]:
    parser = parse_positioned(source)
    if parser.errors:
        raise ValueError("Cannot protect bookmarks in malformed HTML.")
    existing_names = {
        node.attrs["name"]: node for node in parser.root.elements()
        if node.tag == "a" and "name" in node.attrs
    }
    ids = {
        node.attrs["id"]: node for node in parser.root.elements() if node.attrs.get("id")
    }
    additions: list[tuple[int, str]] = []
    required: set[str] = set()
    for node in parser.root.descendants(HEADINGS):
        fragment = node.attrs.get("id")
        if not fragment:
            continue
        names = {fragment}
        decoded = unquote(fragment)
        # Restrict aliases to characters confirmed to survive BookStack saves.
        # Quotes, non-ASCII and whitespace are encoded into duplicate names.
        if re.fullmatch(r"[-a-zA-Z0-9_.,]+", decoded):
            names.add(decoded)
        required.update(names)
        markup = ""
        descendants = {id(child) for child in node.elements()}
        for name in sorted(names):
            owner = existing_names.get(name)
            id_owner = ids.get(name)
            if owner is not None:
                if id(owner) not in descendants:
                    raise ValueError(f"Bookmark name collision: {name}")
                continue
            if id_owner is not None and id_owner is not node:
                raise ValueError(f"Decoded bookmark ID collision: {name}")
            markup += f'<a name="{html.escape(name, quote=True)}"></a>'
        if markup:
            additions.append((parser.spans[id(node)].opening_end, markup))
    result = source
    for position, markup in sorted(additions, reverse=True):
        result = result[:position] + markup + result[position:]
    return result, required


def verify_compatible_save(intended: str, saved: str, names: set[str]) -> bool:
    """Only heading IDs may differ; bookmarks must stay inside the same heading."""
    left = parse_positioned(intended)
    right = parse_positioned(saved)
    if left.errors or right.errors:
        return False
    expected_nodes = list(left.root.elements())
    actual_nodes = list(right.root.elements())
    if len(expected_nodes) != len(actual_nodes):
        return False
    for expected, actual in zip(expected_nodes, actual_nodes):
        if expected.tag != actual.tag or expected.text() != actual.text():
            return False
        expected_attrs, actual_attrs = dict(expected.attrs), dict(actual.attrs)
        if expected.tag in HEADINGS:
            expected_attrs.pop("id", None)
            actual_attrs.pop("id", None)
        if expected_attrs != actual_attrs:
            return False
        # Compare direct children to detect content movement hidden by whitespace joining.
        if [("text", c) if isinstance(c, str) else ("element", c.tag)
            for c in expected.children] != [
            ("text", c) if isinstance(c, str) else ("element", c.tag)
            for c in actual.children
        ]:
            return False
    all_names = [
        node.attrs["name"] for node in actual_nodes
        if node.tag == "a" and "name" in node.attrs
    ]
    return names <= set(all_names) and all(all_names.count(name) == 1 for name in names)


def save_private(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as output:
        os.chmod(path, 0o600)
        json.dump(value, output, indent=2, ensure_ascii=False)
        output.write("\n")


def journal(path: Path, **event: object) -> None:
    with path.open("a", encoding="utf-8") as output:
        os.chmod(path, 0o600)
        output.write(json.dumps(event) + "\n")
        output.flush()
        os.fsync(output.fileno())


async def apply_headings(
    plan_path: Path, output: Path, settings: AuditSettings, *, tables: bool = False,
    reviewed_tables: bool = False,
    approved_layouts: bool = False,
) -> None:
    if sum((tables, reviewed_tables, approved_layouts)) > 1:
        raise ValueError("Select only one approved mode.")
    plan = FormattingPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    snapshot = WikiSnapshot.model_validate_json(
        (plan_path.parent / "snapshot.json").read_text(encoding="utf-8")
    )
    pages = {page.id: page for page in snapshot.pages}
    work: list[tuple[PageProposal, str]] = []
    for proposal in plan.pages:
        if proposal.status != "proposed":
            continue
        page = pages[proposal.page_id]
        if urlsplit(page.url).netloc != urlsplit(settings.bookstack_base_url).netloc:
            raise ValueError("Plan host differs from configured BookStack host.")
        expected, reviewed_after = (
            propose_layout(page) if approved_layouts else
            propose_reviewed_page(page) if reviewed_tables else propose_page(page)
        )
        if expected.edits != proposal.edits or proposal.before_sha256 != page.content_sha256:
            raise ValueError(f"Plan differs from verified source proposals on page {page.id}.")
        if reviewed_tables or approved_layouts:
            if (
                proposal.after_sha256 != expected.after_sha256
                or not all(expected.checks.values())
            ):
                raise ValueError(f"Reviewed table preservation failed on page {page.id}.")
            after = reviewed_after
        else:
            after = table_edits_html(page.html, proposal) if tables else heading_html(page.html, proposal)
        if after != page.html:
            work.append((proposal, after))
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    log = output / "journal.jsonl"
    journal(log, status="started", pages=len(work), atomic_concurrency=False,
            mode="layouts" if approved_layouts else
            "reviewed_tables" if reviewed_tables else "tables" if tables else "headings")
    headers = {"Authorization": f"Token {settings.bookstack_token}"}
    async with httpx.AsyncClient(
        base_url=settings.bookstack_base_url, headers=headers, timeout=30,
    ) as client:
        for proposal, after in work:
            page_id = proposal.page_id
            response = await get_bookstack(client, f"/api/pages/{page_id}")
            before = response.json()
            source = pages[page_id]
            digest = hashlib.sha256(before["html"].encode()).hexdigest()
            if (
                digest != proposal.before_sha256
                or before.get("updated_at") != proposal.source_updated_at
                or before["name"] != source.title
            ):
                journal(log, page_id=page_id, status="skipped_changed_since_preview")
                continue
            if before.get("markdown"):
                journal(log, page_id=page_id, status="skipped_markdown_editor")
                continue
            after, bookmark_names = protect_heading_links(after)
            save_private(output / f"{page_id}-before.json", before)
            save_private(output / f"{page_id}-intended.json", {"html": after})
            latest = (await get_bookstack(client, f"/api/pages/{page_id}")).json()
            if any(latest.get(key) != before.get(key) for key in (
                "html", "updated_at", "revision_count", "name", "book_id", "chapter_id",
                "slug", "priority", "tags", "markdown", "editor",
            )):
                journal(log, page_id=page_id, status="skipped_changed_before_write")
                continue
            journal(log, page_id=page_id, status="write_started")
            try:
                # Never retry writes: a timeout can mean the server already saved.
                result = await client.put(
                    f"/api/pages/{page_id}",
                    json={
                        "html": after,
                        "changelog": "Approved source-specific layout repair" if approved_layouts else
                        "Approved table-header formatting repair" if tables or reviewed_tables
                        else "Approved heading-only formatting repair",
                    },
                )
                result.raise_for_status()
            except httpx.HTTPError as error:
                journal(log, page_id=page_id, status="write_failed_or_uncertain",
                        error_type=type(error).__name__)
                raise
            saved = (await get_bookstack(client, f"/api/pages/{page_id}")).json()
            save_private(output / f"{page_id}-after.json", saved)
            compatible = verify_compatible_save(after, saved["html"], bookmark_names)
            metadata_ok = all(before.get(key) == saved.get(key) for key in (
                "id", "name", "slug", "book_id", "chapter_id", "priority", "tags",
                "markdown", "editor",
            ))
            tables_ok = table_html(
                after if tables or reviewed_tables or approved_layouts else before["html"]
            ) == table_html(saved["html"])
            if not (compatible and metadata_ok and tables_ok):
                journal(log, page_id=page_id, status="verification_failed",
                        compatible_html=compatible, metadata_ok=metadata_ok,
                        tables_unchanged=tables_ok)
                raise ValueError(
                    f"Saved content verification failed on page {page_id}; stopped. "
                    f"Backup and saved response: {output}"
                )
            journal(log, page_id=page_id, status="verified", title=proposal.title,
                    compatibility_bookmarks=sorted(bookmark_names))
            print(f"Verified {page_id}: {proposal.title}", flush=True)
    journal(log, status="completed")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True, help="New private local backup directory.")
    approval = parser.add_mutually_exclusive_group(required=True)
    approval.add_argument("--apply-approved-headings", action="store_true")
    approval.add_argument("--apply-approved-tables", action="store_true")
    approval.add_argument("--apply-approved-reviewed-tables", action="store_true")
    approval.add_argument("--apply-approved-layouts", action="store_true")
    args = parser.parse_args()
    await apply_headings(
        args.plan, args.backup, AuditSettings(), tables=args.apply_approved_tables,
        reviewed_tables=args.apply_approved_reviewed_tables,
        approved_layouts=args.apply_approved_layouts,
    )


if __name__ == "__main__":
    asyncio.run(main())
