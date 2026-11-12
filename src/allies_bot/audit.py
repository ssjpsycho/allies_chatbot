"""Read-only BookStack audit and evidence-linked structured knowledge pilot."""

import argparse
import asyncio
import hashlib
import html
import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urljoin, urlsplit

import httpx
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from allies_bot.ingest import get_bookstack

PILOT_CHAPTERS = {"Orders of Elohim", "Spiritual Effects", "Songs"}
VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr",
}
IGNORED_TAGS = {"script", "style", "head", "noscript"}
FIELD_PATTERNS = {
    "damage_mentions": r"\bdamag\w*|\blower\w*|\breduc\w*",
    "resistance_mentions": r"\bresist\w*",
    "range_mentions": r"\brange\b|\bwithin\b|\bmelee\b",
    "duration_mentions": r"\bduration\b|\bongoing\b|\bturns?\b|\bcycles?\b|\bhours?\b",
    "target_mentions": r"\btargets?\b|\bself\b|\ballies\b|\bunholy elohim\b",
    "eligibility_mentions": (
        r"\b(?:learn|select|choose|access|eligible|only)\w*\b.*"
        r"\b(?:ministering|messenger|minstrel|warrior|order|discipline)\w*\b|"
        r"\b(?:ministering|messenger|minstrel|warrior|order|discipline)\w*\b.*"
        r"\b(?:learn|select|choose|access|eligible|only)\w*\b"
    ),
    "prerequisite_exception_mentions": r"\brequir\w*|\bunless\b|\bexcept\b|\bonly\b|\bif\b",
}


class AuditSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    bookstack_base_url: str = "https://wiki.alliesofmajesty.com"
    bookstack_token: str


class PageSnapshot(BaseModel):
    id: int
    title: str
    url: str
    book: str
    chapter: str | None
    updated_at: str | None
    html: str
    content_sha256: str


class WikiSnapshot(BaseModel):
    schema_version: Literal[1] = 1
    captured_at: str
    base_url: str
    pages: list[PageSnapshot]

    @field_validator("pages")
    @classmethod
    def validate_pages(cls, pages: list[PageSnapshot]) -> list[PageSnapshot]:
        if not pages or len({page.id for page in pages}) != len(pages):
            raise ValueError("Wiki snapshot must contain nonempty, unique page IDs.")
        for page in pages:
            digest = hashlib.sha256(page.html.encode("utf-8")).hexdigest()
            if digest != page.content_sha256:
                raise ValueError(f"Snapshot content hash mismatch for page {page.id}.")
        return pages


@dataclass
class Element:
    tag: str
    attrs: dict[str, str]
    children: list["Element | str"] = field(default_factory=list)

    def text(self) -> str:
        if self.tag in IGNORED_TAGS:
            return ""
        return " ".join(
            " ".join(child.text() if isinstance(child, Element) else child
                     for child in self.children).split()
        )

    def descendants(self, tags: set[str]) -> list["Element"]:
        found = []
        for child in self.children:
            if isinstance(child, Element):
                if child.tag in tags:
                    found.append(child)
                found.extend(child.descendants(tags))
        return found

    def elements(self) -> Iterator["Element"]:
        yield self
        for child in self.children:
            if isinstance(child, Element):
                yield from child.elements()


class SourceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Element("root", {})
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Element(tag, {key: value for key, value in attrs if value is not None})
        self.stack[-1].children.append(node)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


class TableCell(BaseModel):
    text: str
    rowspan: str = "1"
    colspan: str = "1"


class Evidence(BaseModel):
    id: str
    kind: Literal["heading", "paragraph", "table"]
    source_url: str
    heading_path: list[str]
    text: str
    rows: list[list[TableCell]] | None = None
    heading_basis: Literal["html_heading", "progression_label_after_empty_heading"] | None = None


class Progression(BaseModel):
    variant_name: str
    tier: int
    position: int
    evidence_id: str
    row: int
    column: int


class CandidateField(BaseModel):
    status: Literal["unknown", "candidate"] = "unknown"
    quotes: list[str] = []
    evidence_ids: list[str] = []


class PilotRecord(BaseModel):
    id: str
    category: Literal["order", "spiritual_effect", "song", "reference"]
    name: str
    family: str | None
    page_id: int
    page_title: str
    source_url: str
    book: str
    chapter: str | None
    content_sha256: str
    heading_path: list[str]
    classification_basis: str
    review_status: Literal["needs_review"] = "needs_review"
    review_flags: list[str]
    progression: list[Progression]
    fields: dict[str, CandidateField]
    evidence: list[Evidence]


class Finding(BaseModel):
    code: str
    page_id: int
    page_title: str
    source_url: str
    detail: str


class AuditResult(BaseModel):
    schema_version: Literal[1] = 1
    captured_at: str
    pages_audited: int
    pilot_pages: int
    records: int
    counts_by_category: dict[str, int]
    findings: list[Finding]


class PilotResult(BaseModel):
    schema_version: Literal[1] = 1
    captured_at: str
    limitations: list[str]
    records: list[PilotRecord]


def parse_source(page: PageSnapshot) -> tuple[Element, list[Evidence]]:
    parser = SourceParser()
    parser.feed(page.html)
    parser.close()
    evidence: list[Evidence] = []
    headings: list[tuple[int, str]] = []
    heading_anchor = ""
    empty_heading_level: int | None = None

    def walk(node: Element, inherited_anchor: str = "") -> None:
        nonlocal heading_anchor, empty_heading_level
        if node.tag in IGNORED_TAGS:
            return
        anchor = node.attrs.get("id", inherited_anchor)
        rows = None
        heading_basis = None
        if re.fullmatch(r"h[1-6]", node.tag):
            level = int(node.tag[1])
            if not node.text():
                empty_heading_level = level
                return
            empty_heading_level = None
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, node.text()))
            heading_anchor = anchor
            kind = "heading"
            heading_basis = "html_heading"
        elif node.tag == "table":
            kind = "table"
            rows = [
                [
                    TableCell(
                        text=cell.text(),
                        rowspan=cell.attrs.get("rowspan", "1"),
                        colspan=cell.attrs.get("colspan", "1"),
                    )
                    for cell in row.children
                    if isinstance(cell, Element) and cell.tag in {"td", "th"}
                ]
                for row in node.descendants({"tr"})
            ]
        elif node.tag in {"p", "li", "pre", "blockquote", "dt", "dd"}:
            # Traverse nested blocks (especially lists) to avoid quoting a whole
            # list and then duplicating its individual entries.
            if node.descendants({"p", "li", "table"}):
                for child in node.children:
                    if isinstance(child, Element):
                        walk(child, anchor)
                    elif child.strip():
                        append("paragraph", child.strip(), None, anchor)
                return
            kind = "paragraph"
            if empty_heading_level is not None and node.text():
                names = {normalized(entry.variant_name) for entry in progression_entries(evidence)}
                bold_text = " ".join(child.text() for child in node.descendants({"strong", "b"}))
                if (
                    node.tag == "p"
                    and normalized(node.text()) in names
                    and normalized(bold_text) == normalized(node.text())
                ):
                    while headings and headings[-1][0] >= empty_heading_level:
                        headings.pop()
                    headings.append((empty_heading_level, node.text()))
                    heading_anchor = anchor
                    kind = "heading"
                    heading_basis = "progression_label_after_empty_heading"
                empty_heading_level = None
        else:
            for child in node.children:
                if isinstance(child, Element):
                    walk(child, anchor)
                elif child.strip():
                    append("paragraph", child.strip(), None, anchor)
            return
        text = (
            "\n".join(" | ".join(cell.text for cell in row) for row in rows)
            if rows is not None else node.text()
        )
        if text.strip():
            append(kind, text, rows, anchor, heading_basis)

    def append(
        kind: Literal["heading", "paragraph", "table"],
        text: str,
        rows: list[list[TableCell]] | None,
        anchor: str,
        heading_basis: Literal[
            "html_heading", "progression_label_after_empty_heading"
        ] | None = None,
    ) -> None:
        evidence.append(Evidence(
            id=f"bookstack:{page.id}:block:{len(evidence) + 1}",
            kind=kind,
            source_url=page.url + (f"#{anchor or heading_anchor}" if anchor or heading_anchor else ""),
            heading_path=[title for _, title in headings],
            text=text,
            rows=rows,
            heading_basis=heading_basis,
        ))

    walk(parser.root)
    return parser.root, evidence


def normalized(text: str) -> str:
    return " ".join(text.casefold().replace("\u2019", "'").split())


def progression_entries(evidence: list[Evidence]) -> list[Progression]:
    entries = []
    for block in evidence:
        if not block.rows:
            continue
        headers = block.rows[0]
        tiers = {
            index: int(match.group(1))
            for index, cell in enumerate(headers)
            if (match := re.fullmatch(r"Tier\s*:?\s*(\d+)", cell.text, re.IGNORECASE))
        }
        # Do not infer column alignment in merged or irregular tables.
        if not tiers or any(
            cell.rowspan != "1" or cell.colspan != "1" for row in block.rows for cell in row
        ):
            continue
        for row_index, row in enumerate(block.rows[1:], 1):
            if len(row) != len(headers) or not row[0].text.isdigit():
                continue
            for column, tier in tiers.items():
                if row[column].text:
                    entries.append(Progression(
                        variant_name=row[column].text,
                        tier=tier,
                        position=int(row[0].text),
                        evidence_id=block.id,
                        row=row_index,
                        column=column,
                    ))
    return entries


def candidate_fields(evidence: list[Evidence]) -> dict[str, CandidateField]:
    fields = {}
    for name, pattern in FIELD_PATTERNS.items():
        quotes: list[str] = []
        ids: list[str] = []
        for block in evidence:
            if block.kind == "heading":
                continue
            # A table row retains all cells, not an isolated value with no label.
            statements = (
                block.text.splitlines() if block.rows is not None
                else re.split(r"(?<=[.!?])\s+", block.text)
            )
            for statement in statements:
                if re.search(pattern, statement, re.IGNORECASE):
                    if statement not in quotes:
                        quotes.append(statement)
                    if block.id not in ids:
                        ids.append(block.id)
        fields[name] = CandidateField(
            status="candidate" if quotes else "unknown", quotes=quotes, evidence_ids=ids,
        )
    return fields


def discrepancy_flags(name: str, evidence: list[Evidence]) -> list[str]:
    """Flag disjoint labelled values; this is not an automatic rule correction."""
    prose: dict[str, set[str]] = {}
    table: dict[str, set[str]] = {}
    for block in evidence:
        if block.kind == "paragraph":
            destination = prose
            text = block.text
        elif block.rows:
            destination = table
            text = "\n".join(
                " ".join(cell.text for cell in row)
                for row in block.rows
                if row and normalized(row[0].text) == normalized(name)
            )
        else:
            continue
        for label, value in re.findall(r"\b([A-Z][a-z]+):(\d+)\b", text):
            if label != "Tier":
                destination.setdefault(label, set()).add(value)
    return [
        f"Potential prose/table mismatch for {label}: prose {sorted(values)}, "
        f"base table {sorted(table[label])}; verify context and variants."
        for label, values in prose.items()
        if label in table and values.isdisjoint(table[label])
    ]


def extract_records(page: PageSnapshot, evidence: list[Evidence]) -> list[PilotRecord]:
    if page.chapter not in PILOT_CHAPTERS or page.book != "Character Creation & Advancement":
        return []
    family = page.title if page.title.endswith(" Family") else None
    category: Literal["order", "spiritual_effect", "song", "reference"] = "reference"
    if family:
        category = "song" if page.chapter == "Songs" else "spiritual_effect"
    elif page.chapter == "Orders of Elohim" and page.title not in {"Orders", "Multi-Order Elohim"}:
        category = "order"
    groups: list[tuple[str, list[Evidence], list[Progression]]] = []
    entries = progression_entries(evidence)
    if family:
        headings = [block for block in evidence if block.kind == "heading"]
        for heading in headings:
            name = heading.heading_path[-1]
            matched = [
                entry for entry in entries
                if normalized(entry.variant_name) == normalized(name)
                or normalized(entry.variant_name).endswith(" " + normalized(name))
            ]
            # Only progression-linked headings become ability candidates.
            if not matched:
                continue
            start = evidence.index(heading)
            end = next((
                index for index in range(start + 1, len(evidence))
                if evidence[index].kind == "heading"
                and len(evidence[index].heading_path) <= len(heading.heading_path)
            ), len(evidence))
            groups.append((name, evidence[start:end], matched))
        matched_names = {entry.variant_name for _, _, group in groups for entry in group}
        for entry in entries:
            if entry.variant_name not in matched_names:
                groups.append((
                    entry.variant_name,
                    [block for block in evidence if block.id == entry.evidence_id],
                    [entry],
                ))
        # Keep introductory and unrecognised sections visible for review.
        used = {block.id for _, blocks, _ in groups for block in blocks}
        remaining = [block for block in evidence if block.id not in used]
        if remaining:
            groups.append((page.title + " — page context", remaining, []))
    else:
        groups.append((page.title, evidence, []))
    records = []
    for index, (name, blocks, progression) in enumerate(groups, 1):
        context_only = bool(family and not progression)
        flags = ["Automated extraction; verify against the cited source before production use."]
        if any(b.heading_basis == "progression_label_after_empty_heading" for b in blocks):
            flags.append("Section boundary recovered from a bold progression label after an empty heading.")
        if family and (not blocks or blocks[0].kind != "heading") and not context_only:
            flags.append("Progression cell has no matched description heading; variant rules unverified.")
        if progression and len({entry.variant_name for entry in progression}) > 1:
            flags.append("Tier variants share this section; do not apply base values to every variant.")
        flags.extend(discrepancy_flags(name, blocks))
        fields = candidate_fields(blocks)
        if fields["eligibility_mentions"].status == "unknown":
            flags.append("Eligibility is unknown; chapter classification does not establish Order access.")
        # A record must carry the actual table behind every progression cell,
        # even though its rule fields are extracted only from its own section.
        attached = {block.id for block in blocks}
        progression_ids = {entry.evidence_id for entry in progression}
        all_evidence = blocks + [
            block for block in evidence if block.id in progression_ids and block.id not in attached
        ]
        records.append(PilotRecord(
            id=f"bookstack:{page.id}:record:{index}",
            category="reference" if context_only else category,
            name=name,
            family=family,
            page_id=page.id,
            page_title=page.title,
            source_url=blocks[0].source_url if blocks else page.url,
            book=page.book,
            chapter=page.chapter,
            content_sha256=page.content_sha256,
            heading_path=blocks[0].heading_path if blocks else [],
            classification_basis=f"BookStack hierarchy: {page.book} > {page.chapter}",
            review_flags=flags,
            progression=progression,
            fields=fields,
            evidence=all_evidence,
        ))
    return records


def audit_snapshot(snapshot: WikiSnapshot) -> tuple[AuditResult, PilotResult]:
    parsed = {page.id: parse_source(page) for page in snapshot.pages}
    by_id = {page.id: page for page in snapshot.pages}
    by_path = {urlsplit(page.url).path: page for page in snapshot.pages}
    anchors = {
        page_id: {"bkmrk-page-title"} | {
            node.attrs["id"] for node in root.elements() if node.attrs.get("id")
        }
        for page_id, (root, _) in parsed.items()
    }
    findings: list[Finding] = []
    records: list[PilotRecord] = []

    def flag(page: PageSnapshot, code: str, detail: str) -> None:
        findings.append(Finding(
            code=code, page_id=page.id, page_title=page.title, source_url=page.url, detail=detail,
        ))

    for page in snapshot.pages:
        root, evidence = parsed[page.id]
        records.extend(extract_records(page, evidence))
        headings = [block.text for block in evidence if block.kind == "heading"]
        for block in evidence:
            if block.heading_basis == "progression_label_after_empty_heading":
                flag(
                    page, "recovered_visual_heading",
                    f"{block.text!r}: bold progression label after empty HTML heading; description present.",
                )
        if not evidence:
            flag(page, "empty_page", "No indexable text blocks.")
        if not headings and len(root.text()) > 1200:
            flag(page, "long_page_without_headings", "Long page has no semantic HTML headings.")
        for title, count in Counter(normalized(title) for title in headings).items():
            if count > 1:
                flag(page, "duplicate_heading", f"{title!r} occurs {count} times.")
        for block in evidence:
            if block.rows:
                if any(c.colspan != "1" or c.rowspan != "1" for r in block.rows for c in r):
                    flag(page, "merged_table_cells", f"{block.id}: merged cells need manual mapping.")
                if len({len(row) for row in block.rows}) > 1:
                    flag(page, "irregular_table", f"{block.id}: unequal row widths need review.")
                if len(block.text) > 1200:
                    flag(page, "large_table", f"{block.id}: preserve headers when splitting this table.")
        for link in root.descendants({"a"}):
            href = link.attrs.get("href", "")
            url = urlsplit(urljoin(page.url, href))
            if url.netloc != urlsplit(snapshot.base_url).netloc:
                continue
            match = re.fullmatch(r"/link/(\d+)/?", url.path)
            target = by_id.get(int(match.group(1))) if match else by_path.get(url.path)
            if target and url.fragment and unquote(url.fragment) not in anchors[target.id]:
                flag(page, "unresolved_content_anchor", f"{link.text()!r}: {href}")
        if (
            page.title.endswith(" Family") and page.chapter in {"Songs", "Spiritual Effects"}
            and not progression_entries(evidence)
        ):
            flag(page, "unparsed_progression", "No safely aligned progression table was extracted.")
    for record in records:
        for review in record.review_flags:
            if review.startswith("Progression cell"):
                page = by_id[record.page_id]
                flag(page, "unmatched_progression_entry", f"{record.name}: {review}")
            elif review.startswith("Potential prose/table"):
                page = by_id[record.page_id]
                flag(page, "potential_rule_mismatch", f"{record.name}: {review}")
    pilot = PilotResult(
        captured_at=snapshot.captured_at,
        limitations=[
            "Read-only pilot: no BookStack, Qdrant, embedding, or deployment writes.",
            "All records need review. Quotes are evidence candidates, not validated mechanical facts.",
            "Unknown fields remain explicit. Mentions do not establish eligibility, causality or target.",
            ("Only Character Creation & Advancement / Orders of Elohim, Spiritual Effects, Songs "
             "are structured; other pages are audited but excluded from the pilot."),
            "Tier variants may share descriptions. Merged/irregular progression cells are not guessed.",
            "The original-question sweep is not a complete list of eligible ways to damage Resolve.",
            "Snapshot is a sequential API read, not a transactionally consistent wiki backup.",
            "Unresolved anchors are absent from API content; rendered-page validity is not confirmed.",
        ],
        records=records,
    )
    audit = AuditResult(
        captured_at=snapshot.captured_at,
        pages_audited=len(snapshot.pages),
        pilot_pages=len({record.page_id for record in records}),
        records=len(records),
        counts_by_category=dict(Counter(record.category for record in records)),
        findings=findings,
    )
    return audit, pilot


def attribute_candidates(
    records: list[PilotRecord], attribute: str, category: str,
) -> list[dict[str, object]]:
    """Sweep every pilot record, without vector ranking or top-k truncation."""
    results: list[dict[str, object]] = []
    for record in records:
        if record.category != category:
            continue
        quotes = [
            quote for quote in record.fields["damage_mentions"].quotes
            if re.search(rf"\b{re.escape(attribute)}\b", quote, re.IGNORECASE)
        ]
        if quotes:
            results.append({
                "record_id": record.id, "name": record.name,
                "source_url": record.source_url, "quotes": quotes,
                "eligibility": "unverified", "review_status": record.review_status,
            })
    return results


def render_report(audit: AuditResult, pilot: PilotResult) -> str:
    esc = html.escape
    counts = Counter(finding.code for finding in audit.findings)
    parts = [
        "<!doctype html><html lang='en'><meta charset='utf-8'>",
        "<title>Read-only wiki audit and structured pilot</title>",
        ("<style>body{font:16px system-ui;max-width:1100px;margin:2em auto;padding:0 1em}"
        "pre{white-space:pre-wrap;background:#f4f4f4;padding:1em}"
        "details{border:1px solid #ccc;padding:1em;margin:1em 0}"
        "td,th{border:1px solid #ccc;padding:.4em;vertical-align:top}"
         "table{border-collapse:collapse}a{overflow-wrap:anywhere}</style>"),
        "<h1>Read-only wiki audit and structured pilot</h1>",
        (f"<p>Snapshot: {esc(audit.captured_at)}. {audit.pages_audited} pages audited; "
         f"{audit.pilot_pages} pilot pages; {audit.records} review records.</p>"),
        "<pre>" + esc(json.dumps(audit.counts_by_category, indent=2)) + "</pre>",
        "<h2>Limits and review requirements</h2><ul>",
        *[f"<li>{esc(limit)}</li>" for limit in pilot.limitations],
        "</ul><h2>Priority content review</h2><ul>",
        *[
            f"<li><a href='{esc(f.source_url, quote=True)}'>{esc(f.page_title)}</a>: "
            f"{esc(f.detail)}</li>"
            for f in audit.findings
            if f.code in {"unmatched_progression_entry", "potential_rule_mismatch"}
        ],
        "</ul><h2>Audit findings</h2>",
        "<pre>" + esc(json.dumps(dict(counts), indent=2)) + "</pre>",
        "<details><summary>All page findings</summary><ul>",
        *[
            f"<li><a href='{esc(f.source_url, quote=True)}'>{esc(f.page_title)}</a> "
            f"[{esc(f.code)}] {esc(f.detail)}</li>"
            for f in audit.findings
        ],
        "</ul></details><h2>Original-question pilot: Resolve damage candidates</h2>",
        ("<p>All Spiritual Effect candidates with damage/reduction and Resolve in the same "
        "statement or table row. This is a review queue, not proof of damage or eligibility. "
         "Songs are excluded by chapter, not by guessing from names.</p>"),
    ]
    for result in attribute_candidates(pilot.records, "Resolve", "spiritual_effect"):
        parts.append(f"<h3>{esc(str(result['name']))}</h3><pre>"
                     + esc(json.dumps(result, indent=2, ensure_ascii=False)) + "</pre>")
    parts.append("<h2>All structured records</h2>")
    for record in pilot.records:
        parts.extend([
            f"<details><summary>{esc(record.category)}: {esc(record.name)}</summary>",
            f"<p><a href='{esc(record.source_url, quote=True)}'>Open wiki source</a></p>",
            "<pre>" + esc(record.model_dump_json(indent=2)) + "</pre></details>",
        ])
    parts.append("</html>")
    return "\n".join(parts)


async def capture_wiki(settings: AuditSettings) -> WikiSnapshot:
    """Only GET requests are allowed here; no embeddings or vector store are used."""
    pages: list[PageSnapshot] = []
    books: dict[int, tuple[str, str]] = {}
    chapters: dict[int, str] = {}
    base_url = settings.bookstack_base_url.rstrip("/")
    headers = {"Authorization": f"Token {settings.bookstack_token}"}
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=30) as client:
        offset = 0
        while True:
            response = await get_bookstack(client, "/api/pages", params={"count": 100, "offset": offset})
            listing = response.json()
            for summary in listing["data"]:
                response = await get_bookstack(client, f"/api/pages/{summary['id']}")
                page = response.json()
                book_id = page["book_id"]
                if book_id not in books:
                    response = await get_bookstack(client, f"/api/books/{book_id}")
                    book = response.json()
                    books[book_id] = (book["name"], book["slug"])
                chapter_id = page.get("chapter_id")
                if chapter_id and chapter_id not in chapters:
                    response = await get_bookstack(client, f"/api/chapters/{chapter_id}")
                    chapters[chapter_id] = response.json()["name"]
                html = page.get("html") or ""
                book_name, book_slug = books[book_id]
                pages.append(PageSnapshot(
                    id=page["id"],
                    title=page["name"],
                    url=f"{base_url}/books/{book_slug}/page/{page['slug']}",
                    book=book_name,
                    chapter=chapters.get(chapter_id),
                    updated_at=page.get("updated_at"),
                    html=html,
                    content_sha256=hashlib.sha256(html.encode("utf-8")).hexdigest(),
                ))
            offset += len(listing["data"])
            if offset >= listing["total"]:
                break
            if not listing["data"]:
                raise ValueError("BookStack returned an incomplete page listing; audit aborted.")
    if not pages or len({page.id for page in pages}) != len(pages):
        raise ValueError("Wiki snapshot is empty or contains duplicate page IDs; audit aborted.")
    return WikiSnapshot(
        captured_at=datetime.now(UTC).isoformat(),
        base_url=base_url,
        pages=pages,
    )


def write_json(path: Path, value: BaseModel) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(value.model_dump_json(indent=2) + "\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Reuse a snapshot instead of contacting BookStack.")
    parser.add_argument("--output", type=Path, required=True, help="New directory for local audit files.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory; existing audit files will not be overwritten.")
    snapshot = (
        WikiSnapshot.model_validate_json(args.snapshot.read_text(encoding="utf-8"))
        if args.snapshot else await capture_wiki(AuditSettings())
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "snapshot.json", snapshot)
    audit, pilot = audit_snapshot(snapshot)
    write_json(args.output / "audit.json", audit)
    write_json(args.output / "records.json", pilot)
    with (args.output / "report.html").open("x", encoding="utf-8") as output:
        output.write(render_report(audit, pilot))
    print(json.dumps({
        "pages": audit.pages_audited, "pilot_pages": audit.pilot_pages,
        "records": audit.records, "counts_by_category": audit.counts_by_category,
        "findings": dict(Counter(f.code for f in audit.findings)), "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
