"""Prepare local, approval-only wiki formatting previews. Never writes to BookStack."""

import argparse
import asyncio
import difflib
import hashlib
import html
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from allies_bot.audit import (
    IGNORED_TAGS,
    VOID_TAGS,
    AuditSettings,
    Element,
    PageSnapshot,
    SourceParser,
    WikiSnapshot,
    capture_wiki,
    normalized,
    parse_source,
    progression_entries,
    write_json,
)

HEADINGS = {f"h{level}" for level in range(1, 7)}
HEADER_LABELS = {
    "class", "complexity", "allotment", "effect", "weapon", "range", "standard effect",
    "passion effect", "usable by", "song", "benefit", "rank", "social points",
    "maneuver points", "spiritual effects", "songs", "new value", "social points required",
    "progression",
}
BLOCKS = {"p", "div", "table", "ul", "ol", "li", "blockquote", "pre"} | HEADINGS


@dataclass
class Span:
    node: Element
    start: int
    opening_end: int
    closing_start: int | None = None
    end: int | None = None


class PositionedParser(SourceParser):
    """Track original tag ranges so edits never reserialize the page's content."""

    def __init__(self, source: str) -> None:
        super().__init__()
        self.source = source
        self.line_offsets = [0]
        self.line_offsets.extend(match.end() for match in re.finditer("\n", source))
        self.spans: dict[int, Span] = {}
        self.errors: list[str] = []

    def source_position(self) -> int:
        line, column = self.getpos()
        return self.line_offsets[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        start = self.source_position()
        super().handle_starttag(tag, attrs)
        node = self.stack[-1].children[-1] if tag in VOID_TAGS else self.stack[-1]
        if not isinstance(node, Element):
            raise TypeError("Parser did not create an element for a start tag.")
        self.spans[id(node)] = Span(node, start, start + len(self.get_starttag_text()))
        if len(attrs) != len({key for key, _ in attrs}):
            self.errors.append(f"Duplicate attributes at offset {start}.")

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_TAGS:
            self.errors.append(f"Unexpected closing void tag {tag}.")
            return
        if len(self.stack) <= 1 or self.stack[-1].tag != tag:
            self.errors.append(f"Unbalanced closing tag {tag} at offset {self.source_position()}.")
        for node in reversed(self.stack[1:]):
            if node.tag == tag:
                span = self.spans[id(node)]
                span.closing_start = self.source_position()
                span.end = self.source.find(">", self.source_position()) + 1
                break
        super().handle_endtag(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.errors.append(f"Self-closing non-void tag {tag}; browser structure needs review.")
            node = self.stack.pop()
            span = self.spans[id(node)]
            span.closing_start = span.opening_end
            span.end = span.opening_end

    def finish(self) -> None:
        self.feed(self.source)
        self.close()
        if len(self.stack) > 1:
            self.errors.append("Unclosed tags: " + ", ".join(node.tag for node in self.stack[1:]))


class Edit(BaseModel):
    start: int
    end: int
    before: str
    after: str
    reason: str


class PageProposal(BaseModel):
    page_id: int
    title: str
    url: str
    book: str
    chapter: str | None
    source_updated_at: str | None
    before_sha256: str
    after_sha256: str
    status: Literal["proposed", "unchanged", "blocked"]
    edits: list[Edit]
    review_notes: list[str]
    checks: dict[str, bool]


class FormattingPlan(BaseModel):
    schema_version: Literal[1] = 1
    snapshot_captured_at: str
    mode: Literal["preview_only"] = "preview_only"
    pages: list[PageProposal]


def parse_positioned(source: str) -> PositionedParser:
    parser = PositionedParser(source)
    parser.finish()
    return parser


def in_table_nodes(root: Element) -> set[int]:
    return {
        id(node)
        for table in root.descendants({"table"})
        for node in table.elements()
    }


def bold_label(node: Element) -> bool:
    if node.tag != "p" or not node.text() or node.descendants({"img", "table", "br"}):
        return False
    text = " ".join(child.text() for child in node.descendants({"strong", "b"}))
    return normalized(text) == normalized(node.text()) and len(node.text()) <= 120


def apply_edits(source: str, edits: list[Edit]) -> str:
    result = source
    previous = len(source)
    for edit in sorted(edits, key=lambda e: e.start, reverse=True):
        if not 0 <= edit.start <= edit.end <= previous or source[edit.start:edit.end] != edit.before:
            raise ValueError("Overlapping edits or mismatched source range; preview aborted.")
        result = result[:edit.start] + edit.after + result[edit.end:]
        previous = edit.start
    return result


def preservation_checks(before: str, after: str) -> dict[str, bool]:
    old, new = parse_positioned(before), parse_positioned(after)

    def signature(parser: PositionedParser) -> tuple[object, ...]:
        # Changed tag semantics must not alter any attributes except newly added scope.
        nodes = [node for node in parser.root.elements() if node.tag != "root"]
        attrs = [
            tuple((key, value) for key, value in node.attrs.items() if key != "scope")
            for node in nodes
        ]
        cells = [
            (
                node.text(), node.attrs.get("rowspan", "1"), node.attrs.get("colspan", "1"),
            )
            for node in nodes if node.tag in {"td", "th"}
        ]
        return attrs, cells

    old_sig, new_sig = signature(old), signature(new)
    # Compare data segments verbatim, not text reconstructed by a formatter.
    def text_segments(node: Element) -> list[str]:
        return [
            part
            for child in node.children
            for part in (text_segments(child) if isinstance(child, Element) else [child])
        ]

    return {
        "text_segments_identical": text_segments(old.root) == text_segments(new.root),
        "attributes_links_and_anchors_identical": old_sig[0] == new_sig[0],
        "table_cell_text_and_spans_identical": old_sig[1] == new_sig[1],
        "balanced_html": not old.errors and not new.errors,
    }


def propose_page(page: PageSnapshot) -> tuple[PageProposal, str]:
    source = page.html
    parser = parse_positioned(source)
    notes: list[str] = []
    edits: list[Edit] = []
    table_nodes = in_table_nodes(parser.root)
    desired: dict[int, tuple[str, str | None, str]] = {}
    _, evidence = parse_source(page)
    progression_names = {normalized(e.variant_name) for e in progression_entries(evidence)}
    original_levels: list[int] = []
    pending_empty: int | None = None
    semantic_headings = 0

    def set_tag(node: Element, tag: str, scope: str | None, reason: str) -> None:
        desired[id(node)] = tag, scope, reason

    for node in parser.root.elements():
        if id(node) in table_nodes or node.tag in IGNORED_TAGS:
            continue
        if node.tag in HEADINGS:
            level = int(node.tag[1])
            if not node.text():
                if node.descendants({"img", "svg", "iframe", "video"}):
                    notes.append("Heading contains media but no text; left unchanged for review.")
                    continue
                set_tag(node, "span", None, "Replace empty heading with an anchor-preserving span.")
                pending_empty = level
                continue
            pending_empty = None
        elif node.tag == "p" and node.text():
            if pending_empty is not None and bold_label(node) and normalized(node.text()) in progression_names:
                level = pending_empty
                pending_empty = None
            else:
                pending_empty = None
                if bold_label(node):
                    notes.append(
                        f"Possible visual heading {node.text()!r}; meaning/level needs manual review."
                    )
                continue
        else:
            if node.tag in BLOCKS:
                pending_empty = None
            continue
        while original_levels and original_levels[-1] >= level:
            original_levels.pop()
        original_levels.append(level)
        new_tag = f"h{min(6, len(original_levels) + 1)}"
        semantic_headings += 1
        if new_tag != node.tag:
            reason = (
                "Restore progression-linked bold label as a semantic heading."
                if node.tag == "p"
                else "Normalize heading hierarchy below the BookStack page title (H1)."
            )
            set_tag(node, new_tag, None, reason)
    if not semantic_headings and len(parser.root.text()) > 1200:
        notes.append("Long page without section headings; no new wording or section boundaries invented.")

    for table in parser.root.descendants({"table"}):
        rows = table.descendants({"tr"})
        if table.descendants({"table"}):
            notes.append("Nested table left unchanged; requires manual review.")
            continue
        cells = [
            [c for c in row.children if isinstance(c, Element) and c.tag in {"td", "th"}]
            for row in rows
        ]
        if not cells or len(cells[0]) < 2:
            notes.append("Table without a multi-column header left unchanged.")
            continue
        if any(c.attrs.get("rowspan", "1") != "1" or c.attrs.get("colspan", "1") != "1"
               for row in cells for c in row) or len({len(row) for row in cells}) != 1:
            notes.append("Merged or irregular table left unchanged; cell associations need review.")
            continue
        labels = [normalized(c.text()) for c in cells[0]]
        recognized = all(
            not label or label in HEADER_LABELS or re.fullmatch(r"tier\s*:?\s*\d+", label)
            for label in labels
        )
        if (
            not (recognized and sum(bool(label) for label in labels) >= 2)
            and not all(c.tag == "th" for c in cells[0])
        ):
            notes.append("Table header is ambiguous; no headers inferred from its data.")
            continue
        for cell in cells[0]:
            if "scope" not in cell.attrs:
                set_tag(cell, "th", "col", "Mark existing column labels as table headers.")
        for row in cells[1:]:
            first = row[0]
            if (
                first.text() and (bold_cell(first) or first.text().isdigit())
                and "scope" not in first.attrs
            ):
                set_tag(first, "th", "row", "Mark existing row label as a table row header.")

    if parser.errors:
        notes.extend(parser.errors)
        desired.clear()
    for node_id, (tag, scope, reason) in desired.items():
        span = parser.spans[node_id]
        if span.end is None or span.closing_start is None:
            raise ValueError(f"Unclosed editable element in page {page.id}.")
        opening = source[span.start:span.opening_end]
        changed = re.sub(r"^<\s*[a-zA-Z][\w:-]*", "<" + tag, opening, count=1)
        if scope:
            changed = changed[:-1] + f' scope="{scope}">'
        if changed != opening:
            edits.append(Edit(
                start=span.start, end=span.opening_end,
                before=opening, after=changed, reason=reason,
            ))
        closing = source[span.closing_start:span.end]
        new_closing = re.sub(r"^(</\s*)[\w:-]+", r"\g<1>" + tag, closing, count=1)
        if closing != new_closing:
            edits.append(Edit(
                start=span.closing_start, end=span.end,
                before=closing, after=new_closing, reason=reason,
            ))
    after = apply_edits(source, edits)
    checks = preservation_checks(source, after)
    if edits and not all(checks.values()):
        raise ValueError(f"Preservation check failed for page {page.id}: {checks}")
    proposal = PageProposal(
        page_id=page.id, title=page.title, url=page.url, book=page.book, chapter=page.chapter,
        source_updated_at=page.updated_at, before_sha256=page.content_sha256,
        after_sha256=hashlib.sha256(after.encode()).hexdigest(),
        status="blocked" if parser.errors else "proposed" if edits else "unchanged",
        edits=sorted(edits, key=lambda e: e.start),
        review_notes=list(dict.fromkeys(notes)), checks=checks,
    )
    return proposal, after


def bold_cell(node: Element) -> bool:
    text = " ".join(child.text() for child in node.descendants({"strong", "b"}))
    return normalized(text) == normalized(node.text())


STYLE = (
    ":root{color-scheme:dark;background:#0d1117;color:#e6edf3}"
    "body{font:16px system-ui;margin:2em}table{border-collapse:collapse;width:100%}"
    "td,th{border:1px solid #3d4754;padding:.5em;vertical-align:top}"
    "th{background:#1c2633}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
    "background:#161b22;padding:1em;border-radius:6px}"
    "iframe{width:100%;height:650px;border:1px solid #3d4754;background:#0d1117}"
    ".pair{display:grid;grid-template-columns:1fr 1fr;gap:1em}"
    "a{color:#79c0ff;overflow-wrap:anywhere}summary{cursor:pointer;color:#a5d6ff}"
    "@media(max-width:850px){.pair{grid-template-columns:1fr}body{margin:1em}}"
)


def document(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta http-equiv='Content-Security-Policy' "
        'content="default-src \'none\'; style-src \'unsafe-inline\'; frame-src \'self\'">'
        "<meta name='referrer' content='no-referrer'>"
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head>"
        f"<body>{body}</body></html>"
    )


def preview_frame(source: str, title: str) -> str:
    # Wiki scripts, remote images and forms cannot execute or load in the preview.
    content = document(title, source)
    return (
        f'<iframe sandbox="" title="{html.escape(title, quote=True)}" '
        f'srcdoc="{html.escape(content, quote=True)}"></iframe>'
    )


def write_preview(
    snapshot: WikiSnapshot,
    output: Path,
    *,
    proposer: Callable[[PageSnapshot], tuple[PageProposal, str]] = propose_page,
) -> FormattingPlan:
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "snapshot.json", snapshot)
    pages: list[PageProposal] = []
    for page in snapshot.pages:
        proposal, after = proposer(page)
        if proposal.status != "blocked":
            rerun = page.model_copy(update={
                "html": after, "content_sha256": proposal.after_sha256,
            })
            if proposer(rerun)[1] != after:
                raise ValueError(f"Formatting is not idempotent on page {page.id}; preview aborted.")
            proposal.checks["idempotent"] = True
        pages.append(proposal)
        folder = output / "pages" / str(page.id)
        folder.mkdir(parents=True)
        (folder / "before.html").write_text(page.html, encoding="utf-8")
        (folder / "after.html").write_text(after, encoding="utf-8")
        diff = "".join(difflib.unified_diff(
            page.html.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile="before.html", tofile="after.html",
        ))
        (folder / "changes.diff").write_text(diff, encoding="utf-8")
        esc = html.escape
        body = (
            f"<p><a href='../../index.html'>All pages</a></p><h1>{esc(page.title)}</h1>"
            f"<p>{esc(proposal.status)}; original wiki URL: {esc(page.url)}</p>"
            "<p>Preview only. Wording and rules are unchanged. Existing styles are preserved, "
            "so semantic improvements may look identical. Remote media and scripts are blocked. "
            "Nothing here applies edits to BookStack.</p>"
            "<h2>Changes proposed</h2><ul>"
            + "".join(f"<li>{esc(reason)}</li>" for reason in dict.fromkeys(e.reason for e in proposal.edits))
            + "</ul><h2>Manual review</h2><ul>"
            + "".join(f"<li>{esc(note)}</li>" for note in proposal.review_notes)
            + "</ul><div class='pair'><section><h2>Before</h2>"
            + preview_frame(page.html, "Before") + "</section><section><h2>After</h2>"
            + preview_frame(after, "After") + "</section></div>"
            + "<details><summary>Exact HTML diff</summary><pre>" + esc(diff) + "</pre></details>"
            + "<details><summary>Preservation checks</summary><pre>"
            + esc(json.dumps(proposal.checks, indent=2)) + "</pre></details>"
        )
        (folder / "preview.html").write_text(document(page.title, body), encoding="utf-8")
    plan = FormattingPlan(snapshot_captured_at=snapshot.captured_at, pages=pages)
    write_json(output / "plan.json", plan)
    counts = dict(Counter(page.status for page in pages))
    body = (
        "<h1>Wiki formatting proposals -- approval required</h1>"
        "<p>No BookStack, Qdrant or deployment writes. Each original page and proposed version "
        "is saved locally. Page locations, names, wording, rules, links, IDs, table cells and "
        "cell spans are preserved. Only tag semantics and header scope are proposed.</p>"
        "<p>Ambiguous headings/tables remain for manual review. Unchanged does not mean structurally "
        "perfect. Existing broken links are not repaired. No new section titles are invented.</p>"
        f"<p>Snapshot: {html.escape(snapshot.captured_at)}</p><pre>"
        + html.escape(json.dumps(counts, indent=2))
        + "</pre><p>Review and approve before any live edits. A future apply step must recheck "
        "page hashes against BookStack and refuse stale proposals.</p>"
        "<table><tr><th>Page</th><th>Location</th><th>Status</th><th>Tag edits</th><th>Review notes</th></tr>"
    )
    for page in sorted(pages, key=lambda p: (p.status != "proposed", p.book, p.title)):
        body += (
            f"<tr><td><a href='pages/{page.page_id}/preview.html'>{html.escape(page.title)}</a></td>"
            f"<td>{html.escape(page.book)} / {html.escape(page.chapter or '')}</td>"
            f"<td>{page.status}</td><td>{len(page.edits)}</td><td>"
            + "<br>".join(html.escape(note) for note in page.review_notes) + "</td></tr>"
        )
    (output / "index.html").write_text(document("Wiki formatting review", body + "</table>"), encoding="utf-8")
    return plan


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Reuse a snapshot with no network calls.")
    parser.add_argument("--output", type=Path, required=True, help="New local preview directory.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory; existing files will not be overwritten.")
    snapshot = (
        WikiSnapshot.model_validate_json(args.snapshot.read_text(encoding="utf-8"))
        if args.snapshot else await capture_wiki(AuditSettings())
    )
    plan = write_preview(snapshot, args.output)
    print(json.dumps({
        "pages": len(plan.pages), "status_counts": dict(Counter(p.status for p in plan.pages)),
        "pages_with_manual_review": sum(bool(p.review_notes) for p in plan.pages),
        "output": str(args.output), "mode": plan.mode,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
