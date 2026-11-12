"""Offline triage and repair previews for the wiki's remaining manual-review pages."""

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from allies_bot.audit import Element, PageSnapshot, WikiSnapshot
from allies_bot.wiki_format import (
    Edit,
    PageProposal,
    apply_edits,
    document,
    parse_positioned,
    preservation_checks,
    propose_page,
    write_preview,
)

# These are source-observed labels, not guessed titles for unlabelled data.
COLUMN_LABELS = {
    ("", "Height & Width occupied"),
    ("a number of at least:", "gives this many Adjustments:"),
    ("a difference of at least:", "levels or Presence", "is termed as:"),
    ("a difference of at least:", "gives this bonus:"),
    ("a difference of at least:", "gives this many Adjustments:"),
    ("a difference of at least:", "is termed as:"),
    ("Tier", "Maximum Social Attribute Values"),
    ("Action", "AP Cost", "Additional Cost"),
    ("Spiritual Effect Tier", "Complexity"),
    ("Area Presence", "affect on Social Interactions"),
    ("Character Tier", "Base Magnitude"),
    ("Pseudo-Dionysis & Aquinas", "Maimonides"),
    ("Lowest Player", "Group of 3", "Group of 7", "Group of 12"),
    ("Fervor invested", "Skill increase", "Complexity", "Exertion"),
    ("How Relevant are the Spiritual Effects or War Songs in Use?", "Enhancement"),
    ("d12", "Strength of each item (in BP)"),
}
ORDER_HEADERS = {
    ("", "Warrior", "Messenger", "Min. Spirit", "Minstrel", "W-Me", "W-MS", "W-Mi"),
    ("", "Me-MS", "Me-Mi", "Mi-MS", "Tri-W", "Tri-Me", "Tri-MS", "Tri-Mi", "Quad"),
}
AXIS_PAGES = {24, 92, 219}
MINION_PAGES = {350, 351, 352, 353, 374, 375, 376}
EFFECT_ROWS = {45, 138}


class ReviewItem(BaseModel):
    kind: str
    evidence: str
    recommendation: str
    table: int | None = None


class TriagePage(BaseModel):
    page_id: int
    title: str
    url: str
    outcome: str
    tag_edits: int
    items: list[ReviewItem]


def cells_of(table: Element) -> list[list[Element]]:
    return [
        [cell for cell in row.children if isinstance(cell, Element) and cell.tag in {"td", "th"}]
        for row in table.descendants({"tr"})
    ]


def table_pattern(page: PageSnapshot, rows: list[list[Element]]) -> str | None:
    if not rows or len(rows) < 2 or not rows[0]:
        return None
    if len({len(row) for row in rows}) != 1 or any(
        cell.attrs.get("rowspan", "1") != "1" or cell.attrs.get("colspan", "1") != "1"
        for row in rows for cell in row
    ):
        return None
    first = tuple(cell.text() for cell in rows[0])
    if first in COLUMN_LABELS or (page.id in {22, 90} and first in ORDER_HEADERS):
        return "columns"
    if (
        len(first) == 2
        and all(re.fullmatch(r"Mastery Level [1-5]", row[0].text()) for row in rows)
        and len({row[0].text() for row in rows}) == len(rows)
    ):
        return "row_labels"
    if page.id in AXIS_PAGES and first[0] in {
        "Might", "[2(Mi) + Co]", "Control", "Swiftness", "Endurance", "Resolve",
        "Passion", "Rank", "Minion Level", "Power",
    } and all(re.fullmatch(r"[<+≈\d.,k\s-]+", value) for value in first[1:]):
        return "axes"
    if page.id in MINION_PAGES and len(first) == 2 and re.fullmatch(r"ML:\d+", first[1]):
        return "stat_card"
    if (
        page.id in EFFECT_ROWS and len(first) == 4
        and any(word in first[1] for word in ("Affecting", "Manifesting"))
    ):
        return "row_labels"
    if page.id == 76 and first[0] == "Persuaded" and len(first) == 2:
        return "row_labels"
    return None


def inspect_page(page: PageSnapshot) -> tuple[PageProposal, str, TriagePage]:
    original, _ = propose_page(page)
    parser = parse_positioned(page.html)
    edits: list[Edit] = []
    items: list[ReviewItem] = []
    touched: set[int] = set()

    def mark(cell: Element, scope: str) -> None:
        if id(cell) in touched or (cell.tag == "th" and cell.attrs.get("scope") == scope):
            return
        # Never replace pre-existing, contradictory header semantics without review.
        if cell.attrs.get("scope") not in {None, scope}:
            items.append(ReviewItem(
                kind="decision", evidence=cell.text(),
                recommendation="Existing scope differs from proposed association; inspect manually.",
            ))
            return
        touched.add(id(cell))
        span = parser.spans[id(cell)]
        if span.closing_start is None or span.end is None:
            raise ValueError(f"Unclosed table cell on page {page.id}.")
        opening = page.html[span.start:span.opening_end]
        replacement = re.sub(r"^<\s*(td|th)\b", "<th", opening, count=1, flags=re.IGNORECASE)
        if "scope" not in cell.attrs:
            replacement = replacement[:-1] + f' scope="{scope}">'
        reason = "Mark reviewed table associations without adding or moving any cell text."
        if replacement != opening:
            edits.append(Edit(
                start=span.start, end=span.opening_end, before=opening, after=replacement, reason=reason,
            ))
        closing = page.html[span.closing_start:span.end]
        replacement = re.sub(r"(</\s*)(td|th)\b", r"\1th", closing, count=1, flags=re.IGNORECASE)
        if replacement != closing:
            edits.append(Edit(
                start=span.closing_start, end=span.end, before=closing, after=replacement, reason=reason,
            ))

    for index, table in enumerate(parser.root.descendants({"table"}), 1):
        rows = cells_of(table)
        nested = bool(table.descendants({"table"}))
        pattern = None if nested else table_pattern(page, rows)
        if pattern:
            if pattern in {"columns", "axes"}:
                for cell in rows[0]:
                    mark(cell, "col")
            if pattern in {"axes", "row_labels", "stat_card"}:
                for row in rows if pattern != "axes" else rows[1:]:
                    if row[0].text():
                        mark(row[0], "row")
            items.append(ReviewItem(
                kind="repair", table=index,
                evidence=" | ".join(cell.text() for cell in rows[0]),
                recommendation={
                    "columns": "Use the existing first-row labels as column headers.",
                    "axes": "Mark existing threshold/rank columns and attribute rows; keep every number.",
                    "row_labels": "Mark the existing first cell of each row as a row header; "
                                  "the first row is data, not a column-header row.",
                    "stat_card": "Keep this as a stat card; mark its existing left-hand labels "
                                 "as row headers, not its first data row as column headings.",
                }[pattern],
            ))
        elif rows and (
            nested or len({len(row) for row in rows}) > 1
            or any(c.attrs.get("rowspan", "1") != "1" or c.attrs.get("colspan", "1") != "1"
                   for row in rows for c in row)
        ):
            items.append(ReviewItem(
                kind="decision", table=index,
                evidence=json.dumps([[c.text() for c in row] for row in rows[:3]], ensure_ascii=False),
                recommendation="Keep the merged/group-title layout pending review. Confirm its "
                               "logical column groups before changing scope or splitting the table.",
            ))
        elif rows and not all(cell.tag == "th" for cell in rows[0]):
            items.append(ReviewItem(
                kind="decision", table=index,
                evidence=json.dumps([[c.text() for c in row] for row in rows[:2]], ensure_ascii=False),
                recommendation="Determine whether this is a caption, data row, or header. "
                               "Do not invent names for blank columns.",
            ))
    for note in original.review_notes:
        if note.startswith("Long page"):
            items.append(ReviewItem(
                kind="editorial_optional", evidence=note,
                recommendation="Length alone is not a formatting error. Keep current prose. "
                               "New section titles/boundaries would be editorial changes "
                               "requiring a separate proposal, not automatic repair.",
            ))
        elif note.startswith("Possible visual"):
            items.append(ReviewItem(
                kind="keep_emphasis", evidence=note,
                recommendation="Keep as emphasis for now. Formulas, attribute values and bold "
                               "list-item labels are not automatically section headings.",
            ))
        elif note.startswith("Heading contains media"):
            items.append(ReviewItem(
                kind="decision", evidence=note,
                recommendation="Retain the image. Review whether its heading wrapper is intentional; "
                               "do not remove or add text automatically.",
            ))
    decision_guidance = {
        21: (
            "Suggested next proposal: split each mixed table into Social Attributes and Motives "
            "sections using those existing labels. Retain the existing (Skill)/(Comp) labels "
            "for Motives; confirm a label for the Social Attributes numeric column. "
            "Do not fill its blank columns by borrowing the Motives headers."
        ),
        102: (
            "Suggested next proposal: change only the image-only heading wrapper to a normal "
            "block while retaining its image, link and bookmark. No replacement heading text needed."
        ),
        377: (
            "Suggested next proposal: retain grouped encounter tables and their numbers; "
            "treat the spanning first rows as table/group titles, not column headers. "
            "Keep the Tactical/Non-tactical variants distinct. No flattening or new rule labels."
        ),
        378: (
            "Suggested next proposal: retain Archon and Regular Enemy as separate group titles; "
            "mark their second rows (d12 / # of items) as actual column headers. "
            "Preserve the merged title cells and every roll range."
        ),
    }
    if page.id in decision_guidance:
        for item in items:
            if item.kind == "decision":
                item.recommendation = decision_guidance[page.id]
    if parser.errors:
        edits.clear()
        items.append(ReviewItem(
            kind="decision", evidence="; ".join(parser.errors),
            recommendation="Malformed HTML blocks automatic changes.",
        ))
    after = apply_edits(page.html, edits)
    checks = preservation_checks(page.html, after)
    if edits and not all(checks.values()):
        raise ValueError(f"Triage preservation failed on page {page.id}: {checks}")
    decisions = [item for item in items if item.kind == "decision"]
    outcome = "repair_and_decisions" if edits and decisions else (
        "repair_proposed" if edits else "decision_needed" if decisions else "retain_current_structure"
    )
    proposal = PageProposal(
        page_id=page.id, title=page.title, url=page.url, book=page.book, chapter=page.chapter,
        source_updated_at=page.updated_at, before_sha256=page.content_sha256,
        after_sha256=hashlib.sha256(after.encode()).hexdigest(),
        status="blocked" if parser.errors else "proposed" if edits else "unchanged",
        edits=sorted(edits, key=lambda edit: edit.start),
        review_notes=[f"{item.kind}: {item.recommendation}" for item in decisions],
        checks=checks,
    )
    return proposal, after, TriagePage(
        page_id=page.id, title=page.title, url=page.url,
        outcome=outcome, tag_edits=len(edits), items=items,
    )


def propose_reviewed_page(page: PageSnapshot) -> tuple[PageProposal, str]:
    proposal, after, _ = inspect_page(page)
    return proposal, after


def triage(snapshot: WikiSnapshot, output: Path) -> list[TriagePage]:
    flagged = [page for page in snapshot.pages if propose_page(page)[0].review_notes]
    scoped = WikiSnapshot(
        captured_at=snapshot.captured_at, base_url=snapshot.base_url, pages=flagged,
    )
    write_preview(scoped, output, proposer=propose_reviewed_page)
    reviews = [inspect_page(page)[2] for page in flagged]
    data = {
        "mode": "preview_only", "snapshot_captured_at": snapshot.captured_at,
        "pages_scanned": len(snapshot.pages), "flagged_pages": len(reviews),
        "outcomes": dict(Counter(page.outcome for page in reviews)),
        "pages": [page.model_dump() for page in reviews],
    }
    (output / "triage.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    body = (
        "<h1>Remaining wiki pages: focused triage</h1>"
        "<p>Read-only. No live edits. All original flagged pages are accounted for. "
        "Repairs below are new proposals, not covered by the earlier table approval.</p>"
        "<p>Existing prose, formulas and stat cards are not errors simply because they lack "
        "column headings or section titles. No names, rules, values, links, IDs, cell spans "
        "or page locations are changed. All proposed patches passed preservation and "
        "idempotence checks.</p><pre>" + html.escape(json.dumps(data["outcomes"], indent=2))
        + "</pre><h2>Decisions needed</h2>"
    )
    for review in reviews:
        decisions = [item for item in review.items if item.kind == "decision"]
        if not decisions:
            continue
        body += (
            f"<h3><a href='pages/{review.page_id}/preview.html'>{html.escape(review.title)}</a></h3>"
        )
        for item in decisions:
            body += (
                f"<p>Table {item.table if item.table is not None else 'n/a'}: "
                f"{html.escape(item.recommendation)}</p><pre>{html.escape(item.evidence)}</pre>"
            )
    body += "<h2>All flagged pages</h2><table><tr><th>Page</th><th>Outcome</th><th>Findings</th></tr>"
    for review in sorted(reviews, key=lambda p: (p.outcome, p.title)):
        body += (
            f"<tr><td><a href='pages/{review.page_id}/preview.html'>{html.escape(review.title)}</a>"
            f"</td><td>{html.escape(review.outcome)}</td><td>"
        )
        for item in review.items:
            body += (
                f"<details><summary>{html.escape(item.kind)}"
                f"{' / table ' + str(item.table) if item.table is not None else ''}</summary>"
                f"<p>{html.escape(item.recommendation)}</p>"
                f"<pre>{html.escape(item.evidence)}</pre></details>"
            )
        body += "</td></tr>"
    body += "</table><p><a href='previews.html'>Before/after preview index</a></p>"
    (output / "index.html").rename(output / "previews.html")
    (output / "index.html").write_text(document("Manual wiki triage", body), encoding="utf-8")
    # Existing per-page navigation now lands on the focused triage, not a generic index.
    print(json.dumps({key: value for key, value in data.items() if key != "pages"}, indent=2))
    return reviews


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = WikiSnapshot.model_validate_json(args.snapshot.read_text(encoding="utf-8"))
    triage(snapshot, args.output)


if __name__ == "__main__":
    main()
