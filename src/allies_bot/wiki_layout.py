"""Source-scoped proposals for the three explicitly approved layout decisions."""

import hashlib
import re

from allies_bot.audit import PageSnapshot
from allies_bot.wiki_format import (
    Edit,
    PageProposal,
    apply_edits,
    parse_positioned,
)


def propose_layout(page: PageSnapshot) -> tuple[PageProposal, str]:
    parser = parse_positioned(page.html)
    if parser.errors or page.id not in {102, 377, 378}:
        raise ValueError(f"Page {page.id} is not a supported approved layout.")
    edits: list[Edit] = []

    def replace(start: int, end: int, after: str) -> None:
        edits.append(Edit(
            start=start, end=end, before=page.html[start:end], after=after,
            reason="Apply the explicitly approved source-specific layout decision.",
        ))

    if page.id == 102:
        for node in parser.root.descendants({"h5"}):
            if node.attrs.get("id") != "bkmrk-" or node.text():
                continue
            if len(node.descendants({"img"})) != 1:
                raise ValueError("Epic Blessings image wrapper differs from reviewed source.")
            span = parser.spans[id(node)]
            replace(span.start, span.opening_end,
                    re.sub(r"^<h5\b", "<div", page.html[span.start:span.opening_end]))
            replace(span.closing_start, span.end, "</div>")
    else:
        tables = parser.root.descendants({"table"})
        expected = (
            ["For each Elohim", "Average, Non-tactical Players", "Tactical, Experienced Players",
             "For each unholy Elohim", "For each Unholy Elohim",
             "Average, Non-tactical Players", "Tactical, Experienced Players"]
            if page.id == 377 else ["Archon", "Regular Enemy"]
        )
        if len(tables) != (7 if page.id == 377 else 3):
            raise ValueError("Approved table count differs from reviewed source.")
        for table, title in zip(tables, expected):
            captions = table.descendants({"caption"})
            if page.id == 377 and captions:
                if len(captions) != 1 or captions[0].text() != title:
                    raise ValueError("Existing caption differs from approved title.")
                continue
            rows = table.descendants({"tr"})
            cells = [
                child for child in rows[0].children
                if not isinstance(child, str) and child.tag in {"td", "th"}
            ]
            if cells[0].text() != title or any(cell.text() for cell in cells[1:]):
                raise ValueError(f"Approved table title differs: {title}")
            if page.id == 377:
                row_span = parser.spans[id(rows[0])]
                cell_span = parser.spans[id(cells[0])]
                table_span = parser.spans[id(table)]
                if any(set(node.attrs) - {"style", "colspan"} for node in [rows[0], *cells]):
                    raise ValueError("Title row has attributes requiring separate review.")
                inner = page.html[cell_span.opening_end:cell_span.closing_start]
                replace(row_span.start, row_span.end, "")
                replace(table_span.opening_end, table_span.opening_end,
                        f'<caption style="caption-side: top; text-align: center;">{inner}</caption>')
            else:
                headers = [
                    child for child in rows[1].children
                    if not isinstance(child, str) and child.tag in {"td", "th"}
                ]
                if [cell.text() for cell in headers] != ["d12", "# of items"]:
                    raise ValueError("Enemy item column labels differ from reviewed source.")
                for cell in headers:
                    if cell.attrs.get("scope") not in {None, "col"}:
                        raise ValueError("Conflicting existing table scope.")
                    if cell.tag == "th" and cell.attrs.get("scope") == "col":
                        continue
                    span = parser.spans[id(cell)]
                    opening = re.sub(r"^<(td|th)\b", "<th",
                                     page.html[span.start:span.opening_end])
                    if "scope" not in cell.attrs:
                        opening = opening[:-1] + ' scope="col">'
                    replace(span.start, span.opening_end, opening)
                    replace(span.closing_start, span.end, "</th>")
    after = apply_edits(page.html, edits)
    new = parse_positioned(after)

    def targets(tree):
        return {
            (key, node.attrs[key]) for node in tree.root.elements()
            for key in ("id", "name", "href", "src") if key in node.attrs
        }

    def data_rows(tree):
        return [
            page_source[span.start:span.end]
            for table in tree.root.descendants({"table"})
            for row in table.descendants({"tr"})[
                (1 if page.id == 377 and tree is parser and not table.descendants({"caption"}) else 0):
            ]
            if (span := tree.spans[id(row)]).end is not None
            for page_source in [page.html if tree is parser else after]
        ]

    checks = {
        "balanced_html": not new.errors,
        "visible_text_identical": parser.root.text() == new.root.text(),
        "links_images_and_anchors_identical": targets(parser) == targets(new),
        "data_rows_unchanged": page.id != 377 or data_rows(parser) == data_rows(new),
    }
    if not all(checks.values()):
        raise ValueError(f"Layout preservation failed on page {page.id}: {checks}")
    return PageProposal(
        page_id=page.id, title=page.title, url=page.url, book=page.book, chapter=page.chapter,
        source_updated_at=page.updated_at, before_sha256=page.content_sha256,
        after_sha256=hashlib.sha256(after.encode()).hexdigest(),
        status="proposed" if edits else "unchanged",
        edits=sorted(edits, key=lambda edit: edit.start), review_notes=[], checks=checks,
    ), after
