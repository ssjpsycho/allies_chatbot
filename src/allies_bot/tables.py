"""Render HTML tables as independently labelled facts for retrieval."""

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser


@dataclass
class Cell:
    tag: str
    scope: str
    colspan: int
    rowspan: int
    parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join("".join(self.parts).split())


@dataclass
class Table:
    rows: list[list[Cell]] = field(default_factory=list)
    caption: list[str] = field(default_factory=list)
    in_caption: bool = False
    cell: Cell | None = None

    def append(self, text: str) -> None:
        if self.cell is not None:
            self.cell.parts.append(text)
        elif self.in_caption:
            self.caption.append(text)


class TableExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[Table] = []
        self.output: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.stack.append(Table())
        elif self.stack:
            table = self.stack[-1]
            values = dict(attrs)
            if tag == "tr":
                table.rows.append([])
            elif tag in {"td", "th"}:
                if not table.rows:
                    table.rows.append([])

                def span(name: str) -> int:
                    raw = values.get(name) or "1"
                    if not raw.isdigit() or not 1 <= int(raw) <= 1000:
                        raise ValueError(f"Unsupported table {name}: {raw!r}")
                    return int(raw)

                table.cell = Cell(tag, values.get("scope") or "", span("colspan"), span("rowspan"))
                table.rows[-1].append(table.cell)
            elif tag == "caption":
                table.in_caption = True
            elif tag in {"br", "p", "div", "li"}:
                table.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if not self.stack:
            return
        table = self.stack[-1]
        if tag == "table":
            rendered = render_table(self.stack.pop())
            if self.stack:
                self.stack[-1].append(" " + " ".join(rendered) + " ")
            else:
                self.output.extend(rendered)
        elif tag in {"td", "th"}:
            table.cell = None
        elif tag == "caption":
            table.in_caption = False
        elif tag in {"p", "div", "li"}:
            table.append(" ")

    def handle_data(self, data: str) -> None:
        if self.stack:
            self.stack[-1].append(data)


def render_table(table: Table) -> list[str]:
    # Expand only actual spans. Blank cells do not inherit another row's rules.
    grid: list[dict[int, Cell]] = []
    pending: dict[int, tuple[Cell, int]] = {}
    for row in table.rows:
        occupied = {col: cell for col, (cell, _) in pending.items()}
        pending = {col: (cell, left - 1) for col, (cell, left) in pending.items() if left > 1}
        col = 0
        for cell in row:
            while any(col + offset in occupied for offset in range(cell.colspan)):
                col += 1
            for offset in range(cell.colspan):
                occupied[col + offset] = cell
                if cell.rowspan > 1:
                    pending[col + offset] = (cell, cell.rowspan - 1)
            col += cell.colspan
        grid.append(occupied)
    width = max((max(row, default=-1) + 1 for row in grid), default=0)
    base_caption = " ".join("".join(table.caption).split())
    if not base_caption and table.rows and len(table.rows[0]) == 2:
        first, second = table.rows[0]
        if first.text and re.fullmatch(r"ML:\d+", second.text):
            base_caption = f"{first.text} ({second.text})"
    caption = base_caption
    columns: dict[int, list[str]] = {}
    output: list[str] = []
    previous_headers = False
    for row in grid:
        unique = list({id(cell): cell for cell in row.values()}.values())
        if not unique:
            continue
        if len(unique) == 1 and unique[0].colspan == width and width > 1:
            caption = " / ".join(part for part in (base_caption, unique[0].text) if part)
            continue
        header_row = all(
            (not cell.text and row.get(0) is cell)
            or (cell.tag == "th" and cell.scope not in {"row", "rowgroup"})
            for cell in unique
        ) and any(cell.tag == "th" and cell.text for cell in unique)
        if header_row:
            if not previous_headers:
                columns.clear()
            for col, cell in row.items():
                labels = columns.setdefault(col, [])
                if cell.text and cell.text not in labels:
                    labels.append(cell.text)
            previous_headers = True
            continue
        previous_headers = False
        row_headers = [
            cell.text for cell in unique
            if cell.tag == "th" and cell.scope in {"", "row", "rowgroup"} and cell.text
        ]
        if not row_headers and 0 in row and row[0].text:
            row_headers = [row[0].text]
        context = ("Table: " + caption) if caption else "Table"
        if row_headers:
            context += "; Row: " + " / ".join(row_headers)
        for cell in unique:
            labels = list(dict.fromkeys(
                " / ".join(columns[col]) if columns.get(col) else f"Column {col + 1}"
                for col, owner in row.items() if owner is cell
            ))
            label = " / ".join(labels)
            if not re.fullmatch(r"Column \d+", label):
                label = "Column: " + label
            output.append(f"{context}; {label}\t{cell.text or '(empty)'}")
    if not output and columns:
        output.append(f"Table: {caption or '(untitled)'}; Headers\t"
                      + " | ".join(" / ".join(labels) for labels in columns.values()))
    elif not output and caption:
        output.append(f"Table title\t{caption}")
    return output
