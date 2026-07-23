"""Markdown pipe-table parsing and structural cell lookup.

Backs the table-lookup citation type: the agent cites a cell as
(table, row, column, value) and ``lookup_cell`` confirms the named cell
holds that value in the cited section's text.

The corpus (see zon:471 "3.1.13 Building Components") has two kinds of
pipe tables:

- Real matrices: a title row (one non-empty cell), a ``| --- |`` separator,
  a header row whose first cell is empty and the rest are column labels,
  data rows whose first cell is the row label, and often a trailing legend
  row ("P - Permitted • SP - Special Permit Required • N - Not Permitted")
  with a single non-empty cell.
- Layout grids: title-less tables used for page layout, interleaving prose,
  image placeholders, and label/value pairs from two side-by-side panels.
  These parse fine but lookups won't match them (no title, no headers) —
  by design; the agent falls back to short verbatim quotes there.

Pure stdlib, no dependency on app.agent (which imports this module).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_SEPARATOR_CELL_RE = re.compile(r"^:?-{2,}:?$")

# Mirrors the punctuation unification in app.agent._normalize, minus the
# markdown-emphasis stripping (table cells don't carry emphasis markers).
_CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "﻿": "", "​": "",
}
_TRANSLATION = {ord(k): v for k, v in _CHAR_MAP.items()}


def _norm(text: str) -> str:
    """Normalize a cell or label for comparison: NFKC, punctuation-glyph
    unification, whitespace collapse, casefold."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_TRANSLATION)
    return " ".join(text.split()).casefold()


@dataclass
class Table:
    title: str = ""
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    footers: list[str] = field(default_factory=list)


@dataclass
class LookupResult:
    value: str | None
    reason: str | None  # None on success; else table_not_found / row_not_found
    #                     / column_not_found / empty_cell
    table_title: str | None = None  # full title of the matched table


def _parse_row(line: str) -> list[str]:
    inner = line.strip()
    inner = inner.removeprefix("|").removesuffix("|")
    return [cell.strip() for cell in inner.split("|")]


def _is_separator(cells: list[str]) -> bool:
    nonempty = [c for c in cells if c]
    return bool(nonempty) and all(_SEPARATOR_CELL_RE.match(c) for c in nonempty)


def _single_nonempty(cells: list[str]) -> str | None:
    nonempty = [c for c in cells if c]
    return nonempty[0] if len(nonempty) == 1 else None


def parse_tables(text: str) -> list[Table]:
    """Parse every pipe table in ``text`` into Table records."""
    tables: list[Table] = []
    block: list[list[str]] = []

    def flush() -> None:
        nonlocal block
        if block:
            tables.append(_parse_block(block))
            block = []

    for line in text.split("\n"):
        if line.strip().startswith("|"):
            cells = _parse_row(line)
            if not _is_separator(cells):
                block.append(cells)
        else:
            flush()
    flush()
    return tables


def _parse_block(rows: list[list[str]]) -> Table:
    table = Table()
    i = 0

    if i < len(rows) and (title := _single_nonempty(rows[i])) is not None:
        table.title = title
        i += 1

    if i < len(rows):
        cells = rows[i]
        if cells and not cells[0] and sum(1 for c in cells if c) >= 2:
            table.headers = cells
            i += 1

    for cells in rows[i:]:
        single = _single_nonempty(cells)
        if single is not None and table.rows:
            table.footers.append(single)
        else:
            table.rows.append(cells)
    return table


def _match(query: str, label: str) -> bool:
    """Loose label match: normalized equality or substring either way (so
    "Table 3.1.13" matches "Table 3.1.13 Building Components")."""
    nq, nl = _norm(query), _norm(label)
    if not nq or not nl:
        return False
    return nq == nl or nq in nl or nl in nq


def lookup_cell(text: str, table: str, row: str, column: str) -> LookupResult:
    """Find the cell at (row label, column header) in the named table.

    Table is matched by title, row by first-column label, column by header
    label — all via loose normalized matching, exact matches preferred.
    Returns the cell's raw text on success, or a reason string on failure.
    """
    candidates = [t for t in parse_tables(text) if t.title and _match(table, t.title)]
    if not candidates:
        return LookupResult(None, "table_not_found")

    last_reason = "row_not_found"
    for tab in candidates:
        matches = [r for r in tab.rows if r and _match(row, r[0])]
        exact = [r for r in matches if _norm(r[0]) == _norm(row)]
        if exact:
            matches = exact
        if not matches:
            continue
        row_cells = matches[0]

        col_indices = [
            idx for idx, h in enumerate(tab.headers) if h and _match(column, h)
        ]
        exact_cols = [idx for idx in col_indices if _norm(tab.headers[idx]) == _norm(column)]
        if exact_cols:
            col_indices = exact_cols
        if not col_indices:
            last_reason = "column_not_found"
            continue
        idx = col_indices[0]

        if idx >= len(row_cells) or not row_cells[idx]:
            last_reason = "empty_cell"
            continue
        return LookupResult(row_cells[idx], None, table_title=tab.title)

    return LookupResult(None, last_reason)
