"""Tests for app.tables: pipe-table parsing and structural cell lookup.

Includes tests against the real corpus (app/data/sections.json) so the parser
stays honest about the actual shape of the zoning ordinance's tables.

Run with: uv run pytest
"""

from __future__ import annotations

import pytest

from app import law_tools
from app.tables import LookupResult, lookup_cell, parse_tables

SYNTHETIC = """\
Some prose before the table.

| Table 9.9 Widgets |  |  |
| --- | --- | --- |
|  | Small Lot | Large Lot |
| Gizmo | P | N |
| Doodad | SP | P |
| P - Permitted • SP - Special Permit Required • N - Not Permitted |  |  |

Prose after.
"""


def test_parse_synthetic_matrix():
    tables = parse_tables(SYNTHETIC)
    assert len(tables) == 1
    t = tables[0]
    assert t.title == "Table 9.9 Widgets"
    assert t.headers == ["", "Small Lot", "Large Lot"]
    assert t.rows == [["Gizmo", "P", "N"], ["Doodad", "SP", "P"]]
    assert t.footers == ["P - Permitted • SP - Special Permit Required • N - Not Permitted"]


def test_lookup_exact_and_loose_matching():
    assert lookup_cell(SYNTHETIC, "Table 9.9 Widgets", "Gizmo", "Small Lot").value == "P"
    # Partial table title, case-insensitive labels.
    assert lookup_cell(SYNTHETIC, "table 9.9", "gizmo", "large lot").value == "N"
    assert lookup_cell(SYNTHETIC, "Table 9.9", "Doodad", "Small Lot").value == "SP"


def test_lookup_failure_reasons():
    assert lookup_cell(SYNTHETIC, "Table 8.8", "Gizmo", "Small Lot").reason == "table_not_found"
    assert lookup_cell(SYNTHETIC, "Table 9.9", "Sprocket", "Small Lot").reason == "row_not_found"
    assert lookup_cell(SYNTHETIC, "Table 9.9", "Gizmo", "Tiny Lot").reason == "column_not_found"


def test_lookup_value_mismatch_is_callers_problem():
    # lookup_cell reports what the cell holds; comparing against the model's
    # claimed value is the verifier's job.
    result = lookup_cell(SYNTHETIC, "Table 9.9", "Gizmo", "Small Lot")
    assert result == LookupResult("P", None, table_title="Table 9.9 Widgets")


def test_legend_row_is_footer_not_data():
    tables = parse_tables(SYNTHETIC)
    labels = [r[0] for r in tables[0].rows]
    assert not any("P - Permitted" in label for label in labels)


def test_titleless_layout_grid_never_matches():
    grid = (
        "|  |  |  |\n"
        "| --- | --- | --- |\n"
        "| [Image 1] | Projection (max) | 3 ft |\n"
    )
    assert lookup_cell(grid, "anything", "Projection (max)", "x").reason == "table_not_found"


# --- against the real corpus ---

_HAVE_CORPUS = bool(law_tools.SECTIONS)
requires_corpus = pytest.mark.skipif(not _HAVE_CORPUS, reason="sections.json not built")


@requires_corpus
def test_zon471_building_components_matrix():
    text = law_tools.SECTIONS["zon:471"]["text"]
    assert lookup_cell(text, "Table 3.1.13", "Projecting Porch", "Detached House").value == "P"
    assert lookup_cell(text, "Table 3.1.13", "Engaged Porch", "Backyard Cottage").value == "N"
    assert lookup_cell(text, "Table 3.1.13", "Engaged Porch", "Detached Triple Decker").value == "N"
    # Full title works too.
    assert (
        lookup_cell(text, "Table 3.1.13 Building Components", "Rear Addition", "Duplex").value
        == "P"
    )


@requires_corpus
def test_zon484_urban_residence_variant():
    text = law_tools.SECTIONS["zon:484"]["text"]
    assert lookup_cell(text, "Table 3.2.12", "Projecting Porch", "Apartment House").value == "P"


@requires_corpus
def test_zon471_layout_grids_do_not_false_positive():
    # The dimensional standards live in title-less layout grids; a lookup
    # naming a nonexistent table must not match them.
    text = law_tools.SECTIONS["zon:471"]["text"]
    assert lookup_cell(text, "Dimensions", "Projection (max)", "Front").value is None
