"""Deterministic regression tests for app.agent's answer-shape defenses.

These cover the pseudo-XML "spill" failure mode observed 2026-07-10: the model
calls submit_answer but writes the whole tagged document inside the
answer_markdown string ("answer…</answer_markdown><citations>[…]"), leaving
the real citations array empty — which used to stream raw JSON to the reader
and trip the no-citations guardrail retry (visible as the answer resetting and
regenerating). No API access required.

Run with: uv run pytest
"""

from __future__ import annotations

from app.agent import (
    AnswerStreamGuard,
    _build_answer_from_submit,
    _extract_trailing_note,
    _parse_pseudo_submit,
    _salvage_spilled_payload,
)

# Mirrors the real 2026-07-10 incident payload: closing </answer_markdown>
# tag, a <citations> JSON array with NO closing tag, empty citations field.
SPILLED_ANSWER = (
    "Yes, you can keep backyard hens — with a Board of Health permit.\n\n"
    "**Caveat:** Zoning may add district-specific restrictions."
    '</answer_markdown>\n<citations>[{"section_key":"coo:2180","quote":'
    '"No person shall keep hens on their premises without obtaining a '
    'permit from the Board of Health."}]'
)


def test_salvage_recovers_spilled_payload():
    payload = {
        "answer_markdown": SPILLED_ANSWER,
        "citations": [],
        "confidence": "high",
    }
    fixed = _salvage_spilled_payload(payload)

    assert "</answer_markdown>" not in fixed["answer_markdown"]
    assert "<citations>" not in fixed["answer_markdown"]
    assert "section_key" not in fixed["answer_markdown"]
    assert fixed["answer_markdown"].startswith("Yes, you can keep backyard hens")
    # The spilled citations are recovered so the guardrail retry never fires.
    assert fixed["citations"] == [
        {
            "section_key": "coo:2180",
            "quote": (
                "No person shall keep hens on their premises without "
                "obtaining a permit from the Board of Health."
            ),
        }
    ]
    # No <confidence> tag in the spill: the field the model set is kept.
    assert fixed["confidence"] == "high"


def test_salvage_without_closing_answer_tag():
    payload = {
        "answer_markdown": (
            'The answer body.\n<citations>[{"section_key":"zon:1","quote":"q"}]'
        ),
        "citations": [],
        "confidence": "medium",
    }
    fixed = _salvage_spilled_payload(payload)
    assert fixed["answer_markdown"] == "The answer body."
    assert fixed["citations"] == [{"section_key": "zon:1", "quote": "q"}]


def test_salvage_keeps_existing_citations():
    real = [{"section_key": "coo:1", "quote": "real"}]
    payload = {
        "answer_markdown": "Body.</answer_markdown>",
        "citations": real,
        "confidence": "high",
    }
    fixed = _salvage_spilled_payload(payload)
    assert fixed["answer_markdown"] == "Body."
    assert fixed["citations"] == real


def test_salvage_leaves_clean_payload_untouched():
    payload = {
        "answer_markdown": "A perfectly normal answer with a < b comparison.",
        "citations": [{"section_key": "coo:1", "quote": "q"}],
        "confidence": "high",
    }
    assert _salvage_spilled_payload(payload) is payload


def test_salvage_merges_spilled_citations_with_existing():
    # Seen live 2026-07-10 (noise-at-night eval): the real citations array
    # held one (unverifiable) citation while the spill carried the full set.
    # Both must survive so verification can sort them out.
    payload = {
        "answer_markdown": (
            "Quiet hours vary by activity.</answer_markdown>\n"
            '<citations>[{"section_key":"coo:822","quote":"embedded one"},'
            '{"section_key":"coo:823","quote":"embedded two"}]'
        ),
        "citations": [{"section_key": "coo:822", "quote": "from the array"}],
        "confidence": "high",
    }
    fixed = _salvage_spilled_payload(payload)
    assert fixed["citations"] == [
        {"section_key": "coo:822", "quote": "from the array"},
        {"section_key": "coo:822", "quote": "embedded one"},
        {"section_key": "coo:823", "quote": "embedded two"},
    ]


def test_salvage_parses_xml_element_citations():
    # Also seen live 2026-07-10: citations spilled as XML elements, not JSON.
    payload = {
        "answer_markdown": (
            "Answer body.</answer_markdown>\n<citations>\n"
            '<citation section_key="coo:822">\n'
            "<quote>Domestic power tools between 9:00 p.m. and 7:00 a.m.</quote>\n"
            "</citation>\n"
            '<citation section_key="coo:823">\n'
            "<quote>10 PM - 7 AM (residential districts)</quote>\n"
            "</citation>\n"
            "</citations>\n<confidence>high</confidence>"
        ),
        "citations": [],
        "confidence": "high",
    }
    fixed = _salvage_spilled_payload(payload)
    assert fixed["answer_markdown"] == "Answer body."
    assert fixed["citations"] == [
        {
            "section_key": "coo:822",
            "quote": "Domestic power tools between 9:00 p.m. and 7:00 a.m.",
        },
        {"section_key": "coo:823", "quote": "10 PM - 7 AM (residential districts)"},
    ]
    assert fixed["confidence"] == "high"


def test_citation_elements_with_child_section_key_tag():
    from app.agent import _parse_citation_elements

    blob = (
        "<citation><section_key>zon:5</section_key>"
        "<quote>the quote</quote></citation>"
    )
    assert _parse_citation_elements(blob) == [
        {"section_key": "zon:5", "quote": "the quote"}
    ]


def test_parse_pseudo_submit_still_handles_closed_tags():
    text = (
        "<answer_markdown>Body here.</answer_markdown>\n"
        '<citations>[{"section_key":"coo:2","quote":"q2"}]</citations>\n'
        "<confidence>medium</confidence>\n<caveats>Check zoning.</caveats>"
    )
    payload = _parse_pseudo_submit(text)
    assert payload["answer_markdown"] == "Body here."
    assert payload["citations"] == [{"section_key": "coo:2", "quote": "q2"}]
    assert payload["confidence"] == "medium"
    assert payload["caveats"] == "Check zoning."


def test_trailing_note_moves_to_caveats():
    # Seen live 2026-07-10 (short-term-rental eval): a closing "Note:" body
    # paragraph, which the frontend would show alongside the caveats box.
    answer = _build_answer_from_submit(
        {
            "answer_markdown": (
                "You need to register your short-term rental.\n\n"
                "Note: a \"Bed & Breakfast\" is a distinct zoning use category."
            ),
            "citations": [],
            "confidence": "low",
        },
        usage={},
    )
    assert answer.answer_markdown == "You need to register your short-term rental."
    assert answer.caveats == 'a "Bed & Breakfast" is a distinct zoning use category.'


def test_trailing_note_not_duplicated_into_matching_caveats():
    answer = _build_answer_from_submit(
        {
            "answer_markdown": "Answer body.\n\n**Caveat:** check your zoning district.",
            "citations": [],
            "confidence": "low",
            "caveats": "Check your zoning district.",
        },
        usage={},
    )
    assert answer.answer_markdown == "Answer body."
    assert answer.caveats == "Check your zoning district."


def test_extract_trailing_note_leaves_normal_answers_alone():
    body = "First paragraph.\n\nSecond paragraph with note-taking advice."
    assert _extract_trailing_note(body) == (body, None)
    # A lone-paragraph answer is never emptied, even if it looks like a note.
    lone = "Note: the corpus does not address this."
    assert _extract_trailing_note(lone) == (lone, None)


def _feed_chunks(chunks: list[str]) -> tuple[str, AnswerStreamGuard]:
    guard = AnswerStreamGuard()
    return "".join(guard.feed(c) for c in chunks), guard


def test_stream_guard_stops_at_marker():
    out, guard = _feed_chunks(["Answer text.", "</answer_markdown><citations>[…"])
    assert out == "Answer text."
    assert guard.tripped
    assert guard.feed("more junk") == ""


def test_stream_guard_marker_split_across_chunks():
    out, guard = _feed_chunks(["Answer text.</ans", "wer_mark", "down><cit"])
    assert out == "Answer text."
    assert guard.tripped


def test_stream_guard_citations_marker_without_closing_tag():
    out, guard = _feed_chunks(["Body.\n<cit", 'ations>[{"section_key":'])
    assert out == "Body.\n"
    assert guard.tripped


def test_stream_guard_passes_ordinary_angle_brackets():
    out, guard = _feed_chunks(["hens must be kept in pens <", " 10 ft tall"])
    assert out == "hens must be kept in pens < 10 ft tall"
    assert not guard.tripped


def test_stream_guard_false_alarm_prefix_is_released():
    # "</an" looks like the start of </answer_markdown> but turns out to be
    # ordinary text — it must be released, not swallowed.
    out, guard = _feed_chunks(["a </an", "gle> b"])
    assert out == "a </angle> b"
    assert not guard.tripped


# --- table-lookup citation verification (2026-07: table-citations feature) ---

_FAKE_SECTION_TEXT = (
    "1. Components are permitted as specified on Table 1.1.\n"
    "\n"
    "| Table 1.1 Things |  |  |\n"
    "| --- | --- | --- |\n"
    "|  | Small Lot | Large Lot |\n"
    "| Foo Porch | P | N |\n"
    "| P - Permitted • N - Not Permitted |  |  |\n"
)

_FAKE_SECTIONS = {
    "test:1": {
        "key": "test:1",
        "corpus": "test",
        "heading_path": ["Things"],
        "title": "1.1 Things",
        "text": _FAKE_SECTION_TEXT,
        "url": "https://example.test/#secid-1",
    }
}


def _patched_verify(monkeypatch, citations):
    from app import law_tools
    from app.agent import _verify_citations

    monkeypatch.setattr(law_tools, "SECTIONS", _FAKE_SECTIONS)
    return _verify_citations(citations)


def test_table_citation_verifies_against_parsed_table(monkeypatch):
    kept, dropped, details = _patched_verify(
        monkeypatch,
        [
            {
                "section_key": "test:1",
                "table": "Table 1.1",
                "row": "Foo Porch",
                "column": "Small Lot",
                "value": "P",
            }
        ],
    )
    assert dropped == 0 and details == []
    (cite,) = kept
    assert cite.verified
    assert cite.quote == ""
    # The full table title from the corpus is attached for display.
    assert cite.table == "Table 1.1 Things"
    assert (cite.row, cite.column, cite.value) == ("Foo Porch", "Small Lot", "P")


def test_table_citation_value_mismatch_is_dropped(monkeypatch):
    kept, dropped, details = _patched_verify(
        monkeypatch,
        [
            {
                "section_key": "test:1",
                "table": "Table 1.1",
                "row": "Foo Porch",
                "column": "Large Lot",
                "value": "P",  # cell actually holds N
            }
        ],
    )
    assert kept == [] and dropped == 1
    assert "value mismatch" in details[0]


def test_table_citation_unknown_table_is_dropped_with_reason(monkeypatch):
    kept, dropped, details = _patched_verify(
        monkeypatch,
        [
            {
                "section_key": "test:1",
                "table": "Table 9.9",
                "row": "Foo Porch",
                "column": "Small Lot",
                "value": "P",
            }
        ],
    )
    assert kept == [] and dropped == 1
    assert "table_not_found" in details[0]


def test_failed_lookup_falls_back_to_quote(monkeypatch):
    kept, dropped, details = _patched_verify(
        monkeypatch,
        [
            {
                "section_key": "test:1",
                "table": "Table 9.9",  # wrong table -> lookup fails
                "row": "Foo Porch",
                "column": "Small Lot",
                "value": "P",
                "quote": "Components are permitted as specified on Table 1.1.",
            }
        ],
    )
    assert dropped == 0
    (cite,) = kept
    assert cite.verified and cite.quote.startswith("Components are permitted")


def test_quote_citations_still_verify(monkeypatch):
    kept, dropped, details = _patched_verify(
        monkeypatch,
        [
            {"section_key": "test:1", "quote": "permitted as specified on Table 1.1"},
            {"section_key": "test:1", "quote": "this text does not appear"},
            {"section_key": "test:404", "quote": "whatever"},
        ],
    )
    assert len(kept) == 1 and dropped == 2
    assert any("not found verbatim" in d for d in details)
    assert any("unknown section key" in d for d in details)
