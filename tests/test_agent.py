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
