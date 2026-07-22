"""The Somerville municipal-law Q&A agent.

A manual Anthropic tool loop over the retrieval tools in ``law_tools``. The
model researches the corpus with ``search_law`` / ``get_sections`` /
``get_wiki_page`` and finishes by calling the terminal ``submit_answer`` tool.
Every citation the model returns is verified deterministically (no LLM) against
the exact text of the cited section before it is surfaced to the user.

CLI:  ``uv run python -m app.agent "question here"``
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import anthropic
import httpx

from . import law_tools

# --- API usage constants (see app/DESIGN.md Component 3) ---
# LAW_QA_MODEL overrides the answering model (e.g. for A/B testing).
# Sonnet 5 won the 2026-07-09 A/B vs Opus 4.8: equal eval pass rate,
# ~33% cheaper, much better tail latency (p90 38s vs 59s).
MODEL = os.environ.get("LAW_QA_MODEL", "claude-sonnet-5")
# 32k: citation-heavy questions (e.g. the noise ordinance) were hitting a 16k
# cap mid-submit_answer, which degrades to an uncited low-confidence answer.
MAX_TOKENS = 32_000
MAX_ITERATIONS = 12
NUDGE_EXTRA_ITERATIONS = 2

DATA_DIR = Path(__file__).resolve().parent / "data"

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VerifiedCitation:
    quote: str
    section_key: str
    url: str | None
    verified: bool


@dataclass
class Answer:
    answer_markdown: str
    citations: list[VerifiedCitation]
    confidence: str
    caveats: str | None
    dropped_citations: int
    usage: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Incremental streaming extractor for the answer_markdown field
# ---------------------------------------------------------------------------

_WS = " \t\n\r"


class AnswerMarkdownExtractor:
    """Incrementally extract the value of the JSON string field
    ``"answer_markdown"`` from a stream of ``input_json_delta`` fragments.

    Feed each ``partial_json`` chunk to :meth:`feed`; it returns the newly
    completed, JSON-unescaped portion of the answer string (possibly ""). A
    trailing incomplete escape sequence (``\\``, ``\\uXXXX``, or a lone high
    surrogate awaiting its low surrogate) is held back until the next chunk
    completes it. Extraction stops at the closing unescaped quote.

    Pure and self-contained: no I/O, no dependency on the SDK. Tested in the
    scratchpad against realistic submit_answer JSON split at every boundary.
    """

    _KEY = '"answer_markdown"'
    _ESCAPES = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }

    def __init__(self) -> None:
        self._buf = ""
        self._pos = 0
        self._state = "search"  # search -> instring -> done

    @property
    def done(self) -> bool:
        return self._state == "done"

    def feed(self, partial_json: str) -> str:
        if self._state == "done":
            return ""
        self._buf += partial_json
        if self._state == "search":
            start = self._find_value_start()
            if start is None:
                return ""
            self._pos = start
            self._state = "instring"
        if self._state == "instring":
            return self._consume_string()
        return ""

    def _find_value_start(self) -> int | None:
        """Index just after the opening quote of the answer_markdown value, or
        None if the key/colon/opening-quote have not all arrived yet."""
        buf = self._buf
        ki = buf.find(self._KEY)
        if ki == -1:
            return None
        j = ki + len(self._KEY)
        n = len(buf)
        while j < n and buf[j] in _WS:
            j += 1
        if j >= n or buf[j] != ":":
            return None
        j += 1
        while j < n and buf[j] in _WS:
            j += 1
        if j >= n:
            return None
        if buf[j] != '"':
            return None
        return j + 1

    def _consume_string(self) -> str:
        buf = self._buf
        n = len(buf)
        i = self._pos
        out: list[str] = []
        while i < n:
            c = buf[i]
            if c == '"':
                self._pos = i + 1
                self._state = "done"
                break
            if c != "\\":
                out.append(c)
                i += 1
                continue
            # escape sequence starting at i
            if i + 1 >= n:
                break  # incomplete: hold back the lone backslash
            e = buf[i + 1]
            if e != "u":
                out.append(self._ESCAPES.get(e, e))
                i += 2
                continue
            # \uXXXX
            if i + 6 > n:
                break  # incomplete hex escape
            hi = _parse_hex(buf[i + 2 : i + 6])
            if hi is None:
                out.append(buf[i : i + 6])  # malformed; emit literally
                i += 6
                continue
            if 0xD800 <= hi <= 0xDBFF:
                # high surrogate: need the following \uXXXX low surrogate
                if i + 12 > n:
                    break  # hold back the whole surrogate pair
                if buf[i + 6] == "\\" and buf[i + 7] == "u":
                    lo = _parse_hex(buf[i + 8 : i + 12])
                    if lo is not None and 0xDC00 <= lo <= 0xDFFF:
                        cp = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                        out.append(chr(cp))
                        i += 12
                        continue
                out.append(chr(hi))  # lone high surrogate
                i += 6
                continue
            out.append(chr(hi))
            i += 6
        else:
            # loop exhausted without a closing quote
            self._pos = i
            return "".join(out)
        if self._state != "done":
            self._pos = i
        return "".join(out)


def _parse_hex(s: str) -> int | None:
    try:
        return int(s, 16)
    except ValueError:
        return None


# Markers of a submit_answer document spilled into the answer_markdown string
# (the model occasionally writes "answer text</answer_markdown><citations>[…]"
# inside the field instead of using the separate fields).
_SPILL_MARKERS = ("</answer_markdown>", "<citations>", "<confidence>", "<caveats>")


class AnswerStreamGuard:
    """Filter streamed answer text so pseudo-XML spillover never reaches the
    reader.

    Feed each chunk of extracted answer_markdown text to :meth:`feed`; it
    returns the chunk with everything from the first spill marker onward
    suppressed. A trailing partial marker (e.g. ``"</answ"`` split across
    chunks) is held back until it either completes into a marker or turns out
    to be ordinary text.
    """

    def __init__(self) -> None:
        self._tail = ""
        self.tripped = False

    def feed(self, text: str) -> str:
        if self.tripped:
            return ""
        buf = self._tail + text
        cut = min(
            (i for i in (buf.find(m) for m in _SPILL_MARKERS) if i != -1),
            default=None,
        )
        if cut is not None:
            self.tripped = True
            self._tail = ""
            return buf[:cut]
        hold = 0
        longest = max(len(m) for m in _SPILL_MARKERS)
        for k in range(1, min(len(buf), longest - 1) + 1):
            if any(m.startswith(buf[-k:]) for m in _SPILL_MARKERS):
                hold = k
        self._tail = buf[-hold:] if hold else ""
        return buf[:-hold] if hold else buf


# ---------------------------------------------------------------------------
# Tool definitions exposed to the model
# ---------------------------------------------------------------------------

SUBMIT_ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer_markdown", "citations", "confidence"],
    "properties": {
        "answer_markdown": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["quote", "section_key"],
                "properties": {
                    "quote": {"type": "string"},
                    "section_key": {"type": "string"},
                },
            },
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "caveats": {"type": "string"},
    },
}

TOOLS = [
    {
        "name": "search_law",
        "description": (
            "Full-text (BM25) search over every section of the Somerville "
            "corpus (Charter, Code of Ordinances, appendices, and the zoning "
            "ordinance). Returns the best-matching sections with their key, "
            "title, heading path, and a short snippet; top results also "
            "include the complete section text ('text', as many as fit a "
            "size budget). Text returned here counts as fetched — quote from "
            "it directly instead of re-fetching with get_sections. Search "
            "with the legal vocabulary the code actually uses (e.g. "
            "'domestic fowl' rather than 'chickens'); issue several searches "
            "with synonyms if the first is thin."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms (legal vocabulary works best).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 12).",
                },
            },
        },
    },
    {
        "name": "get_sections",
        "description": (
            "Fetch the full verbatim text of one or more sections by key "
            "(e.g. 'coo:826' or 'zon:88'). Use it for sections whose full "
            "text you have NOT already received inline from search_law or "
            "get_wiki_page — always read the actual section text before "
            "making a legal claim or quoting from it; snippets from search "
            "are not enough."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["keys"],
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Section keys to fetch.",
                }
            },
        },
    },
    {
        "name": "get_wiki_page",
        "description": (
            "Fetch a pregenerated plain-language topic page by slug (from the "
            "<topic_index>). Returns the page body plus the complete text of "
            "its referenced sections (as many as fit a size budget; any left "
            "out are listed in 'omitted_section_keys' — fetch those with "
            "get_sections only if you still need them). If the topic index "
            "is empty or has no matching slug, skip this and use search_law "
            "instead."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["topic_slug"],
            "properties": {
                "topic_slug": {
                    "type": "string",
                    "description": "The topic slug from the <topic_index>.",
                }
            },
        },
    },
    {
        "name": "submit_answer",
        "description": (
            "Submit your final answer. Call this exactly once, when you have "
            "read the relevant sections and are ready to answer. Provide the "
            "'answer_markdown' field FIRST, before 'citations', 'confidence', "
            "and 'caveats' — the answer text is streamed to the reader as you "
            "write it, so emitting it first lets them start reading sooner. "
            "Every legal claim in answer_markdown must be backed by a citation "
            "whose 'quote' is copied VERBATIM from the fetched section text "
            "(quotes are verified by exact substring match — paraphrases are "
            "dropped). Keep each quote to the shortest passage that proves the "
            "claim — usually one sentence, at most about 40 words; never quote "
            "a whole subsection when one clause carries the point. "
            "'caveats' is displayed as a highlighted note under the "
            "answer: omit it unless there is one substantive question-specific "
            "point, and never duplicate it as a closing note inside "
            "answer_markdown."
        ),
        "input_schema": SUBMIT_ANSWER_SCHEMA,
        "strict": True,
        # Fine-grained tool streaming (NOT a beta): stream input_json_delta
        # fragments eagerly so answer_markdown reaches the reader token by
        # token. Coexists with strict: true.
        "eager_input_streaming": True,
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_ROLE_AND_WORKFLOW = """\
You are a legal research assistant for the municipal law of Somerville, \
Massachusetts. You help residents understand the City's Charter, Code of \
Ordinances, appendices, and Zoning Ordinance in plain language.

Ground rules:
- Answer ONLY from the provided corpus, reached through your tools. Never rely \
on outside knowledge of what a law "usually" says, and never invent a section, \
number, or quote. If the corpus does not address the question, say so plainly \
and set confidence to "low" rather than guessing.
- Confidence rubric: "high" = the corpus directly answers the question and you \
are quoting the governing section(s); "medium" = the corpus addresses the \
topic but the answer requires interpretation, depends on facts you don't have \
(like a zoning district), or the governing text is ambiguous; "low" = \
REQUIRED whenever the corpus does not directly address the question, even if \
you can offer helpful adjacent context. An answer whose core claim is "this \
corpus doesn't cover that" is always confidence "low". Inferring an answer \
from provisions that govern a DIFFERENT scenario (e.g. concluding a use is \
banned from a rule about displays or businesses when the question is about \
private conduct) is adjacent context, not coverage — that is "low", no \
matter how confident the practical conclusion.
- Every legal claim you make MUST be supported by a citation containing an \
EXACT, verbatim quote copied from the text of a section you have fetched. \
Quotes are checked by exact substring match against the section text; a \
paraphrased or approximate quote will be dropped and will not support your \
answer. Quote only from complete section text you have actually retrieved — \
the full 'text' field returned by search_law or get_wiki_page, or a \
get_sections result. Never quote from a search snippet.
- Write for residents: clear, direct, plain language. Lead with the bottom-line \
answer, then the supporting detail.
- Be brief. Lead with the bottom line, and keep the whole answer under about \
250 words unless the law genuinely requires more detail. Do not restate the \
question or pad the answer with generic advice.
- Do NOT append a legal-advice disclaimer or a "verify against the official \
code" reminder to your answers — the site already displays that disclaimer \
alongside every answer.
- Question-specific caveats (e.g. "the zoning ordinance may add \
district-specific restrictions — check your district on the official zoning \
map") belong ONLY in the 'caveats' field, which is rendered as a highlighted \
"Note" box below your answer. Never write such a point as a closing "Note:", \
"Caveat:", or "one more thing" paragraph inside answer_markdown — the reader \
would see the caveat twice. Most answers need no caveat at all; when one is \
warranted, keep it to one or two sentences and do not repeat anything already \
said in the answer.
- State-law questions (Massachusetts General Laws, "M.G.L.") are outside this \
corpus, which covers only Somerville's own municipal law. If a question turns \
on state law, say that it is out of scope here.
- If the question is about a specific address or property, explain that the \
answer depends on that property's zoning district, point the reader to the \
City's official zoning map, and — where useful — answer conditionally for the \
common districts.

Workflow:
1. Consult the <topic_index> below first. If a topic clearly matches, call \
get_wiki_page with its slug to find the relevant section keys. If the topic \
index is empty or nothing matches, go straight to search_law.
2. Remember that legal vocabulary differs from everyday speech (for example \
the code says "domestic fowl", not "chickens"; "leaf blowers", not "yard \
equipment"). Search with legal synonyms, and run several searches if needed. \
Relevant rules may span the Code of Ordinances, appendices, and the Zoning \
Ordinance — check across corpora.
3. Make sure you have the FULL text of every section you rely on. search_law \
and get_wiki_page already inline the complete text of their top results — if \
that text answers the question, do NOT re-fetch it; only call get_sections \
for sections you have merely as a snippet or key. Use the \
<table_of_contents> to orient yourself and to find neighboring sections.
4. When you have read enough to answer, finish by calling submit_answer with \
your plain-language answer and verbatim-quoted citations. Do not answer in \
free-form text — always end with submit_answer. Prefer finishing in few \
turns: if the first tool results already contain the governing text, submit \
your answer immediately.
"""


def _build_system_blocks() -> list[dict]:
    """Assemble the system prompt as a list of text blocks.

    Order (stable text first so the whole prefix caches): role + workflow,
    then <topic_index> (from the wiki), then <table_of_contents> from
    data/toc.txt with the 1-hour cache breakpoint on it.
    """
    topic_index = law_tools.wiki_index()
    if topic_index.strip():
        topic_block = (
            "<topic_index>\n"
            "Pregenerated topic pages (fetch one with get_wiki_page using its "
            "slug):\n\n" + topic_index + "\n</topic_index>"
        )
    else:
        topic_block = (
            "<topic_index>\n"
            "(No topic pages are available yet. Ignore get_wiki_page and use "
            "search_law to find relevant sections.)\n"
            "</topic_index>"
        )

    toc_path = DATA_DIR / "toc.txt"
    toc_text = toc_path.read_text(encoding="utf-8") if toc_path.exists() else ""
    toc_block = (
        "<table_of_contents>\n"
        "Every section in the corpus, one line per section (indented by depth) "
        "as: title [key]. Use these keys with get_sections.\n\n"
        + toc_text
        + "\n</table_of_contents>"
    )

    return [
        {"type": "text", "text": _ROLE_AND_WORKFLOW},
        {"type": "text", "text": topic_block},
        {
            "type": "text",
            "text": toc_block,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
    ]


# ---------------------------------------------------------------------------
# Citation verification (deterministic, no LLM)
# ---------------------------------------------------------------------------

_EMPHASIS_CHARS = str.maketrans("", "", "*_`")

# Curly quotes / apostrophes -> straight; dashes -> hyphen; nbsp -> space.
_CHAR_MAP = {
    "‘": "'",  # left single quote
    "’": "'",  # right single quote / apostrophe
    "‚": "'",
    "‛": "'",
    "′": "'",  # prime
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "„": '"',
    "″": '"',
    "–": "-",  # en dash
    "—": "-",  # em dash
    "−": "-",  # minus sign
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
    " ": " ",  # non-breaking space
    " ": " ",  # thin space
    " ": " ",
    " ": " ",  # narrow no-break space
    " ": " ",
    "﻿": "",  # zero-width no-break space / BOM
    "​": "",  # zero-width space
}
_TRANSLATION = {ord(k): v for k, v in _CHAR_MAP.items()}


def _normalize(text: str) -> str:
    """Normalize for substring comparison.

    A citation counts as "verified" iff its quote matches the section text
    verbatim, modulo: whitespace-run collapse, punctuation-glyph unification
    (curly quotes/apostrophes, dashes/hyphens, and non-breaking spaces mapped
    to their plain-ASCII equivalents via NFKC plus an explicit char map), and
    stripped markdown emphasis markers (* _ `). Case is NOT normalized —
    quotes must match the section text's original casing.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_TRANSLATION)
    text = text.translate(_EMPHASIS_CHARS)
    text = " ".join(text.split())
    return text


def _verify_citations(
    raw_citations: list[dict],
) -> tuple[list[VerifiedCitation], int]:
    """Verify each citation's quote against the cited section text.

    Returns (kept_citations, dropped_count). A citation is kept iff the
    normalized quote is a substring of the normalized text of its cited
    section. The section's url is attached to kept citations.
    """
    kept: list[VerifiedCitation] = []
    dropped = 0
    for cite in raw_citations:
        quote = (cite.get("quote") or "").strip()
        section_key = (cite.get("section_key") or "").strip()

        section = law_tools.SECTIONS.get(section_key)
        verified = False
        url = None
        if section is not None:
            url = section.get("url")
            norm_quote = _normalize(quote)
            norm_text = _normalize(section.get("text", "") or "")
            verified = bool(norm_quote) and norm_quote in norm_text

        if verified:
            kept.append(
                VerifiedCitation(
                    quote=quote,
                    section_key=section_key,
                    url=url,
                    verified=True,
                )
            )
        else:
            dropped += 1
    return kept, dropped


def _salvage_spilled_payload(payload: dict) -> dict:
    """Recover a submit_answer payload whose answer_markdown contains the
    whole pseudo-XML document ("answer…</answer_markdown><citations>[…]…")
    instead of just the answer text.

    The spilled citations/confidence/caveats are parsed out of the string and
    used to fill fields the model left empty, so the answer flows through the
    normal citation-verification path instead of showing raw markup to the
    reader — and instead of tripping the no-citations guardrail retry.
    """
    text = payload.get("answer_markdown") or ""
    if not any(m in text for m in _SPILL_MARKERS):
        return payload
    doc = text if "<answer_markdown>" in text else "<answer_markdown>" + text
    pseudo = _parse_pseudo_submit(doc)
    if pseudo is None:
        return payload
    merged = dict(payload)
    # If the spill had no closing </answer_markdown> tag the parsed body still
    # carries the trailing markup — cut it at the first remaining marker.
    body = pseudo["answer_markdown"]
    cut = min(
        (i for i in (body.find(m) for m in _SPILL_MARKERS) if i != -1),
        default=None,
    )
    if cut is not None:
        body = body[:cut].rstrip()
    merged["answer_markdown"] = body
    # Merge recovered citations with any the model put in the real array —
    # the spill often carries the full set while the array holds a partial
    # (and possibly unverifiable) subset, so keep both and let verification
    # sort them out.
    recovered = [c for c in (pseudo.get("citations") or []) if isinstance(c, dict)]
    if recovered:
        existing = [c for c in (merged.get("citations") or []) if isinstance(c, dict)]
        seen = {(c.get("section_key"), c.get("quote")) for c in existing}
        merged["citations"] = existing + [
            c for c in recovered if (c.get("section_key"), c.get("quote")) not in seen
        ]
    if not merged.get("caveats") and pseudo.get("caveats"):
        merged["caveats"] = pseudo["caveats"]
    # _parse_pseudo_submit defaults confidence to "low"; only take its value
    # when the spill actually carried a <confidence> tag.
    if "<confidence>" in text:
        merged["confidence"] = pseudo["confidence"]
    return merged


def _extract_trailing_note(answer_markdown: str) -> tuple[str, str | None]:
    """Split off a closing "Note:"/"Caveat:" paragraph from the answer body.

    The prompt forbids these (the caveats field is the note UI), but the model
    still writes them sometimes — relocating mechanically beats hoping. Only a
    LAST paragraph is moved, and never a lone-paragraph answer.
    """
    paras = answer_markdown.rstrip().split("\n\n")
    if len(paras) < 2:
        return answer_markdown, None
    last = paras[-1].strip()
    lead = last.lstrip("*_ ").lower()
    if not lead.startswith(("note:", "notes:", "caveat:", "caveats:")):
        return answer_markdown, None
    note = last.split(":", 1)[1].strip().lstrip("*_ ").strip()
    return "\n\n".join(paras[:-1]).rstrip(), note or None


def _build_answer_from_submit(payload: dict, usage: dict) -> Answer:
    """Turn a validated submit_answer payload into a verified Answer."""
    answer_markdown = payload.get("answer_markdown", "") or ""
    confidence = payload.get("confidence", "low") or "low"
    caveats = payload.get("caveats") or None
    raw_citations = payload.get("citations") or []

    # A closing Note:/Caveat: paragraph belongs in the caveats field — move it
    # there, unless the caveats field already says the same thing.
    answer_markdown, trailing_note = _extract_trailing_note(answer_markdown)
    if trailing_note:
        if not caveats:
            caveats = trailing_note
        elif _normalize(trailing_note).lower() not in _normalize(caveats).lower():
            caveats = f"{caveats}\n\n{trailing_note}"

    citations, dropped = _verify_citations(raw_citations)

    # Hard floor: no verified citation, no confidence. Whether the model
    # offered none (and survived the one rejection retry) or all of them
    # failed verification, an uncited legal answer is never presented as
    # better than "low".
    if not citations and confidence != "low":
        confidence = "low"
        if raw_citations:
            note = (
                "None of the provided citations could be verified against the "
                "official section text, so this answer could not be confirmed. "
                "Please verify against the official code."
            )
        else:
            note = (
                "This answer was produced without verifiable citations to the "
                "official section text, so it could not be confirmed. Please "
                "verify against the official code."
            )
        caveats = f"{caveats}\n\n{note}" if caveats else note

    return Answer(
        answer_markdown=answer_markdown,
        citations=citations,
        confidence=confidence,
        caveats=caveats,
        dropped_citations=dropped,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _execute_tool(name: str, tool_input: dict) -> str:
    """Run a retrieval tool and return a JSON string result."""
    try:
        if name == "search_law":
            query = tool_input.get("query", "")
            limit = tool_input.get("limit", 12) or 12
            results = law_tools.search_law(query, limit=int(limit))
            return json.dumps({"results": results}, ensure_ascii=False)
        if name == "get_sections":
            keys = tool_input.get("keys", []) or []
            results = law_tools.get_sections(list(keys))
            return json.dumps({"sections": results}, ensure_ascii=False)
        if name == "get_wiki_page":
            slug = tool_input.get("topic_slug", "")
            bundle = law_tools.get_wiki_page_bundle(slug)
            if bundle is None:
                return json.dumps(
                    {"error": f"no wiki page for slug '{slug}'"},
                    ensure_ascii=False,
                )
            return json.dumps({"topic_slug": slug, **bundle}, ensure_ascii=False)
        return json.dumps({"error": f"unknown tool '{name}'"}, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover - defensive
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)


def _clean_heading(text: str) -> str:
    label = (text or "").strip().rstrip(".:").strip()
    return label if len(label) <= 60 else label[:57] + "…"


def _sections_detail(keys: list[str]) -> str:
    """Human-readable summary of a get_sections call, for progress display.

    Labels come from each section's last heading (for headerless sections
    that is the nearest real ancestor heading). Duplicate labels collapse,
    and when three or more distinct labels share a common ancestor heading
    the whole fetch is summarized as that ancestor plus a section count —
    e.g. "Board of Health Regulations for the Keeping of Hens (5 sections)".
    """
    recs = [r for r in law_tools.get_sections(keys) if r and not r.get("error")]
    if not recs:
        return ", ".join(keys)

    paths = [r.get("heading_path") or [] for r in recs]
    labels: list[str] = []
    for r in recs:
        label = law_tools.section_label(r.get("key", ""))
        if label and label not in labels:
            labels.append(label)

    if len(labels) <= 2:
        return ", ".join(labels) or ", ".join(keys)

    # Many distinct labels: prefer the deepest heading shared by all of them.
    common: list[str] = []
    for parts in zip(*paths):
        if all(p == parts[0] for p in parts):
            common.append(parts[0])
        else:
            break
    if common:
        return f"{_clean_heading(common[-1])} ({len(recs)} sections)"
    return f"{labels[0]}, {labels[1]}, +{len(labels) - 2} more"


def _tool_detail(name: str, tool_input: dict) -> str:
    if name == "search_law":
        return str(tool_input.get("query", ""))
    if name == "get_sections":
        keys = tool_input.get("keys", []) or []
        return _sections_detail(keys) if keys else ""
    if name == "get_wiki_page":
        return str(tool_input.get("topic_slug", ""))
    return ""


def _accumulate_usage(total: dict, usage) -> None:
    if usage is None:
        return
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        val = getattr(usage, key, None)
        total[key] = total.get(key, 0) + (val or 0)


def _text_from_content(content) -> str:
    parts = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n\n".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _stream_turn(client, request_params: dict, emit: Callable[[dict], None]):
    """Run one streaming model turn.

    Consumes the SDK stream events, and while the terminal ``submit_answer``
    tool's input streams in, incrementally extracts its ``answer_markdown``
    string and emits ``{"type": "answer_delta", "text": ...}`` events so the UI
    can render the answer as it is written. Only ``submit_answer`` triggers
    deltas — other tool_use blocks are ignored by the extractor.

    Returns ``(final_message, streamed_text)`` where ``final_message`` is the
    fully-accumulated response (so all existing post-loop logic — tool handling,
    citation verification, usage accumulation — works unchanged) and
    ``streamed_text`` is the concatenation of every answer_delta emitted ("" if
    none) — usable as a salvage body when the turn is cut off mid-submit.
    """
    extractor: AnswerMarkdownExtractor | None = None
    guard = AnswerStreamGuard()
    submit_index = None
    streamed_parts: list[str] = []

    with client.messages.stream(**request_params) as stream:
        for event in stream:
            etype = getattr(event, "type", None)
            if etype == "content_block_start":
                block = getattr(event, "content_block", None)
                if (
                    getattr(block, "type", None) == "tool_use"
                    and getattr(block, "name", None) == "submit_answer"
                ):
                    extractor = AnswerMarkdownExtractor()
                    guard = AnswerStreamGuard()
                    submit_index = getattr(event, "index", None)
                    streamed_parts = []
                    sources_pending_emitted = False
            elif etype == "content_block_delta":
                if extractor is not None and getattr(event, "index", None) == submit_index:
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "type", None) == "input_json_delta":
                        text = extractor.feed(getattr(delta, "partial_json", "") or "")
                        if text:
                            text = guard.feed(text)
                        if text:
                            streamed_parts.append(text)
                            emit({"type": "answer_delta", "text": text})
                        if extractor.done and not sources_pending_emitted:
                            # The visible answer is fully streamed; the model
                            # is now generating the citations JSON. Let the UI
                            # narrate the gap instead of appearing stalled.
                            sources_pending_emitted = True
                            emit({"type": "sources_pending"})
        response = stream.get_final_message()

    return response, "".join(streamed_parts)


def ask(
    question: str,
    history: list[dict] | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> Answer:
    """Answer a Somerville municipal-law question with verified citations."""
    client = anthropic.Anthropic()
    system_blocks = _build_system_blocks()

    messages: list[dict] = list(history or [])
    messages.append({"role": "user", "content": question})

    usage_total: dict = {}

    def emit(event: dict) -> None:
        if on_event is not None:
            try:
                on_event(event)
            except Exception:  # pragma: no cover - never let UI break the loop
                pass

    nudged = False
    citation_nudged = False
    turn_retried = False
    cutoff_retried = False
    iteration = 0
    max_iterations = MAX_ITERATIONS

    while iteration < max_iterations:
        iteration += 1
        emit({"type": "thinking"})

        # Track whether THIS turn streamed any answer text, so a mid-stream
        # failure can reset the reader's provisional answer before the retry.
        turn_streamed = False

        def emit_tracking(event: dict) -> None:
            nonlocal turn_streamed
            if event.get("type") == "answer_delta":
                turn_streamed = True
            emit(event)

        try:
            response, streamed_answer = _stream_turn(
                client,
                {
                    "model": MODEL,
                    "max_tokens": MAX_TOKENS,
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "low"},
                    "system": system_blocks,
                    "tools": TOOLS,
                    "messages": messages,
                },
                emit_tracking,
            )
        except (anthropic.APIError, httpx.HTTPError) as exc:
            # Transient stream/connection failures (the SDK's own retries only
            # cover the initial request, not a stream that dies midway): retry
            # the turn once before giving up. The failed turn contributed
            # nothing to `messages`, so a plain re-send is safe.
            if turn_retried is False:
                turn_retried = True
                iteration -= 1
                if turn_streamed:
                    emit({"type": "answer_reset"})
                continue
            return Answer(
                answer_markdown=(
                    "Sorry — I ran into an error contacting the model and "
                    "could not complete this request."
                ),
                citations=[],
                confidence="low",
                caveats=f"API error: {type(exc).__name__}: {exc}",
                dropped_citations=0,
                usage=usage_total,
            )

        _accumulate_usage(usage_total, getattr(response, "usage", None))
        stop_reason = response.stop_reason

        # A paused server turn: re-send with the assistant content appended and
        # continue without adding a user message.
        if stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue

        if stop_reason == "max_tokens" and not cutoff_retried:
            # A runaway generation hit the 32k output cap (seen ~2/40 eval
            # questions: the model spirals while composing submit_answer and
            # the result degrades to an uncited low-confidence answer).
            # The cut-off turn contributed nothing to `messages`, so drop it
            # and retry once with an explicit brevity instruction instead of
            # shipping the degraded answer.
            cutoff_retried = True
            if turn_streamed:
                emit({"type": "answer_reset"})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was cut off at the maximum "
                        "output length and has been discarded. Answer again, "
                        "much more concisely: keep the answer under 250 "
                        "words, cite only the few sections that directly "
                        "answer the question, and keep each citation quote "
                        "to a single short sentence or clause. Do not "
                        "re-fetch sections you have already read."
                    ),
                }
            )
            continue

        if stop_reason in ("max_tokens", "refusal"):
            text = _text_from_content(response.content)
            if stop_reason == "refusal":
                caveat = (
                    "The model declined to answer this request. Please "
                    "consult the official code or the City directly."
                )
                body = text or "I'm not able to answer this question."
            else:
                caveat = (
                    "The response was cut off before it could be completed "
                    "(max tokens reached). This answer may be incomplete; "
                    "please verify against the official code."
                )
                pseudo = _parse_pseudo_submit(text) if text else None
                if streamed_answer:
                    # Cut off mid-submit_answer: the extracted answer text that
                    # already streamed to the reader is the best body we have.
                    body = streamed_answer
                elif pseudo is not None:
                    body = pseudo.get("answer_markdown") or text
                else:
                    body = text or (
                        "I wasn't able to finish composing an answer for this "
                        "question."
                    )
            return Answer(
                answer_markdown=body,
                citations=[],
                confidence="low",
                caveats=caveat,
                dropped_citations=0,
                usage=usage_total,
            )

        tool_uses = [
            b for b in response.content if getattr(b, "type", None) == "tool_use"
        ]

        # Terminal tool: verify and return. Do not execute anything else.
        submit = next((b for b in tool_uses if b.name == "submit_answer"), None)
        if submit is not None:
            payload = submit.input if isinstance(submit.input, dict) else {}
            payload = _salvage_spilled_payload(payload)
            # Guardrail: an answer with no citations is rejected once at ANY
            # confidence. Gating this on confidence above 'low' left an escape
            # hatch — the model would drop to 'low' instead of citing, even
            # with the governing text quoted in its answer body (seen in eval
            # flakes: graffiti-removal, dog-leash). A genuinely-uncovered
            # topic survives the retry: the second uncited submit is accepted.
            if not payload.get("citations") and not citation_nudged:
                citation_nudged = True
                # Provisional answer text was already streamed to the UI; clear
                # it so the retry streams fresh deltas onto a clean slate.
                if streamed_answer:
                    emit({"type": "answer_reset"})
                messages.append({"role": "assistant", "content": response.content})
                # The assistant turn may contain OTHER tool_use blocks besides
                # submit_answer (e.g. a search_law call issued in parallel).
                # Every tool_use needs a matching tool_result in the follow-up
                # message or the API 400s — so execute the non-submit tools
                # normally (same as the regular tool-execution path below,
                # including on_event) and use the rejection message as
                # submit_answer's own (is_error) result. All results go in
                # this single user message.
                tool_results = []
                for block in tool_uses:
                    if block.id == submit.id:
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": submit.id,
                                "content": (
                                    "Rejected: your answer has an empty "
                                    "citations array. If your answer makes "
                                    "legal claims — especially if it quotes "
                                    "section text — call submit_answer again "
                                    "with the same answer plus those verbatim "
                                    "quotes in the citations array (not "
                                    "embedded in the answer text), and keep "
                                    "your original confidence. Lowering "
                                    "confidence is NOT a substitute for "
                                    "citing text you have already read. Only "
                                    "if the corpus genuinely does not address "
                                    "the question should you resubmit with "
                                    "no citations and confidence 'low'."
                                ),
                                "is_error": True,
                            }
                        )
                        continue
                    detail = _tool_detail(block.name, block.input or {})
                    emit({"type": "tool", "name": block.name, "detail": detail})
                    result = _execute_tool(block.name, block.input or {})
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )
                messages.append({"role": "user", "content": tool_results})
                continue
            return _build_answer_from_submit(payload, usage_total)

        if stop_reason == "tool_use" and tool_uses:
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in tool_uses:
                detail = _tool_detail(block.name, block.input or {})
                emit({"type": "tool", "name": block.name, "detail": detail})
                result = _execute_tool(block.name, block.input or {})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )
            messages.append({"role": "user", "content": tool_results})
            continue

        # end_turn (or any stop with no tool call) and no submit_answer:
        # nudge once, allowing a couple of extra iterations.
        if not nudged:
            nudged = True
            max_iterations = min(
                MAX_ITERATIONS + NUDGE_EXTRA_ITERATIONS,
                iteration + NUDGE_EXTRA_ITERATIONS,
            )
            # An assistant turn with an EMPTY content list is itself invalid
            # on the next API call (400) — only append it when non-empty; the
            # API allows consecutive user messages, so the nudge alone is
            # fine on its own.
            if response.content:
                messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Reply by calling the submit_answer tool — an actual "
                        "tool call, not the answer written out as text or "
                        "XML-style tags."
                    ),
                }
            )
            continue

        # Already nudged and still no submit_answer: synthesize from the text.
        return _synthesize_answer(response.content, usage_total)

    # Iteration cap exhausted without a submit_answer.
    return _synthesize_answer(None, usage_total)


def _parse_pseudo_submit(text: str) -> dict | None:
    """Salvage a submit_answer-shaped payload from pseudo-XML text.

    The model occasionally writes its final answer as plain text with XML-ish
    tags mirroring the submit_answer schema (<answer_markdown>…<citations>…)
    instead of calling the tool. Extract the fields so the answer can run
    through the normal citation-verification path rather than showing raw
    markup to the reader. Returns None if the text isn't in that shape.
    """
    m = re.search(
        r"<answer_markdown>\s*(.*?)\s*(?:</answer_markdown>|\Z)", text, re.DOTALL
    )
    if not m:
        return None
    payload: dict = {"answer_markdown": m.group(1)}

    cite_match = re.search(
        r"<citations>\s*(.*?)\s*(?:</citations>|(?=<(?:confidence|caveats)>)|\Z)",
        text,
        re.DOTALL,
    )
    if cite_match:
        blob = cite_match.group(1)
        try:
            citations = json.loads(blob)
        except ValueError:
            citations = None
        if isinstance(citations, list):
            payload["citations"] = [c for c in citations if isinstance(c, dict)]
        else:
            # Not JSON: the model also writes citations as XML elements,
            # <citation section_key="…"><quote>…</quote></citation>.
            elements = _parse_citation_elements(blob)
            if elements:
                payload["citations"] = elements

    conf_match = re.search(r"<confidence>\s*(high|medium|low)\s*</confidence>", text)
    payload["confidence"] = conf_match.group(1) if conf_match else "low"

    caveat_match = re.search(r"<caveats>\s*(.*?)\s*</caveats>", text, re.DOTALL)
    if caveat_match:
        payload["caveats"] = caveat_match.group(1)
    return payload


def _parse_citation_elements(blob: str) -> list[dict]:
    """Parse XML-element-style citations the model sometimes writes instead
    of the JSON array: <citation section_key="…"><quote>…</quote></citation>
    (section_key may also appear as a child tag instead of an attribute)."""
    cites: list[dict] = []
    for m in re.finditer(
        r"<citation\b([^>]*)>(.*?)</citation>", blob, re.DOTALL
    ):
        attrs, body = m.group(1), m.group(2)
        key_m = re.search(r'section_key\s*=\s*"([^"]+)"', attrs) or re.search(
            r"<section_key>\s*(.*?)\s*</section_key>", body, re.DOTALL
        )
        quote_m = re.search(r"<quote>\s*(.*?)\s*</quote>", body, re.DOTALL)
        if key_m and quote_m:
            cites.append(
                {"section_key": key_m.group(1).strip(), "quote": quote_m.group(1)}
            )
    return cites


def _synthesize_answer(content, usage: dict) -> Answer:
    """Fallback Answer when the model never called submit_answer."""
    text = _text_from_content(content) if content else ""
    pseudo = _parse_pseudo_submit(text) if text else None
    if pseudo is not None:
        # Tagged text mirroring the submit_answer schema: recover the fields
        # and verify citations exactly as if the tool had been called.
        return _build_answer_from_submit(pseudo, usage)
    if not text:
        text = (
            "I wasn't able to complete a fully-formed answer for this "
            "question. Please try rephrasing, or consult the official code."
        )
    return Answer(
        answer_markdown=text,
        citations=[],
        confidence="low",
        caveats=(
            "This answer was assembled without the model's structured "
            "citation step, so its legal claims are unverified. Please verify "
            "against the official code."
        ),
        dropped_citations=0,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _pretty_print(answer: Answer) -> None:
    print("\n" + "=" * 72)
    print("ANSWER")
    print("=" * 72)
    print(answer.answer_markdown.strip())

    print("\n" + "-" * 72)
    print(f"CITATIONS ({len(answer.citations)} verified, "
          f"{answer.dropped_citations} dropped)")
    print("-" * 72)
    if not answer.citations:
        print("(none)")
    for c in answer.citations:
        mark = "✓" if c.verified else "✗"
        print(f"\n{mark} [{c.section_key}] {c.url or ''}")
        print(f"  “{c.quote}”")

    print("\n" + "-" * 72)
    print(f"Confidence: {answer.confidence}")
    if answer.caveats:
        print(f"Caveats: {answer.caveats}")
    print(f"Token usage: {json.dumps(answer.usage)}")
    print("=" * 72 + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask a question about Somerville municipal law."
    )
    parser.add_argument("question", help="The question to ask.")
    args = parser.parse_args(argv)

    def on_event(event: dict) -> None:
        if event.get("type") == "tool":
            name = event.get("name", "")
            detail = event.get("detail", "")
            sys.stderr.write(f"  [tool] {name}: {detail}\n")
            sys.stderr.flush()

    answer = ask(args.question, on_event=on_event)
    _pretty_print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
