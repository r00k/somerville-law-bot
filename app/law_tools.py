"""Retrieval tools over the indexed law corpus (app/data/sections.json) and
the generated topic wiki (app/wiki/*.md).

Pure stdlib. BM25 index is built once at import time from module state.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
WIKI_DIR = Path(__file__).resolve().parent / "wiki"

TOKEN_RE = re.compile(r"[a-z0-9]+")

K1 = 1.5
B = 0.75

SNIPPET_WIDTH = 300
GET_SECTIONS_CHAR_CAP = 150_000


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _load_sections() -> dict[str, dict]:
    path = DATA_DIR / "sections.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_bm25_index(sections: dict[str, dict]):
    """Returns (postings, doc_len, avgdl, idf).
    postings: token -> {section_key: term_frequency}
    """
    postings: dict[str, dict[str, int]] = {}
    doc_len: dict[str, int] = {}
    doc_freq: dict[str, int] = {}

    for key, sec in sections.items():
        blob = " ".join(
            [
                sec.get("title", "") or "",
                " ".join(sec.get("heading_path", []) or []),
                sec.get("text", "") or "",
            ]
        )
        tokens = _tokenize(blob)
        doc_len[key] = len(tokens)
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        for tok, count in tf.items():
            postings.setdefault(tok, {})[key] = count
            doc_freq[tok] = doc_freq.get(tok, 0) + 1

    n_docs = len(sections)
    avgdl = (sum(doc_len.values()) / n_docs) if n_docs else 0.0
    idf = {
        tok: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        for tok, df in doc_freq.items()
    }
    return postings, doc_len, avgdl, idf


# --- module state, built once at import ---
SECTIONS: dict[str, dict] = _load_sections()
_POSTINGS, _DOC_LEN, _AVGDL, _IDF = _build_bm25_index(SECTIONS)


def _make_snippet(text: str, query_tokens: list[str], width: int = SNIPPET_WIDTH) -> str:
    if not text:
        return ""
    q_set = set(query_tokens)
    positions = [m.start() for m in TOKEN_RE.finditer(text.lower()) if m.group() in q_set]

    if not positions:
        snippet = text[:width].strip()
        return snippet + ("…" if len(text) > width else "")

    half = width // 2
    best_pos = positions[0]
    best_count = -1
    for p in positions:
        lo, hi = p - half, p + half
        count = sum(1 for x in positions if lo <= x <= hi)
        if count > best_count:
            best_count = count
            best_pos = p

    start = max(0, best_pos - half)
    end = min(len(text), start + width)
    start = max(0, end - width)
    snippet = text[start:end].strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def search_law(query: str, limit: int = 12) -> list[dict]:
    """BM25 over (title + heading_path + text), tokenized lowercase alnum.
    Returns [{key, title, heading_path, score, snippet}] — snippet is ~300
    chars around the best matching region.
    """
    query_tokens = _tokenize(query)
    if not query_tokens or not SECTIONS:
        return []

    scores: dict[str, float] = {}
    for tok in set(query_tokens):
        post = _POSTINGS.get(tok)
        idf = _IDF.get(tok, 0.0)
        if not post or idf <= 0:
            continue
        for key, tf in post.items():
            dl = _DOC_LEN.get(key, 0)
            norm = (1 - B + B * (dl / _AVGDL)) if _AVGDL else 1.0
            denom = tf + K1 * norm
            score = idf * (tf * (K1 + 1)) / denom if denom else 0.0
            scores[key] = scores.get(key, 0.0) + score

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]

    results = []
    for key, score in ranked:
        sec = SECTIONS[key]
        results.append(
            {
                "key": key,
                "title": sec.get("title", ""),
                "heading_path": sec.get("heading_path", []),
                "score": round(score, 4),
                "snippet": _make_snippet(sec.get("text", ""), query_tokens),
            }
        )
    return results


def get_sections(keys: list[str]) -> list[dict]:
    """Full records for the given keys (invalid keys -> {'key': k, 'error': ...}).
    Cap combined returned text at ~150_000 chars; if over, truncate later
    sections and mark 'truncated': true.
    """
    results = []
    running_total = 0

    for key in keys:
        sec = SECTIONS.get(key)
        if sec is None:
            results.append({"key": key, "error": "unknown section key"})
            continue

        text = sec.get("text", "") or ""
        remaining = GET_SECTIONS_CHAR_CAP - running_total
        truncated = False
        if remaining <= 0:
            text = ""
            truncated = True
        elif len(text) > remaining:
            text = text[:remaining]
            truncated = True
        running_total += len(text)

        record = {
            "key": sec["key"],
            "corpus": sec["corpus"],
            "heading_path": sec["heading_path"],
            "title": sec["title"],
            "text": text,
            "url": sec["url"],
        }
        if truncated:
            record["truncated"] = True
        results.append(record)

    return results


def _parse_frontmatter(raw: str) -> dict[str, str] | None:
    """Minimal flat `key: value` frontmatter parser (no PyYAML dependency)."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", raw, re.DOTALL)
    if not m:
        return None
    fm: dict[str, str] = {}
    for line in m.group(1).split("\n"):
        if not line.strip() or ":" not in line:
            continue
        k, _, v = line.partition(":")
        fm[k.strip()] = v.strip()
    return fm


def get_wiki_page(topic_slug: str) -> str | None:
    """Return the markdown body of app/wiki/{topic_slug}.md if it exists."""
    path = WIKI_DIR / f"{topic_slug}.md"
    if not path.exists() or not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8")
    m = re.match(r"^---\s*\n.*?\n---\s*\n?(.*)$", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw.strip()


def wiki_index() -> str:
    """Concatenated frontmatter summaries of all wiki pages: one block per
    topic with slug, title, synonyms, and section keys. Used in the system
    prompt. Returns '' if wiki/ is empty (the agent must work without it).
    """
    if not WIKI_DIR.exists() or not WIKI_DIR.is_dir():
        return ""

    blocks = []
    for path in sorted(WIKI_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        fm = _parse_frontmatter(raw)
        if not fm:
            continue
        lines = [
            f"slug: {fm.get('slug', path.stem)}",
            f"title: {fm.get('title', '')}",
        ]
        if fm.get("synonyms"):
            lines.append(f"synonyms: {fm['synonyms']}")
        if fm.get("sections"):
            lines.append(f"sections: {fm['sections']}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


_GENERIC_HEADINGS = {
    "definitions", "definition", "purpose", "applicability", "general",
    "general provisions", "enforcement", "penalties", "penalty",
    "severability", "fees", "scope", "intent", "authority", "findings",
}


def section_label(key: str) -> str:
    """Human-readable display label for a section.

    Walks the heading path from the most specific end and returns the first
    heading that isn't a generic subsection name ("Definitions", "Purpose",
    ...), so headerless subsections inherit their parent regulation's name
    instead of a body-text fallback title.
    """
    rec = SECTIONS.get(key)
    if not rec:
        return key

    def clean(text: str) -> str:
        label = (text or "").strip().rstrip(".:").strip()
        return label if len(label) <= 70 else label[:67] + "…"

    for heading in reversed(rec.get("heading_path") or []):
        cleaned = clean(heading)
        if cleaned and cleaned.casefold() not in _GENERIC_HEADINGS:
            return cleaned
    return clean(rec.get("title") or "") or key
