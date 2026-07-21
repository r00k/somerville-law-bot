"""Parses the Somerville law corpus markdown files into app/data/sections.json
and app/data/toc.txt.

Run from the repo root with: uv run python -m app.indexer
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"

READABLE_BASE = "https://somervillelawbot.com"

# (corpus prefix, source markdown filename, readable-HTML filename, display name)
# The third field is the URL path the app serves each readable edition at
# (see app/server.py), not the generated filename.
CORPORA = [
    ("coo", "somerville-law-non-zoning.md", "code",
     "CODE OF ORDINANCES (non-zoning)"),
    ("zon", "somerville-zoning.md", "zoning",
     "ZONING ORDINANCE"),
]

SECID_RE = re.compile(r"<!--\s*secid:(\d+)\s*-->")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
COMMENT_LINE_RE = re.compile(r"^<!--.*-->\s*$")
BLANK_RUN_RE = re.compile(r"\n{3,}")

FALLBACK_TITLE_LEN = 100


class ParsedSection(NamedTuple):
    key: str
    corpus: str
    heading_path: list[str]
    title: str
    text: str
    url: str


def _clean_body(lines: list[str]) -> str:
    """Drop HTML-comment-only lines (stray tocid/secid remnants) and collapse
    excess blank lines."""
    kept = [line for line in lines if not COMMENT_LINE_RE.match(line.strip())]
    body = "\n".join(kept)
    body = BLANK_RUN_RE.sub("\n\n", body)
    return body.strip()


def _fallback_title(body: str) -> str:
    """When a section has no markdown heading of its own, use a short prefix
    of its body text as a display/search title instead of leaving it blank."""
    if not body:
        return ""
    first_line = body.split("\n", 1)[0].strip()
    first_line = first_line.strip('"“” ')
    if len(first_line) > FALLBACK_TITLE_LEN:
        first_line = first_line[:FALLBACK_TITLE_LEN].rsplit(" ", 1)[0] + "…"
    return first_line


def parse_corpus(path: Path, prefix: str, url_base: str) -> dict[str, ParsedSection]:
    raw = path.read_text(encoding="utf-8")
    parts = SECID_RE.split(raw)
    # parts = [preamble, secid, content, secid, content, ...]
    # preamble (parts[0]) is document text before the first secid marker; ignored.
    ids = parts[1::2]
    contents = parts[2::2]

    sections: dict[str, ParsedSection] = {}
    stack: list[tuple[int, str]] = []  # (heading level, title)

    for secid, content in zip(ids, contents):
        lines = content.split("\n")
        idx = 0
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1

        heading_title: str | None = None
        heading_level: int | None = None
        if idx < len(lines):
            m = HEADING_RE.match(lines[idx])
            if m:
                heading_level = len(m.group(1))
                heading_title = m.group(2).strip()
                idx += 1

        body = _clean_body(lines[idx:])

        if heading_title is not None:
            while stack and stack[-1][0] >= heading_level:
                stack.pop()
            stack.append((heading_level, heading_title))
            heading_path = [t for _, t in stack]
            title = heading_title
        else:
            heading_path = [t for _, t in stack]
            title = _fallback_title(body)

        key = f"{prefix}:{secid}"
        sections[key] = ParsedSection(
            key=key,
            corpus=prefix,
            heading_path=heading_path,
            title=title,
            text=body,
            url=f"{url_base}#secid-{secid}",
        )

    return sections


def build_toc(sections_by_corpus: dict[str, dict[str, ParsedSection]]) -> str:
    lines: list[str] = []
    for prefix, _, _, display_name in CORPORA:
        sections = sections_by_corpus[prefix]
        lines.append(f"=== {display_name} ===")
        for sec in sections.values():
            depth = max(len(sec.heading_path) - 1, 0)
            indent = "  " * depth
            title = sec.title or "(untitled)"
            lines.append(f"{indent}{title} [{sec.key}]")
    return "\n".join(lines) + "\n"


def build_index() -> tuple[dict[str, ParsedSection], str]:
    all_sections: dict[str, ParsedSection] = {}
    sections_by_corpus: dict[str, dict[str, ParsedSection]] = {}
    for prefix, filename, readable_html, _ in CORPORA:
        path = REPO_ROOT / filename
        url_base = f"{READABLE_BASE}/{readable_html}"
        parsed = parse_corpus(path, prefix, url_base)
        sections_by_corpus[prefix] = parsed
        all_sections.update(parsed)
    toc_text = build_toc(sections_by_corpus)
    return all_sections, toc_text


def _run_sanity_checks(sections: dict[str, ParsedSection]) -> None:
    assert len(sections) > 3000, f"expected > 3000 sections, got {len(sections)}"

    corpora_present = {sec.corpus for sec in sections.values()}
    assert corpora_present == {"coo", "zon"}, f"missing corpora: {corpora_present}"

    leaf_blower_hits = [
        sec for sec in sections.values() if "leaf blower" in sec.title.lower()
    ]
    assert leaf_blower_hits, "no section title mentions 'leaf blower'"


def main() -> None:
    sections, toc_text = build_index()
    _run_sanity_checks(sections)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sections_path = DATA_DIR / "sections.json"
    toc_path = DATA_DIR / "toc.txt"

    sections_json = {key: sec._asdict() for key, sec in sections.items()}
    sections_path.write_text(
        json.dumps(sections_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    toc_path.write_text(toc_text, encoding="utf-8")

    per_corpus_counts = {}
    for sec in sections.values():
        per_corpus_counts[sec.corpus] = per_corpus_counts.get(sec.corpus, 0) + 1

    print(f"Wrote {sections_path} ({len(sections)} sections)")
    for prefix, count in sorted(per_corpus_counts.items()):
        print(f"  {prefix}: {count} sections")
    print(f"Wrote {toc_path} ({len(toc_text.splitlines())} lines)")


if __name__ == "__main__":
    main()
