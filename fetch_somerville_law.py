#!/usr/bin/env python3
"""Fetch and consolidate Somerville's non-zoning municipal law into one Markdown.

By default this script combines:

- PART I CHARTER AND RELATED LEGISLATION
- PART II CODE OF ORDINANCES
- APPENDICES B, D, and E

It excludes Somerville's separate zoning publication.

The script also makes a best-effort attempt to download a host-exported PDF for a
selected tocid (default: PART II).
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from combine_law import combine, DEFAULT_OUTPUT, DEFAULT_ZONING

BASE_URL = "https://online.encodeplus.com/regs/somerville-ma-coo"
PART_II_TOCID = "001.004"

# Default consolidated scope for non-zoning municipal law.
DEFAULT_TOCIDS: tuple[str, ...] = (
    "001.003",  # PART I CHARTER AND RELATED LEGISLATION
    "001.004",  # PART II CODE OF ORDINANCES
    "001.008",  # APPENDIX B RULES OF THE CITY COUNCIL
    "001.012",  # APPENDIX D BOARD OF HEALTH REGULATIONS
    "001.013",  # APPENDIX E TRAFFIC COMMISSION REGULATIONS
)


@dataclass(frozen=True)
class TocDocument:
    tocid: str
    source_url: str
    title: str
    breadcrumbs: list[str]
    section_count: int
    rendered_markdown: str
    raw_html: str


def clean_text(value: str) -> str:
    """Normalize whitespace for stable plain-text/markdown rendering."""
    value = html.unescape(value)
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    return value.strip()


def collapse_blank_lines(value: str) -> str:
    """Collapse excessive blank lines after markdown assembly."""
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip() + "\n"


def remove_non_substantive_lines(value: str) -> str:
    """Remove publication metadata lines that are not part of substantive law text."""
    metadata_patterns = (
        re.compile(r"^\(\s*Ord\.\s*No\..*\)\s*$", flags=re.IGNORECASE),
        re.compile(r"^Effective\s+on:\s*.+$", flags=re.IGNORECASE),
        re.compile(r"^\*?\s*editor(?:'|’)?s?\s+note(?:\(s\))?\s*[-—–:].*$", flags=re.IGNORECASE),
    )
    parenthetical_history_token = re.compile(
        r"\b("
        r"ord\."
        r"|acts?\s+\d{4}"
        r"|code\s+\d{4}"
        r"|ords?\."
        r"|supp\."
        r"|approved"
        r"|adopted"
        r"|amended"
        r"|pending\s+repeal"
        r"|res\.?\s*no\.?"
        r")\b",
        flags=re.IGNORECASE,
    )

    kept: list[str] = []
    for line in value.splitlines():
        stripped = line.strip()
        if stripped and any(pattern.match(stripped) for pattern in metadata_patterns):
            continue
        if stripped.startswith("(") and stripped.endswith(")"):
            inside = stripped[1:-1].strip()
            if inside and parenthetical_history_token.search(inside):
                continue
        kept.append(line)
    return "\n".join(kept)


def md_escape_cell(value: str) -> str:
    return value.replace("|", "\\|")


def node_text(node: Tag) -> str:
    return clean_text(node.get_text(" ", strip=True))


def render_table(table: Tag) -> str:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        row = [md_escape_cell(node_text(cell)) for cell in cells]
        rows.append(row)

    if not rows:
        return ""

    width = max(len(r) for r in rows)
    normalized = [r + [""] * (width - len(r)) for r in rows]

    # Use first row as header when table has any <th>, otherwise use blank header.
    has_header_cells = bool(table.find("th"))
    if has_header_cells:
        header = normalized[0]
        body = normalized[1:]
    else:
        header = ["" for _ in range(width)]
        body = normalized

    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for row in body:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n\n"


def li_label(li: Tag) -> str | None:
    num = li.find(class_="li-num")
    if num:
        txt = node_text(num)
        return txt if txt else None
    return None


def render_list(list_tag: Tag, indent: int = 0) -> str:
    lines: list[str] = []
    ordered = list_tag.name == "ol"
    index = 1

    for li in list_tag.find_all("li", recursive=False):
        marker = f"{index}." if ordered else "-"
        custom_label = li_label(li)

        content_parts: list[str] = []
        nested_blocks: list[str] = []

        for child in li.children:
            if isinstance(child, NavigableString):
                text = clean_text(str(child))
                if text:
                    content_parts.append(text)
                continue

            if not isinstance(child, Tag):
                continue

            if child.name in {"ol", "ul"}:
                nested_blocks.append(render_list(child, indent + 2).rstrip())
                continue

            if "li-num" in child.get("class", []):
                continue

            if "li-cont" in child.get("class", []):
                text = node_text(child)
                if text:
                    content_parts.append(text)
                continue

            if child.name == "p":
                text = node_text(child)
                if text:
                    content_parts.append(text)
                continue

            if child.name == "table":
                nested_blocks.append(render_table(child).rstrip())
                continue

            text = node_text(child)
            if text:
                content_parts.append(text)

        joined = " ".join(part for part in content_parts if part)
        if custom_label and joined and not joined.startswith(custom_label):
            joined = f"{custom_label} {joined}"
        if not joined:
            joined = custom_label or ""

        prefix = " " * indent + marker + " "
        lines.append(prefix + joined)

        for block in nested_blocks:
            if block:
                for block_line in block.splitlines():
                    lines.append(" " * (indent + 2) + block_line)

        if ordered:
            index += 1

    return "\n".join(lines) + "\n\n" if lines else ""


def render_children(nodes: Iterable[Tag | NavigableString], heading_shift: int = 0) -> str:
    parts: list[str] = []

    for node in nodes:
        if isinstance(node, NavigableString):
            text = clean_text(str(node))
            if text:
                parts.append(text + "\n\n")
            continue

        if not isinstance(node, Tag):
            continue

        if node.name in {"script", "style", "noscript"}:
            continue

        if node.get("id") == "sectionBanner":
            continue

        classes = set(node.get("class", []))
        if "mini-TOC" in classes:
            continue

        if node.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(node.name[1]) + heading_shift
            level = min(level, 6)
            heading = node_text(node)
            if heading:
                parts.append(f"{'#' * level} {heading}\n\n")
            continue

        if node.name == "p":
            text = node_text(node)
            if text:
                parts.append(text + "\n\n")
            continue

        if node.name in {"ol", "ul"}:
            parts.append(render_list(node))
            continue

        if node.name == "table":
            parts.append(render_table(node))
            continue

        if node.name == "section":
            parts.append(render_section(node, heading_shift=heading_shift))
            continue

        parts.append(render_children(list(node.children), heading_shift=heading_shift))

    return "".join(parts)


def render_section(section: Tag, heading_shift: int = 0) -> str:
    secid_attr = section.get("data-secid")
    secid_text = f"<!-- secid:{secid_attr} -->\n" if secid_attr else ""
    body = render_children(list(section.children), heading_shift=heading_shift)
    if not body.strip():
        return ""
    return secid_text + body + "\n"


def parse_export_component_url(export_html: str) -> str | None:
    match = re.search(r"\$\.get\('([^']*component\.aspx[^']*)'", export_html)
    if not match:
        return None
    return match.group(1)


def try_download_toc_pdf(
    session: requests.Session, output_pdf: Path, tocid: str, timeout_seconds: int = 90
) -> tuple[bool, str]:
    """Attempt PDF export for a tocid via host-provided export flow.

    Returns (success, message).
    """
    export_url = f"{BASE_URL}/export2doc.aspx?pdf=1&tocid={tocid}"
    resp = session.get(export_url, timeout=45)
    resp.raise_for_status()

    content_type = (resp.headers.get("content-type") or "").lower()
    # The host sometimes responds with the PDF immediately, bypassing polling JS.
    if "pdf" in content_type:
        output_pdf.write_bytes(resp.content)
        return True, f"Downloaded PDF directly from {export_url}"

    component_path = parse_export_component_url(resp.text)
    if not component_path:
        return False, "Could not locate export polling endpoint in export page."

    component_url = urljoin(BASE_URL + "/", component_path)
    deadline = time.time() + timeout_seconds
    last_msg = "Export did not become ready in time."

    while time.time() < deadline:
        poll_resp = session.get(component_url, timeout=45)
        poll_resp.raise_for_status()
        body = poll_resp.text.strip()
        if not body:
            time.sleep(2)
            continue

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            last_msg = f"Unexpected export poll response: {body[:200]}"
            time.sleep(2)
            continue

        if payload.get("Failed"):
            return False, payload.get("Msg") or "Export endpoint reported failure."

        if not payload.get("Ready"):
            if payload.get("Msg"):
                last_msg = str(payload.get("Msg"))
            time.sleep(2)
            continue

        export_file = payload.get("File")
        if not export_file:
            return False, "Export was marked ready but no file path was returned."

        file_url = urljoin(export_url, export_file)
        file_resp = session.get(file_url, timeout=120)
        file_resp.raise_for_status()
        content_type = file_resp.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not file_url.lower().endswith(".pdf"):
            return False, f"Export file was not a PDF (content-type={content_type!r})."

        output_pdf.write_bytes(file_resp.content)
        return True, f"Downloaded PDF from {file_url}"

    return False, last_msg


def fetch_toc_html(session: requests.Session, tocid: str) -> str:
    source_url = f"{BASE_URL}/doc-view.aspx?tocid={tocid}"
    resp = session.get(source_url, timeout=120)
    resp.raise_for_status()
    return resp.text


def extract_breadcrumbs(soup: BeautifulSoup) -> list[str]:
    container = soup.select_one("div.breadCrumbs")
    if not container:
        return []
    crumbs = [clean_text(a.get_text(" ", strip=True)) for a in container.find_all("a")]
    return [c for c in crumbs if c]


def parse_html_document(html_text: str) -> tuple[str, list[str], str, int]:
    soup = BeautifulSoup(html_text, "html.parser")
    the_page = soup.select_one("#thePage")
    if not the_page:
        raise RuntimeError("Could not find #thePage in source HTML.")

    sections = the_page.find_all("section", recursive=False)
    # Fallback for unexpected structure.
    if not sections:
        sections = the_page.find_all("section")

    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else "Untitled"
    breadcrumbs = extract_breadcrumbs(soup)

    body_parts: list[str] = []
    for section in sections:
        body_parts.append(render_section(section, heading_shift=1))

    return title, breadcrumbs, "".join(body_parts), len(sections)


def build_markdown_document(documents: list[TocDocument]) -> tuple[str, int]:
    if not documents:
        raise RuntimeError("No documents were provided for markdown build.")

    retrieved = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    total_sections = sum(doc.section_count for doc in documents)

    header_lines = [
        "# Somerville Municipal Law (Non-Zoning)",
        "",
        "This document consolidates Somerville's **Part I Charter**, **Part II Code of Ordinances**, and **Appendices B, D, and E** from the enCodePlus municipal code publication.",
        "",
        f"- Source publication root: `{BASE_URL}/doc-viewer.aspx?tocid=001`",
        f"- Retrieved: `{retrieved}`",
        "- Scope: `PART I + PART II + APPENDICES B, D, E` (non-zoning)",
        "- Excluded: Separate Somerville zoning publication",
        "",
        "Included publication roots:",
    ]

    for doc in documents:
        header_lines.append(f"- `{doc.tocid}` — {doc.title}")

    header_lines.extend(["", "---", ""])

    body_parts: list[str] = []
    for idx, doc in enumerate(documents):
        breadcrumbs_text = " > ".join(doc.breadcrumbs)
        body_parts.append(
            f"<!-- tocid:{doc.tocid} title:{doc.title} source:{doc.source_url} breadcrumbs:{breadcrumbs_text} -->\n"
        )
        body_parts.append(doc.rendered_markdown)
        if idx != len(documents) - 1:
            body_parts.append("\n---\n\n")

    markdown = "\n".join(header_lines) + "".join(body_parts)
    markdown = remove_non_substantive_lines(markdown)
    markdown = collapse_blank_lines(markdown)
    return markdown, total_sections


def build_raw_html_bundle(documents: list[TocDocument]) -> str:
    parts: list[str] = []
    for doc in documents:
        parts.append(f"<!-- BEGIN TOCID {doc.tocid}: {doc.title} ({doc.source_url}) -->")
        parts.append(doc.raw_html.rstrip())
        parts.append(f"<!-- END TOCID {doc.tocid} -->")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def parse_tocids_arg(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("No tocid values were provided.")

    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate Somerville's non-zoning municipal law into one Markdown file."
    )
    parser.add_argument(
        "--tocids",
        default=",".join(DEFAULT_TOCIDS),
        help="Comma-separated tocid roots to include, in order (default: %(default)s)",
    )
    parser.add_argument(
        "--markdown-output",
        default="somerville-law-non-zoning.md",
        help="Output Markdown file path (default: %(default)s)",
    )
    parser.add_argument(
        "--html-output",
        default="somerville-law-non-zoning.raw.html",
        help="Optional raw HTML output path for auditability (default: %(default)s)",
    )
    parser.add_argument(
        "--pdf-output",
        default="somerville-code-ordinances-part-ii.pdf",
        help="Best-effort PDF output path for --pdf-tocid (default: %(default)s)",
    )
    parser.add_argument(
        "--pdf-tocid",
        default=PART_II_TOCID,
        help="tocid to use for best-effort host PDF export (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-pdf-attempt",
        action="store_true",
        help="Skip host PDF export attempt and only produce Markdown/HTML outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tocids = parse_tocids_arg(args.tocids)

    md_output = Path(args.markdown_output)
    raw_html_output = Path(args.html_output)
    pdf_output = Path(args.pdf_output)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
    )

    print(f"[info] Source publication: {BASE_URL}/doc-viewer.aspx?tocid=001")
    print(f"[info] Included tocids: {', '.join(tocids)}")

    if not args.skip_pdf_attempt:
        print(f"[info] Attempting host-provided PDF export for tocid {args.pdf_tocid}...")
        ok, msg = try_download_toc_pdf(session, pdf_output, args.pdf_tocid)
        if ok:
            size_kb = pdf_output.stat().st_size / 1024
            print(f"[ok] PDF export succeeded: {pdf_output} ({size_kb:.1f} KiB)")
        else:
            print(f"[warn] PDF export unavailable: {msg}")
    else:
        print("[info] PDF export attempt skipped by flag.")

    documents: list[TocDocument] = []
    for tocid in tocids:
        source_url = f"{BASE_URL}/doc-view.aspx?tocid={tocid}"
        print(f"[info] Fetching canonical HTML for tocid {tocid}...")
        html_text = fetch_toc_html(session, tocid)
        title, breadcrumbs, rendered_markdown, section_count = parse_html_document(html_text)
        documents.append(
            TocDocument(
                tocid=tocid,
                source_url=source_url,
                title=title,
                breadcrumbs=breadcrumbs,
                section_count=section_count,
                rendered_markdown=rendered_markdown,
                raw_html=html_text,
            )
        )
        print(f"[ok] Parsed {tocid}: title={title!r}, top-level sections={section_count}")

    raw_html_bundle = build_raw_html_bundle(documents)
    raw_html_output.write_text(raw_html_bundle, encoding="utf-8")
    print(f"[ok] Saved raw HTML: {raw_html_output}")

    print("[info] Converting HTML to consolidated Markdown...")
    markdown, section_count = build_markdown_document(documents)
    md_output.write_text(markdown, encoding="utf-8")
    print(f"[ok] Saved markdown: {md_output}")
    print(f"[ok] Top-level sections extracted across all documents: {section_count}")

    combine(md_output, Path(DEFAULT_ZONING), Path(DEFAULT_OUTPUT))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.HTTPError as exc:
        print(f"[error] HTTP error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except requests.RequestException as exc:
        print(f"[error] Request error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("[error] Interrupted.", file=sys.stderr)
        raise SystemExit(130)
