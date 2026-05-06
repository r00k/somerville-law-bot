#!/usr/bin/env python3
"""Fetch and consolidate Somerville's zoning ordinance into one Markdown file.

This script targets Somerville's zoning publication on enCodePlus and produces a
local, searchable corpus optimized for LLM use:

- markdown with section markers and image placeholders,
- raw HTML bundle for auditability,
- image manifest JSON with source references.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

BASE_URL = "https://online.encodeplus.com/regs/somerville-ma"
ROOT_TOCID = "001"
MEDIA_TAGS = {"img", "figure", "svg", "object", "embed", "iframe"}


@dataclass(frozen=True)
class ImageReference:
    index: int
    secid: str | None
    tag: str
    src: str | None
    absolute_src: str | None
    alt: str | None
    title: str | None
    caption: str | None


@dataclass
class RenderContext:
    current_secid: str | None = None
    next_image_index: int = 1
    image_refs: list[ImageReference] = field(default_factory=list)


@dataclass(frozen=True)
class TocDocument:
    tocid: str
    source_url: str
    title: str
    breadcrumbs: list[str]
    section_count: int
    rendered_markdown: str
    raw_html: str
    image_refs: list[ImageReference]


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
    """Optionally remove publication metadata lines that are not substantive law."""
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


def media_src(media: Tag) -> str | None:
    if media.name == "img":
        src = media.get("src") or media.get("data-src")
    elif media.name == "object":
        src = media.get("data") or media.get("src")
    else:
        src = media.get("src")
    if not src:
        return None
    return clean_text(src)


def media_caption(media: Tag) -> str | None:
    if media.name == "figure":
        figcaption = media.find("figcaption")
        if figcaption:
            caption = node_text(figcaption)
            return caption or None

    parent_figure = media.find_parent("figure")
    if parent_figure:
        figcaption = parent_figure.find("figcaption")
        if figcaption:
            caption = node_text(figcaption)
            return caption or None
    return None


def render_media_placeholder(media: Tag, ctx: RenderContext) -> str:
    tag = media.name.lower()
    src = media_src(media)
    absolute_src = urljoin(BASE_URL + "/", src) if src else None
    alt = clean_text(media.get("alt", "")) or None
    title = clean_text(media.get("title", "")) or None
    caption = media_caption(media)

    index = ctx.next_image_index
    ctx.next_image_index += 1

    ctx.image_refs.append(
        ImageReference(
            index=index,
            secid=ctx.current_secid,
            tag=tag,
            src=src,
            absolute_src=absolute_src,
            alt=alt,
            title=title,
            caption=caption,
        )
    )

    kind = "Image" if tag in {"img", "figure", "svg"} else "Embedded media"
    fields = [f"{kind} {index}"]
    if alt:
        fields.append(f'alt="{alt}"')
    if caption and caption != alt:
        fields.append(f'caption="{caption}"')
    if src:
        fields.append(f"src={src}")

    return "[" + "; ".join(fields) + "]"


def iter_top_level_media(container: Tag) -> Iterable[Tag]:
    for media in container.find_all(MEDIA_TAGS):
        ancestor = media.parent
        nested = False
        while isinstance(ancestor, Tag) and ancestor is not container:
            if ancestor.name in MEDIA_TAGS:
                nested = True
                break
            ancestor = ancestor.parent
        if not nested:
            yield media


def extract_text_and_media(node: Tag, ctx: RenderContext) -> tuple[str, list[str]]:
    text_parts: list[str] = []
    media_parts: list[str] = []

    for child in node.children:
        if isinstance(child, NavigableString):
            text = clean_text(str(child))
            if text:
                text_parts.append(text)
            continue

        if not isinstance(child, Tag):
            continue

        if child.name in {"script", "style", "noscript"}:
            continue

        if child.name in MEDIA_TAGS:
            media_parts.append(render_media_placeholder(child, ctx))
            continue

        text = node_text(child)
        if text:
            text_parts.append(text)

        for media in iter_top_level_media(child):
            media_parts.append(render_media_placeholder(media, ctx))

    joined = " ".join(part for part in text_parts if part).strip()
    return joined, media_parts


def render_table_cell(cell: Tag, ctx: RenderContext) -> str:
    text = node_text(cell)
    media_parts = [render_media_placeholder(media, ctx) for media in iter_top_level_media(cell)]
    if media_parts:
        combined = f"{text} {' '.join(media_parts)}".strip() if text else " ".join(media_parts)
        return md_escape_cell(combined)
    return md_escape_cell(text)


def render_table(table: Tag, ctx: RenderContext) -> str:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        row = [render_table_cell(cell, ctx) for cell in cells]
        rows.append(row)

    if not rows:
        return ""

    width = max(len(r) for r in rows)
    normalized = [r + [""] * (width - len(r)) for r in rows]

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


def render_list(list_tag: Tag, ctx: RenderContext, indent: int = 0) -> str:
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
                nested_blocks.append(render_list(child, ctx, indent + 2).rstrip())
                continue

            if "li-num" in child.get("class", []):
                continue

            if child.name == "table":
                nested_blocks.append(render_table(child, ctx).rstrip())
                continue

            if child.name in MEDIA_TAGS:
                content_parts.append(render_media_placeholder(child, ctx))
                continue

            if "li-cont" in child.get("class", []) or child.name == "p":
                text, media = extract_text_and_media(child, ctx)
                if text:
                    content_parts.append(text)
                content_parts.extend(media)
                continue

            text = node_text(child)
            if text:
                content_parts.append(text)

            for media in iter_top_level_media(child):
                content_parts.append(render_media_placeholder(media, ctx))

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


def render_children(nodes: Iterable[Tag | NavigableString], ctx: RenderContext, heading_shift: int = 0) -> str:
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
            text, media = extract_text_and_media(node, ctx)
            if text:
                parts.append(text + "\n\n")
            for media_placeholder in media:
                parts.append(media_placeholder + "\n\n")
            continue

        if node.name in {"ol", "ul"}:
            parts.append(render_list(node, ctx))
            continue

        if node.name == "table":
            parts.append(render_table(node, ctx))
            continue

        if node.name == "section":
            parts.append(render_section(node, ctx, heading_shift=heading_shift))
            continue

        if node.name in MEDIA_TAGS:
            parts.append(render_media_placeholder(node, ctx) + "\n\n")
            continue

        parts.append(render_children(list(node.children), ctx, heading_shift=heading_shift))

    return "".join(parts)


def render_section(section: Tag, ctx: RenderContext, heading_shift: int = 0) -> str:
    secid_attr = section.get("data-secid")
    secid_text = f"<!-- secid:{secid_attr} -->\n" if secid_attr else ""

    previous_secid = ctx.current_secid
    ctx.current_secid = str(secid_attr) if secid_attr else None
    try:
        body = render_children(list(section.children), ctx, heading_shift=heading_shift)
    finally:
        ctx.current_secid = previous_secid

    if not body.strip():
        return ""
    return secid_text + body + "\n"


def parse_export_component_url(export_html: str) -> str | None:
    match = re.search(r"\$\.get\('([^']*component\.aspx[^']*)'", export_html)
    if not match:
        return None
    return match.group(1)


def try_download_toc_pdf(
    session: requests.Session, output_pdf: Path, tocid: str, timeout_seconds: int = 120
) -> tuple[bool, str]:
    """Attempt PDF export for a tocid via host-provided export flow."""
    export_url = f"{BASE_URL}/export2doc.aspx?pdf=1&tocid={tocid}"
    resp = session.get(export_url, timeout=45)
    resp.raise_for_status()

    content_type = (resp.headers.get("content-type") or "").lower()
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
        file_resp = session.get(file_url, timeout=180)
        file_resp.raise_for_status()
        content_type = file_resp.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not file_url.lower().endswith(".pdf"):
            return False, f"Export file was not a PDF (content-type={content_type!r})."

        output_pdf.write_bytes(file_resp.content)
        return True, f"Downloaded PDF from {file_url}"

    return False, last_msg


def fetch_toc_html(session: requests.Session, tocid: str) -> str:
    source_url = f"{BASE_URL}/doc-view.aspx?tocid={tocid}"
    resp = session.get(source_url, timeout=180)
    resp.raise_for_status()
    return resp.text


def extract_breadcrumbs(soup: BeautifulSoup) -> list[str]:
    container = soup.select_one("div.breadCrumbs")
    if not container:
        return []
    crumbs = [clean_text(a.get_text(" ", strip=True)) for a in container.find_all("a")]
    return [c for c in crumbs if c]


def parse_html_document(html_text: str) -> tuple[str, list[str], str, int, list[ImageReference]]:
    soup = BeautifulSoup(html_text, "html.parser")
    the_page = soup.select_one("#thePage")
    if not the_page:
        raise RuntimeError("Could not find #thePage in source HTML.")

    sections = the_page.find_all("section", recursive=False)
    if not sections:
        sections = the_page.find_all("section")

    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else "Untitled"
    breadcrumbs = extract_breadcrumbs(soup)

    ctx = RenderContext()
    body_parts: list[str] = []
    for section in sections:
        body_parts.append(render_section(section, ctx, heading_shift=1))

    return title, breadcrumbs, "".join(body_parts), len(sections), ctx.image_refs


def build_markdown_document(document: TocDocument, strip_metadata: bool) -> tuple[str, int]:
    retrieved = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    image_count = len(document.image_refs)

    header_lines = [
        "# Somerville Zoning Ordinance",
        "",
        "This document is a text-first extract of Somerville's zoning ordinance from enCodePlus, optimized for local search and LLM workflows.",
        "",
        f"- Source publication root: `{BASE_URL}/doc-viewer.aspx#secid--1`",
        f"- Source tocid: `{document.tocid}`",
        f"- Retrieved: `{retrieved}`",
        f"- Section count extracted: `{document.section_count}`",
        f"- Image placeholders emitted: `{image_count}`",
        "",
        "Image handling notes:",
        "- Images are represented inline with placeholders like `[Image N; ...]`.",
        "- Detailed image references are available in the companion JSON manifest.",
        "",
        "---",
        "",
    ]

    breadcrumbs_text = " > ".join(document.breadcrumbs)
    body = (
        f"<!-- tocid:{document.tocid} title:{document.title} source:{document.source_url} breadcrumbs:{breadcrumbs_text} -->\n"
        f"{document.rendered_markdown}"
    )

    markdown = "\n".join(header_lines) + body
    if strip_metadata:
        markdown = remove_non_substantive_lines(markdown)
    markdown = collapse_blank_lines(markdown)
    return markdown, document.section_count


def build_raw_html_bundle(document: TocDocument) -> str:
    parts = [
        f"<!-- BEGIN TOCID {document.tocid}: {document.title} ({document.source_url}) -->",
        document.raw_html.rstrip(),
        f"<!-- END TOCID {document.tocid} -->",
        "",
    ]
    return "\n".join(parts).rstrip() + "\n"


def build_image_manifest(image_refs: list[ImageReference]) -> list[dict[str, str | int | None]]:
    return [
        {
            "index": ref.index,
            "secid": ref.secid,
            "tag": ref.tag,
            "src": ref.src,
            "absolute_src": ref.absolute_src,
            "alt": ref.alt,
            "title": ref.title,
            "caption": ref.caption,
        }
        for ref in image_refs
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate Somerville's zoning ordinance into one Markdown file."
    )
    parser.add_argument(
        "--tocid",
        default=ROOT_TOCID,
        help="tocid root to include (default: %(default)s)",
    )
    parser.add_argument(
        "--markdown-output",
        default="somerville-zoning.md",
        help="Output Markdown file path (default: %(default)s)",
    )
    parser.add_argument(
        "--html-output",
        default="somerville-zoning.raw.html",
        help="Raw HTML output path for auditability (default: %(default)s)",
    )
    parser.add_argument(
        "--images-output",
        default="somerville-zoning.images.json",
        help="Image manifest JSON output path (default: %(default)s)",
    )
    parser.add_argument(
        "--pdf-output",
        default="somerville-zoning.pdf",
        help="Best-effort PDF output path for --tocid (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-pdf-attempt",
        action="store_true",
        help="Skip host PDF export attempt and only produce Markdown/HTML/JSON outputs.",
    )
    parser.add_argument(
        "--strip-metadata",
        action="store_true",
        help="Apply metadata-line cleanup (off by default for conservative zoning extraction).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    md_output = Path(args.markdown_output)
    raw_html_output = Path(args.html_output)
    images_output = Path(args.images_output)
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

    print(f"[info] Source publication: {BASE_URL}/doc-viewer.aspx#secid--1")
    print(f"[info] Included tocid: {args.tocid}")

    if not args.skip_pdf_attempt:
        print(f"[info] Attempting host-provided PDF export for tocid {args.tocid}...")
        ok, msg = try_download_toc_pdf(session, pdf_output, args.tocid)
        if ok:
            size_kb = pdf_output.stat().st_size / 1024
            print(f"[ok] PDF export succeeded: {pdf_output} ({size_kb:.1f} KiB)")
        else:
            print(f"[warn] PDF export unavailable: {msg}")
    else:
        print("[info] PDF export attempt skipped by flag.")

    source_url = f"{BASE_URL}/doc-view.aspx?tocid={args.tocid}"
    print(f"[info] Fetching canonical HTML for tocid {args.tocid}...")
    html_text = fetch_toc_html(session, args.tocid)
    title, breadcrumbs, rendered_markdown, section_count, image_refs = parse_html_document(html_text)

    document = TocDocument(
        tocid=args.tocid,
        source_url=source_url,
        title=title,
        breadcrumbs=breadcrumbs,
        section_count=section_count,
        rendered_markdown=rendered_markdown,
        raw_html=html_text,
        image_refs=image_refs,
    )
    print(
        f"[ok] Parsed tocid {args.tocid}: title={title!r}, top-level sections={section_count}, image placeholders={len(image_refs)}"
    )

    raw_html_bundle = build_raw_html_bundle(document)
    raw_html_output.write_text(raw_html_bundle, encoding="utf-8")
    print(f"[ok] Saved raw HTML: {raw_html_output}")

    print("[info] Converting HTML to consolidated Markdown...")
    markdown, extracted_sections = build_markdown_document(document, strip_metadata=args.strip_metadata)
    md_output.write_text(markdown, encoding="utf-8")
    print(f"[ok] Saved markdown: {md_output}")
    print(f"[ok] Top-level sections extracted: {extracted_sections}")

    image_manifest = build_image_manifest(image_refs)
    images_output.write_text(json.dumps(image_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] Saved image manifest: {images_output}")

    raw_ids = re.findall(r"data-secid=['\"](\d+)['\"]", raw_html_bundle)
    md_ids = re.findall(r"<!--\s*secid:(\d+)\s*-->", markdown)
    print(
        f"[ok] Section marker completeness: raw_total={len(raw_ids)} md_total={len(md_ids)} sequence_exact={raw_ids == md_ids}"
    )

    from combine_law import combine, DEFAULT_OUTPUT, DEFAULT_NON_ZONING

    combine(Path(DEFAULT_NON_ZONING), md_output, Path(DEFAULT_OUTPUT))

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
