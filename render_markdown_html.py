#!/usr/bin/env python3
"""Render a local Markdown file to styled, readable standalone HTML.

This renderer is intentionally self-contained (no third-party markdown package).
It supports headings, paragraphs, lists, tables, horizontal rules, and inline
formatting used by the Somerville law markdown output.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
LIST_RE = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\s-]", "", value).strip().lower()
    slug = re.sub(r"\s+", "-", slug)
    return slug or "section"


def split_table_row(line: str) -> list[str]:
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    cells = re.split(r"(?<!\\)\|", inner)
    return [cell.strip().replace("\\|", "|") for cell in cells]


def apply_inline_formatting(text: str) -> str:
    escaped = html.escape(text)

    def link_sub(match: re.Match[str]) -> str:
        label = match.group(1)
        url = match.group(2)
        return f'<a href="{url}">{label}</a>'

    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_sub, escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
    return escaped


def render_table(lines: list[str]) -> str:
    if len(lines) < 2:
        return ""

    header = split_table_row(lines[0])
    body_rows = [split_table_row(line) for line in lines[2:]]

    parts = ["<table>", "<thead>", "<tr>"]
    for cell in header:
        parts.append(f"<th>{apply_inline_formatting(cell)}</th>")
    parts.extend(["</tr>", "</thead>"])

    if body_rows:
        parts.append("<tbody>")
        for row in body_rows:
            parts.append("<tr>")
            for cell in row:
                parts.append(f"<td>{apply_inline_formatting(cell)}</td>")
            parts.append("</tr>")
        parts.append("</tbody>")

    parts.append("</table>")
    return "\n".join(parts)


def render_list(lines: list[str]) -> str:
    items: list[tuple[int, bool, str]] = []
    for line in lines:
        m = LIST_RE.match(line)
        if not m:
            continue
        spaces, marker, content = m.groups()
        level = len(spaces.replace("\t", "  ")) // 2
        ordered = marker.endswith(".")
        items.append((level, ordered, content.strip()))

    if not items:
        return ""

    out: list[str] = []
    stack: list[dict[str, object]] = []

    def close_one() -> None:
        top = stack.pop()
        if top["li_open"]:
            out.append("</li>")
        out.append(f"</{top['tag']}>")

    for level, ordered, content in items:
        desired_tag = "ol" if ordered else "ul"

        while stack and (
            level < int(stack[-1]["level"])
            or (level == int(stack[-1]["level"]) and desired_tag != str(stack[-1]["tag"]))
        ):
            close_one()

        if not stack or level > int(stack[-1]["level"]):
            start_level = int(stack[-1]["level"]) + 1 if stack else 0
            for current_level in range(start_level, level + 1):
                tag = desired_tag if current_level == level else "ul"
                out.append(f"<{tag}>")
                stack.append({"level": current_level, "tag": tag, "li_open": False})
        elif stack[-1]["li_open"]:
            out.append("</li>")
            stack[-1]["li_open"] = False

        out.append(f"<li>{apply_inline_formatting(content)}")
        stack[-1]["li_open"] = True

    while stack:
        close_one()

    return "\n".join(out)


def markdown_to_html(markdown_text: str) -> tuple[str, list[tuple[int, str, str]]]:
    lines = markdown_text.splitlines()
    idx = 0
    rendered: list[str] = []
    toc: list[tuple[int, str, str]] = []
    slug_counts: dict[str, int] = {}

    def next_slug(text: str) -> str:
        base = slugify(text)
        count = slug_counts.get(base, 0)
        slug_counts[base] = count + 1
        return base if count == 0 else f"{base}-{count + 1}"

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        if not stripped:
            idx += 1
            continue

        if stripped.startswith("<!--") and stripped.endswith("-->"):
            secid_match = re.match(r"^<!--\s*secid:(\d+)\s*-->$", stripped)
            if secid_match:
                secid = secid_match.group(1)
                rendered.append(f'<a id="secid-{secid}" class="secid-anchor" data-secid="{secid}"></a>')
            idx += 1
            continue

        heading_match = HEADING_RE.match(line)
        if heading_match:
            hashes, heading_text = heading_match.groups()
            level = len(hashes)
            heading_text = heading_text.strip()
            heading_id = next_slug(heading_text)
            rendered.append(f"<h{level} id=\"{heading_id}\">{apply_inline_formatting(heading_text)}</h{level}>")
            if 2 <= level <= 5:
                toc.append((level, heading_text, heading_id))
            idx += 1
            continue

        if stripped == "---":
            rendered.append("<hr>")
            idx += 1
            continue

        if stripped.startswith("|") and idx + 1 < len(lines) and re.match(r"^\|?\s*:?-{3,}", lines[idx + 1].strip()):
            table_lines = [line]
            idx += 1
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                table_lines.append(lines[idx])
                idx += 1
            rendered.append(render_table(table_lines))
            continue

        if LIST_RE.match(line):
            list_lines = [line]
            idx += 1
            while idx < len(lines) and lines[idx].strip() and LIST_RE.match(lines[idx]):
                list_lines.append(lines[idx])
                idx += 1
            rendered.append(render_list(list_lines))
            continue

        paragraph_lines = [stripped]
        idx += 1
        while idx < len(lines):
            nxt = lines[idx].strip()
            if not nxt:
                break
            if HEADING_RE.match(lines[idx]) or LIST_RE.match(lines[idx]) or nxt.startswith("<!--") or nxt == "---":
                break
            if nxt.startswith("|") and idx + 1 < len(lines) and re.match(r"^\|?\s*:?-{3,}", lines[idx + 1].strip()):
                break
            paragraph_lines.append(nxt)
            idx += 1
        paragraph_text = " ".join(paragraph_lines)
        rendered.append(f"<p>{apply_inline_formatting(paragraph_text)}</p>")

    return "\n".join(rendered), toc


def build_toc_tree(toc: list[tuple[int, str, str]]) -> list[dict]:
    """Nest flat (level, label, target) headings into a tree by heading level."""
    roots: list[dict] = []
    stack: list[dict] = []
    for level, label, target in toc:
        node = {"level": level, "label": label, "target": target, "children": []}
        while stack and stack[-1]["level"] >= level:
            stack.pop()
        (stack[-1]["children"] if stack else roots).append(node)
        stack.append(node)

    # A lone root (e.g. a single document-title heading above every part) adds
    # a nesting level without aiding navigation: keep it as a plain link and
    # promote its children to the top level.
    if len(roots) == 1 and roots[0]["children"]:
        root = roots[0]
        roots = [{**root, "children": []}] + root["children"]
    return roots


def render_toc_nodes(nodes: list[dict]) -> str:
    parts = ["<ul>"]
    for node in nodes:
        link = f'<a href="#{node["target"]}">{html.escape(node["label"])}</a>'
        if node["children"]:
            parts.append(f'<li class="lvl-{node["level"]}"><details><summary>{link}</summary>')
            parts.append(render_toc_nodes(node["children"]))
            parts.append("</details></li>")
        else:
            parts.append(f'<li class="lvl-{node["level"]}">{link}</li>')
    parts.append("</ul>")
    return "\n".join(parts)


TOC_SCRIPT = """<script>
(function () {
  function tocLinkFor(id) {
    return document.querySelector('.toc a[href="#' + id.replace(/"/g, '\\\\"') + '"]');
  }

  function nearestTocHeadingBefore(el) {
    var headings = document.querySelectorAll("main h2[id], main h3[id], main h4[id], main h5[id]");
    var best = null;
    for (var i = 0; i < headings.length; i++) {
      var h = headings[i];
      if (h === el || h.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING) {
        best = h;
      } else {
        break;
      }
    }
    return best;
  }

  function expandFor(hash) {
    if (!hash || hash.length < 2) return;
    var id;
    try { id = decodeURIComponent(hash.slice(1)); } catch (e) { return; }
    var link = tocLinkFor(id);
    if (!link) {
      // Deep link into content without its own ToC entry (e.g. a section
      // anchor): expand the branch of the nearest heading above it.
      var target = document.getElementById(id);
      if (!target) return;
      var heading = nearestTocHeadingBefore(target);
      if (heading) link = tocLinkFor(heading.id);
    }
    if (!link) return;
    for (var d = link.closest("details"); d; d = d.parentElement.closest("details")) {
      d.open = true;
    }
    link.scrollIntoView({ block: "nearest" });
  }

  expandFor(location.hash);
  window.addEventListener("hashchange", function () { expandFor(location.hash); });

  // On a document this large the browser jumps to the hash target while
  // content is still rendering, so late layout shifts leave the viewport a
  // chapter away from the anchor. Re-scroll once everything has loaded (and
  // once more shortly after, for font/layout settling).
  function rescrollToHash() {
    if (!location.hash || location.hash.length < 2) return;
    var id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch (e) { return; }
    var target = document.getElementById(id);
    if (target) target.scrollIntoView({ block: "start" });
  }
  window.addEventListener("load", function () {
    rescrollToHash();
    setTimeout(rescrollToHash, 300);
  });
})();
</script>"""


def build_html_document(title: str, body_html: str, toc: list[tuple[int, str, str]]) -> str:
    toc_html = render_toc_nodes(build_toc_tree(toc))

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f4ee;
      --paper: #ffffff;
      --ink: #1f2933;
      --muted: #5f6c7b;
      --rule: #d8dee5;
      --accent: #0a5a8f;
      --toc-bg: #f0efe9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top left, #efece2 0%, var(--bg) 35%);
      color: var(--ink);
      font-family: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
      line-height: 1.62;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 1.25rem;
      max-width: 1300px;
      margin: 0 auto;
      padding: 1rem;
    }}
    .toc {{
      position: sticky;
      top: 0.75rem;
      align-self: start;
      max-height: calc(100vh - 1.5rem);
      overflow: auto;
      padding: 1rem;
      border: 1px solid var(--rule);
      border-radius: 12px;
      background: var(--toc-bg);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    }}
    .toc h2 {{ margin: 0 0 0.75rem 0; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
    .toc ul {{ list-style: none; margin: 0; padding: 0; }}
    .toc ul ul {{ padding-left: 0.9rem; }}
    .toc li {{ margin: 0.25rem 0; font-size: 0.9rem; }}
    .toc a {{ color: #134e7a; text-decoration: none; }}
    .toc a:hover {{ text-decoration: underline; }}
    .toc details {{ margin: 0; }}
    .toc summary {{ cursor: pointer; }}
    .toc summary::marker {{ color: var(--muted); font-size: 0.8em; }}
    main {{
      background: var(--paper);
      border: 1px solid var(--rule);
      border-radius: 14px;
      padding: clamp(1rem, 2vw, 2rem);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06);
    }}
    h1, h2, h3, h4, h5, h6 {{
      font-family: "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif;
      line-height: 1.25;
      margin-top: 1.45em;
      margin-bottom: 0.45em;
      color: #17212c;
      scroll-margin-top: 1rem;
    }}
    h1 {{ font-size: clamp(1.7rem, 2.2vw, 2.25rem); border-bottom: 2px solid var(--rule); padding-bottom: 0.35rem; margin-top: 0.25rem; }}
    h2 {{ font-size: 1.45rem; border-bottom: 1px solid var(--rule); padding-bottom: 0.25rem; }}
    h3 {{ font-size: 1.2rem; color: #0f3552; }}
    h4 {{ font-size: 1.1rem; color: #1a3a50; text-transform: uppercase; letter-spacing: 0.03em; }}
    h5 {{ font-size: 1.05rem; color: #1a3a50; margin-top: 2em; padding-bottom: 0.15rem; border-bottom: 1px solid #e4e8ec; }}
    h6 {{ font-size: 0.92rem; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.02em; margin-top: 1.5em; }}
    p {{ margin: 0.65em 0; }}
    a {{ color: var(--accent); }}
    hr {{ border: 0; border-top: 1px solid var(--rule); margin: 1.8rem 0; }}
    ul, ol {{ margin: 0.55em 0 0.75em 1.25rem; padding-left: 0.75rem; }}
    li {{ margin: 0.25em 0; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 0.9em;
      background: #eef3f7;
      border: 1px solid #d6e1ea;
      border-radius: 4px;
      padding: 0.08rem 0.28rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 1rem 0 1.3rem;
      font-size: 0.95rem;
      background: #fbfcfd;
    }}
    th, td {{
      border: 1px solid #cfd8e2;
      padding: 0.4rem 0.5rem;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #eef3f8; font-family: "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif; }}
    .secid-anchor {{ display: block; position: relative; top: -0.2rem; visibility: hidden; }}

    @media (min-width: 1060px) {{
      .layout {{ grid-template-columns: 280px minmax(0, 1fr); }}
    }}

    @media (max-width: 1059px) {{
      .toc {{ position: static; max-height: none; }}
    }}

    @media print {{
      body {{ background: #fff; }}
      .layout {{ max-width: none; padding: 0; display: block; }}
      .toc {{ display: none; }}
      main {{ box-shadow: none; border: 0; border-radius: 0; padding: 0; }}
      a {{ color: inherit; text-decoration: none; }}
    }}
  </style>
</head>
<body>
  <div class=\"layout\">
    <nav class=\"toc\" aria-label=\"On this page\">
      <h2>Navigation</h2>
      {toc_html}
    </nav>
    <main>
      {body_html}
    </main>
  </div>
  {TOC_SCRIPT}
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render markdown to readable standalone HTML.")
    parser.add_argument(
        "--input",
        default="somerville-law-non-zoning.md",
        help="Input markdown path (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="somerville-law-non-zoning.readable.html",
        help="Output HTML path (default: %(default)s)",
    )
    parser.add_argument(
        "--title",
        default="Somerville Municipal Law (Readable Edition)",
        help="HTML <title> value",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    markdown_text = input_path.read_text(encoding="utf-8")
    body_html, toc = markdown_to_html(markdown_text)
    html_doc = build_html_document(args.title, body_html, toc)
    output_path.write_text(html_doc, encoding="utf-8")

    print(f"[ok] Rendered {input_path} -> {output_path}")
    print(f"[ok] TOC entries: {len(toc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
