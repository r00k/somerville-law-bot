"""Generates app/wiki/*.md topic pages via the Anthropic API.

Two phases:
  1. Assign  — claude-sonnet-5 maps a curated list of resident-facing topics
               to relevant section keys by reading app/data/toc.txt in chunks.
  2. Write   — claude-opus-4-8 (adaptive thinking) writes each topic page
               from the full text of that topic's assigned sections
               (app/data/sections.json).

Usage:
    uv run python -m app.wiki_build                    # build all topics
    uv run python -m app.wiki_build --topics a,b        # build a subset
    uv run python -m app.wiki_build --force             # rebuild existing pages
    uv run python -m app.wiki_build --dry-run           # phase 1 only, no writes

Requires ANTHROPIC_API_KEY in the environment (read by the SDK) and
app/data/sections.json + app/data/toc.txt — run `uv run python -m app.indexer`
first if those don't exist yet.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic

APP_DIR = Path(__file__).resolve().parent
SECTIONS_PATH = APP_DIR / "data" / "sections.json"
TOC_PATH = APP_DIR / "data" / "toc.txt"
WIKI_DIR = APP_DIR / "wiki"

ASSIGN_MODEL = "claude-sonnet-5"
WRITE_MODEL = "claude-opus-4-8"

TOC_CHUNK_LINES = 1500
MAX_SECTION_TEXT_CHARS = 120_000
MAX_RATE_LIMIT_ATTEMPTS = 3

# $ per million tokens: (input, output). Cache reads bill at 0.1x input;
# cache writes (5-minute ephemeral TTL, the default we use) bill at 1.25x input.
PRICING = {
    ASSIGN_MODEL: (3.0, 15.0),
    WRITE_MODEL: (5.0, 25.0),
}

FOOTER_LINE = (
    "*This summary is for general orientation only. The cited section text "
    "above is authoritative — always verify anything that matters against "
    "the linked sections, not this summary.*"
)


@dataclass(frozen=True)
class Topic:
    slug: str
    title: str
    synonyms: list[str]


# Curated starter list (~40 topics) from app/DESIGN.md Component 5. Synonyms
# aim at what residents actually type, not just legal vocabulary.
TOPICS: list[Topic] = [
    Topic("animals-chickens", "Animals & Chickens", ["chickens", "domestic fowl", "backyard chickens", "roosters", "livestock", "pets"]),
    Topic("leaf-blowers", "Leaf Blowers", ["leaf blower", "yard equipment", "landscaping noise", "gas blower", "electric blower"]),
    Topic("noise", "Noise", ["noise ordinance", "loud noise", "quiet hours", "construction noise", "noise complaint"]),
    Topic("snow-sidewalks", "Snow & Sidewalks", ["snow removal", "shoveling", "sidewalk snow", "ice removal", "snow shoveling"]),
    Topic("trash-recycling-composting", "Trash, Recycling & Composting", ["garbage", "trash pickup", "recycling bin", "compost", "waste collection", "trash day"]),
    Topic("parking-permits", "Parking Permits", ["resident parking", "parking sticker", "visitor parking", "permit parking"]),
    Topic("street-cleaning", "Street Cleaning", ["street sweeping", "alternate side parking", "sweeping schedule"]),
    Topic("adus-additions", "ADUs & Additions", ["accessory dwelling unit", "in-law apartment", "third story", "home addition", "granny flat"]),
    Topic("fences", "Fences", ["fence height", "fencing", "backyard fence", "property line fence"]),
    Topic("trees", "Trees", ["tree removal", "tree cutting", "street trees", "tree protection", "tree permit"]),
    Topic("short-term-rentals", "Short-Term Rentals", ["airbnb", "vrbo", "vacation rental", "short term rental registration"]),
    Topic("home-businesses", "Home Businesses", ["home occupation", "home business", "working from home", "cottage business"]),
    Topic("building-permits", "Building Permits", ["permit application", "renovation permit", "construction permit", "remodel"]),
    Topic("signs", "Signs", ["signage", "business sign", "sign permit", "storefront sign"]),
    Topic("pools", "Pools", ["swimming pool", "backyard pool", "pool fence", "above ground pool"]),
    Topic("rodent-control", "Rodent Control", ["rats", "mice", "pest control", "rodent"]),
    Topic("dumpsters", "Dumpsters", ["dumpster permit", "construction dumpster", "roll off container"]),
    Topic("food-trucks", "Food Trucks", ["mobile food vendor", "food cart", "street vendor"]),
    Topic("restaurants-licensing", "Restaurants & Licensing", ["restaurant permit", "food establishment", "common victualler"]),
    Topic("alcohol", "Alcohol", ["liquor license", "beer and wine", "alcohol permit", "bar license"]),
    Topic("tobacco-smoking", "Tobacco & Smoking", ["smoking ban", "vaping", "e-cigarette", "smoke free", "cigarettes"]),
    Topic("marijuana", "Marijuana", ["cannabis", "weed", "marijuana dispensary", "cannabis retail"]),
    Topic("bicycles-ebikes", "Bicycles & E-Bikes", ["bike lane", "electric bike", "scooter", "bicycle parking"]),
    Topic("sidewalks-obstructions", "Sidewalks & Obstructions", ["sidewalk obstruction", "blocking sidewalk", "sandwich board", "sidewalk cafe"]),
    Topic("yard-sales", "Yard Sales", ["garage sale", "tag sale", "stoop sale", "yard sale permit"]),
    Topic("fireworks", "Fireworks", ["firecrackers", "fireworks ban", "sparklers"]),
    Topic("dogs", "Dogs", ["dog leash", "dog license", "off leash", "dog park", "dog walking"]),
    Topic("cats", "Cats", ["cat license", "feral cats", "cat ownership"]),
    Topic("bees", "Bees", ["beekeeping", "honeybees", "backyard hives"]),
    Topic("demolition", "Demolition", ["demolition permit", "tear down", "demolition delay"]),
    Topic("historic-districts", "Historic Districts", ["historic preservation", "landmark", "historic district commission"]),
    Topic("affordable-housing", "Affordable Housing", ["inclusionary housing", "affordable unit", "low income housing"]),
    Topic("condo-conversion", "Condo Conversion", ["condominium conversion", "condo conversion ordinance", "converting apartments to condos", "condo conversion permit"]),
    Topic("tenants-rental-registration", "Tenants & Rental Registration", ["tenant rights", "landlord", "rental registration", "eviction", "lease"]),
    Topic("hours-of-operation", "Hours of Operation", ["business hours", "closing time", "operating hours"]),
    Topic("elections", "Elections", ["voting", "ballot", "municipal election", "voter registration"]),
    Topic("city-council", "City Council", ["city councilor", "council meeting", "ward councilor"]),
    Topic("mayor", "Mayor", ["mayor's term", "mayoral election", "office of the mayor"]),
    Topic("taxes-fees", "Taxes & Fees", ["property tax", "excise tax", "municipal fees", "tax rate"]),
    Topic("wage-theft", "Wage Theft", ["unpaid wages", "wage enforcement", "labor violations"]),
    Topic("plastic-bags-polystyrene", "Plastic Bags & Polystyrene", ["plastic bag ban", "styrofoam", "single use plastic", "polystyrene ban"]),
    Topic("utilities-digging", "Utilities & Digging", ["street opening permit", "digging permit", "utility work", "excavation"]),
    Topic("graffiti", "Graffiti", ["graffiti removal", "vandalism", "tagging"]),
    Topic("abandoned-vehicles", "Abandoned Vehicles", ["junk car", "abandoned car", "unregistered vehicle"]),
]

TOPICS_BY_SLUG = {t.slug: t for t in TOPICS}

ASSIGN_INSTRUCTIONS = (
    "You are indexing a municipal law table of contents to build a resident-facing topic "
    "wiki for the City of Somerville, MA. You will be given the full topic list once "
    "(cached) and then the table of contents in sequential chunks. For each chunk, decide "
    "which topics are relevant to which section keys appearing in that chunk. A section "
    "may be relevant to multiple topics; a topic may have zero relevant sections in a "
    "given chunk — that's expected. Only flag a section if its heading or position "
    "plausibly deals with that topic; err toward inclusion on genuinely plausible matches, "
    "since a missed section creates a gap in the wiki, but do not flag sections that are "
    "only tangentially related. Municipal law vocabulary differs from how residents talk "
    "about it (e.g. \"chickens\" maps to \"domestic fowl\"; \"AirBnB\" maps to \"short-term "
    "rental\") — use those synonyms when matching. Respond with a JSON object mapping "
    "EVERY topic slug to an array of section keys (formatted exactly as they appear in the "
    "chunk, e.g. \"coo:1234\" or \"zon:56\") found in this chunk. Use an empty array for "
    "topics with no matches in this chunk."
)

WRITE_SYSTEM_INSTRUCTIONS = (
    "You are writing one page of a plain-language topic wiki about Somerville, MA "
    "municipal law, for residents with no legal background. You will be given a topic "
    "(with common synonyms) and the full text of the law sections identified as relevant "
    "to it. Write an 8-25 line plain-language summary covering: what's regulated, the key "
    "rules with specific section numbers cited inline (e.g. \"Sec. 9-120\"), common "
    "gotchas or exceptions, and cross-references to related topics where relevant. Base "
    "every claim strictly on the provided section text — never invent or infer a rule "
    "that isn't there. If the provided sections don't clearly answer a predictable "
    "resident question, say so plainly rather than guessing. If no sections were provided "
    "or they're clearly irrelevant, say the corpus doesn't appear to address this topic "
    "directly, in 1-3 lines."
)

PAGE_FORMAT_TEMPLATE = (
    "Output ONLY the markdown body of the wiki page (plain paragraphs and/or a short "
    "list) — no frontmatter, no top-level heading, no code fence. Do not add your own "
    "disclaimer or footer line; the caller appends a standard one automatically. Just "
    "write the 8-25 line summary itself."
)


def load_data() -> tuple[dict, str]:
    missing = [p for p in (SECTIONS_PATH, TOC_PATH) if not p.exists()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        sys.exit(
            f"Missing required data file(s): {names}\n"
            "Run `uv run python -m app.indexer` first to build the section index, "
            "then re-run this script."
        )
    sections = json.loads(SECTIONS_PATH.read_text())
    toc_text = TOC_PATH.read_text()
    return sections, toc_text


def chunk_toc(toc_text: str, chunk_lines: int) -> list[str]:
    lines = toc_text.splitlines()
    return ["\n".join(lines[i : i + chunk_lines]) for i in range(0, len(lines), chunk_lines)]


def build_assign_schema(topics: list[Topic]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [t.slug for t in topics],
        "properties": {
            t.slug: {"type": "array", "items": {"type": "string"}}
            for t in topics
        },
    }


def build_topic_list_block(topics: list[Topic]) -> str:
    lines = ["Topics (slug: title — synonyms):"]
    for t in topics:
        lines.append(f"- {t.slug}: {t.title} — {', '.join(t.synonyms)}")
    return "\n".join(lines)


def track_usage(usage_totals: dict, model: str, usage) -> None:
    """Accumulate usage for `model` into usage_totals and print running totals/cost."""
    totals = usage_totals.setdefault(
        model, {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}
    )
    totals["input"] += usage.input_tokens or 0
    totals["output"] += usage.output_tokens or 0
    totals["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
    totals["cache_creation"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
    cost = estimate_cost(usage_totals)
    print(
        f"    [usage] {model}: +in={usage.input_tokens} +out={usage.output_tokens} "
        f"+cache_read={getattr(usage, 'cache_read_input_tokens', 0) or 0} "
        f"+cache_write={getattr(usage, 'cache_creation_input_tokens', 0) or 0} "
        f"| running cost~${cost:.3f}"
    )


def estimate_cost(usage_totals: dict) -> float:
    total = 0.0
    for model, t in usage_totals.items():
        in_price, out_price = PRICING[model]
        total += t["input"] / 1_000_000 * in_price
        total += t["output"] / 1_000_000 * out_price
        total += t["cache_read"] / 1_000_000 * (in_price * 0.1)
        total += t["cache_creation"] / 1_000_000 * (in_price * 1.25)
    return total


def call_with_retry(fn, **kwargs):
    """Call fn(**kwargs), retrying on RateLimitError (respecting retry-after), max 3 tries.
    Any other exception surfaces immediately."""
    for attempt in range(1, MAX_RATE_LIMIT_ATTEMPTS + 1):
        try:
            return fn(**kwargs)
        except anthropic.RateLimitError as e:
            if attempt >= MAX_RATE_LIMIT_ATTEMPTS:
                raise
            retry_after = 5.0
            try:
                retry_after = float(e.response.headers.get("retry-after", "5"))
            except Exception:
                pass
            print(
                f"    [rate limit] attempt {attempt}/{MAX_RATE_LIMIT_ATTEMPTS}; "
                f"waiting {retry_after:.0f}s...",
                file=sys.stderr,
            )
            time.sleep(retry_after)
    raise RuntimeError("unreachable")  # pragma: no cover


def phase1_assign(
    client: anthropic.Anthropic, topics: list[Topic], toc_text: str, usage_totals: dict
) -> dict[str, list[str]]:
    chunks = chunk_toc(toc_text, TOC_CHUNK_LINES)
    schema = build_assign_schema(topics)
    system = [
        {"type": "text", "text": ASSIGN_INSTRUCTIONS},
        {
            "type": "text",
            "text": build_topic_list_block(topics),
            "cache_control": {"type": "ephemeral"},
        },
    ]

    assignments: dict[str, set[str]] = {t.slug: set() for t in topics}

    for i, chunk in enumerate(chunks, start=1):
        print(f"  [assign] chunk {i}/{len(chunks)} ({len(chunk.splitlines())} lines)...")
        user_content = f"Table of contents chunk {i} of {len(chunks)}:\n\n{chunk}"
        response = call_with_retry(
            client.messages.create,
            model=ASSIGN_MODEL,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user_content}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        track_usage(usage_totals, ASSIGN_MODEL, response.usage)
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        for slug, keys in data.items():
            if slug in assignments:
                assignments[slug].update(keys)

    return {slug: sorted(keys) for slug, keys in assignments.items()}


def build_sections_text(sections: dict, keys: list[str]) -> tuple[str, bool]:
    parts: list[str] = []
    total = 0
    truncated = False
    for key in keys:
        rec = sections.get(key)
        if rec is None:
            continue
        heading = " > ".join(rec.get("heading_path") or [rec.get("title", key)])
        block = (
            f"### [{key}] {rec.get('title', key)}\n{heading}\n\n{rec.get('text', '')}\n\n"
        )
        if total + len(block) > MAX_SECTION_TEXT_CHARS:
            truncated = True
            break
        parts.append(block)
        total += len(block)
    return "".join(parts), truncated


def phase2_write(
    client: anthropic.Anthropic,
    topic: Topic,
    section_keys: list[str],
    sections: dict,
    usage_totals: dict,
) -> str:
    sections_text, truncated = build_sections_text(sections, section_keys)
    if truncated:
        sections_text += (
            "\n[NOTE: section text truncated at 120,000 characters; some assigned "
            "sections were omitted from this prompt.]\n"
        )
    if not sections_text:
        sections_text = "(no relevant sections were found for this topic)"

    system = [
        {"type": "text", "text": WRITE_SYSTEM_INSTRUCTIONS},
        {"type": "text", "text": PAGE_FORMAT_TEMPLATE, "cache_control": {"type": "ephemeral"}},
    ]
    user = (
        f"Topic: {topic.title} (slug: {topic.slug})\n"
        f"Synonyms residents might use: {', '.join(topic.synonyms)}\n\n"
        f"Relevant section text:\n\n{sections_text}"
    )

    response = call_with_retry(
        client.messages.create,
        model=WRITE_MODEL,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    track_usage(usage_totals, WRITE_MODEL, response.usage)
    text = next(b.text for b in response.content if b.type == "text")
    return text.strip()


def render_page(topic: Topic, section_keys: list[str], body_markdown: str) -> str:
    frontmatter = (
        "---\n"
        f"slug: {topic.slug}\n"
        f"title: {topic.title}\n"
        f"synonyms: {', '.join(topic.synonyms)}\n"
        f"sections: {', '.join(section_keys)}\n"
        "---\n"
    )
    body = body_markdown.rstrip()
    return f"{frontmatter}{body}\n\n{FOOTER_LINE}\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate app/wiki/*.md topic pages via the Anthropic API."
    )
    parser.add_argument(
        "--topics",
        help="Comma-separated topic slugs to build (default: all ~40 curated topics).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild pages even if app/wiki/{slug}.md already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run phase 1 (assign) only: print the topic->sections assignment as JSON "
            "and exit without calling opus or writing any files."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    selected_topics = TOPICS
    if args.topics:
        requested = [s.strip() for s in args.topics.split(",") if s.strip()]
        unknown = [s for s in requested if s not in TOPICS_BY_SLUG]
        if unknown:
            sys.exit(f"Unknown topic slug(s): {', '.join(unknown)}")
        selected_topics = [TOPICS_BY_SLUG[s] for s in requested]

    sections, toc_text = load_data()

    client = anthropic.Anthropic()
    usage_totals: dict = {}

    print(
        f"Phase 1: assigning sections to {len(selected_topics)} topic(s) from "
        f"{len(toc_text.splitlines())} lines of TOC..."
    )
    assignments = phase1_assign(client, selected_topics, toc_text, usage_totals)

    print("\nAssignment summary:")
    for t in selected_topics:
        print(f"  {t.slug}: {len(assignments.get(t.slug, []))} section(s)")

    if args.dry_run:
        print("\n--dry-run: assignment (topic -> section keys):")
        print(json.dumps(assignments, indent=2))
        print(f"\nEstimated cost so far: ${estimate_cost(usage_totals):.3f}")
        return

    WIKI_DIR.mkdir(parents=True, exist_ok=True)

    print("\nPhase 2: writing wiki pages...")
    written = 0
    skipped = 0
    for t in selected_topics:
        out_path = WIKI_DIR / f"{t.slug}.md"
        if out_path.exists() and not args.force:
            print(f"  [skip] {t.slug} (already exists; use --force to rebuild)")
            skipped += 1
            continue
        keys = assignments.get(t.slug, [])
        print(f"  [write] {t.slug} ({len(keys)} section(s))...")
        body = phase2_write(client, t, keys, sections, usage_totals)
        out_path.write_text(render_page(t, keys, body))
        written += 1

    print(
        f"\nDone. Wrote {written} page(s), skipped {skipped}. "
        f"Total estimated cost: ${estimate_cost(usage_totals):.3f}"
    )


if __name__ == "__main__":
    main()
