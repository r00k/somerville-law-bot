# Somerville Law Q&A — Design Spec

A web app where Somerville residents ask questions about municipal law and get
cited, verified answers from an LLM agent. This spec is the contract between
components; implementers should follow the interfaces exactly.

## Why this design (context for implementers)

A previous version of this app (removed in commit `6f01b69`) failed on
questions like "can I raise chickens" because its retrieval searched TOC
headings of one corpus only. The law says "domestic fowl", not "chickens", and
relevant sections span the Code of Ordinances, Board of Health regulations,
and the zoning ordinance. This design fixes that with (a) a full cross-corpus
section index, (b) a pregenerated topic wiki that maps common-language topics
and synonyms to section IDs, and (c) deterministic citation verification so
the model can never fabricate law text.

## Source data (already in repo root, do not modify)

- `somerville-law-non-zoning.md` — Charter, Code of Ordinances, Appendices B/D/E
- `somerville-zoning.md` — Zoning ordinance
- Sections are delimited by `<!-- secid:N -->` HTML comments followed by
  markdown headings (`#`–`######`). secids are NOT unique across the two files
  — always qualify with corpus prefix.
- Readable HTML editions are published at:
  - `https://somervillelawbot.com/code#secid-{N}`
  - `https://somervillelawbot.com/zoning#secid-{N}`
  (anchors `id="secid-{N}"` exist in the published HTML)

## Directory layout

```
app/
  __init__.py
  indexer.py        # builds data/sections.json + data/toc.txt from corpus md
  law_tools.py      # search / fetch functions used as agent tools
  agent.py          # Anthropic tool loop + citation verification
  server.py         # FastAPI app: /api/ask (SSE), static frontend, rate limits
  wiki_build.py     # generates wiki/*.md topic pages via the Anthropic API
  wiki/             # generated topic pages (committed)
  data/             # generated index artifacts (committed)
  static/index.html # frontend (single file, no build step)
evals/
  questions.yaml
  run.py
logs/               # runtime QA logs (gitignored)
```

Run everything from the **repo root** with `uv run python -m app.indexer` etc.
Dependencies are declared in the root `pyproject.toml` (anthropic, fastapi,
uvicorn, pyyaml). Do not add dependencies without noting it.

## Component 1: indexer.py

Parses both corpus files into `app/data/sections.json` and `app/data/toc.txt`.

Section key format: `"coo:{secid}"` for non-zoning, `"zon:{secid}"` for zoning.

`sections.json` schema — a JSON object mapping section key → :

```json
{
  "key": "coo:1234",
  "corpus": "coo",
  "heading_path": ["PART II CODE OF ORDINANCES", "Chapter 9 ...", "ARTICLE V. ...", "DIVISION 3. LEAF BLOWERS", "Sec. 9-120. Leaf blowers regulated."],
  "title": "Sec. 9-120. Leaf blowers regulated.",
  "text": "<full markdown text of the section, headings excluded>",
  "url": "https://somervillelawbot.com/code#secid-1234"
}
```

Parsing rules:
- A section starts at `<!-- secid:N -->`. Its heading is the first markdown
  heading after the marker (if any). Its text runs until the next secid marker.
- Maintain the heading hierarchy from heading levels to build `heading_path`
  (the trail of ancestor headings at lower levels, ending with this section's
  own heading). Content before the first secid marker (document preamble) is
  ignored.
- Some sections are containers with little text; keep them (they matter for
  the TOC) but text may be short/empty.

`toc.txt`: a compact plain-text tree of every section, one line per section:
indentation by depth, then `title` then ` [key]`. This is injected into the
agent's system prompt, so keep it compact (no URLs, no text).

CLI: `python -m app.indexer` prints counts and writes both files.
Sanity assertions: > 3000 sections total, leaf-blower section findable, both
corpora present.

## Component 2: law_tools.py

Pure-Python retrieval over `sections.json` (load once at import into module
state). No third-party search deps — implement BM25 scoring in-module.

```python
def search_law(query: str, limit: int = 12) -> list[dict]:
    """BM25 over (title + heading_path + text), tokenized lowercase alnum.
    Returns [{key, title, heading_path, score, snippet}] — snippet is ~300
    chars around the best matching region."""

def get_sections(keys: list[str]) -> list[dict]:
    """Full records for the given keys (invalid keys -> {'key': k, 'error': ...}).
    Cap combined returned text at ~150_000 chars; if over, truncate later
    sections and mark 'truncated': true."""

def get_wiki_page(topic_slug: str) -> str | None:
    """Return the markdown body of app/wiki/{topic_slug}.md if it exists."""

def wiki_index() -> str:
    """Concatenated frontmatter summaries of all wiki pages: one block per
    topic with slug, title, synonyms, and section keys. Used in the system
    prompt. Returns '' if wiki/ is empty (the agent must work without it)."""
```

## Component 3: agent.py

The Q&A agent. Uses the `anthropic` SDK. **API usage must follow these exact
shapes — do not improvise from memory:**

- Model: `claude-opus-4-8` (constant `MODEL` at module top).
- Thinking: `thinking={"type": "adaptive"}`. Do NOT pass `budget_tokens`,
  `temperature`, `top_p`, or `top_k` (they 400 on this model).
- `max_tokens=16000`.
- System prompt is a list of text blocks; put
  `"cache_control": {"type": "ephemeral", "ttl": "1h"}` on the LAST system
  block so the whole prefix (tools + system) caches.
- Tool loop: manual loop on `client.messages.create(...)`; continue while
  `response.stop_reason == "tool_use"`; append
  `{"role": "assistant", "content": response.content}` then a user message
  containing ALL `tool_result` blocks (one per tool_use, matched by
  `tool_use_id`), in a single user message. Handle `pause_turn` by re-sending
  and continuing. Cap at 12 iterations.

Tools exposed to the model:
1. `search_law(query)` — from law_tools.
2. `get_sections(keys)` — from law_tools.
3. `get_wiki_page(topic_slug)` — from law_tools.
4. `submit_answer` — terminal tool; the model MUST end by calling it. When the
   model calls submit_answer, do not execute anything: validate, verify
   citations, and return. Define with `strict: true` and this schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["answer_markdown", "citations", "confidence"],
  "properties": {
    "answer_markdown": {"type": "string"},
    "citations": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["quote", "section_key", "why_relevant"],
        "properties": {
          "quote": {"type": "string"},
          "section_key": {"type": "string"},
          "why_relevant": {"type": "string"}
        }
      }
    },
    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    "caveats": {"type": "string"}
  }
}
```

System prompt contents (assembled in this order; stable text first for
caching):
1. Role: legal research assistant for Somerville, MA municipal law. Answers
   ONLY from the provided corpus via tools. Plain language for residents.
   Every legal claim needs a citation with an EXACT verbatim quote from
   section text. If the corpus doesn't address the question, say so
   (confidence: low) rather than guessing. Question about a specific
   address/property: explain that the answer depends on the zoning district,
   suggest the city's official zoning map, and answer conditionally for
   common districts if useful. Always note this is not legal advice.
   State-law questions (M.G.L.) are out of corpus — say so.
2. Workflow guidance: consult the topic index first; use get_wiki_page for
   routing when a topic matches; law vocabulary differs from common speech
   (chickens → "domestic fowl") so search with legal synonyms too; read the
   actual sections before answering; quote only from section text you have
   fetched; finish with submit_answer.
3. `<topic_index>` — law_tools.wiki_index()
4. `<table_of_contents>` — contents of app/data/toc.txt
   (cache_control on this block)

Public interface:

```python
@dataclass
class Answer:
    answer_markdown: str
    citations: list[VerifiedCitation]  # quote, section_key, why_relevant, url, verified: bool
    confidence: str
    caveats: str | None
    dropped_citations: int            # citations that failed verification
    usage: dict                       # summed input/output/cache tokens across the loop

def ask(question: str, history: list[dict] | None = None,
        on_event: Callable[[dict], None] | None = None) -> Answer
```

`on_event` receives progress events during the loop for the SSE stream:
`{"type": "tool", "name": "search_law", "detail": "<query or keys>"}` and
`{"type": "thinking"}` at loop starts. Keep payloads small.

Citation verification (in agent.py, deterministic, no LLM):
- Normalize both quote and section text: collapse all whitespace runs to a
  single space, unify curly/straight quotes and dashes, casefold.
- Verified iff normalized quote is a substring of the normalized text of the
  cited section. Failing citations are dropped from the returned list;
  `dropped_citations` counts them. If ALL citations fail, downgrade
  confidence to "low" and append a caveat.
- Attach `url` from the section record to each verified citation.

CLI for manual testing: `python -m app.agent "question here"` pretty-prints
the answer, citations with ✓/✗, confidence, and token usage.

## Component 4: server.py + static/index.html

FastAPI app, run with `uv run uvicorn app.server:app`.

- `POST /api/ask` — body `{"question": str}`. Responds with
  `text/event-stream`: events `{"type": "tool", ...}` as they happen, then a
  final `{"type": "answer", ...}` event with the full Answer (citations
  include url + verified), then `{"type": "done"}`. Run the (synchronous)
  agent in a thread and bridge events through a queue.
- `GET /` — serves `static/index.html`. `GET /healthz` — `{"ok": true}`.
- Rate limiting (in-memory, no deps): per-IP sliding window, default 10
  questions/hour (env `RATE_LIMIT_PER_HOUR`), and a global daily cap, default
  200/day (env `DAILY_QUESTION_CAP`). Over limit → 429 with a friendly JSON
  message. Trust `X-Forwarded-For` first value when present.
- Question length cap: 1000 chars → 400.
- Logging: append JSONL to `logs/qa-YYYY-MM-DD.jsonl`: timestamp, request id,
  ip hash (sha256 truncated), question, answer_markdown, citations (keys +
  verified), confidence, dropped_citations, usage, latency_ms, error if any.
  Create logs/ if missing.

Frontend (`static/index.html`, single self-contained file, no frameworks, no
CDN):
- Clean, civic, trustworthy look. Title "Ask Somerville Law". Subtitle
  explaining what it is. Example question chips (leaf blower in July / raise
  chickens / mayor's term / shovel sidewalk snow).
- Textarea + Ask button. While working, show live progress line from SSE tool
  events ("Searching: leaf blower…", "Reading Sec. 9-120…").
- Render answer markdown (small built-in renderer: paragraphs, bold, lists,
  links ok). Citations as cards: quote (blockquote), section title link →
  readable-edition URL, why_relevant. Confidence badge (green/yellow/gray).
- Permanent footer disclaimer: informational only, not legal advice, verify
  with the official code; link to the official enCodePlus publication and to
  the repo.
- Respect `prefers-color-scheme` (light + dark).

## Component 5: wiki_build.py

Generates `app/wiki/*.md`. Runs offline with `ANTHROPIC_API_KEY`. Two phases:

1. **Assign**: for a curated starter list of ~40 topics (hardcode in the
   script: animals/chickens, leaf blowers, noise, snow & sidewalks, trash &
   recycling & composting, parking permits, street cleaning, ADUs & additions,
   fences, trees, short-term rentals, home businesses, building permits,
   signs, pools, rodent control, dumpsters, food trucks, restaurants &
   licensing, alcohol, tobacco & smoking, marijuana, bicycles & e-bikes,
   sidewalks & obstructions, yard sales, fireworks, dogs, cats, bees,
   demolition, historic districts, affordable housing, condo conversion,
   tenants & rental registration, hours of operation, elections, city
   council, mayor, taxes & fees, wage theft, plastic bags & polystyrene,
   utilities & digging, graffiti, abandoned vehicles): send batches of TOC
   lines to `claude-sonnet-5` asking which section keys are relevant to which
   topics (JSON out). Union results per topic.
2. **Write**: per topic, fetch the assigned sections' full text (cap 120K
   chars) and ask `claude-opus-4-8` (thinking adaptive) to write the page.

Page format:

```markdown
---
slug: leaf-blowers
title: Leaf Blowers
synonyms: leafblower, blower, yard equipment, landscaping noise
sections: coo:1234, coo:1235, zon:88
---
<8-25 line plain-language summary: what's regulated, key rules with section
numbers, common gotchas, cross-references to related topics>
```

Requirements: use prompt caching (`cache_control` on the shared prefix),
be resumable (skip topics whose file already exists unless `--force`),
`--topics a,b` to build a subset, print running token/cost totals.
The generated body must instruct readers/agents that section text is
authoritative, not the summary (one footer line).

## Component 6: evals/

`questions.yaml`: ~20 entries:

```yaml
- id: leaf-blower-july
  question: Can I use a gas-powered leaf blower in July?
  expect_answer_contains_any: ["no", "not permitted", "prohibited"]
  expect_cited_sections_any: ["coo:*leaf*"]   # glob against section key OR title
  expect_confidence_at_least: medium
```

Include: leaf blower July (no); raise chickens (fowl/health regs); mayor term
(2 years); snow shoveling requirements; parking permit; trash/put out time;
dog leash; fence height; ADU/third story (conditional-on-district answer OK);
short-term rental; smoking in parks; plastic bags; noise at night hours;
yard sale permit; question with no corpus answer ("can I own a tiger?" →
low confidence acceptable) etc. Derive expected sections by actually
grepping the corpus while writing the file — do not guess section numbers.

`run.py`: `uv run python evals/run.py [--only id1,id2]` — runs each question
through `agent.ask`, evaluates checks (answer substring any-match,
case-insensitive; cited-section glob against keys and titles; confidence
ordering high>medium>low; citation verification must have ≥1 verified
citation unless the eval sets `allow_no_citations: true`), prints a table of
pass/fail with reasons and total cost, exits nonzero on failures. Sequential
is fine.

## Conventions

- Python 3.11+, type hints, no classes where functions do.
- Match existing repo style (stdlib-leaning, small files, argparse CLIs).
- No secrets in code; read `ANTHROPIC_API_KEY` from env via the SDK default.
- Do not modify files outside your assigned component. Do not commit.
