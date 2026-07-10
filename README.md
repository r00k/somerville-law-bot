# Somerville Municipal Law Consolidation

Consolidated, machine-friendly extracts of Somerville municipal law from enCodePlus — plus an LLM-powered Q&A web app for asking questions about it.

## What This Is

This repository fetches official municipal-law content from Somerville's enCodePlus publications, normalizes it into machine-friendly Markdown, and renders a human-friendly HTML reading edition.

## Who This Is For

- Researchers, civic technologists, and policy teams who want consolidated non-zoning and zoning law corpora.
- Developers who want a reproducible fetch + transform pipeline.

## Just Want to Read the Law?

The generated outputs are committed to this repo — no need to run anything. Open them directly:

- **Readable HTML (recommended for browsing):**
  - [Non-zoning law (Charter, Code of Ordinances, Appendices B/D/E)](https://r00k.github.io/somerville-ordinances/somerville-law-non-zoning.readable.html)
  - [Zoning ordinance](https://r00k.github.io/somerville-ordinances/somerville-zoning.readable.html)
- **Markdown (best for search, diffing, LLM ingestion):**
  - [somerville-law-non-zoning.md](somerville-law-non-zoning.md)
  - [somerville-zoning.md](somerville-zoning.md)
  - [somerville-law-combined.md](somerville-law-combined.md) — both of the above concatenated, regenerated automatically by the fetch scripts
- **Raw HTML (auditable, as fetched from enCodePlus):**
  - [somerville-law-non-zoning.raw.html](somerville-law-non-zoning.raw.html)
  - [somerville-zoning.raw.html](somerville-zoning.raw.html)
- **PDF:**
  - [rules-of-the-council.pdf](rules-of-the-council.pdf) — Rules of the City Council (Appendix B), exported from enCodePlus

Check the commit history to see when the law was last refreshed, and [CHANGELOG.md](CHANGELOG.md) for a summary of substantive legal changes observed between refreshes.

## Quick Start

```bash
python3 -m pip install -r requirements.txt

# Non-zoning law
python3 fetch_somerville_law.py
python3 render_markdown_html.py

# Zoning ordinance (text-first with image placeholders)
python3 fetch_somerville_zoning.py --skip-pdf-attempt --strip-metadata
python3 render_markdown_html.py \
  --input somerville-zoning.md \
  --output somerville-zoning.readable.html \
  --title 'Somerville Zoning Ordinance (Readable Edition)'

# Combined corpus (also regenerated automatically by each fetch script)
python3 combine_law.py
```

Pass `--skip-pdf-attempt` to `fetch_somerville_law.py` to skip the best-effort host PDF export and only produce Markdown/HTML.

Expected outputs:

- `somerville-law-non-zoning.md`
- `somerville-law-non-zoning.raw.html`
- `somerville-law-non-zoning.readable.html`
- `somerville-zoning.md`
- `somerville-zoning.raw.html`
- `somerville-zoning.images.json`
- `somerville-zoning.readable.html`
- `somerville-law-combined.md` (regenerated automatically whenever either fetch script runs, when both source Markdown files are present)

## Somerville Law Bot (Q&A web app)

`app/` contains a web app where residents ask plain-language questions ("Can I use a gas-powered leaf blower in July?", "Can I raise chickens?") and get answers grounded in the corpus. Every legal claim carries a citation whose quote is verified verbatim against the ordinance text before display, deep-linked into the readable editions. Powered by Claude Opus 4.8 via the Anthropic API.

```bash
uv sync
export ANTHROPIC_API_KEY=sk-ant-...

# One-time (already committed, rerun after a corpus refresh):
uv run python -m app.indexer          # rebuild section index
uv run python -m app.wiki_build       # regenerate topic wiki pages (costs API tokens)

# Run the app
uv run uvicorn app.server:app         # then open http://127.0.0.1:8000

# Ask from the CLI
uv run python -m app.agent "How long is the mayor's term?"

# Run the eval suite (20 live questions, costs API tokens)
uv run python evals/run.py
```

Architecture: `app/indexer.py` parses both corpora into a 3,346-section index; `app/law_tools.py` provides BM25 search and section fetch as agent tools; `app/wiki/` holds 44 pregenerated topic pages that map resident vocabulary ("chickens") to legal vocabulary ("domestic fowl") and route the agent across corpora; `app/agent.py` runs the tool loop and rejects any citation whose quote doesn't appear verbatim in the cited section; `app/server.py` serves the frontend with SSE progress streaming, per-IP rate limits, and JSONL question logging. See `app/DESIGN.md` for the full spec.

After refreshing the law corpus, rerun the indexer, regenerate wiki pages for changed topics (see `CHANGELOG.md`), and rerun the evals.

## Known Limitations

- This is a transformed convenience corpus, not a replacement for the official publication.
- If enCodePlus changes source structure, parser logic may need updates.
- Internal text may still reference excluded appendices or external materials; those references are part of included legal text.
- Zoning images are represented as placeholders by default; image binaries are not downloaded locally.

## Legal Disclaimer

This repository is for informational and research use only and does not constitute legal advice. Check the official municipal code before doing something serious with this information.
