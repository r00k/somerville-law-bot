# Somerville Law Bot

**[somervillelawbot.com](https://somervillelawbot.com)** — ask plain-language questions about Somerville, MA municipal law ("Can I raise chickens?", "How soon do I have to shovel my sidewalk?") and get answers grounded in the actual ordinance text. Every legal claim carries a citation whose quote is verified verbatim against the corpus before display, deep-linked into a readable edition of the law.

This repo contains the whole thing: the pipeline that fetches and normalizes the law from Somerville's official enCodePlus publications, the section index and search tools, the Claude-powered Q&A agent, the web app, and the eval suite that keeps the answers honest.

## How It Works

- `fetch_somerville_law.py` / `fetch_somerville_zoning.py` fetch official content from enCodePlus and normalize it into Markdown; `render_markdown_html.py` produces the readable HTML editions that citations link into.
- `app/indexer.py` parses both corpora into a 3,346-section index (`app/data/sections.json`).
- `app/law_tools.py` provides BM25 search and section fetch as agent tools.
- `app/wiki/` holds 44 pregenerated topic pages that map resident vocabulary ("chickens") to legal vocabulary ("domestic fowl") and route the agent across corpora.
- `app/agent.py` runs the tool loop (Claude Sonnet 5 by default, override with `LAW_QA_MODEL`) and rejects any citation whose quote doesn't appear verbatim in the cited section.
- `app/server.py` serves the frontend with SSE progress streaming, per-IP and global rate limits, JSONL question logging, and the readable corpus pages themselves.

See `app/DESIGN.md` for the full spec.

## Running Locally

```bash
uv sync
printf 'ANTHROPIC_API_KEY=sk-ant-...\n' > .env  # loaded automatically; gitignored

# Run the app
uv run uvicorn app.server:app             # then open http://127.0.0.1:8000

# Frontend/design work: auto-reload the browser on static or server changes
DEV_RELOAD=1 uv run uvicorn app.server:app --reload

# Ask from the CLI
uv run python -m app.agent "How long is the mayor's term?"
```

## Evals

`evals/questions.yaml` holds 30 live questions — unequivocal lookups graded on exactness, multi-hop judgment questions that must cite every provision in the chain, and hallucination traps where inventing a number is a failure.

```bash
uv run python evals/run.py                # all questions, 4 in parallel (costs API tokens)
uv run python evals/run.py --parallel 8   # more concurrency
uv run python evals/run.py --only council-composition,late-night-music
uv run python evals/run.py --json         # also write logs/eval-results-{date}.json
```

Answers are stochastic: rerun affected questions before trusting a single pass, and read the failing `answer_markdown` (via `--json`) before deciding whether the answer or the check is wrong.

## Deployment

Deployed on Railway; pushes to `main` auto-deploy. `railway.json` sets the start command (single uvicorn worker — the in-memory rate limiter is per-process), a `/healthz` health check, and restart-on-failure.

Service configuration (Railway environment variables):

- `ANTHROPIC_API_KEY` — required.
- `TRUST_PROXY=1` — required behind Railway's proxy so per-IP rate limiting keys on real client IPs.
- `RATE_LIMIT_PER_HOUR` (default 10) and `DAILY_QUESTION_CAP` (default 200) — abuse/cost caps.

A Railway volume mounted at `/app/logs` persists the JSONL Q&A logs across deploys.

## Refreshing the Law Corpus

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

After a refresh: rerun `uv run python -m app.indexer`, regenerate wiki pages for changed topics with `uv run python -m app.wiki_build` (costs API tokens; see `CHANGELOG.md` for what changed), and rerun the evals.

## Just Want to Read the Law?

The generated outputs are committed — no need to run anything:

- **Readable HTML (what citations link to):**
  - [Non-zoning law (Charter, Code of Ordinances, Appendices B/D/E)](https://somervillelawbot.com/somerville-law-non-zoning.readable.html)
  - [Zoning ordinance](https://somervillelawbot.com/somerville-zoning.readable.html)
- **Markdown (best for search, diffing, LLM ingestion):** [somerville-law-non-zoning.md](somerville-law-non-zoning.md), [somerville-zoning.md](somerville-zoning.md), [somerville-law-combined.md](somerville-law-combined.md)
- **Raw HTML (auditable, as fetched):** [somerville-law-non-zoning.raw.html](somerville-law-non-zoning.raw.html), [somerville-zoning.raw.html](somerville-zoning.raw.html)
- **PDF:** [rules-of-the-council.pdf](rules-of-the-council.pdf) — Rules of the City Council (Appendix B)

Check the commit history for when the law was last refreshed, and [CHANGELOG.md](CHANGELOG.md) for substantive legal changes observed between refreshes.

## Known Limitations

- This is a transformed convenience corpus, not a replacement for the official publication.
- If enCodePlus changes source structure, parser logic may need updates.
- Internal text may still reference excluded appendices or external materials; those references are part of included legal text.
- Zoning images are represented as placeholders by default; image binaries are not downloaded locally.

## Legal Disclaimer

This site and repository are for informational and research use only and do not constitute legal advice. Check the official municipal code before doing something serious with this information.
