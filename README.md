# Somerville Municipal Law Consolidation

Consolidated, machine-friendly extracts of Somerville municipal law from enCodePlus.

## What This Is

This repository fetches official municipal-law content from Somerville's enCodePlus publications, normalizes it into machine-friendly Markdown, and renders a human-friendly HTML reading edition.

## Who This Is For

- Researchers, civic technologists, and policy teams who want consolidated non-zoning and zoning law corpora.
- Developers who want a reproducible fetch + transform pipeline.

## Just Want to Read the Law?

The generated outputs are committed to this repo — no need to run anything. Open them directly:

- **Readable HTML (recommended for browsing):**
  - [Non-zoning law (Charter, Code of Ordinances, Appendices B/D/E)](somerville-law-non-zoning.readable.html)
  - [Zoning ordinance](somerville-zoning.readable.html)
- **Markdown (best for search, diffing, LLM ingestion):**
  - [somerville-law-non-zoning.md](somerville-law-non-zoning.md)
  - [somerville-zoning.md](somerville-zoning.md)
  - [somerville-law-combined.md](somerville-law-combined.md) — both of the above concatenated, regenerated automatically by the fetch scripts
- **Raw HTML (auditable, as fetched from enCodePlus):**
  - [somerville-law-non-zoning.raw.html](somerville-law-non-zoning.raw.html)
  - [somerville-zoning.raw.html](somerville-zoning.raw.html)

Check the commit history to see when the law was last refreshed.

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
```

Expected outputs:

- `somerville-law-non-zoning.md`
- `somerville-law-non-zoning.raw.html`
- `somerville-law-non-zoning.readable.html`
- `somerville-zoning.md`
- `somerville-zoning.raw.html`
- `somerville-zoning.images.json`
- `somerville-zoning.readable.html`
- `somerville-law-combined.md` (regenerated automatically whenever either fetch script runs, when both source Markdown files are present)

## Known Limitations

- This is a transformed convenience corpus, not a replacement for the official publication.
- If enCodePlus changes source structure, parser logic may need updates.
- Internal text may still reference excluded appendices or external materials; those references are part of included legal text.
- Zoning images are represented as placeholders by default; image binaries are not downloaded locally.

## Legal Disclaimer

This repository is for informational and research use only and does not constitute legal advice.
