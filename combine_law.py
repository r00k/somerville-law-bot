#!/usr/bin/env python3
"""Combine the non-zoning and zoning Markdown files into a single corpus.

Reads `somerville-law-non-zoning.md` and `somerville-zoning.md` (when both
exist) and writes `somerville-law-combined.md`. Safe to call repeatedly; if
either source file is missing, the combined file is left untouched and a note
is printed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_NON_ZONING = "somerville-law-non-zoning.md"
DEFAULT_ZONING = "somerville-zoning.md"
DEFAULT_OUTPUT = "somerville-law-combined.md"
SEPARATOR = "\n\n---\n\n"


def combine(non_zoning: Path, zoning: Path, output: Path) -> bool:
    missing = [str(p) for p in (non_zoning, zoning) if not p.exists()]
    if missing:
        print(f"[info] Skipping combined corpus; missing: {', '.join(missing)}")
        return False

    parts = [
        non_zoning.read_text(encoding="utf-8").rstrip(),
        SEPARATOR,
        zoning.read_text(encoding="utf-8").rstrip(),
        "\n",
    ]
    output.write_text("".join(parts), encoding="utf-8")
    print(f"[ok] Wrote combined corpus: {output}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--non-zoning", default=DEFAULT_NON_ZONING)
    parser.add_argument("--zoning", default=DEFAULT_ZONING)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    combine(Path(args.non_zoning), Path(args.zoning), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
