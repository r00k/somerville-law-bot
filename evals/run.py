"""Eval runner for the Somerville law Q&A agent (app/agent.py).

Runs every question in evals/questions.yaml through app.agent.ask, checks the
result against each question's expectations, prints a pass/fail table with
failure reasons, and prints a total token usage / cost summary.

Usage:
    uv run python evals/run.py                    # run all questions (4 in parallel)
    uv run python evals/run.py --only leaf-blower-july,mayor-term
    uv run python evals/run.py --parallel 8        # more concurrency (watch rate limits)
    uv run python evals/run.py --parallel 1        # sequential
    uv run python evals/run.py --json              # also write logs/eval-results-{date}.json

Exits nonzero if any question fails (or errors).

Note: importing app.agent is deferred to run_evals() (not module load time),
so this file parses and --help works even before app/agent.py exists.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # Allow `uv run python evals/run.py` (a direct script invocation, not
    # `-m app.x`) to still find the `app` package.
    sys.path.insert(0, str(REPO_ROOT))

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.yaml"
LOGS_DIR = REPO_ROOT / "logs"

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

# Standard (non-introductory) $ per million tokens, keyed by model-ID prefix.
# The agent's model is app.agent.MODEL (default claude-sonnet-5, overridable
# via LAW_QA_MODEL), so pricing is resolved from it at runtime. Cache reads
# bill at 0.1x the input rate. DESIGN.md does not specify a separate
# multiplier for cache *writes* in the eval cost summary, so those tokens are
# counted at the normal input rate here.
MODEL_PRICING_PER_MTOK = {
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
DEFAULT_PRICING_PER_MTOK = MODEL_PRICING_PER_MTOK["claude-sonnet-5"]
CACHE_READ_MULTIPLIER = 0.1


def _pricing_per_mtok() -> tuple[float, float]:
    """(input, output) $ per MTok for the model the agent actually uses."""
    try:
        from app.agent import MODEL
    except Exception:
        return DEFAULT_PRICING_PER_MTOK
    for prefix, rates in MODEL_PRICING_PER_MTOK.items():
        if MODEL.startswith(prefix):
            return rates
    return DEFAULT_PRICING_PER_MTOK


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute-or-dict access, since Answer/citation objects from
    app.agent may be dataclass instances or plain dicts depending on how
    that module ended up implementing the DESIGN.md interface."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass
class EvalOutcome:
    id: str
    question: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    confidence: str | None = None
    dropped_citations: int = 0
    cited_sections: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    answer_markdown: str = ""
    caveats: str | None = None
    error: str | None = None
    latency_s: float = 0.0


def load_questions(only: set[str] | None = None) -> list[dict]:
    raw = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8")) or []
    if only:
        ids_present = {q["id"] for q in raw}
        missing = only - ids_present
        if missing:
            raise SystemExit(f"--only referenced unknown id(s): {sorted(missing)}")
        raw = [q for q in raw if q["id"] in only]
    return raw


def _glob_match_any(patterns: list[str], candidates: list[str]) -> bool:
    for pattern in patterns:
        for candidate in candidates:
            if candidate and fnmatch.fnmatch(candidate.lower(), pattern.lower()):
                return True
    return False


# Raw submit_answer markup that must never reach a rendered answer. Seen in
# the wild when the model writes the whole pseudo-XML document inside the
# answer_markdown string instead of using the separate fields.
_MARKUP_MARKERS = (
    "<answer_markdown>",
    "</answer_markdown>",
    "<citations>",
    "</citations>",
    "<confidence>",
    "<caveats>",
    '"section_key"',
)


def _check_well_formed(answer_markdown: str, caveats: str | None) -> list[str]:
    """Structural checks applied to EVERY question, regardless of its spec:
    no raw submit_answer markup in the rendered fields, and no trailing
    Note:/Caveat: paragraph duplicating what belongs in the caveats field."""
    reasons: list[str] = []
    for name, blob in (("answer_markdown", answer_markdown or ""), ("caveats", caveats or "")):
        marker = next((m for m in _MARKUP_MARKERS if m in blob), None)
        if marker:
            reasons.append(f"{name} contains raw submit_answer markup ({marker!r})")
    paragraphs = [p.strip() for p in (answer_markdown or "").split("\n\n") if p.strip()]
    if paragraphs:
        last = paragraphs[-1].lstrip("*_ ").lower()
        if last.startswith(("note:", "notes:", "caveat:", "caveats:")):
            reasons.append(
                "answer_markdown ends with a Note:/Caveat: paragraph "
                "(question-specific caveats belong in the caveats field)"
            )
    return reasons


def _check_answer_contains(spec: dict, answer_markdown: str) -> str | None:
    """Returns a failure reason string, or None if the check passes/is absent."""
    expected = spec.get("expect_answer_contains_any")
    if not expected:
        return None
    haystack = (answer_markdown or "").lower()
    if any(needle.lower() in haystack for needle in expected):
        return None
    return f"answer_markdown did not contain any of {expected!r}"


def _check_answer_contains_all(spec: dict, answer_markdown: str) -> list[str]:
    """expect_answer_contains_all is a list of groups, each a string or a list
    of alternative strings. Every group must match (AND of ORs)."""
    expected = spec.get("expect_answer_contains_all")
    if not expected:
        return []
    haystack = (answer_markdown or "").lower()
    reasons: list[str] = []
    for group in expected:
        alternatives = group if isinstance(group, list) else [group]
        if not any(alt.lower() in haystack for alt in alternatives):
            reasons.append(
                f"answer_markdown did not contain any of {alternatives!r} "
                "(expect_answer_contains_all)"
            )
    return reasons


def _check_answer_not_contains(spec: dict, answer_markdown: str) -> str | None:
    banned = spec.get("expect_answer_not_contains")
    if not banned:
        return None
    haystack = (answer_markdown or "").lower()
    hit = next((n for n in banned if n.lower() in haystack), None)
    if hit is None:
        return None
    return f"answer_markdown contains banned substring {hit!r} (expect_answer_not_contains)"


def _titles_for(section_keys: list[str]) -> dict[str, str]:
    if not section_keys:
        return {}
    from app.law_tools import get_sections  # lazy: only needed when checking citations

    records = get_sections(section_keys)
    return {rec.get("key", ""): rec.get("title", "") or "" for rec in records}


def _check_cited_sections(spec: dict, citations: list) -> tuple[str | None, list[str]]:
    keys = [str(_get(c, "section_key", "")) for c in citations]
    keys = [k for k in keys if k]
    expected = spec.get("expect_cited_sections_any")
    if not expected:
        return None, keys
    titles = _titles_for(keys)
    candidates: list[str] = []
    for k in keys:
        candidates.append(k)
        title = titles.get(k, "")
        if title:
            candidates.append(title)
    if _glob_match_any(expected, candidates):
        return None, keys
    return f"no citation matched any of expect_cited_sections_any={expected!r} (got keys={keys!r})", keys


def _check_cited_sections_all(spec: dict, citations: list) -> list[str]:
    """expect_cited_sections_all is a list of groups, each a string or a list
    of alternative glob patterns. Every group must match some citation's key
    or title (AND of ORs) — for questions whose point is a multi-hop lookup."""
    expected = spec.get("expect_cited_sections_all")
    if not expected:
        return []
    keys = [str(_get(c, "section_key", "")) for c in citations]
    keys = [k for k in keys if k]
    titles = _titles_for(keys)
    candidates: list[str] = []
    for k in keys:
        candidates.append(k)
        title = titles.get(k, "")
        if title:
            candidates.append(title)
    reasons: list[str] = []
    for group in expected:
        patterns = group if isinstance(group, list) else [group]
        if not _glob_match_any(patterns, candidates):
            reasons.append(
                f"no citation matched any of {patterns!r} "
                f"(expect_cited_sections_all, got keys={keys!r})"
            )
    return reasons


def _check_table_citation(spec: dict, citations: list) -> str | None:
    """expect_table_citation: true requires at least one verified table-lookup
    citation (row + value present) — for questions whose governing law is a
    table cell, so a regression to prose-only quoting is visible."""
    if not spec.get("expect_table_citation"):
        return None
    for c in citations:
        if _get(c, "row", None) and _get(c, "value", None) and _get(c, "verified", True):
            return None
    return "expected >=1 verified table-lookup citation (expect_table_citation)"


def _check_citation_count(spec: dict, citations: list) -> str | None:
    if spec.get("allow_no_citations"):
        return None
    verified_count = sum(1 for c in citations if _get(c, "verified", True))
    if verified_count >= 1:
        return None
    return "expected >=1 verified citation (allow_no_citations not set)"


def _check_confidence(spec: dict, confidence: str | None) -> str | None:
    rank = CONFIDENCE_RANK.get((confidence or "").lower())
    at_least = spec.get("expect_confidence_at_least")
    at_most = spec.get("expect_confidence_at_most")
    if rank is None:
        if at_least or at_most:
            return f"confidence {confidence!r} is not one of {sorted(CONFIDENCE_RANK)}"
        return None
    if at_least is not None:
        want = CONFIDENCE_RANK.get(at_least.lower())
        if want is not None and rank < want:
            return f"confidence {confidence!r} is below expect_confidence_at_least={at_least!r}"
    if at_most is not None:
        want = CONFIDENCE_RANK.get(at_most.lower())
        if want is not None and rank > want:
            return f"confidence {confidence!r} is above expect_confidence_at_most={at_most!r}"
    return None


def evaluate(spec: dict, answer: Any) -> EvalOutcome:
    reasons: list[str] = []

    answer_markdown = _get(answer, "answer_markdown", "") or ""
    confidence = _get(answer, "confidence", None)
    citations = _get(answer, "citations", []) or []
    dropped_citations = _get(answer, "dropped_citations", 0) or 0
    usage = _get(answer, "usage", {}) or {}

    reasons.extend(_check_well_formed(answer_markdown, _get(answer, "caveats", None)))

    if reason := _check_answer_contains(spec, answer_markdown):
        reasons.append(reason)

    reasons.extend(_check_answer_contains_all(spec, answer_markdown))

    if reason := _check_answer_not_contains(spec, answer_markdown):
        reasons.append(reason)

    cited_reason, cited_keys = _check_cited_sections(spec, citations)
    if cited_reason:
        reasons.append(cited_reason)

    reasons.extend(_check_cited_sections_all(spec, citations))

    if reason := _check_table_citation(spec, citations):
        reasons.append(reason)

    if reason := _check_citation_count(spec, citations):
        reasons.append(reason)

    if reason := _check_confidence(spec, confidence):
        reasons.append(reason)

    return EvalOutcome(
        id=spec["id"],
        question=spec["question"],
        passed=not reasons,
        reasons=reasons,
        confidence=confidence,
        dropped_citations=dropped_citations,
        cited_sections=cited_keys,
        usage=usage,
        answer_markdown=answer_markdown,
        caveats=_get(answer, "caveats", None),
    )


def _sum_usage(outcomes: list[EvalOutcome]) -> dict[str, int]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    for outcome in outcomes:
        for key in totals:
            totals[key] += int(outcome.usage.get(key, 0) or 0)
    return totals


def _cost_for(totals: dict[str, int]) -> float:
    input_per_mtok, output_per_mtok = _pricing_per_mtok()
    input_cost = totals["input_tokens"] / 1_000_000 * input_per_mtok
    output_cost = totals["output_tokens"] / 1_000_000 * output_per_mtok
    cache_write_cost = totals["cache_creation_input_tokens"] / 1_000_000 * input_per_mtok
    cache_read_cost = (
        totals["cache_read_input_tokens"] / 1_000_000 * input_per_mtok * CACHE_READ_MULTIPLIER
    )
    return input_cost + output_cost + cache_write_cost + cache_read_cost


def print_table(outcomes: list[EvalOutcome]) -> None:
    id_width = max([len(o.id) for o in outcomes] + [10])
    header = f"{'ID'.ljust(id_width)}  {'RESULT':6}  {'CONF':6}  REASONS"
    print(header)
    print("-" * len(header))
    for o in outcomes:
        status = "PASS" if o.passed else "FAIL"
        conf = o.confidence or "-"
        if o.error:
            reasons_str = f"ERROR: {o.error}"
        else:
            reasons_str = "; ".join(o.reasons) if o.reasons else ""
        print(f"{o.id.ljust(id_width)}  {status:6}  {conf:6}  {reasons_str}")


def print_summary(outcomes: list[EvalOutcome]) -> None:
    passed = sum(1 for o in outcomes if o.passed)
    total = len(outcomes)
    totals = _sum_usage(outcomes)
    cost = _cost_for(totals)
    print()
    print(f"{passed}/{total} passed")
    print(
        "tokens: "
        f"input={totals['input_tokens']:,} "
        f"output={totals['output_tokens']:,} "
        f"cache_write={totals['cache_creation_input_tokens']:,} "
        f"cache_read={totals['cache_read_input_tokens']:,}"
    )
    print(f"estimated cost: ${cost:.4f}")


def run_evals(specs: list[dict], parallel: int = 1) -> list[EvalOutcome]:
    from app.agent import ask  # deferred: only needed to actually run evals

    def run_one(spec: dict) -> EvalOutcome:
        t0 = time.monotonic()
        try:
            answer = ask(spec["question"])
            outcome = evaluate(spec, answer)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            outcome = EvalOutcome(
                id=spec["id"],
                question=spec["question"],
                passed=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        outcome.latency_s = round(time.monotonic() - t0, 2)
        return outcome

    outcomes_by_id: dict[str, EvalOutcome] = {}
    if parallel <= 1:
        for spec in specs:
            outcome = run_one(spec)
            outcomes_by_id[outcome.id] = outcome
            print(f"[{'PASS' if outcome.passed else 'FAIL'}] {outcome.id}")
    else:
        # ask() is thread-safe: it builds its own Anthropic client per call
        # and app.law_tools is read-only module state after import.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [pool.submit(run_one, spec) for spec in specs]
            for future in as_completed(futures):
                outcome = future.result()
                outcomes_by_id[outcome.id] = outcome
                print(f"[{'PASS' if outcome.passed else 'FAIL'}] {outcome.id}")
    # Report in questions.yaml order regardless of completion order.
    return [outcomes_by_id[spec["id"]] for spec in specs]


def write_json(outcomes: list[EvalOutcome], path: Path) -> None:
    payload = {
        "date": date.today().isoformat(),
        "results": [asdict(o) for o in outcomes],
        "usage_totals": _sum_usage(outcomes),
        "estimated_cost_usd": round(_cost_for(_sum_usage(outcomes)), 4),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of question ids to run (default: all).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also write results to logs/eval-results-{date}.json",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of questions to run concurrently (default: 4; use 1 for sequential).",
    )
    args = parser.parse_args()

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    specs = load_questions(only)
    if not specs:
        raise SystemExit("no questions to run")

    outcomes = run_evals(specs, parallel=max(1, args.parallel))

    print()
    print_table(outcomes)
    print_summary(outcomes)

    if args.json:
        json_path = LOGS_DIR / f"eval-results-{date.today().isoformat()}.json"
        write_json(outcomes, json_path)

    if any(not o.passed for o in outcomes):
        sys.exit(1)


if __name__ == "__main__":
    main()
