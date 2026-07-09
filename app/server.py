"""FastAPI server for the Somerville law Q&A app.

Serves the static frontend, streams answers from the agent over SSE, applies
in-memory rate limiting, and logs each Q&A exchange to a daily JSONL file.

Run from the repo root with: uv run uvicorn app.server:app
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import queue
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
LOG_DIR = REPO_ROOT / "logs"

MAX_QUESTION_LENGTH = 1000

RATE_LIMIT_PER_HOUR = int(os.environ.get("RATE_LIMIT_PER_HOUR", "10"))
DAILY_QUESTION_CAP = int(os.environ.get("DAILY_QUESTION_CAP", "200"))

_HOUR_SECONDS = 3600
_DAY_SECONDS = 86400

app = FastAPI(title="Ask Somerville Law")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- rate limiting (in-memory, per-process; no third-party deps) ---

_rate_lock = threading.Lock()
_ip_requests: dict[str, deque[float]] = {}
_global_requests: deque[float] = deque()


def _check_and_record(ip: str) -> tuple[bool, str | None]:
    """Records the request if allowed. Returns (allowed, reason_if_blocked)."""
    now = time.time()
    with _rate_lock:
        dq = _ip_requests.setdefault(ip, deque())
        while dq and now - dq[0] > _HOUR_SECONDS:
            dq.popleft()
        while _global_requests and now - _global_requests[0] > _DAY_SECONDS:
            _global_requests.popleft()

        if len(dq) >= RATE_LIMIT_PER_HOUR:
            return False, "rate_limit_ip"
        if len(_global_requests) >= DAILY_QUESTION_CAP:
            return False, "rate_limit_global"

        dq.append(now)
        _global_requests.append(now)
        return True, None


def _client_ip(request: Request) -> str:
    """Trust X-Forwarded-For's first value when present, else the socket peer."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    if request.client:
        return request.client.host
    return "unknown"


# --- logging ---


def _hash_ip(ip: str) -> str:
    return hashlib.sha256(ip.encode("utf-8")).hexdigest()[:16]


def _log_qa(
    *,
    request_id: str,
    ip: str,
    question: str,
    answer_markdown: str | None,
    citations: list[dict],
    confidence: str | None,
    dropped_citations: int | None,
    usage: dict | None,
    latency_ms: int,
    error: str | None,
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = LOG_DIR / f"qa-{today}.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "ip_hash": _hash_ip(ip),
        "question": question,
        "answer_markdown": answer_markdown,
        "citations": citations,
        "confidence": confidence,
        "dropped_citations": dropped_citations,
        "usage": usage,
        "latency_ms": latency_ms,
        "error": error,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# --- SSE plumbing ---


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


def _to_plain_dict(obj: Any) -> Any:
    """Best-effort conversion of a dataclass (or dataclass-like object) to a
    plain dict, recursing into dataclass fields. Falls back to vars()/the
    object itself so the server degrades gracefully if agent.py's concrete
    types differ slightly from the spec.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: _to_plain_dict(v) for k, v in vars(obj).items()}
    return obj


def _answer_to_dict(answer: Any) -> dict:
    d = _to_plain_dict(answer)
    if isinstance(d.get("citations"), list):
        d["citations"] = [_to_plain_dict(c) for c in d["citations"]]
        _attach_section_titles(d["citations"])
    return d


def _attach_section_titles(citations: list[dict]) -> None:
    """Add the human-readable section title to each citation for display."""
    try:
        from app.law_tools import get_sections
    except ImportError:
        return
    keys = [c.get("section_key") for c in citations if c.get("section_key")]
    if not keys:
        return
    titles = {
        rec.get("key"): rec.get("title")
        for rec in get_sections(list(dict.fromkeys(keys)))
        if not rec.get("error")
    }
    for c in citations:
        title = titles.get(c.get("section_key"))
        if title:
            c["section_title"] = title


def _run_agent(
    ask_fn: Callable[..., Any],
    question: str,
    event_queue: "queue.Queue[tuple[str, Any]]",
) -> None:
    def on_event(evt: dict) -> None:
        event_queue.put(("event", evt))

    try:
        answer = ask_fn(question, history=None, on_event=on_event)
        event_queue.put(("answer", answer))
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as a generic error
        event_queue.put(("error", exc))
    finally:
        event_queue.put(("done", None))


async def _event_stream(ask_fn: Callable[..., Any], question: str, ip: str, request_id: str):
    event_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
    thread = threading.Thread(target=_run_agent, args=(ask_fn, question, event_queue), daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()
    start = time.monotonic()

    answer_dict: dict | None = None
    error_message: str | None = None

    while True:
        kind, payload = await loop.run_in_executor(None, event_queue.get)
        if kind == "event":
            yield _sse(payload)
        elif kind == "answer":
            answer_dict = _answer_to_dict(payload)
            yield _sse({"type": "answer", **answer_dict})
        elif kind == "error":
            error_message = str(payload)
            yield _sse(
                {
                    "type": "error",
                    "message": "Something went wrong while answering your question. Please try again.",
                }
            )
        elif kind == "done":
            yield _sse({"type": "done"})
            break

    latency_ms = int((time.monotonic() - start) * 1000)
    log_citations = []
    for c in (answer_dict or {}).get("citations", []) or []:
        if isinstance(c, dict):
            log_citations.append({"section_key": c.get("section_key"), "verified": c.get("verified")})

    _log_qa(
        request_id=request_id,
        ip=ip,
        question=question,
        answer_markdown=(answer_dict or {}).get("answer_markdown"),
        citations=log_citations,
        confidence=(answer_dict or {}).get("confidence"),
        dropped_citations=(answer_dict or {}).get("dropped_citations"),
        usage=(answer_dict or {}).get("usage"),
        latency_ms=latency_ms,
        error=error_message,
    )


# --- routes ---


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/ask")
async def api_ask(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_json", "message": "Request body must be valid JSON."},
        )

    question = body.get("question") if isinstance(body, dict) else None
    if not isinstance(question, str) or not question.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_request", "message": "A non-empty 'question' field is required."},
        )
    if len(question) > MAX_QUESTION_LENGTH:
        return JSONResponse(
            status_code=400,
            content={
                "error": "question_too_long",
                "message": f"Questions are limited to {MAX_QUESTION_LENGTH} characters.",
            },
        )
    question = question.strip()

    ip = _client_ip(request)
    allowed, reason = _check_and_record(ip)
    if not allowed:
        if reason == "rate_limit_ip":
            message = (
                f"You've reached the limit of {RATE_LIMIT_PER_HOUR} questions per hour. "
                "Please try again later."
            )
        else:
            message = "This service has reached its daily question limit. Please try again tomorrow."
        return JSONResponse(status_code=429, content={"error": reason, "message": message})

    try:
        # Lazy import: app.agent may be developed concurrently and might not
        # exist yet. Importing here (not at module scope) lets this module
        # load cleanly regardless.
        from app.agent import ask  # noqa: F401  (Answer not referenced directly here)
    except ImportError:
        return JSONResponse(
            status_code=503,
            content={
                "error": "agent_unavailable",
                "message": "The Q&A agent is not available right now. Please try again shortly.",
            },
        )

    request_id = uuid.uuid4().hex[:12]
    return StreamingResponse(
        _event_stream(ask, question, ip, request_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
