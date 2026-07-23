"""FastAPI server for the Somerville law Q&A app.

Serves the static frontend, streams answers from the agent over SSE, applies
in-memory rate limiting, and logs each Q&A exchange to a daily JSONL file.

Run from the repo root with: uv run uvicorn app.server:app

For local development, ``.env`` is loaded automatically without overwriting
variables already present in the process environment.

Environment variables:
  RATE_LIMIT_PER_HOUR  Per-IP sliding-window limit (default 10).
  DAILY_QUESTION_CAP   Global daily request cap (default 200).
  TRUST_PROXY          Set to "1" ONLY when this app runs behind a trusted
                       reverse proxy that overwrites/strips any inbound
                       X-Forwarded-For header. When set, the first value of
                       X-Forwarded-For identifies the client for rate limiting.
                       When unset (the default), the direct connection address
                       (request.client.host) is used, because X-Forwarded-For is
                       client-spoofable and would otherwise defeat per-IP limits.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
LOG_DIR = REPO_ROOT / "logs"

# Keep the simple documented launch command reliable in local development.
# Explicit process environment variables still win in deployed environments.
load_dotenv(REPO_ROOT / ".env", override=False)

MAX_QUESTION_LENGTH = 1000

RATE_LIMIT_PER_HOUR = int(os.environ.get("RATE_LIMIT_PER_HOUR", "10"))
DAILY_QUESTION_CAP = int(os.environ.get("DAILY_QUESTION_CAP", "200"))

_HOUR_SECONDS = 3600
_DAY_SECONDS = 86400

# Cap on the number of per-IP buckets kept in memory. Empty buckets are pruned
# on every check; this cap bounds worst-case growth from a flood of distinct
# IPs between the arrival of their requests and the next expiry sweep.
_MAX_IP_BUCKETS = 50_000

app = FastAPI(title="Somerville Law Bot")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- rate limiting (in-memory, per-process; no third-party deps) ---

_rate_lock = threading.Lock()
_ip_requests: dict[str, deque[float]] = {}
_global_requests: deque[float] = deque()


def _sweep_ip_buckets_locked(now: float) -> None:
    """Expire stale timestamps across every per-IP bucket and drop empties.

    Caller must hold _rate_lock. A bucket only becomes empty (and thus leaks)
    once its own IP's timestamps have all aged out, so we expire in place here
    before deleting. Used as the periodic cap-triggered sweep and exposed to
    tests via _sweep_empty_ip_buckets(). In-memory best-effort; kept simple.
    """
    for key in list(_ip_requests.keys()):
        dq = _ip_requests[key]
        while dq and now - dq[0] > _HOUR_SECONDS:
            dq.popleft()
        if not dq:
            del _ip_requests[key]


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

        # Drop this bucket if it ended up empty (defensive; we just appended),
        # then run a full expiry+prune sweep if the dict has grown past the cap
        # so that buckets from IPs that never return can't accumulate forever.
        if not dq:
            _ip_requests.pop(ip, None)
        if len(_ip_requests) > _MAX_IP_BUCKETS:
            _sweep_ip_buckets_locked(now)
        return True, None


def _sweep_empty_ip_buckets() -> None:
    """Force a full expiry+prune sweep of every per-IP bucket (thread-safe)."""
    now = time.time()
    with _rate_lock:
        _sweep_ip_buckets_locked(now)


def _client_ip(request: Request) -> str:
    """Identify the client for rate limiting.

    Uses the direct connection address (request.client.host) by default because
    X-Forwarded-For is client-spoofable and trusting it would defeat per-IP
    rate limiting. Only when TRUST_PROXY=1 (i.e. this app sits behind a trusted
    reverse proxy that overwrites/strips inbound XFF) do we honor the first
    value of X-Forwarded-For. See the module docstring.
    """
    if os.environ.get("TRUST_PROXY") == "1":
        xff = request.headers.get("x-forwarded-for")
        if xff:
            first = xff.split(",")[0].strip()
            if first:
                return first
    if request.client:
        return request.client.host
    return "unknown"


# --- logging ---


# Salting makes the hashes impractical to brute-force back to addresses
# (the IPv4 space is small) while keeping same-client correlation. Set
# IP_HASH_SALT to a long random value in production; unset, hashes are
# unsalted and therefore reversible by enumeration.
_IP_HASH_SALT = os.environ.get("IP_HASH_SALT", "")


def _hash_ip(ip: str) -> str:
    return hashlib.sha256((_IP_HASH_SALT + ip).encode("utf-8")).hexdigest()[:16]


def _version_stamp() -> dict:
    """Model + deploy identifiers so log records are attributable."""
    try:
        from app.agent import MODEL  # lazy: agent import is deferred elsewhere too

        model = MODEL
    except Exception:
        model = os.environ.get("LAW_QA_MODEL")
    return {
        "model": model,
        "git_sha": (os.environ.get("RAILWAY_GIT_COMMIT_SHA") or "")[:12] or None,
    }


def _log_qa(
    *,
    request_id: str,
    ip: str,
    question: str,
    answer_markdown: str | None,
    caveats: str | None,
    citations: list[dict],
    confidence: str | None,
    dropped_citations: int | None,
    dropped_detail: list[str] | None,
    usage: dict | None,
    latency_ms: int,
    error: str | None,
    client_disconnected: bool = False,
    tool_trace: list[dict] | None = None,
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = LOG_DIR / f"qa-{today}.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "ip_hash": _hash_ip(ip),
        **_version_stamp(),
        "question": question,
        "tool_trace": tool_trace or [],
        "answer_markdown": answer_markdown,
        "caveats": caveats,
        "citations": citations,
        "confidence": confidence,
        "dropped_citations": dropped_citations,
        "dropped_detail": dropped_detail or [],
        "usage": usage,
        "latency_ms": latency_ms,
        "error": error,
        "client_disconnected": client_disconnected,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# Rejected requests are logged too (abuse is invisible otherwise), but capped
# per ip-hash per hour so a flood can't spam the log file itself.
_REJECTION_LOG_CAP_PER_HOUR = 5
_rejection_log_lock = threading.Lock()
_rejection_log_counts: dict[tuple[str, int], int] = {}


def _log_rejection(*, ip: str, reason: str) -> None:
    hour = int(time.time() // _HOUR_SECONDS)
    ip_hash = _hash_ip(ip)
    with _rejection_log_lock:
        for key in [k for k in _rejection_log_counts if k[1] != hour]:
            del _rejection_log_counts[key]
        count = _rejection_log_counts.get((ip_hash, hour), 0)
        if count >= _REJECTION_LOG_CAP_PER_HOUR:
            return
        _rejection_log_counts[(ip_hash, hour)] = count + 1
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = LOG_DIR / f"rejections-{today}.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ip_hash": ip_hash,
        "reason": reason,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# Reader feedback (thumbs up/down on answers). Same per-ip-hash hourly cap
# idea as rejections, just roomier — feedback is cheap but shouldn't be a
# spam vector either.
_FEEDBACK_LOG_CAP_PER_HOUR = 20
_feedback_log_lock = threading.Lock()
_feedback_log_counts: dict[tuple[str, int], int] = {}


def _log_feedback(*, ip: str, request_id: str, verdict: str) -> bool:
    """Returns False when this ip-hash is over its hourly feedback cap."""
    hour = int(time.time() // _HOUR_SECONDS)
    ip_hash = _hash_ip(ip)
    with _feedback_log_lock:
        for key in [k for k in _feedback_log_counts if k[1] != hour]:
            del _feedback_log_counts[key]
        count = _feedback_log_counts.get((ip_hash, hour), 0)
        if count >= _FEEDBACK_LOG_CAP_PER_HOUR:
            return False
        _feedback_log_counts[(ip_hash, hour)] = count + 1
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = LOG_DIR / f"feedback-{today}.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ip_hash": ip_hash,
        "request_id": request_id,
        "verdict": verdict,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return True


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
    """Add the human-readable section label to each citation for display."""
    try:
        from app.law_tools import section_label
    except ImportError:
        return
    for c in citations:
        key = c.get("section_key")
        if key:
            label = section_label(key)
            if label and label != key:
                c["section_title"] = label


def _run_agent(
    ask_fn: Callable[..., Any],
    question: str,
    loop: asyncio.AbstractEventLoop,
    async_queue: "asyncio.Queue[tuple[str, Any]]",
    holder: dict[str, Any],
) -> None:
    """Run the synchronous agent in a worker thread and deliver events to the
    request's asyncio.Queue via the event loop. No executor thread is parked on
    the consuming side, so concurrency is not bounded by the executor pool.

    Results are also stashed in `holder` so the request handler's finally block
    can log a completed answer even if the client disconnected before it was
    streamed out.
    """

    def deliver(item: tuple[str, Any]) -> None:
        # call_soon_threadsafe is the only safe way to touch the loop / queue
        # from this worker thread.
        loop.call_soon_threadsafe(async_queue.put_nowait, item)

    def on_event(evt: dict) -> None:
        if evt.get("type") == "tool":
            holder.setdefault("tool_trace", []).append(
                {"name": evt.get("name"), "detail": evt.get("detail")}
            )
        deliver(("event", evt))

    try:
        answer = ask_fn(question, history=None, on_event=on_event)
        holder["answer"] = answer
        deliver(("answer", answer))
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as a generic error
        holder["error"] = exc
        deliver(("error", exc))
    finally:
        deliver(("done", None))


async def _event_stream(ask_fn: Callable[..., Any], question: str, ip: str, request_id: str):
    loop = asyncio.get_running_loop()
    async_queue: "asyncio.Queue[tuple[str, Any]]" = asyncio.Queue()
    holder: dict[str, Any] = {}
    thread = threading.Thread(
        target=_run_agent,
        args=(ask_fn, question, loop, async_queue, holder),
        daemon=True,
    )
    thread.start()

    start = time.monotonic()

    answer_dict: dict | None = None
    error_message: str | None = None
    completed = False

    try:
        while True:
            kind, payload = await async_queue.get()
            if kind == "event":
                yield _sse(payload)
            elif kind == "answer":
                answer_dict = _answer_to_dict(payload)
                yield _sse({"type": "answer", "request_id": request_id, **answer_dict})
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
                completed = True
                break
    finally:
        # Always log, even if the client disconnected mid-stream (which cancels
        # this generator) while the agent thread kept working. Use whatever the
        # shared holder captured if we never streamed the answer/error out.
        latency_ms = int((time.monotonic() - start) * 1000)

        if answer_dict is None and holder.get("answer") is not None:
            try:
                answer_dict = _answer_to_dict(holder["answer"])
            except Exception:  # noqa: BLE001 - logging must not raise
                answer_dict = None
        if error_message is None and holder.get("error") is not None:
            error_message = str(holder["error"])

        log_citations = []
        for c in (answer_dict or {}).get("citations", []) or []:
            if isinstance(c, dict):
                log_citations.append(
                    {"section_key": c.get("section_key"), "verified": c.get("verified")}
                )

        _log_qa(
            request_id=request_id,
            ip=ip,
            question=question,
            answer_markdown=(answer_dict or {}).get("answer_markdown"),
            caveats=(answer_dict or {}).get("caveats"),
            citations=log_citations,
            confidence=(answer_dict or {}).get("confidence"),
            dropped_citations=(answer_dict or {}).get("dropped_citations"),
            dropped_detail=(answer_dict or {}).get("dropped_detail"),
            usage=(answer_dict or {}).get("usage"),
            latency_ms=latency_ms,
            error=error_message,
            client_disconnected=not completed,
            tool_trace=holder.get("tool_trace"),
        )


# --- routes ---


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


# --- dev live reload (enabled only with DEV_RELOAD=1; never set in prod) ---

DEV_RELOAD = os.environ.get("DEV_RELOAD") == "1"
_SERVER_STARTED_AT = time.time()

_DEV_RELOAD_SCRIPT = """
<script>
/* dev live reload: polls /dev/reload-state and refreshes when static files
   change or the server restarts. Injected only when DEV_RELOAD=1. */
(async () => {
  let last = null;
  for (;;) {
    try {
      const r = await fetch("/dev/reload-state", { cache: "no-store" });
      const { token } = await r.json();
      if (last !== null && token !== last) location.reload();
      last = token;
    } catch (e) {
      /* server restarting; keep polling */
    }
    await new Promise((res) => setTimeout(res, 500));
  }
})();
</script>
"""


def _static_mtime_token() -> str:
    latest = 0.0
    for path in STATIC_DIR.rglob("*"):
        if path.is_file():
            latest = max(latest, path.stat().st_mtime)
    return f"{_SERVER_STARTED_AT:.0f}-{latest:.6f}"


@app.get("/dev/reload-state")
async def dev_reload_state():
    if not DEV_RELOAD:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    return {"token": _static_mtime_token()}


@app.get("/")
async def index():
    if DEV_RELOAD:
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        if "</body>" in html:
            html = html.replace("</body>", _DEV_RELOAD_SCRIPT + "</body>", 1)
        else:
            html += _DEV_RELOAD_SCRIPT
        return HTMLResponse(html)
    return FileResponse(STATIC_DIR / "index.html")


# The readable corpus pages that citation URLs point to (/ordinances#secid-N
# and /zoning#secid-N). Served by this app so citations work on the app's own
# domain. /code and the *.readable.html filenames stay routable because they
# appear in previously-given answers and logs.
_READABLE_PAGES = {
    "ordinances": "somerville-law-non-zoning.readable.html",
    "zoning": "somerville-zoning.readable.html",
}


@app.get("/ordinances")
@app.get("/code")
async def ordinances_page() -> FileResponse:
    return FileResponse(REPO_ROOT / _READABLE_PAGES["ordinances"], media_type="text/html")


@app.get("/zoning")
async def zoning_page() -> FileResponse:
    return FileResponse(REPO_ROOT / _READABLE_PAGES["zoning"], media_type="text/html")


@app.get("/{page_name}.readable.html")
async def readable_page(page_name: str):
    filename = f"{page_name}.readable.html"
    if filename not in _READABLE_PAGES.values():
        return JSONResponse(status_code=404, content={"error": "not_found"})
    return FileResponse(REPO_ROOT / filename, media_type="text/html")


@app.post("/api/feedback")
async def api_feedback(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_json", "message": "Request body must be valid JSON."},
        )
    request_id = body.get("request_id")
    verdict = body.get("verdict")
    if (
        not isinstance(request_id, str)
        or not (1 <= len(request_id) <= 32)
        or not request_id.isalnum()
        or verdict not in ("up", "down")
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_feedback", "message": "Invalid feedback payload."},
        )
    accepted = _log_feedback(ip=_client_ip(request), request_id=request_id, verdict=verdict)
    return {"ok": accepted}


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

    # Report configuration failures before consuming rate-limit budget or
    # starting an SSE response, where they would otherwise become a vague
    # mid-stream error in the browser.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return JSONResponse(
            status_code=503,
            content={
                "error": "anthropic_not_configured",
                "message": (
                    "The Q&A service is not configured. For local development, "
                    "add ANTHROPIC_API_KEY to the repository's .env file and restart the server."
                ),
            },
        )

    # Confirm the agent is importable BEFORE consuming any rate-limit budget, so
    # a 503 (agent unavailable) never burns per-IP or global quota.
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
        _log_rejection(ip=ip, reason=reason or "rate_limited")
        return JSONResponse(status_code=429, content={"error": reason, "message": message})

    request_id = uuid.uuid4().hex[:12]
    return StreamingResponse(
        _event_stream(ask, question, ip, request_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
