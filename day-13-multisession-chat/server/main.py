"""
main.py
----------
The FastAPI server. Exposes:
    POST /api/chat                          -- send a message, get a reply
    GET  /api/sessions                        -- list all active sessions
    POST /api/sessions/{session_id}/title       -- Day 13: generate a short AI title
    POST /api/sessions/{session_id}/regenerate   -- Day 13: regenerate the last response
    GET  /docs                                    -- auto-generated Swagger UI

Run locally:
    export GEMINI_API_KEY="your-key-here"
    uvicorn main:app --reload --port 8000

For testing WITHOUT a real API key/network access:
    export USE_MOCK_LLM=true
    uvicorn main:app --reload --port 8000

CORS (Day 12): the React dev server runs on a DIFFERENT origin
(http://localhost:5173) than this API (http://127.0.0.1:8000). Browsers
enforce the Same-Origin Policy by default -- a fetch() from the React app
to this API would be BLOCKED by the browser itself, before the request
even reaches this server, unless this server explicitly tells the browser
"requests from that specific origin are allowed." CORSMiddleware below is
what sends that permission back in every response's headers.

*** A DELIBERATE ARCHITECTURE CHANGE IN DAY 13, DOCUMENTED HONESTLY ***
Days 11-12: the SERVER owned session ID generation. A client-provided
session_id that didn't exist on the server was treated as an error (404
"session_not_found") -- a reasonable rule when the server is the only
thing that ever creates IDs.

Day 13 flips this: the task requires the FRONTEND to generate the ID via
crypto.randomUUID() the moment "New Chat" is clicked -- BEFORE any
message is ever sent, so the UI can register and display the new session
immediately. This means the very first /api/chat request for a new
session now legitimately carries a session_id the server has never seen.

The fix: POST /api/chat now LAZILY CREATES a session using whatever
session_id the client provides, if it doesn't already exist, rather than
404ing. This is a genuine, common real-world pattern (client-generated
UUIDs for new resources) precisely because UUIDs are cryptographically
random enough that accidental collisions are not a realistic concern --
this is NOT the same as "accept any ID a client makes up to imitate
someone else's existing resource," since a truly unknown ID is, by
definition, presumed to be a legitimate NEW session under this design.
"""

import time
import json
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from models import (
    ChatRequest, ChatResponse, SessionListResponse, SessionSummary, ErrorResponse,
    TitleResponse, RegenerateRequest,
)
import session_store
from llm_client import generate_reply, generate_reply_stream, generate_title, MODEL
from logging_config import log_request, log_error

app = FastAPI(
    title="Chat API",
    description="A FastAPI backend exposing LLM generation over HTTP with "
                 "server-side session management across stateless connections.",
    version="1.0.0",
)

# CORS: allow the React dev server (Vite's default port 5173) to call this
# API from the browser. In a real production deployment, allow_origins
# would list the actual deployed frontend's domain(s), NOT a wildcard --
# allow_origins=["*"] combined with allow_credentials=True is explicitly
# disallowed by the CORS spec itself (browsers will reject it), and even
# where technically permitted, a wildcard defeats the entire purpose of
# CORS as an access-control mechanism.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    """
    Catches any exception NOT already handled by a specific try/except in
    a route -- ensures a genuinely unexpected server-side failure still
    returns a clean, structured 500 response instead of leaking a raw
    Python traceback to the client (a real security/professionalism
    concern for any public-facing API).
    """
    log_error(now_iso(), None, "internal_server_error", str(exc), 500)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred while processing the request.",
            status_code=500,
        ).model_dump(),
    )


@app.post("/api/chat", response_model=ChatResponse, responses={
    400: {"model": ErrorResponse, "description": "Invalid request (e.g. empty message)"},
    500: {"model": ErrorResponse, "description": "Internal server or LLM call failure"},
})
async def chat(request: ChatRequest):
    """
    Sends a message and returns the model's reply.

    - If session_id is omitted, the server generates a new one.
    - If session_id is provided but unrecognized, Day 13 LAZILY CREATES a
      new session using that exact ID (see module docstring) -- this is
      the expected path for a brand-new session whose ID the frontend
      generated via crypto.randomUUID() before this first message was sent.
    - If session_id is provided and already exists, the message is
      appended to that session's existing history.
    - Every turn appends BOTH the user's message and the model's reply
      to the session history, so the next request in the same session
      has full context.
    """
    start_time = time.perf_counter()

    if request.message.strip() == "":
        log_error(now_iso(), request.session_id, "empty_message",
                   "Message cannot be blank or whitespace-only.", 400)
        raise HTTPException(status_code=400, detail={
            "error": "empty_message",
            "message": "Message cannot be blank or whitespace-only.",
            "status_code": 400,
        })

    if request.session_id is None:
        # No ID provided at all -- server generates one (legacy path,
        # still supported for any client that doesn't pre-generate an ID).
        session_id = session_store.create_session()
    elif not session_store.session_exists(request.session_id):
        # Day 13: an unrecognized session_id is now treated as a NEW
        # session using the client's own ID (lazy creation), not an
        # error -- see the module docstring's "architecture change" note.
        session_id = session_store.create_session(session_id=request.session_id)
    else:
        session_id = request.session_id

    session_store.append_message(session_id, "user", request.message)
    history = session_store.get_history(session_id)

    try:
        reply_text, usage = generate_reply(history)
    except Exception as e:
        log_error(now_iso(), session_id, "llm_call_failed", str(e), 500)
        raise HTTPException(status_code=500, detail={
            "error": "llm_call_failed",
            "message": f"The model call failed: {e}",
            "status_code": 500,
        })

    session_store.append_message(session_id, "model", reply_text)

    latency_ms = (time.perf_counter() - start_time) * 1000

    log_request(
        timestamp=now_iso(),
        session_id=session_id,
        model=MODEL,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        latency_ms=latency_ms,
    )

    updated_history = session_store.get_history(session_id)
    message_count = len([m for m in updated_history if m["role"] != "system"])

    return ChatResponse(
        session_id=session_id,
        response=reply_text,
        message_count=message_count,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming counterpart to POST /api/chat, carried forward from the Day
    12 improvement -- same session logic (including Day 13's lazy
    creation for a client-generated session_id), but the model's reply is
    sent piece-by-piece via Server-Sent Events as it's generated, instead
    of waiting for the complete response.

    SSE event types:
      - "chunk": {"text": "..."} for each piece of text as it arrives
      - "done": {"session_id", "message_count", "latency_ms"} once, at
        the very end, after the full reply is saved to session history
      - "error": {"message": "..."} if generation fails mid-stream
    """
    start_time = time.perf_counter()

    if request.message.strip() == "":
        raise HTTPException(status_code=400, detail={
            "error": "empty_message",
            "message": "Message cannot be blank or whitespace-only.",
            "status_code": 400,
        })

    if request.session_id is None:
        session_id = session_store.create_session()
    elif not session_store.session_exists(request.session_id):
        session_id = session_store.create_session(session_id=request.session_id)
    else:
        session_id = request.session_id

    session_store.append_message(session_id, "user", request.message)
    history = session_store.get_history(session_id)

    def event_stream():
        full_text = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            for piece in generate_reply_stream(history):
                if isinstance(piece, tuple) and piece[0] == "__usage__":
                    usage = piece[1]
                    continue
                full_text += piece
                yield f"event: chunk\ndata: {json.dumps({'text': piece})}\n\n"
        except Exception as e:
            log_error(now_iso(), session_id, "llm_call_failed", str(e), 500)
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
            return

        session_store.append_message(session_id, "model", full_text)
        latency_ms = (time.perf_counter() - start_time) * 1000

        log_request(
            timestamp=now_iso(), session_id=session_id, model=MODEL,
            prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"], latency_ms=latency_ms,
        )

        updated_history = session_store.get_history(session_id)
        message_count = len([m for m in updated_history if m["role"] != "system"])

        yield f"event: done\ndata: {json.dumps({'session_id': session_id, 'message_count': message_count, 'latency_ms': round(latency_ms, 2)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/sessions", response_model=SessionListResponse)
async def list_sessions():
    """Lists every active session with its message count, timestamps, and title."""
    sessions = session_store.list_sessions()
    return SessionListResponse(
        active_sessions=len(sessions),
        sessions=[SessionSummary(**s) for s in sessions],
    )


@app.post("/api/sessions/{session_id}/title", response_model=TitleResponse, responses={
    404: {"model": ErrorResponse, "description": "session_id does not exist"},
    400: {"model": ErrorResponse, "description": "Session has no exchange to summarize yet"},
    500: {"model": ErrorResponse, "description": "Internal server or LLM call failure"},
})
async def generate_session_title(session_id: str):
    """
    Day 13: generates a short (3-5 word) AI title summarizing a session's
    FIRST exchange, and stores it on the session. Unlike POST /api/chat,
    this endpoint operates on an EXISTING session only -- there is no
    sensible "lazy creation" here, since generating a title requires an
    actual first exchange to already exist. A genuinely unknown
    session_id is therefore still a real 404 here.
    """
    if not session_store.session_exists(session_id):
        log_error(now_iso(), session_id, "session_not_found",
                   f"No session with ID '{session_id}' exists.", 404)
        raise HTTPException(status_code=404, detail={
            "error": "session_not_found",
            "message": f"No session with ID '{session_id}' exists.",
            "status_code": 404,
        })

    history = session_store.get_history(session_id)
    user_messages = [m for m in history if m["role"] == "user"]
    model_messages = [m for m in history if m["role"] == "model"]

    if not user_messages or not model_messages:
        raise HTTPException(status_code=400, detail={
            "error": "no_exchange_yet",
            "message": "Session needs at least one completed exchange before a title can be generated.",
            "status_code": 400,
        })

    try:
        title = generate_title(user_messages[0]["content"], model_messages[0]["content"])
    except Exception as e:
        log_error(now_iso(), session_id, "llm_call_failed", str(e), 500)
        raise HTTPException(status_code=500, detail={
            "error": "llm_call_failed",
            "message": f"Title generation failed: {e}",
            "status_code": 500,
        })

    session_store.set_title(session_id, title)
    return TitleResponse(session_id=session_id, title=title)


@app.post("/api/sessions/{session_id}/regenerate", response_model=ChatResponse, responses={
    404: {"model": ErrorResponse, "description": "session_id does not exist"},
    400: {"model": ErrorResponse, "description": "No user message available to regenerate"},
    500: {"model": ErrorResponse, "description": "Internal server or LLM call failure"},
})
async def regenerate_response(session_id: str):
    """
    Day 13: regenerates the model's response to the LAST user message in
    a session. Pops the previous (user, model) exchange from history via
    session_store.pop_last_turn(), then re-sends the recovered user
    message through the normal generation path -- so the session ends up
    with exactly one user message and one (new) model reply for that
    turn, not a duplicated user message with two stacked replies.
    """
    start_time = time.perf_counter()

    if not session_store.session_exists(session_id):
        log_error(now_iso(), session_id, "session_not_found",
                   f"No session with ID '{session_id}' exists.", 404)
        raise HTTPException(status_code=404, detail={
            "error": "session_not_found",
            "message": f"No session with ID '{session_id}' exists.",
            "status_code": 404,
        })

    try:
        last_user_message = session_store.pop_last_turn(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "error": "nothing_to_regenerate",
            "message": str(e),
            "status_code": 400,
        })

    session_store.append_message(session_id, "user", last_user_message)
    history = session_store.get_history(session_id)

    try:
        reply_text, usage = generate_reply(history)
    except Exception as e:
        log_error(now_iso(), session_id, "llm_call_failed", str(e), 500)
        raise HTTPException(status_code=500, detail={
            "error": "llm_call_failed",
            "message": f"The model call failed: {e}",
            "status_code": 500,
        })

    session_store.append_message(session_id, "model", reply_text)
    latency_ms = (time.perf_counter() - start_time) * 1000

    log_request(
        timestamp=now_iso(), session_id=session_id, model=MODEL,
        prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"], latency_ms=latency_ms,
    )

    updated_history = session_store.get_history(session_id)
    message_count = len([m for m in updated_history if m["role"] != "system"])

    return ChatResponse(
        session_id=session_id, response=reply_text,
        message_count=message_count, latency_ms=round(latency_ms, 2),
    )


@app.get("/")
async def root():
    """A simple root endpoint confirming the server is up."""
    return {"status": "ok", "message": "Chat API is running. See /docs for interactive API documentation."}
