"""
main.py
----------
The FastAPI server. Exposes:
    POST /api/chat      -- send a message, get a reply, session managed automatically
    GET  /api/sessions   -- list all active sessions with message counts
    GET  /docs            -- auto-generated Swagger UI (FastAPI provides this for free)

Run locally:
    export GEMINI_API_KEY="your-key-here"
    uvicorn main:app --reload --port 8000

Then visit http://127.0.0.1:8000/docs for the interactive Swagger UI, or
run test_client.py against it for automated verification.

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
"""

import time
import json
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from models import ChatRequest, ChatResponse, SessionListResponse, SessionSummary, ErrorResponse
import session_store
from llm_client import generate_reply, generate_reply_stream, MODEL
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
    404: {"model": ErrorResponse, "description": "session_id provided but does not exist"},
    500: {"model": ErrorResponse, "description": "Internal server or LLM call failure"},
})
async def chat(request: ChatRequest):
    """
    Sends a message and returns the model's reply.

    - If session_id is omitted, a new session is created (with the
      system prompt seeded as the first history entry).
    - If session_id is provided but doesn't exist, returns 404.
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
        session_id = session_store.create_session()
    else:
        if not session_store.session_exists(request.session_id):
            log_error(now_iso(), request.session_id, "session_not_found",
                       f"No session with ID '{request.session_id}' exists.", 404)
            raise HTTPException(status_code=404, detail={
                "error": "session_not_found",
                "message": f"No session with ID '{request.session_id}' exists. "
                           f"Omit session_id to start a new one.",
                "status_code": 404,
            })
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
    The streaming counterpart to POST /api/chat -- same session logic
    (check/create session, append user message, append final reply), but
    the model's reply is sent to the client piece-by-piece AS IT ARRIVES,
    using Server-Sent Events (SSE), instead of waiting for the complete
    response.

    WHY THIS DOESN'T MAKE THE MODEL RESPOND FASTER (Day 8's finding,
    revisited): the TOTAL time to generate the full reply is the same
    either way -- streaming only changes WHEN the user starts SEEING
    text, not how fast the underlying generation happens. The UX benefit
    is purely about perceived responsiveness.

    SSE FORMAT: each event is sent as a "event: <type>\\ndata: <json>\\n\\n"
    block. Two event types are used here:
      - "chunk": {"text": "..."} for each piece of text as it arrives
      - "done": {"session_id", "message_count", "latency_ms"} sent once,
        at the very end, after the full reply has been accumulated and
        saved to the session history -- this is where the metadata that
        used to come back in one single JSON response (Day 11/12) now
        arrives, since it isn't known until generation is complete.
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
    """Lists every active session with its message count and timestamps."""
    sessions = session_store.list_sessions()
    return SessionListResponse(
        active_sessions=len(sessions),
        sessions=[SessionSummary(**s) for s in sessions],
    )


@app.get("/")
async def root():
    """A simple root endpoint confirming the server is up."""
    return {"status": "ok", "message": "Chat API is running. See /docs for interactive API documentation."}
