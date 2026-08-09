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
"""

import time
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from models import ChatRequest, ChatResponse, SessionListResponse, SessionSummary, ErrorResponse
import session_store
from llm_client import generate_reply, MODEL
from logging_config import log_request, log_error

app = FastAPI(
    title="Chat API",
    description="A FastAPI backend exposing LLM generation over HTTP with "
                 "server-side session management across stateless connections.",
    version="1.0.0",
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
