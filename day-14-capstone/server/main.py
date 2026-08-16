"""
main.py
----------
The FastAPI capstone application. Exposes:
    POST   /api/session/start        -- create a session, stream the first riddle/question/greeting
    POST   /api/chat/stream           -- submit a guess/answer/hint/give_up/question, stream the response
    DELETE /api/session/{session_id}   -- explicit termination (Back button)
    GET    /api/monitoring/recent       -- last N monitoring log entries

Run locally:
    export GEMINI_API_KEY="your-key-here"    (or USE_MOCK_LLM=true)
    uvicorn main:app --reload --port 8000
    (or just: ./start.sh)
"""

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from models import StartSessionRequest, ChatRequest
import session_store
import activity_engine
from activities import VALID_ACTIVITIES
from safety import is_blatantly_inappropriate, SAFE_REDIRECT_MESSAGE
from monitoring import log_llm_request, TTFTTimer, LOG_FILE

SWEEP_INTERVAL_SECONDS = 10


async def _background_session_sweeper():
    """
    Requirement #2's server-side enforcement: periodically removes any
    session inactive for more than 60 seconds, so expired session data
    genuinely does not persist -- independent of whether the frontend's
    own inactivity timer runs correctly.
    """
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
        expired = session_store.sweep_expired_sessions()
        if expired:
            print(f"[session sweep] removed {len(expired)} expired session(s): {[s[:8] for s in expired]}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_background_session_sweeper())
    yield
    task.cancel()


app = FastAPI(
    title="Learning Adventures API",
    description="An educational chatbot backend for children: Brain Buster, Quick Fire, and Ask & Explore.",
    version="1.0.0",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = ["http://localhost:5174", "http://127.0.0.1:5174"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
    expose_headers=["X-Session-Id"],
)


def _sse_error(message):
    return f"event: error\ndata: {json.dumps({'message': message})}\n\n"


def _run_activity_stream(session_id, activity, generator_fn, user_prompt_for_log):
    """
    Wraps an activity_engine generator into a full SSE response,
    handling monitoring (requirement #8) uniformly across every activity
    and every kind of turn (start, guess, hint, give_up, ask).
    """
    def event_stream():
        timer = TTFTTimer()
        timer.start()
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        first_chunk_sent = False

        try:
            for piece in generator_fn:
                if isinstance(piece, tuple):
                    if piece[0] == "__usage__":
                        usage = piece[1]
                    elif piece[0] == "__new_item__":
                        yield "event: new_item\ndata: {}\n\n"
                    continue

                if not first_chunk_sent:
                    timer.mark_first_chunk()
                    first_chunk_sent = True
                yield f"event: chunk\ndata: {json.dumps({'text': piece})}\n\n"
        except Exception as e:
            yield _sse_error(str(e))
            return

        total_time_ms = timer.elapsed_ms()
        log_llm_request(
            session_id=session_id,
            activity=activity,
            user_prompt=user_prompt_for_log,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            ttft_ms=timer.ttft_ms(),
            total_time_ms=total_time_ms,
        )

        yield f"event: done\ndata: {json.dumps({'session_id': session_id, 'total_time_ms': round(total_time_ms, 2)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/session/start")
async def start_session(request: StartSessionRequest):
    """Creates a new session and streams the first riddle/question/greeting."""
    if request.activity not in VALID_ACTIVITIES:
        raise HTTPException(status_code=400, detail={
            "error": "invalid_activity", "message": f"Unknown activity: {request.activity}", "status_code": 400,
        })

    session_id = session_store.create_session(request.activity)

    if request.activity == "brain_buster":
        generator = activity_engine.start_brain_buster(session_id)
    elif request.activity == "quick_fire":
        generator = activity_engine.start_quick_fire(session_id)
    else:
        generator = activity_engine.start_ask_explore(session_id)

    response = _run_activity_stream(session_id, request.activity, generator, "[session start]")
    response.headers["X-Session-Id"] = session_id
    return response


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Handles every subsequent turn: a guess, an answer, a hint request, giving up, or a free-form question."""
    if not session_store.session_exists(request.session_id):
        raise HTTPException(status_code=404, detail={
            "error": "session_not_found",
            "message": "This session has ended or expired. Please start a new activity.",
            "status_code": 404,
        })

    session_store.touch_session(request.session_id)
    session = session_store.get_session(request.session_id)
    activity = session["activity"]

    if request.message and is_blatantly_inappropriate(request.message):
        def safe_stream():
            for word in SAFE_REDIRECT_MESSAGE.split(" "):
                yield f"event: chunk\ndata: {json.dumps({'text': word + ' '})}\n\n"
            yield f"event: done\ndata: {json.dumps({'session_id': request.session_id, 'total_time_ms': 0})}\n\n"
        return StreamingResponse(safe_stream(), media_type="text/event-stream")

    if activity == "brain_buster":
        action = request.action or "guess"
        generator = activity_engine.handle_brain_buster_turn(request.session_id, action, request.message)
    elif activity == "quick_fire":
        generator = activity_engine.handle_quick_fire_turn(request.session_id, request.message)
    else:
        generator = activity_engine.handle_ask_explore_turn(request.session_id, request.message)

    return _run_activity_stream(request.session_id, activity, generator, request.message)


@app.delete("/api/session/{session_id}")
async def end_session(session_id: str):
    """Requirement #2: explicit termination (Back button) -- no session data persists after this."""
    session_store.terminate_session(session_id)
    return {"status": "terminated", "session_id": session_id}


@app.get("/api/monitoring/recent")
async def recent_monitoring(limit: int = 20):
    """Returns the most recent monitoring log entries."""
    try:
        with open(LOG_FILE) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return {"entries": []}
    recent = [json.loads(line) for line in lines[-limit:]]
    return {"entries": recent}


@app.get("/")
async def root():
    return {"status": "ok", "message": "Learning Adventures API is running. See /docs for the interactive API documentation."}
