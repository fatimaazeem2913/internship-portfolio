"""
models.py
------------
Pydantic models defining the request and response schemas for every
endpoint. Keeping these in a separate module (rather than defining them
inline in main.py) is standard FastAPI practice -- it lets the schemas be
imported and reused by tests, and keeps main.py focused on routing logic.

FastAPI uses these models for THREE things simultaneously:
    1. Request validation -- a request that doesn't match the schema is
       automatically rejected with a 422 error, before your handler code
       ever runs.
    2. Response serialization -- your handler returns a Python object;
       FastAPI converts it to JSON matching the response_model's shape.
    3. Automatic documentation -- the Swagger UI at /docs reads these
       models directly to generate interactive, accurate API docs with
       zero extra work.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for POST /api/chat."""
    session_id: Optional[str] = Field(
        default=None,
        description="Existing session ID to continue a conversation. "
                    "Omit to start a new session (one will be generated).",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="The user's message to send to the model.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": None,
                "message": "What's the capital of France?",
            }
        }


class ChatResponse(BaseModel):
    """Response body for a successful POST /api/chat."""
    session_id: str = Field(description="The session ID -- save this to continue the conversation.")
    response: str = Field(description="The model's generated reply.")
    message_count: int = Field(description="Total messages in this session's history after this turn.")
    latency_ms: float = Field(description="How long this request took to process, in milliseconds.")


class Message(BaseModel):
    """A single message in a session's history -- used internally and in session detail views."""
    role: str = Field(description="'system', 'user', or 'model'.")
    content: str
    timestamp: datetime


class SessionSummary(BaseModel):
    """One entry in the GET /api/sessions listing."""
    session_id: str
    message_count: int
    created_at: datetime
    last_active_at: datetime
    title: Optional[str] = Field(
        default=None,
        description="Day 13: AI-generated short title, or null if not yet generated "
                    "(happens after the first successful exchange in a session).",
    )


class TitleResponse(BaseModel):
    """Response body for POST /api/sessions/{session_id}/title (Day 13)."""
    session_id: str
    title: str = Field(description="The generated 3-5 word title.")


class RegenerateRequest(BaseModel):
    """
    Reserved for future use -- POST /api/sessions/{session_id}/regenerate
    currently takes no request body (the session_id in the URL path is
    sufficient, since it operates on that session's own last message).
    Kept as an explicit empty model rather than omitted entirely, so the
    endpoint's shape is self-documenting in the Swagger UI.
    """
    pass


class SessionListResponse(BaseModel):
    """Response body for GET /api/sessions."""
    active_sessions: int
    sessions: list[SessionSummary]


class ErrorResponse(BaseModel):
    """
    Structured error response used consistently across every endpoint's
    error paths (400, 404, 500) -- rather than letting each error path
    return an ad-hoc shape, every error in this API has the same
    predictable structure, which is exactly what a real API consumer
    needs to write reliable error-handling code against.
    """
    error: str = Field(description="A short, machine-readable error code, e.g. 'session_not_found'.")
    message: str = Field(description="A human-readable explanation of what went wrong.")
    status_code: int
