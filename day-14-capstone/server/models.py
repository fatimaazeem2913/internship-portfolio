"""
models.py
------------
Pydantic request/response schemas for the capstone app's endpoints.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    activity: Literal["brain_buster", "quick_fire", "ask_explore"] = Field(
        description="Which activity to start a new session for."
    )


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(default="", description="The child's typed message/guess/answer.")
    action: Optional[Literal["guess", "answer", "hint", "give_up", "ask"]] = Field(
        default=None,
        description="Explicit action type -- avoids parsing user intent from free text "
                    "(Day 10's principle: structured actions beat implicit intent parsing). "
                    "'guess'/'answer' use `message` as the child's attempt; 'hint' and "
                    "'give_up' need no message; 'ask' is Ask & Explore's free-form question.",
    )


class ErrorResponse(BaseModel):
    error: str
    message: str
    status_code: int
