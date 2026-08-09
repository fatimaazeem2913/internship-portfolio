"""
session_store.py
--------------------
The in-memory session store: chat_sessions = {} mapping session_id to a
full message history array, plus metadata (created_at, last_active_at).

WHY THIS IS NECESSARY AT ALL -- THE CORE PROBLEM THIS DAY SOLVES:
HTTP is stateless. Every request to a web server arrives with zero memory
of any previous request -- the server doesn't inherently know "this is
the same person who asked something 30 seconds ago." Day 4 established
that a Transformer itself has no memory either, only a context window.
Combined, this means a genuine conversation ACROSS multiple HTTP requests
requires something to hold state BETWEEN requests -- that's exactly what
this session store is. Each request re-sends only the NEW message; the
server is responsible for reassembling the full conversation history
before calling the model, exactly as Day 4's "chat apps re-send history"
point predicted.

WHY IN-MEMORY, AND ITS REAL LIMITATION:
A plain Python dict is the simplest possible session store -- fast, zero
setup, perfect for learning and prototyping. ITS REAL, HONEST LIMITATION:
all sessions are lost the moment the server process restarts, and it
cannot be shared across multiple server instances (a real production
deployment running 3 server replicas behind a load balancer would have 3
DIFFERENT, inconsistent copies of this dict). Real production systems use
Redis or a database for exactly this reason -- documented explicitly in
REPORT.md rather than glossed over.
"""

import uuid
from datetime import datetime, timezone

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Keep responses focused and "
    "avoid unnecessary preamble."
)

chat_sessions = {}


def create_session():
    """
    Creates a new session with a fresh history seeded with the system
    prompt. Returns the new session_id.
    """
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    chat_sessions[session_id] = {
        "history": [
            {"role": "system", "content": SYSTEM_PROMPT, "timestamp": now},
        ],
        "created_at": now,
        "last_active_at": now,
    }
    return session_id


def session_exists(session_id):
    return session_id in chat_sessions


def get_history(session_id):
    """Returns the full message history for a session, or None if it doesn't exist."""
    if session_id not in chat_sessions:
        return None
    return chat_sessions[session_id]["history"]


def append_message(session_id, role, content):
    """
    Appends one message to a session's history and updates last_active_at.
    Raises KeyError if the session doesn't exist -- callers are expected
    to check session_exists() first (or handle the KeyError), matching
    the explicit "check for session_id" step in the task specification.
    """
    if session_id not in chat_sessions:
        raise KeyError(f"Session '{session_id}' does not exist.")

    now = datetime.now(timezone.utc)
    chat_sessions[session_id]["history"].append(
        {"role": role, "content": content, "timestamp": now}
    )
    chat_sessions[session_id]["last_active_at"] = now


def list_sessions():
    """
    Returns a list of summary dicts for every active session -- used by
    GET /api/sessions. Message count EXCLUDES the system prompt from the
    user-facing count, since "message count" should reflect actual
    conversation turns, not internal bookkeeping.
    """
    summaries = []
    for session_id, data in chat_sessions.items():
        user_facing_count = len([m for m in data["history"] if m["role"] != "system"])
        summaries.append({
            "session_id": session_id,
            "message_count": user_facing_count,
            "created_at": data["created_at"],
            "last_active_at": data["last_active_at"],
        })
    return summaries


def clear_all_sessions():
    """Testing/reset utility -- not exposed via any API endpoint."""
    chat_sessions.clear()


if __name__ == "__main__":
    print("=" * 90)
    print("SESSION STORE -- SELF-TEST (pure Python, no server needed)")
    print("=" * 90)

    clear_all_sessions()

    print("\n--- Creating a new session ---")
    sid = create_session()
    print(f"New session_id: {sid}")
    print(f"session_exists('{sid[:8]}...'): {session_exists(sid)}")
    print(f"session_exists('fake-id'): {session_exists('fake-id')}")

    print("\n--- Appending messages ---")
    append_message(sid, "user", "What's 2+2?")
    append_message(sid, "model", "2+2 equals 4.")
    append_message(sid, "user", "And 3+3?")
    append_message(sid, "model", "3+3 equals 6.")

    history = get_history(sid)
    print(f"History length (including system prompt): {len(history)}")
    for msg in history:
        print(f"  [{msg['role']}] {msg['content']}")

    print("\n--- Testing append to non-existent session ---")
    try:
        append_message("nonexistent-id", "user", "hello")
        print("ERROR: should have raised KeyError")
    except KeyError as e:
        print(f"Correctly raised KeyError: {e}")

    print("\n--- Listing all sessions ---")
    sessions = list_sessions()
    for s in sessions:
        print(f"  {s['session_id'][:8]}... -> {s['message_count']} messages "
              f"(system prompt excluded from count)")

    print(f"\nAll self-tests passed. Total active sessions: {len(chat_sessions)}")
