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


def create_session(session_id=None):
    """
    Creates a new session with a fresh history seeded with the system
    prompt. Returns the session_id used.

    Day 13 CHANGE: accepts an OPTIONAL client-provided session_id --
    the frontend now generates its own ID via crypto.randomUUID() (a
    real, standard browser API for cryptographically random unique IDs,
    not a library) so the ID exists in the UI's state immediately, before
    the very first API call is even made. If no session_id is provided,
    the server generates one itself (Day 11/12's original behavior),
    preserving backward compatibility.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    now = datetime.now(timezone.utc)
    chat_sessions[session_id] = {
        "history": [
            {"role": "system", "content": SYSTEM_PROMPT, "timestamp": now},
        ],
        "created_at": now,
        "last_active_at": now,
        "title": None,  # Day 13: set later via generate_title()
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
            "title": data.get("title"),
        })
    return summaries


def pop_last_turn(session_id):
    """
    Day 13 addition, for the "regenerate response" feature: removes the
    LAST user+model exchange from a session's history (in that order --
    the model reply first if present, then the user message beneath it),
    returning the popped user message's content so the caller can resend
    it to get a genuinely NEW response.

    WHY THIS IS NECESSARY (not just re-POSTing the same message again):
    if regenerate simply sent another chat request with the same user
    text, the session history would end up with the SAME user message
    appearing twice in a row, with two different model replies stacked
    after it -- a real, honestly worse conversation history than what the
    user intended ("give me a different answer to the same question,"
    not "add another back-and-forth"). Popping the old exchange first
    keeps the history clean: exactly one user message, followed by
    exactly one (new) model reply, per turn.

    Raises KeyError if the session doesn't exist, ValueError if there is
    no user message to regenerate (e.g. the session only has the seeded
    system prompt so far).
    """
    if session_id not in chat_sessions:
        raise KeyError(f"Session '{session_id}' does not exist.")

    history = chat_sessions[session_id]["history"]

    # Pop the trailing model reply, if the last turn completed successfully
    if history and history[-1]["role"] == "model":
        history.pop()

    if not history or history[-1]["role"] != "user":
        raise ValueError("No user message available to regenerate a response for.")

    popped_user_message = history.pop()
    chat_sessions[session_id]["last_active_at"] = datetime.now(timezone.utc)
    return popped_user_message["content"]


def set_title(session_id, title):
    """Stores the AI-generated short title (Day 13) for a session."""
    if session_id not in chat_sessions:
        raise KeyError(f"Session '{session_id}' does not exist.")
    chat_sessions[session_id]["title"] = title


def get_title(session_id):
    if session_id not in chat_sessions:
        return None
    return chat_sessions[session_id].get("title")


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
              f"(system prompt excluded from count), title={s['title']}")

    print("\n--- Day 13: client-provided session_id ---")
    client_id = "client-generated-uuid-1234"
    returned_id = create_session(session_id=client_id)
    assert returned_id == client_id, "Server should use the client-provided ID, not generate its own"
    print(f"Server correctly used the client-provided ID: {returned_id}")

    print("\n--- Day 13: set/get title ---")
    set_title(sid, "Basic Arithmetic Questions")
    assert get_title(sid) == "Basic Arithmetic Questions"
    print(f"Title correctly set and retrieved: '{get_title(sid)}'")

    print("\n--- Day 13: pop_last_turn (for regenerate) ---")
    history_before = len(get_history(sid))
    popped_message = pop_last_turn(sid)
    history_after = len(get_history(sid))
    assert popped_message == "And 3+3?", f"Expected to pop 'And 3+3?', got '{popped_message}'"
    assert history_after == history_before - 2, "Should have removed exactly 2 messages (user + model)"
    print(f"Popped user message: '{popped_message}'")
    print(f"History length: {history_before} -> {history_after} (removed 1 user + 1 model message)")

    print("\n--- Day 13: pop_last_turn with nothing to pop ---")
    clear_all_sessions()
    empty_sid = create_session()
    try:
        pop_last_turn(empty_sid)
        print("ERROR: should have raised ValueError")
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")

    print(f"\nAll self-tests passed. Total active sessions: {len(chat_sessions)}")
