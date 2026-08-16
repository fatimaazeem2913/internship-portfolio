"""
session_store.py
--------------------
In-memory session management (requirement #9: "in-memory session
management without a database") with 60-second inactivity expiry
(requirement #2).

WHY `used_answers` IS TRACKED SEPARATELY FROM `history`:
Requirement #7 caps conversation CONTEXT sent to the model at the 6 most
recent messages. A long hint back-and-forth on one riddle could push an
EARLIER riddle's answer out of that 6-message window well before the
session ends -- if "no repeats" relied purely on the model remembering
prior riddles from context, repeats would become likely exactly when a
session runs long. Tracking `used_answers` as separate, compact metadata
(passed to every generation call regardless of the 6-message window)
guarantees no-repeat behavior for the FULL session, not just the last
few turns.

60-SECOND EXPIRY: `last_active_at` is updated on every real interaction.
A background sweep (started in main.py) periodically removes any session
whose `last_active_at` is more than 60 seconds old -- enforced HERE,
server-side, not just via a client-side timer (which a client could fail
to run, e.g. a closed browser tab) -- the same "don't rely on the client
alone" principle as Day 9's client + server-side validation.
"""

import uuid
from datetime import datetime, timezone, timedelta

SESSION_TIMEOUT_SECONDS = 60

sessions = {}


def create_session(activity):
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    sessions[session_id] = {
        "session_id": session_id,
        "activity": activity,
        "history": [],
        "used_answers": [],
        "game_state": {
            "current_answer": None,
            "current_hints": [],
            "hints_given": 0,
            "current_fact": None,
        },
        "created_at": now,
        "last_active_at": now,
    }
    return session_id


def session_exists(session_id):
    return session_id in sessions


def get_session(session_id):
    return sessions.get(session_id)


def touch_session(session_id):
    """Updates last_active_at -- called on every real interaction, resetting the 60s inactivity clock."""
    if session_id in sessions:
        sessions[session_id]["last_active_at"] = datetime.now(timezone.utc)


def append_message(session_id, role, content):
    if session_id not in sessions:
        raise KeyError(f"Session '{session_id}' does not exist.")
    sessions[session_id]["history"].append({
        "role": role, "content": content, "timestamp": datetime.now(timezone.utc),
    })


def get_context_messages(session_id, limit=6):
    """
    Requirement #7: "maintain only the six most recent messages as
    conversation context." Returns just the last `limit` messages -- the
    FULL history is still kept for potential frontend display, but only
    this trimmed slice is ever sent to the LLM as context.
    """
    if session_id not in sessions:
        return []
    return sessions[session_id]["history"][-limit:]


def set_game_state(session_id, **kwargs):
    if session_id not in sessions:
        raise KeyError(f"Session '{session_id}' does not exist.")
    sessions[session_id]["game_state"].update(kwargs)


def get_game_state(session_id):
    if session_id not in sessions:
        return None
    return sessions[session_id]["game_state"]


def add_used_answer(session_id, answer):
    if session_id not in sessions:
        raise KeyError(f"Session '{session_id}' does not exist.")
    sessions[session_id]["used_answers"].append(answer)


def get_used_answers(session_id):
    if session_id not in sessions:
        return []
    return sessions[session_id]["used_answers"]


def terminate_session(session_id):
    """Explicit termination (e.g. the Back button). Fully removes the session; no data persists."""
    sessions.pop(session_id, None)


def sweep_expired_sessions():
    """
    Removes every session whose last_active_at is older than
    SESSION_TIMEOUT_SECONDS. Called periodically by a background task in
    main.py. Returns the list of expired session_ids removed, for logging.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=SESSION_TIMEOUT_SECONDS)
    expired = [sid for sid, data in sessions.items() if data["last_active_at"] < cutoff]
    for sid in expired:
        sessions.pop(sid, None)
    return expired


def clear_all_sessions():
    """Testing utility."""
    sessions.clear()


if __name__ == "__main__":
    print("=" * 90)
    print("SESSION STORE -- SELF-TEST (pure Python, no API needed)")
    print("=" * 90)

    clear_all_sessions()

    print("\n--- Creating a Brain Buster session ---")
    sid = create_session("brain_buster")
    print(f"session_id: {sid}, exists: {session_exists(sid)}")

    print("\n--- Appending messages and checking the 6-message context cap ---")
    for i in range(10):
        append_message(sid, "user" if i % 2 == 0 else "model", f"message {i}")
    full_history_len = len(get_session(sid)["history"])
    context_len = len(get_context_messages(sid))
    print(f"Full history length: {full_history_len} (all messages kept)")
    print(f"Context sent to LLM: {context_len} (correctly capped at 6)")
    assert full_history_len == 10 and context_len == 6

    print("\n--- Game state: tracking a riddle's answer and hints ---")
    set_game_state(sid, current_answer="sun", current_hints=["hint1", "hint2", "hint3"], hints_given=0)
    state = get_game_state(sid)
    print(f"Game state: {state}")
    assert state["current_answer"] == "sun"

    print("\n--- used_answers persists independent of the 6-message context window ---")
    add_used_answer(sid, "sun")
    add_used_answer(sid, "moon")
    print(f"Used answers: {get_used_answers(sid)}")
    assert get_used_answers(sid) == ["sun", "moon"]

    print("\n--- 60-second expiry sweep (simulated with a manually-aged session) ---")
    old_sid = create_session("quick_fire")
    sessions[old_sid]["last_active_at"] = datetime.now(timezone.utc) - timedelta(seconds=61)
    expired = sweep_expired_sessions()
    print(f"Expired sessions removed: {expired}")
    assert old_sid in expired
    assert not session_exists(old_sid)
    assert session_exists(sid), "The fresh session should NOT have been swept"
    print(f"Fresh session '{sid[:8]}...' correctly still exists: {session_exists(sid)}")

    print("\n--- Explicit termination (e.g. Back button) ---")
    terminate_session(sid)
    print(f"session_exists after termination: {session_exists(sid)}")
    assert not session_exists(sid)

    print(f"\nAll self-tests passed. Total active sessions: {len(sessions)}")
