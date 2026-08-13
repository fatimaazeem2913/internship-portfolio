"""
test_client.py
------------------
A comprehensive test suite exercising every endpoint and every required
error case (400, 404, 422, 500) using FastAPI's own TestClient -- the
standard, officially-recommended way to test a FastAPI application.

TestClient calls the ASGI application directly, in-process, using the
exact same routing, validation, and response logic a real deployed
server would use -- it is NOT a mock of the API; it exercises the real
main.py application code path for path routing, Pydantic validation,
session management, and error handling. The only thing it doesn't
exercise is the actual network transport layer (sockets/HTTP framing),
which is FastAPI/Starlette's own well-tested responsibility, not this
project's.

For a genuine over-the-network test (e.g. via curl or the Swagger UI),
run the server for real:
    export USE_MOCK_LLM=true
    uvicorn main:app --reload --port 8000
and visit http://127.0.0.1:8000/docs, or use test_via_curl.sh in this
same folder.

Run this test suite directly:
    export USE_MOCK_LLM=true
    python3 test_client.py
"""

from fastapi.testclient import TestClient
import main
import session_store

client = TestClient(main.app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    return "PASS", r.status_code, r.json()


def test_new_session_chat():
    r = client.post("/api/chat", json={"message": "What's 2+2?"})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert "session_id" in body
    assert body["message_count"] == 2
    return "PASS", r.status_code, body


def test_continue_session(session_id):
    r = client.post("/api/chat", json={
        "session_id": session_id, "message": "And what about 3+3?",
    })
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert body["session_id"] == session_id, "Session ID should be preserved across turns"
    assert body["message_count"] == 4, f"Expected 4 messages after 2 turns, got {body['message_count']}"
    return "PASS", r.status_code, body


def test_400_empty_message():
    r = client.post("/api/chat", json={"message": "   "})
    assert r.status_code == 400, f"Expected 400, got {r.status_code}: {r.text}"
    return "PASS", r.status_code, r.json()


def test_422_missing_required_field():
    r = client.post("/api/chat", json={"session_id": "whatever"})
    assert r.status_code == 422, f"Expected 422 (Pydantic validation), got {r.status_code}: {r.text}"
    return "PASS", r.status_code, r.json()


def test_client_provided_session_id_lazy_creation():
    """
    Day 13's core behavior change: a session_id the server has never seen
    should be LAZILY CREATED, not rejected with 404 -- this is the exact
    path a real "New Chat" button takes (crypto.randomUUID() generated
    client-side, before any message has been sent).
    """
    import uuid
    client_generated_id = str(uuid.uuid4())
    r = client.post("/api/chat", json={
        "session_id": client_generated_id, "message": "First message in a brand new session",
    })
    assert r.status_code == 200, f"Expected 200 (lazy creation), got {r.status_code}: {r.text}"
    body = r.json()
    assert body["session_id"] == client_generated_id, (
        "Server should use the EXACT client-provided ID, not generate a different one"
    )
    return "PASS", r.status_code, body


def test_404_on_title_for_unknown_session():
    """Unlike /api/chat, the title endpoint still correctly 404s for a truly unknown session,
    since generating a title requires a real existing exchange."""
    r = client.post("/api/sessions/genuinely-unknown-session-id/title")
    assert r.status_code == 404, f"Expected 404, got {r.status_code}: {r.text}"
    return "PASS", r.status_code, r.json()


def test_list_sessions():
    r = client.get("/api/sessions")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    body = r.json()
    assert "active_sessions" in body
    assert "sessions" in body
    assert body["active_sessions"] >= 1
    return "PASS", r.status_code, body


def test_openapi_docs_available():
    r_docs = client.get("/docs")
    r_schema = client.get("/openapi.json")
    assert r_docs.status_code == 200
    assert r_schema.status_code == 200
    schema = r_schema.json()
    assert "/api/chat" in schema["paths"]
    assert "/api/sessions" in schema["paths"]
    return "PASS", r_docs.status_code, {"paths_documented": list(schema["paths"].keys())}


def test_latency_is_measured():
    r = client.post("/api/chat", json={"message": "Quick test"})
    body = r.json()
    assert "latency_ms" in body
    assert body["latency_ms"] > 0, "Latency should be a real, positive measurement"
    return "PASS", r.status_code, {"latency_ms": body["latency_ms"]}


def test_title_generation():
    """Full flow: create session, exchange one message, generate title, verify it's stored."""
    r1 = client.post("/api/chat", json={"message": "What's the best way to learn Python?"})
    session_id = r1.json()["session_id"]

    r2 = client.post(f"/api/sessions/{session_id}/title")
    assert r2.status_code == 200, f"Expected 200, got {r2.status_code}: {r2.text}"
    title_body = r2.json()
    assert title_body["session_id"] == session_id
    assert len(title_body["title"]) > 0, "Title should not be empty"

    # Verify the title is actually STORED and shows up in the sessions list
    r3 = client.get("/api/sessions")
    matching = [s for s in r3.json()["sessions"] if s["session_id"] == session_id]
    assert len(matching) == 1
    assert matching[0]["title"] == title_body["title"], (
        "The title returned by the generate endpoint should match what's stored and listed"
    )
    return "PASS", r2.status_code, title_body


def test_regenerate_response():
    """
    Sends one message, regenerates it, and verifies: (1) message_count did
    NOT grow (the old exchange was popped, not just appended-after), and
    (2) the session still has exactly 1 user message in its history, not 2.
    """
    r1 = client.post("/api/chat", json={"message": "Tell me a fact about the ocean"})
    session_id = r1.json()["session_id"]
    count_after_first = r1.json()["message_count"]

    r2 = client.post(f"/api/sessions/{session_id}/regenerate")
    assert r2.status_code == 200, f"Expected 200, got {r2.status_code}: {r2.text}"
    count_after_regenerate = r2.json()["message_count"]

    assert count_after_regenerate == count_after_first, (
        f"Regenerate should NOT grow message_count (popped old exchange first): "
        f"before={count_after_first}, after={count_after_regenerate}"
    )

    history = session_store.get_history(session_id)
    user_messages = [m for m in history if m["role"] == "user"]
    assert len(user_messages) == 1, (
        f"Expected exactly 1 user message after regenerate (not duplicated), got {len(user_messages)}"
    )
    return "PASS", r2.status_code, r2.json()


def test_regenerate_404_unknown_session():
    r = client.post("/api/sessions/genuinely-unknown-session-id/regenerate")
    assert r.status_code == 404, f"Expected 404, got {r.status_code}: {r.text}"
    return "PASS", r.status_code, r.json()


def test_five_simultaneous_sessions_no_cross_contamination():
    """
    *** THE CRITICAL DAY 13 REQUIREMENT ***
    Creates 5 independent sessions, sends a DIFFERENT, UNIQUE message to
    each, then verifies EVERY session's history contains ONLY its own
    messages -- none of the other 4 sessions' content leaked in. This is
    a real, direct test of server-side isolation, not an assumption.
    """
    import uuid

    session_messages = {}
    session_ids = []

    for i in range(5):
        sid = str(uuid.uuid4())
        unique_message = f"UNIQUE_MARKER_SESSION_{i}_TOPIC_{['cats','rockets','pasta','violins','glaciers'][i]}"
        r = client.post("/api/chat", json={"session_id": sid, "message": unique_message})
        assert r.status_code == 200, f"Session {i} setup failed: {r.status_code}: {r.text}"
        session_ids.append(sid)
        session_messages[sid] = unique_message

    # Now verify EVERY session's history contains ONLY its own marker,
    # and explicitly confirm none of the OTHER 4 sessions' markers appear.
    contamination_found = []
    for sid in session_ids:
        history = session_store.get_history(sid)
        history_text = " ".join(m["content"] for m in history)

        own_marker = session_messages[sid]
        assert own_marker in history_text, f"Session {sid[:8]} is missing its OWN message!"

        for other_sid, other_marker in session_messages.items():
            if other_sid != sid and other_marker in history_text:
                contamination_found.append((sid, other_sid))

    assert not contamination_found, f"CROSS-CONTAMINATION DETECTED: {contamination_found}"

    return "PASS", 200, {
        "sessions_tested": len(session_ids),
        "cross_contamination_found": len(contamination_found),
        "sample_session_ids": [s[:8] + "..." for s in session_ids],
    }


def test_500_llm_failure(monkeypatch_target):
    """
    Simulates a genuine LLM-call failure (network error, provider outage,
    quota exceeded) to verify the 500 path returns a clean, structured
    error rather than crashing the server or leaking a raw traceback.
    """
    import llm_client
    original = llm_client.generate_reply

    def broken_generate_reply(history):
        raise RuntimeError("Simulated provider outage for testing.")

    llm_client.generate_reply = broken_generate_reply
    main.generate_reply = broken_generate_reply
    try:
        r = client.post("/api/chat", json={"message": "This should fail"})
        assert r.status_code == 500, f"Expected 500, got {r.status_code}: {r.text}"
        result = ("PASS", r.status_code, r.json())
    finally:
        llm_client.generate_reply = original
        main.generate_reply = original
    return result


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out("FASTAPI APPLICATION -- FULL TEST SUITE (via FastAPI's own TestClient)")
    out("=" * 90)

    session_store.clear_all_sessions()

    tests = [
        ("Root health check", test_root, ()),
        ("New session chat (no session_id)", test_new_session_chat, ()),
    ]

    results = {}
    passed = 0
    total = 0

    for label, fn, args in tests:
        total += 1
        try:
            status, code, body = fn(*args)
            out(f"\n[{status}] {label}")
            out(f"  HTTP {code}: {body}")
            results[label] = body
            passed += 1
        except AssertionError as e:
            out(f"\n[FAIL] {label}: {e}")

    session_id = results.get("New session chat (no session_id)", {}).get("session_id")
    remaining_tests = [
        ("Continue existing session", test_continue_session, (session_id,)),
        ("400: empty/whitespace message", test_400_empty_message, ()),
        ("422: missing required field", test_422_missing_required_field, ()),
        ("Day 13: client-provided session_id lazily creates session", test_client_provided_session_id_lazy_creation, ()),
        ("GET /api/sessions listing", test_list_sessions, ()),
        ("Swagger UI + OpenAPI schema available", test_openapi_docs_available, ()),
        ("Latency is genuinely measured", test_latency_is_measured, ()),
        ("500: simulated LLM call failure", test_500_llm_failure, (None,)),
        ("Day 13: title generation (full flow)", test_title_generation, ()),
        ("Day 13: 404 on title for unknown session", test_404_on_title_for_unknown_session, ()),
        ("Day 13: regenerate response", test_regenerate_response, ()),
        ("Day 13: 404 on regenerate for unknown session", test_regenerate_404_unknown_session, ()),
        ("Day 13: 5 SIMULTANEOUS SESSIONS -- NO CROSS-CONTAMINATION", test_five_simultaneous_sessions_no_cross_contamination, ()),
    ]

    for label, fn, args in remaining_tests:
        total += 1
        try:
            status, code, body = fn(*args)
            out(f"\n[{status}] {label}")
            out(f"  HTTP {code}: {body}")
            passed += 1
        except AssertionError as e:
            out(f"\n[FAIL] {label}: {e}")
        except Exception as e:
            out(f"\n[ERROR] {label}: {type(e).__name__}: {e}")

    out(f"\n\n{'='*90}")
    out(f"SUMMARY: {passed}/{total} tests passed")
    out("=" * 90)

    with open("outputs/test_client_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nSaved to outputs/test_client_results.txt")
