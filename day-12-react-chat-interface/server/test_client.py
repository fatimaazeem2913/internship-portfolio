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


def test_404_unknown_session():
    r = client.post("/api/chat", json={
        "session_id": "this-session-does-not-exist-12345", "message": "Hello?",
    })
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
        ("404: unknown session_id", test_404_unknown_session, ()),
        ("GET /api/sessions listing", test_list_sessions, ()),
        ("Swagger UI + OpenAPI schema available", test_openapi_docs_available, ()),
        ("Latency is genuinely measured", test_latency_is_measured, ()),
        ("500: simulated LLM call failure", test_500_llm_failure, (None,)),
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
