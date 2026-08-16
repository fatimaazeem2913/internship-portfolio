"""
test_client.py
------------------
Comprehensive test suite using FastAPI's TestClient (Day 11's
established, standard testing approach), verifying every requirement
against the real application logic.

Run: USE_MOCK_LLM=true python3 test_client.py
"""

from fastapi.testclient import TestClient
import main
import session_store
import json

client = TestClient(main.app)


def parse_sse(response_text):
    events = []
    for block in response_text.strip().split("\n\n"):
        if not block.strip():
            continue
        lines = block.split("\n")
        event_type, data = None, None
        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event_type:
            events.append((event_type, data))
    return events


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    return "PASS", r.status_code, r.json()


def test_start_brain_buster():
    r = client.post("/api/session/start", json={"activity": "brain_buster"})
    assert r.status_code == 200
    session_id = r.headers.get("x-session-id")
    assert session_id and session_store.session_exists(session_id)
    events = parse_sse(r.text)
    chunk_events = [e for e in events if e[0] == "chunk"]
    done_events = [e for e in events if e[0] == "done"]
    assert len(chunk_events) > 0, "Should stream at least one text chunk"
    assert len(done_events) == 1, "Should end with exactly one 'done' event"
    return "PASS", session_id, {"chunks": len(chunk_events), "riddle_text": "".join(c[1]["text"] for c in chunk_events)}


def test_start_quick_fire():
    r = client.post("/api/session/start", json={"activity": "quick_fire"})
    session_id = r.headers.get("x-session-id")
    game_state = session_store.get_game_state(session_id)
    assert game_state["current_answer"] is not None
    return "PASS", session_id, {"answer_stored": game_state["current_answer"]}


def test_start_ask_explore():
    r = client.post("/api/session/start", json={"activity": "ask_explore"})
    session_id = r.headers.get("x-session-id")
    events = parse_sse(r.text)
    greeting = "".join(e[1]["text"] for e in events if e[0] == "chunk")
    assert len(greeting) > 0
    return "PASS", session_id, {"greeting": greeting}


def test_invalid_activity():
    r = client.post("/api/session/start", json={"activity": "not_a_real_activity"})
    assert r.status_code == 422, f"Expected 422 (Pydantic Literal validation), got {r.status_code}"
    return "PASS", r.status_code, "Pydantic correctly rejected an invalid activity"


def test_brain_buster_correct_guess_advances():
    r1 = client.post("/api/session/start", json={"activity": "brain_buster"})
    session_id = r1.headers.get("x-session-id")
    correct_answer = session_store.get_game_state(session_id)["current_answer"]

    r2 = client.post("/api/chat/stream", json={"session_id": session_id, "message": correct_answer, "action": "guess"})
    events = parse_sse(r2.text)
    new_item_events = [e for e in events if e[0] == "new_item"]
    assert len(new_item_events) == 1, "A correct guess should trigger exactly one 'new_item' (a new riddle)"

    new_answer = session_store.get_game_state(session_id)["current_answer"]
    used = session_store.get_used_answers(session_id)
    assert len(used) == 2, f"Should have 2 used answers after 1 correct guess, got {len(used)}"
    return "PASS", session_id, {"old_answer": correct_answer, "new_answer": new_answer, "used_answers": used}


def test_brain_buster_incorrect_guess_allows_retry():
    r1 = client.post("/api/session/start", json={"activity": "brain_buster"})
    session_id = r1.headers.get("x-session-id")
    correct_answer = session_store.get_game_state(session_id)["current_answer"]

    r2 = client.post("/api/chat/stream", json={
        "session_id": session_id, "message": "definitely_wrong_guess_xyz", "action": "guess",
    })
    events = parse_sse(r2.text)
    new_item_events = [e for e in events if e[0] == "new_item"]
    assert len(new_item_events) == 0, "An INCORRECT guess should NOT trigger a new riddle -- same riddle continues"

    same_answer = session_store.get_game_state(session_id)["current_answer"]
    assert same_answer == correct_answer, "The riddle's answer should be unchanged after a wrong guess"
    return "PASS", session_id, {"answer_unchanged": same_answer == correct_answer}


def test_brain_buster_hints_and_reveal():
    r1 = client.post("/api/session/start", json={"activity": "brain_buster"})
    session_id = r1.headers.get("x-session-id")

    for i in range(1, 4):
        r = client.post("/api/chat/stream", json={"session_id": session_id, "message": "", "action": "hint"})
        assert r.status_code == 200

    used = session_store.get_used_answers(session_id)
    assert len(used) == 2, f"After 3 hints exhausted, should have moved to a new riddle (2 used answers), got {len(used)}"
    return "PASS", session_id, {"used_answers_after_3_hints": used}


def test_brain_buster_give_up():
    r1 = client.post("/api/session/start", json={"activity": "brain_buster"})
    session_id = r1.headers.get("x-session-id")

    r2 = client.post("/api/chat/stream", json={"session_id": session_id, "message": "", "action": "give_up"})
    events = parse_sse(r2.text)
    new_item_events = [e for e in events if e[0] == "new_item"]
    assert len(new_item_events) == 1, "Give up should immediately reveal and move to a new riddle"
    return "PASS", session_id, "PASS -- give_up correctly triggers reveal + new riddle"


def test_quick_fire_always_advances():
    r1 = client.post("/api/session/start", json={"activity": "quick_fire"})
    session_id = r1.headers.get("x-session-id")

    r2 = client.post("/api/chat/stream", json={"session_id": session_id, "message": "wrong_answer_xyz"})
    events = parse_sse(r2.text)
    new_item_events = [e for e in events if e[0] == "new_item"]
    assert len(new_item_events) == 1, "Quick Fire should ALWAYS advance to a new question, even on a wrong answer"
    return "PASS", session_id, "PASS -- incorrect Quick Fire answer still advances (per requirement #4)"


def test_ask_explore_conversation():
    r1 = client.post("/api/session/start", json={"activity": "ask_explore"})
    session_id = r1.headers.get("x-session-id")

    r2 = client.post("/api/chat/stream", json={"session_id": session_id, "message": "Why is the sky blue?"})
    assert r2.status_code == 200
    events = parse_sse(r2.text)
    answer = "".join(e[1]["text"] for e in events if e[0] == "chunk")
    assert len(answer) > 0
    return "PASS", session_id, {"answer": answer}


def test_safety_filter_blocks_before_llm_call():
    r1 = client.post("/api/session/start", json={"activity": "ask_explore"})
    session_id = r1.headers.get("x-session-id")

    r2 = client.post("/api/chat/stream", json={"session_id": session_id, "message": "shut up you stupid bot"})
    assert r2.status_code == 200
    events = parse_sse(r2.text)
    reply = "".join(e[1]["text"] for e in events if e[0] == "chunk")
    assert "kind and fun" in reply, "Safety redirect message should have been returned, not a real LLM answer"

    history = session_store.get_session(session_id)["history"]
    assert not any("shut up" in m["content"] for m in history), (
        "The abusive message should not have reached the conversation history"
    )
    return "PASS", session_id, {"redirect_message": reply}


def test_six_message_context_cap():
    r1 = client.post("/api/session/start", json={"activity": "ask_explore"})
    session_id = r1.headers.get("x-session-id")

    for i in range(5):
        client.post("/api/chat/stream", json={"session_id": session_id, "message": f"question number {i}"})

    full_history = session_store.get_session(session_id)["history"]
    context = session_store.get_context_messages(session_id)
    assert len(full_history) > 6, "Full history should have grown beyond 6 messages"
    assert len(context) == 6, f"Context sent to the LLM must be capped at exactly 6, got {len(context)}"
    return "PASS", session_id, {"full_history_length": len(full_history), "context_length": len(context)}


def test_404_unknown_session():
    r = client.post("/api/chat/stream", json={"session_id": "genuinely-unknown-id", "message": "hello"})
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    return "PASS", r.status_code, r.json()


def test_explicit_termination():
    r1 = client.post("/api/session/start", json={"activity": "ask_explore"})
    session_id = r1.headers.get("x-session-id")
    assert session_store.session_exists(session_id)

    r2 = client.delete(f"/api/session/{session_id}")
    assert r2.status_code == 200
    assert not session_store.session_exists(session_id), "Session must be fully removed after termination"
    return "PASS", r2.status_code, "Session correctly no longer exists after DELETE"


def test_60_second_expiry_sweep():
    from datetime import datetime, timezone, timedelta

    sid = session_store.create_session("ask_explore")
    session_store.sessions[sid]["last_active_at"] = datetime.now(timezone.utc) - timedelta(seconds=61)

    expired = session_store.sweep_expired_sessions()
    assert sid in expired
    assert not session_store.session_exists(sid), "No session data should persist after 60s inactivity"
    return "PASS", sid, "Session correctly swept after simulated 60s+ inactivity"


def test_monitoring_log_has_required_fields():
    r1 = client.post("/api/session/start", json={"activity": "ask_explore"})
    session_id = r1.headers.get("x-session-id")
    client.post("/api/chat/stream", json={"session_id": session_id, "message": "test monitoring"})

    r2 = client.get("/api/monitoring/recent?limit=1")
    entries = r2.json()["entries"]
    assert len(entries) >= 1
    latest = entries[-1]

    required_fields = ["timestamp", "session_id", "activity", "user_prompt",
                        "input_tokens", "output_tokens", "total_tokens", "ttft_ms", "total_response_time_ms"]
    missing = [f for f in required_fields if f not in latest]
    assert not missing, f"Missing required monitoring fields: {missing}"
    return "PASS", session_id, latest


def test_openapi_docs_available():
    r_docs = client.get("/docs")
    r_schema = client.get("/openapi.json")
    assert r_docs.status_code == 200
    assert r_schema.status_code == 200
    return "PASS", 200, "Swagger UI + OpenAPI schema both available"


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out("LEARNING ADVENTURES CAPSTONE -- FULL BACKEND TEST SUITE")
    out("=" * 90)

    session_store.clear_all_sessions()

    tests = [
        ("Root health check", test_root),
        ("Start Brain Buster session", test_start_brain_buster),
        ("Start Quick Fire session", test_start_quick_fire),
        ("Start Ask & Explore session", test_start_ask_explore),
        ("Invalid activity rejected (422)", test_invalid_activity),
        ("Brain Buster: CORRECT guess advances to new riddle", test_brain_buster_correct_guess_advances),
        ("Brain Buster: INCORRECT guess allows retry (no new riddle)", test_brain_buster_incorrect_guess_allows_retry),
        ("Brain Buster: 3 hints exhausted -> auto-reveal + new riddle", test_brain_buster_hints_and_reveal),
        ("Brain Buster: give_up -> immediate reveal + new riddle", test_brain_buster_give_up),
        ("Quick Fire: BOTH correct/incorrect always advance", test_quick_fire_always_advances),
        ("Ask & Explore: real conversational answer", test_ask_explore_conversation),
        ("Safety filter blocks abusive input BEFORE any LLM call", test_safety_filter_blocks_before_llm_call),
        ("6-message context cap enforced (full history still kept)", test_six_message_context_cap),
        ("404 on genuinely unknown session_id", test_404_unknown_session),
        ("Explicit session termination (Back button)", test_explicit_termination),
        ("60-second inactivity expiry sweep", test_60_second_expiry_sweep),
        ("Monitoring log contains ALL required fields", test_monitoring_log_has_required_fields),
        ("Swagger UI + OpenAPI schema available", test_openapi_docs_available),
    ]

    passed = 0
    for label, fn in tests:
        try:
            status, detail1, detail2 = fn()
            out(f"\n[{status}] {label}")
            out(f"  {detail1}")
            out(f"  {detail2}")
            passed += 1
        except AssertionError as e:
            out(f"\n[FAIL] {label}: {e}")
        except Exception as e:
            out(f"\n[ERROR] {label}: {type(e).__name__}: {e}")

    out(f"\n\n{'='*90}")
    out(f"SUMMARY: {passed}/{len(tests)} tests passed")
    out("=" * 90)

    with open("outputs/test_client_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nSaved to outputs/test_client_results.txt")
