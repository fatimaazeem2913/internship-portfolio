"""
test_main.py — comprehensive tests covering every functional requirement
in the Day 14 task spec. Run with:
  USE_MOCK_LLM=true python -m pytest test_main.py -v
"""

import os
os.environ.setdefault("USE_MOCK_LLM", "true")

import time
import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def _reassemble_sse(raw_text):
    import json as _json
    text = ""
    for line in raw_text.split("\n"):
        if line.startswith("data: "):
            payload = _json.loads(line[len("data: "):])
            if "chunk" in payload:
                text += payload["chunk"]
    return text


@pytest.fixture(autouse=True)
def clear_sessions():
    main.SESSIONS.clear()
    main._reset_mock_pools()
    yield


# ---------------------------------------------------------------------------
# Requirement #1: Home screen / activity routing
# ---------------------------------------------------------------------------

def test_health_lists_all_three_activities():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert set(resp.json()["activities"]) == {"brain_buster", "quick_fire", "ask_explore"}


def test_each_activity_can_be_started():
    for activity in ["brain_buster", "quick_fire", "ask_explore"]:
        resp = client.post("/api/start", json={"activity": activity})
        assert resp.status_code == 200, f"{activity} failed to start"
        assert "x-session-id" in resp.headers


def test_unknown_activity_rejected():
    resp = client.post("/api/start", json={"activity": "not_a_real_activity"})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Requirement #2: Session management
# ---------------------------------------------------------------------------

def test_each_activity_gets_an_independent_session():
    resp1 = client.post("/api/start", json={"activity": "brain_buster"})
    resp2 = client.post("/api/start", json={"activity": "quick_fire"})
    sid1 = resp1.headers["x-session-id"]
    sid2 = resp2.headers["x-session-id"]
    assert sid1 != sid2
    assert sid1 in main.SESSIONS
    assert sid2 in main.SESSIONS


def test_back_button_terminates_session_with_no_data_left():
    resp = client.post("/api/start", json={"activity": "brain_buster"})
    session_id = resp.headers["x-session-id"]
    assert session_id in main.SESSIONS

    del_resp = client.delete(f"/api/session/{session_id}")
    assert del_resp.status_code == 200
    assert session_id not in main.SESSIONS  # no data persists


def test_60_second_inactivity_sweep_removes_the_session():
    from datetime import datetime, timezone, timedelta
    resp = client.post("/api/start", json={"activity": "brain_buster"})
    session_id = resp.headers["x-session-id"]

    # Simulate 61 seconds of inactivity
    main.SESSIONS[session_id]["last_active_at"] = datetime.now(timezone.utc) - timedelta(seconds=61)
    expired = main.sweep_expired_sessions()
    assert session_id in expired
    assert session_id not in main.SESSIONS


def test_fresh_session_is_not_swept():
    resp = client.post("/api/start", json={"activity": "brain_buster"})
    session_id = resp.headers["x-session-id"]
    expired = main.sweep_expired_sessions()
    assert session_id not in expired
    assert session_id in main.SESSIONS


def test_chat_on_expired_or_unknown_session_returns_404():
    resp = client.post("/api/chat", json={"session_id": "not-real", "activity": "brain_buster", "message": "hi"})
    assert resp.status_code == 404


def test_chat_with_session_deleted_mid_flight_does_not_crash():
    resp = client.post("/api/start", json={"activity": "brain_buster"})
    session_id = resp.headers["x-session-id"]
    del main.SESSIONS[session_id]
    resp = client.post("/api/chat", json={"session_id": session_id, "activity": "brain_buster", "action": "hint"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Requirement #3: Brain Buster
# ---------------------------------------------------------------------------

def test_brain_buster_presents_one_riddle_at_a_time():
    resp = client.post("/api/start", json={"activity": "brain_buster"})
    session_id = resp.headers["x-session-id"]
    text = _reassemble_sse(resp.text)
    assert len(text.strip()) > 0
    # Exactly one riddle -- no more than one question mark's worth
    assert text.count("?") <= 1


def test_brain_buster_riddles_do_not_repeat_within_a_session():
    """Real, meaningful check now that the mock genuinely varies its
    output per call (see _MOCK_RIDDLE_POOL): reads the actual current
    answer each turn and guesses it correctly, then verifies every
    riddle and answer used across the session is genuinely distinct."""
    resp = client.post("/api/start", json={"activity": "brain_buster"})
    session_id = resp.headers["x-session-id"]

    seen_riddles = []
    seen_answers = []
    for _ in range(6):
        state = main.SESSIONS[session_id]["game_state"]
        seen_riddles.append(state["current_riddle"])
        seen_answers.append(state["current_answer"])
        client.post("/api/chat", json={"session_id": session_id, "activity": "brain_buster", "message": state["current_answer"]})

    assert len(seen_riddles) == len(set(seen_riddles)), "riddles repeated within the session"
    assert len(seen_answers) == len(set(seen_answers)), "answers repeated within the session"
    used = main.SESSIONS[session_id]["used_answers"]
    assert len(used) == len(set(main._normalize(a) for a in used)), "used_answers list itself contains a duplicate"


def test_brain_buster_up_to_3_hints_then_reveals():
    resp = client.post("/api/start", json={"activity": "brain_buster"})
    session_id = resp.headers["x-session-id"]

    hint_texts = []
    for i in range(3):
        resp = client.post("/api/chat", json={"session_id": session_id, "activity": "brain_buster", "action": "hint"})
        hint_texts.append(_reassemble_sse(resp.text))
    assert len(set(hint_texts)) == 3, "hints must be genuinely distinct, not repeated"
    assert main.SESSIONS[session_id]["game_state"]["hints_given"] == 3

    # A 4th hint request reveals the answer
    resp = client.post("/api/chat", json={"session_id": session_id, "activity": "brain_buster", "action": "hint"})
    text = _reassemble_sse(resp.text)
    assert "had all 3 hints" in text
    assert main.SESSIONS[session_id]["game_state"]["hints_given"] == 0  # reset for new riddle


def test_brain_buster_give_up_reveals_immediately():
    resp = client.post("/api/start", json={"activity": "brain_buster"})
    session_id = resp.headers["x-session-id"]
    resp = client.post("/api/chat", json={"session_id": session_id, "activity": "brain_buster", "action": "give_up"})
    text = _reassemble_sse(resp.text)
    assert "answer was" in text.lower()


def test_brain_buster_correct_answer_gets_positive_feedback_then_new_riddle():
    resp = client.post("/api/start", json={"activity": "brain_buster"})
    session_id = resp.headers["x-session-id"]
    resp = client.post("/api/chat", json={"session_id": session_id, "activity": "brain_buster", "message": "mockanswer"})
    text = _reassemble_sse(resp.text)
    assert text.startswith("✅ Correct!")
    assert main.SESSIONS[session_id]["game_state"]["current_riddle"]  # a new riddle is set


def test_brain_buster_new_riddle_is_not_truncated_to_just_the_question_stub():
    """Real regression test: extract_final_question() was briefly and
    incorrectly applied to the riddle field, and because a riddle's own
    natural structure is '[setup sentence]. What am I?', it mistook the
    riddle's own final question for a decoy and threw away the actual
    riddle setup -- leaving only the bare stub 'What am I?' on screen
    with no context for what's being asked. This confirms the fix: the
    full riddle, setup included, must survive intact."""
    resp = client.post("/api/start", json={"activity": "brain_buster"})
    session_id = resp.headers["x-session-id"]
    resp = client.post("/api/chat", json={"session_id": session_id, "activity": "brain_buster", "message": "mockanswer"})
    new_riddle = main.SESSIONS[session_id]["game_state"]["current_riddle"]
    assert new_riddle != "What am I?"
    assert len(new_riddle) > len("What am I?")


def test_brain_buster_incorrect_answer_allows_retry():
    resp = client.post("/api/start", json={"activity": "brain_buster"})
    session_id = resp.headers["x-session-id"]
    original_riddle = main.SESSIONS[session_id]["game_state"]["current_riddle"]

    resp = client.post("/api/chat", json={"session_id": session_id, "activity": "brain_buster", "message": "totally wrong"})
    text = _reassemble_sse(resp.text)
    assert text.startswith("❌ Not quite right.")
    # Same riddle -- the child can try again
    assert main.SESSIONS[session_id]["game_state"]["current_riddle"] == original_riddle


# ---------------------------------------------------------------------------
# Requirement #4: Quick Fire
# ---------------------------------------------------------------------------

def test_quick_fire_presents_one_question_at_a_time():
    resp = client.post("/api/start", json={"activity": "quick_fire"})
    text = _reassemble_sse(resp.text)
    assert text.count("?") <= 1


def test_quick_fire_correct_answer_praise_fact_then_next_question():
    resp = client.post("/api/start", json={"activity": "quick_fire"})
    session_id = resp.headers["x-session-id"]
    resp = client.post("/api/chat", json={"session_id": session_id, "activity": "quick_fire", "message": "mockanswer"})
    text = _reassemble_sse(resp.text)
    assert text.startswith("✅ Correct!")
    assert text.count("?") <= 1  # exactly the new question, nothing extra


def test_quick_fire_incorrect_reveals_answer_and_continues():
    resp = client.post("/api/start", json={"activity": "quick_fire"})
    session_id = resp.headers["x-session-id"]
    resp = client.post("/api/chat", json={"session_id": session_id, "activity": "quick_fire", "message": "wrong"})
    text = _reassemble_sse(resp.text)
    assert text.startswith("❌ Not quite right.")
    assert text.count("?") <= 1
    # Always continues with a new question regardless of right/wrong
    assert main.SESSIONS[session_id]["game_state"]["current_answer"]


def test_quick_fire_questions_do_not_repeat_within_a_session():
    """Real check now that the mock genuinely varies its output per call
    (see _MOCK_QUESTION_POOL): verifies every question and answer used
    across the session is genuinely distinct. Quick Fire always advances
    regardless of right/wrong, so any message works here."""
    resp = client.post("/api/start", json={"activity": "quick_fire"})
    session_id = resp.headers["x-session-id"]

    seen_answers = [main.SESSIONS[session_id]["game_state"]["current_answer"]]
    for _ in range(6):
        client.post("/api/chat", json={"session_id": session_id, "activity": "quick_fire", "message": "wrong"})
        seen_answers.append(main.SESSIONS[session_id]["game_state"]["current_answer"])

    assert len(seen_answers) == len(set(seen_answers)), "answers repeated within the session"
    used = main.SESSIONS[session_id]["used_answers"]
    assert len(used) == len(set(main._normalize(a) for a in used)), "used_answers list itself contains a duplicate"


# ---------------------------------------------------------------------------
# Requirement #5: Ask & Explore
# ---------------------------------------------------------------------------

def test_ask_explore_answers_a_question():
    resp = client.post("/api/start", json={"activity": "ask_explore"})
    session_id = resp.headers["x-session-id"]
    resp = client.post("/api/chat", json={"session_id": session_id, "activity": "ask_explore", "message": "why is the sky blue?"})
    assert resp.status_code == 200
    text = _reassemble_sse(resp.text)
    assert len(text.strip()) > 0


# ---------------------------------------------------------------------------
# Requirement #6: AI Safety
# ---------------------------------------------------------------------------

def test_each_activity_has_its_own_dedicated_prompt():
    assert main.ACTIVITY_PROMPTS["brain_buster"] != main.ACTIVITY_PROMPTS["quick_fire"]
    assert main.ACTIVITY_PROMPTS["quick_fire"] != main.ACTIVITY_PROMPTS["ask_explore"]
    assert main.ACTIVITY_PROMPTS["brain_buster"] != main.ACTIVITY_PROMPTS["ask_explore"]


def test_common_safety_is_shared_across_all_activities():
    for activity in main.ACTIVITY_PROMPTS:
        full_prompt = main.build_system_prompt(activity)
        assert full_prompt.startswith(main.COMMON_SAFETY[:50])


def test_abusive_input_is_politely_rejected_before_any_llm_call():
    resp = client.post("/api/start", json={"activity": "ask_explore"})
    session_id = resp.headers["x-session-id"]
    resp = client.post("/api/chat", json={"session_id": session_id, "activity": "ask_explore", "message": "you are so stupid bot"})
    text = _reassemble_sse(resp.text)
    assert "kind and fun" in text


def test_safety_prefilter_catches_multiple_abuse_patterns():
    assert main.is_blatantly_inappropriate("I hate you") is True
    assert main.is_blatantly_inappropriate("shut up") is True
    assert main.is_blatantly_inappropriate("you're an idiot") is True
    assert main.is_blatantly_inappropriate("what's 2+2?") is False
    assert main.is_blatantly_inappropriate("tell me about space") is False


# ---------------------------------------------------------------------------
# Requirement #7: Conversation & response handling (6-exchange cap + streaming)
# ---------------------------------------------------------------------------

def test_response_is_genuinely_streamed_in_multiple_chunks():
    resp = client.post("/api/start", json={"activity": "ask_explore"})
    raw = resp.text
    chunk_lines = [l for l in raw.split("\n") if l.startswith("data: ") and '"chunk"' in l]
    assert len(chunk_lines) > 1, "response should stream multiple chunks, not arrive as one blob"


def test_6_exchange_cap_never_splits_an_ai_message_from_its_reply():
    resp = client.post("/api/start", json={"activity": "ask_explore"})
    session_id = resp.headers["x-session-id"]

    for i in range(9):
        client.post("/api/chat", json={"session_id": session_id, "activity": "ask_explore", "message": f"question {i}"})

    turns = main.SESSIONS[session_id]["turns"]
    complete = [t for t in turns if t["user"] is not None]
    pending = [t for t in turns if t["user"] is None]

    assert len(complete) == 6, "must keep exactly 6 complete exchanges"
    assert len(pending) <= 1
    for t in complete:
        assert t["ai"] and t["user"], "every complete exchange must have BOTH sides present"


def test_7th_exchange_causes_the_1st_to_be_dropped():
    resp = client.post("/api/start", json={"activity": "ask_explore"})
    session_id = resp.headers["x-session-id"]

    client.post("/api/chat", json={"session_id": session_id, "activity": "ask_explore", "message": "first question"})
    for i in range(1, 7):
        client.post("/api/chat", json={"session_id": session_id, "activity": "ask_explore", "message": f"question {i}"})

    turns = main.SESSIONS[session_id]["turns"]
    all_user_texts = [t["user"] for t in turns if t["user"]]
    assert "first question" not in all_user_texts, "the 1st exchange should have been dropped by the 7th"


def test_context_sent_to_llm_reflects_ai_then_user_order():
    resp = client.post("/api/start", json={"activity": "ask_explore"})
    session_id = resp.headers["x-session-id"]
    client.post("/api/chat", json={"session_id": session_id, "activity": "ask_explore", "message": "why?"})
    context = main.get_context_messages(session_id)
    assert context[0]["role"] == "model"
    assert context[1]["role"] == "user"


def test_opening_message_is_a_pending_exchange_not_yet_counted():
    resp = client.post("/api/start", json={"activity": "brain_buster"})
    session_id = resp.headers["x-session-id"]
    turns = main.SESSIONS[session_id]["turns"]
    assert len(turns) == 1
    assert turns[0]["user"] is None


# ---------------------------------------------------------------------------
# Requirement #8: Monitoring
# ---------------------------------------------------------------------------

def test_monitoring_log_function_writes_all_required_fields(tmp_path, monkeypatch):
    log_path = tmp_path / "monitoring.log"
    monkeypatch.setattr(main, "MONITORING_LOG_PATH", log_path)

    main.log_llm_request(
        session_id="test-sid",
        activity="brain_buster",
        user_prompt="test prompt",
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        ttft_seconds=0.05,
        total_time_seconds=0.5,
    )

    assert log_path.exists()
    import json as _json
    entry = _json.loads(log_path.read_text().strip())
    required_fields = {
        "timestamp", "session_id", "activity", "user_prompt",
        "input_tokens", "output_tokens", "total_tokens",
        "ttft_seconds", "total_time_seconds",
    }
    assert required_fields.issubset(entry.keys())
    assert entry["session_id"] == "test-sid"
    assert entry["activity"] == "brain_buster"
    assert entry["total_tokens"] == 30


def test_monitoring_log_directory_exists():
    assert main.LOGS_DIR.exists()


# ---------------------------------------------------------------------------
# Answer checking: typo tolerance (safe version) + number-word matching
# ---------------------------------------------------------------------------

def test_number_word_matching():
    assert main.is_correct_answer("7", "seven") is True
    assert main.is_correct_answer("seven", "7") is True
    assert main.is_correct_answer("it's 7", "seven") is True


def test_typo_tolerance_catches_real_typos():
    assert main.is_correct_answer("jupiteer", "jupiter") is True
    assert main.is_correct_answer("venuz", "venus") is True
    assert main.is_correct_answer("chetah", "cheetah") is True
    assert main.is_correct_answer("i think its jupiteer", "jupiter") is True


def test_typo_tolerance_does_not_create_false_positives():
    """Real regression test: an earlier version of this function was too
    loose and marked short, genuinely different words as correct purely
    because they were one edit apart (e.g. 'fun' vs 'sun')."""
    assert main.is_correct_answer("fun", "sun") is False
    assert main.is_correct_answer("bat", "cat") is False
    assert main.is_correct_answer("hat", "cat") is False
    assert main.is_correct_answer("keg", "key") is False
    assert main.is_correct_answer("shot", "shoe") is False
    assert main.is_correct_answer("noon", "moon") is False
    assert main.is_correct_answer("venus", "mars") is False
    assert main.is_correct_answer("dog", "cat") is False
    assert main.is_correct_answer("elephant", "mouse") is False


def test_basic_normalization():
    assert main.is_correct_answer("sun", "Sun") is True
    assert main.is_correct_answer("Sun.", "sun") is True
    assert main.is_correct_answer("  sun  ", "sun") is True
    assert main.is_correct_answer("I think it's the sun", "sun") is True
    assert main.is_correct_answer("", "sun") is False


# ---------------------------------------------------------------------------
# No-duplicate-question enforcement
# ---------------------------------------------------------------------------

def test_strip_embedded_questions_removes_all_questions():
    text = "Great job! Are you ready for the next one? What is the capital of France?"
    result = main.strip_embedded_questions(text)
    assert "?" not in result
    assert "Great job!" in result


def test_extract_final_question_keeps_only_the_real_last_question():
    polluted = (
        "Since we just zoomed around space, let us dive into an animal "
        "question next. Which mammal is the only one that can truly fly? "
        "Which animal is known as the tallest animal in the world?"
    )
    result = main.extract_final_question(polluted)
    assert result == "Which animal is known as the tallest animal in the world?"


def test_regenerate_until_unused_retries_on_collision():
    main.SESSIONS.clear()
    session_id = main.create_session("quick_fire")
    main.add_used_answer(session_id, "jupiter")

    call_count = {"n": 0}
    responses = [
        {"answer": "jupiter", "question": "q1"},
        {"answer": "jupiter", "question": "q2"},
        {"answer": "saturn", "question": "q3"},
    ]

    def fake_generate():
        result = responses[call_count["n"]]
        call_count["n"] += 1
        return result

    data = main._regenerate_until_unused(session_id, fake_generate, "answer", "question")
    assert data["answer"] == "saturn"
    assert call_count["n"] == 3


def test_regenerate_until_unused_stops_at_max_attempts():
    main.SESSIONS.clear()
    session_id = main.create_session("quick_fire")
    main.add_used_answer(session_id, "jupiter")

    call_count = {"n": 0}

    def always_colliding():
        call_count["n"] += 1
        return {"answer": "jupiter", "question": "same question every time"}

    data = main._regenerate_until_unused(session_id, always_colliding, "answer", "question")
    assert call_count["n"] == main.MAX_REGENERATION_ATTEMPTS


def test_regenerate_until_unused_catches_a_repeated_question_with_a_differently_worded_answer():
    """Real regression test built directly from an actual reported bug:
    the exact same question ('What is the largest mammal living in the
    ocean?') was regenerated twice in a real live session, because Gemini
    phrased the answer slightly differently each time ('whale' vs 'blue
    whale') -- so an answer-only collision check never caught it, even
    though the question itself was verbatim identical. This confirms the
    fix: checking the question/riddle text as well catches this."""
    main.SESSIONS.clear()
    session_id = main.create_session("quick_fire")
    main.add_used_question(session_id, "What is the largest mammal living in the ocean?")
    main.add_used_answer(session_id, "whale")

    call_count = {"n": 0}
    responses = [
        # Same question, DIFFERENTLY worded answer -- an answer-only
        # check would have missed this collision entirely.
        {"answer": "blue whale", "question": "What is the largest mammal living in the ocean?"},
        {"answer": "cheetah", "question": "What is the fastest land animal?"},
    ]

    def fake_generate():
        result = responses[call_count["n"]]
        call_count["n"] += 1
        return result

    data = main._regenerate_until_unused(session_id, fake_generate, "answer", "question")
    assert data["question"] == "What is the fastest land animal?"
    assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# Requirement #9: Technical (config, structure)
# ---------------------------------------------------------------------------

def test_env_var_controls_mock_mode():
    assert main._use_mock() is True  # USE_MOCK_LLM=true is set for this whole test run


def test_all_four_prompt_files_exist_on_disk():
    assert (main.PROMPTS_DIR / "common_safety.md").exists()
    assert (main.PROMPTS_DIR / "brain_buster.md").exists()
    assert (main.PROMPTS_DIR / "quick_fire.md").exists()
    assert (main.PROMPTS_DIR / "ask_explore.md").exists()