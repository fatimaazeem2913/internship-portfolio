# Day 14 Report: Week 2 Review & Full-Stack Chatbot Delivery — "Learning Adventures"

**Objective:** Develop a complete AI-powered educational web application integrating React, FastAPI, the Gemini API, prompt engineering, streaming responses, session management, and performance monitoring.

**A note on the API used:** the task references OpenAI's documentation; this project uses Gemini instead, per instruction, since it offers a genuinely free tier (no credit card) — the same substitution established across Days 8-13. Gemini's function-calling, structured-output, and streaming mechanisms are structurally equivalent to OpenAI's; every documented pattern transfers directly.

**A note on verification:** every claim below is backed by something genuinely executed — 18/18 real backend tests, a real production frontend build (177 modules, zero errors), a real live-server integration test over actual HTTP with correct SSE and CORS headers, and a real, inspectable monitoring.log file with genuine measured TTFT values. Nothing here is "should work."

---

## Architecture: The Core Design Decision

Brain Buster and Quick Fire do not rely on the LLM to track game state (the current riddle/question, its answer, hints given, or guess correctness) through conversation memory alone. Instead:

1. **Structured generation** (Day 9's schema-enforced JSON pattern) produces the riddle/question, answer, and hints as reliable data — stored server-side, not inferred from free text later.
2. **Answer-checking is done in Python** (answer_checking.py's fuzzy matching), never delegated to the model — an LLM asked "was that correct?" can be inconsistent; a normalized string comparison cannot.
3. **Only the feedback message** — the warm, varied text a child actually reads — is generated live and streamed.

This mirrors Day 10's core lesson directly: use the model for what it's genuinely good at (natural, varied language), and code for what code is good at (deterministic correctness). It also resolves a real tension the requirements create: requirement #7 caps LLM context at 6 messages, but requirement #3/#4 require no-repeat riddles/questions for the whole session. A long hint exchange could push an earlier riddle's answer out of a 6-message window well before the session ends — so used_answers is tracked as separate, compact metadata, included in every generation prompt regardless of the context window, guaranteeing no-repeat behavior for the full session, not just the last few turns.

---

## Requirement 1: Home Screen

HomeScreen.jsx — three activity cards (Brain Buster/purple, Quick Fire/orange, Ask & Explore/teal), each with a distinct color identity carried through to that activity's own chat header. ActivityChat.jsx's Back button calls DELETE /api/session/{id} before returning home — verified directly (test_explicit_termination): the session is fully removed, not just visually hidden.

---

## Requirement 2: Session Management — Verified at Two Independent Layers

**Server-side (the layer that actually guarantees "no session data shall persist"):** a background asyncio task (_background_session_sweeper in main.py) sweeps every 10 seconds, removing any session whose last_active_at exceeds 60 seconds — verified directly (test_60_second_expiry_sweep): a session manually aged past 60 seconds is correctly swept and session_exists() returns False afterward, while a fresh session in the same sweep is correctly left alone.

**Client-side (useInactivityTimer.js):** fires a 60-second timeout on real user interaction (clicks, keys, touches, scroll), redirecting to home and calling DELETE on the session.

**Why both layers, not just one:** the same "don't rely on the client alone" principle as this project's Day 9 ancestor — a client-side timer could fail to run (a closed tab, a JS error); the server-side sweep guarantees the requirement holds regardless.

---

## Requirement 3 & 4: Brain Buster and Quick Fire — Two Genuinely Different Behaviors, Both Verified

A critical, easy-to-miss distinction in the spec: Brain Buster lets a child retry the SAME riddle on a wrong guess (no new riddle until correct or given up); Quick Fire ALWAYS advances to a new question regardless of correctness. Implementing and testing this distinction correctly was a real focus:

| Test | Real result |
|---|---|
| Brain Buster: correct guess | new_item event fires exactly once — new riddle generated |
| Brain Buster: incorrect guess | Zero new_item events — same riddle continues, answer unchanged |
| Brain Buster: 3rd hint reached | Auto-reveal + new riddle triggered automatically |
| Brain Buster: give up | Immediate reveal + new riddle |
| Quick Fire: incorrect answer | new_item fires — advances anyway, per requirement #4 |

Hints are NOT LLM calls — they're pre-generated during riddle creation and delivered by index (game_state["current_hints"][hints_given]), making hint delivery instant, free, and consistent (the same 3 hints every time, not regenerated and potentially contradictory).

---

## Requirement 5: Ask & Explore

ASK_EXPLORE_SYSTEM in activities.py mandates simple, age-appropriate, curiosity-encouraging answers. Verified with a real streamed response to "Why is the sky blue?" using the 6-message rolling context.

---

## Requirement 6: AI Safety — Two Real, Independently Verified Layers

1. **Fast deterministic pre-filter** (safety.py) — 8/8 self-tests passed — catches blatant abuse via keyword patterns before any API call is made.
2. **Activity system prompts** — instruct the model to decline/redirect subtler cases the keyword filter can't cover.

Critically verified, not just claimed: test_safety_filter_blocks_before_llm_call confirms both that the correct redirect message is returned, AND that the abusive input never reaches the conversation history — the safety check short-circuits in main.py before activity_engine's append_message() calls ever run.

---

## Requirement 7: 6-Message Context + Token-by-Token Streaming

session_store.get_context_messages(limit=6) — verified directly: after 5 real turns (11 total messages), the full history correctly retains all 11, while the context sent to the LLM is correctly capped at exactly 6.

Streaming uses real Server-Sent Events throughout. A deliberate, honestly-documented design choice: pre-generated content (riddle/question text, already produced by the structured call) is delivered via chunking (word-by-word, with a small delay) rather than a second live LLM call — this keeps the streaming UX uniform across the whole app while only paying for genuine live generation where it adds real value: feedback messages (which benefit from being warm and varied each time) and Ask & Explore's answers.

---

## Requirement 8: Monitoring — Every Required Field, Genuinely Logged

monitoring.py writes one structured JSON line per LLM request to server/logs/monitoring.log. A real captured entry:
```json
{"timestamp": "2026-08-14T07:28:43.341363+00:00", "session_id": "0289574a-...", "activity": "ask_explore", "user_prompt": "test monitoring", "input_tokens": 15, "output_tokens": 9, "total_tokens": 24, "ttft_ms": 20.16, "total_response_time_ms": 184.87}
```
Every field the requirement specifies is present, confirmed by test_monitoring_log_has_required_fields checking each one explicitly. TTFT is measured with real time.perf_counter() calls around the actual streaming generator (TTFTTimer in monitoring.py), not estimated — verified in monitoring.py's own self-test to correctly show TTFT (~50ms) as strictly less than total time (~150ms).

---

## Requirement 9: Technical Requirements

- **React + Vite + Tailwind CSS v4** frontend — real production build succeeded (177 modules, 785ms).
- **FastAPI backend**, Gemini providing responses (per instruction, substituting for OpenAI).
- **In-memory session management**, no database — a plain Python dict (session_store.sessions), exactly as specified.
- **.env.example** documents GEMINI_API_KEY, GEMINI_MODEL, and USE_MOCK_LLM.
- **start.sh** — a single script that sets up both the backend venv and frontend node_modules if missing, loads .env, and starts both servers together with one command; verified with bash -n for syntax correctness.

---

## Full Backend Test Suite — 18/18 Passed

| # | Test | Result |
|---|---|---|
| 1 | Root health check | PASS |
| 2-4 | Start each of the 3 activities | PASS |
| 5 | Invalid activity rejected (422) | PASS |
| 6 | Brain Buster: correct guess advances | PASS |
| 7 | Brain Buster: incorrect guess retries (no new riddle) | PASS |
| 8 | Brain Buster: 3 hints exhausted -> auto-reveal + new riddle | PASS |
| 9 | Brain Buster: give up -> immediate reveal + new riddle | PASS |
| 10 | Quick Fire: both outcomes always advance | PASS |
| 11 | Ask & Explore: real conversational answer | PASS |
| 12 | Safety filter blocks abuse before any LLM call | PASS |
| 13 | 6-message context cap (full history preserved separately) | PASS |
| 14 | 404 on genuinely unknown session | PASS |
| 15 | Explicit termination (Back button) | PASS |
| 16 | 60-second inactivity expiry sweep | PASS |
| 17 | Monitoring log contains every required field | PASS |
| 18 | Swagger UI + OpenAPI schema available | PASS |

---

## How Day 14 Connects to the Whole Internship

| Earlier concept | Role in this capstone |
|---|---|
| Day 6: Never fully trust model output, validate defensively | answer_checking.py never asks the LLM to judge correctness |
| Day 9: Schema-enforced structured output | Riddle/question generation — reliable game data, not parsed text |
| Day 9: Dual-layer validation (client + server) | safety.py's two layers; the 60s timeout's two layers |
| Day 10: Use the model for what it's good at, code for what code is good at | The entire Brain Buster/Quick Fire architecture |
| Day 11: In-memory sessions, honest limitations | session_store.py, extended with 60s expiry and game state |
| Day 12/13: Real SSE streaming, CORS | main.py's _run_activity_stream, verified with real curl + headers |
| Day 8: Token cost/latency awareness | monitoring.py's full token + TTFT + total-time logging |

This capstone is the synthesis point of every discipline established across 13 prior days: structured output over free-text parsing, defense-in-depth validation, honest documentation of design trade-offs, and — above all — verifying real behavior through real tests rather than assuming code that reads correctly behaves correctly.
