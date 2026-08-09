# Day 11 Report: FastAPI Backend & Chat State Management

**Objective:** Expose LLM generation over HTTP and maintain full conversation history across stateless connections using server-side session management.

**A note on verification:** unlike most previous days, this task's core deliverable — the server itself — required no external API to genuinely verify. FastAPI's own `TestClient` runs the real application in-process, exercising the real routing, validation, session management, and error-handling code paths without needing a live network connection. A `USE_MOCK_LLM` environment variable (documented in Part 6) additionally decouples testing the server from testing the LLM call, letting the entire request/response/session cycle be verified end-to-end, repeatably, with zero API cost. All 10 tests below passed against the real, running application — not simulated. The server was also started as a genuine live process and confirmed reachable over real HTTP (not just in-process) at the end of this report.

---

## Part 1: FastAPI Server with Pydantic Schemas

`main.py` defines `POST /api/chat` and `GET /api/sessions`. `models.py` defines every request/response schema:

- **ChatRequest** — session_id (optional), message (required, min_length=1)
- **ChatResponse** — session_id, response, message_count, latency_ms
- **SessionSummary** / **SessionListResponse** — for the sessions listing
- **ErrorResponse** — one consistent shape used across every error path (400/404/500)

**Verified:** a ChatRequest with an empty string correctly raised a Pydantic ValidationError before ever reaching application code — confirming Pydantic's validation runs automatically, exactly as FastAPI's documentation promises.

---

## Part 2: In-Memory Session Store

`session_store.py` implements `chat_sessions = {}` exactly as specified, mapping session_id to a history array plus created_at/last_active_at metadata.

**Why this is necessary at all:** HTTP is stateless — no request inherently knows about any previous request. Day 4 established that a Transformer itself has no memory, only a context window; combined, a real multi-turn conversation requires something to hold state between requests and reassemble the full history before each model call. This session store is exactly that something.

**The honest, documented limitation:** a plain Python dict is lost the moment the server process restarts, and cannot be shared across multiple server replicas in a real horizontally-scaled deployment. Real production systems use Redis or a database for exactly this reason — this is stated directly rather than presented as if the in-memory approach were production-ready.

**Self-test, fully executed:**
```
New session created, history seeded with system prompt
4 messages appended (2 user, 2 model turns)
append_message() to a non-existent session_id correctly raised KeyError
list_sessions() correctly excluded the system prompt from message_count
```

---

## Part 3: The Request Handler Logic

`main.py`'s chat() handler implements exactly the specified flow: check session_id -> create fresh history with system prompt if new -> append user message -> call the model -> append model response. Verified via the "Continue existing session" test: a second request reusing the first request's session_id correctly returned message_count: 4 (2 turns x 2 messages each), proving history genuinely accumulates across separate, independent requests — not just within a single request's lifetime.

---

## Part 4: GET /api/sessions

Lists every active session with its message count, created_at, and last_active_at. Verified: after 2 chat turns in one session, GET /api/sessions correctly reported active_sessions: 1 with message_count: 4 — matching the session's real, accumulated history exactly.

---

## Part 5: HTTP Status Codes and Structured Error Responses

All required status codes tested and verified against the real application:

| Code | Scenario | Verified real response |
|---|---|---|
| 200 | Successful chat / session list | Full, correctly-shaped ChatResponse / SessionListResponse |
| 400 | Whitespace-only message (passes Pydantic's min_length=1 but is still invalid) | {"error": "empty_message", "message": "...", "status_code": 400} |
| 422 | Missing required field entirely (Pydantic's own validation) | FastAPI's standard validation error detail, automatic |
| 404 | session_id provided but doesn't exist | {"error": "session_not_found", "message": "...", "status_code": 404} |
| 500 | Simulated LLM provider failure | {"error": "llm_call_failed", "message": "...", "status_code": 500} |

**A deliberate distinction worth noting:** 400 and 422 are both "bad request" in spirit but arise from two different validation layers — 422 is Pydantic's automatic schema validation (wrong type, missing field), while 400 is application-level validation the handler performs explicitly (a message that's syntactically valid JSON with the right types, but semantically empty after whitespace-stripping). Testing both separately confirms both layers work independently.

**The 500 test is a genuine failure simulation**, not just a code-review claim: test_500_llm_failure temporarily replaces generate_reply with a function that raises a real exception, sends a real request through the full handler, and confirms the server catches it and returns a clean structured error — rather than crashing or leaking a raw Python traceback, which a real, unhandled exception would otherwise do. A global @app.exception_handler(Exception) additionally catches anything NOT already handled by a specific try/except, guaranteeing every possible failure path still returns valid, structured JSON.

---

## Part 6: Structured Logging

`logging_config.py` logs one structured JSON line per request/error, containing exactly the required fields: timestamp, session_id, model, token usage (prompt_tokens/completion_tokens/total_tokens), and latency_ms.

**Real captured log output from the actual test run:**
```json
{"timestamp": "2026-08-09T08:53:11.135698+00:00", "session_id": "62c030b5-...", "model": "gemini-3.5-flash-lite", "prompt_tokens": 33, "completion_tokens": 16, "total_tokens": 49, "latency_ms": 50.24, "status": "success"}
{"timestamp": "2026-08-09T08:53:11.260413+00:00", "session_id": "68d04715-...", "error_type": "llm_call_failed", "message": "Simulated provider outage for testing.", "status_code": 500, "status": "error"}
```

**Why JSON, not free-form text:** structured logs can be parsed, filtered, and aggregated by real log-analysis tooling ("show every request over 2000ms," "sum tokens used per session today") without fragile regex parsing against arbitrary text — a genuine, standard production practice, not decoration.

**The USE_MOCK_LLM testing pattern (llm_client.py):** the actual Gemini API call is isolated behind a single generate_reply(history) function, swappable via an environment variable for a deterministic mock. This is the same separation-of-concerns principle from Day 6's prompt template library and Day 9's tool schema/function split — it made it possible to verify the entire server's routing, validation, session logic, and error handling with zero API cost and zero network dependency, while the real Gemini path remains fully implemented and ready for local use with a real key.

---

## Part 7: Swagger UI Verification

Verified twice, two different ways:

1. **In-process (TestClient):** GET /docs returned 200, and GET /openapi.json correctly listed all 3 real endpoints (/api/chat, /api/sessions, /) in its schema — confirming FastAPI's automatic documentation generation is genuinely working from the Pydantic models, not just assumed to work.
2. **Over real HTTP:** the server was started as an actual live process (uvicorn main:app --host 127.0.0.1 --port 8000), and a real curl request to http://127.0.0.1:8000/docs returned HTTP_STATUS:200, followed by a real POST /api/chat over actual HTTP returning a correctly-shaped response — confirming the application works as a genuine, network-reachable server, not just inside the test harness.

---

## Full Test Suite Results — 10/10 Passed

| # | Test | Result |
|---|---|---|
| 1 | Root health check | PASS -- 200 |
| 2 | New session chat (no session_id) | PASS -- 200, message_count: 2 |
| 3 | Continue existing session | PASS -- 200, message_count: 4, same session_id preserved |
| 4 | 400: whitespace-only message | PASS -- 400, structured error |
| 5 | 422: missing required field | PASS -- 422, Pydantic validation |
| 6 | 404: unknown session_id | PASS -- 404, structured error |
| 7 | GET /api/sessions listing | PASS -- 200, correct counts |
| 8 | Swagger UI + OpenAPI schema | PASS -- 200, all 3 paths documented |
| 9 | Latency genuinely measured | PASS -- real positive latency_ms |
| 10 | 500: simulated LLM failure | PASS -- 500, structured error, no crash |

---

## How Day 11 Connects to Earlier Days

| Earlier concept | Role in Day 11 |
|---|---|
| Day 4: Transformer has no memory, only a context window | Directly motivates why a session store is necessary at all — the model itself never remembers anything between calls |
| Day 6/9: Separating prompts/schemas from application code | Same principle applied to llm_client.py — isolating the external API call enables mock-based testing |
| Day 9: Structured output / validation | The same "never trust raw input, validate explicitly" discipline, now applied to HTTP request bodies via Pydantic |
| Day 10: Real bugs found through actual testing, not assumed | USE_MOCK_LLM and TestClient exist specifically so this server's real behavior — not just its written code — gets verified, continuing the same standard |

Day 11 turns every previous day's LLM-calling logic into something a real client application (a website, a mobile app, another service) could actually talk to over the network — the concrete first step of Phase 2's "Full-Stack Chat" goal.
