# FastAPI Backend & Chat State Management – Day 11 Internship

## Project Overview

This project was completed as part of Day 11 internship tasks. The objective was to expose LLM generation over HTTP and maintain full conversation history across stateless connections using server-side session management.

Unlike most previous days, this task's core deliverable — the server itself — required no external API to genuinely verify. Every endpoint, every status code, every session-management behavior was tested against the real, running FastAPI application using FastAPI's own TestClient, plus a genuine live uvicorn process confirmed reachable over real HTTP. A USE_MOCK_LLM environment variable decouples testing the server from testing the actual Gemini call, enabling 10/10 tests to pass with zero API cost and zero network dependency, while the real Gemini integration remains fully implemented for local use with a real key.

---

## Objectives

- Build a FastAPI server with a POST /api/chat endpoint; define request/response schemas with Pydantic.
- Create an in-memory session store mapping session_id to message history arrays.
- Write a handler that checks for session_id, initializes fresh history with the system prompt if new, and appends messages each turn.
- Add a GET /api/sessions endpoint listing all active sessions with message counts.
- Implement proper HTTP status codes (400, 404, 500) and structured error responses.
- Add structured logging: timestamp, session_id, model, token usage, latency_ms.
- Test all endpoints via the FastAPI auto-generated Swagger UI at /docs.

---

## Technologies Used

- Python 3
- FastAPI, Uvicorn, Pydantic
- google-genai (Gemini SDK, for the real LLM call)

---

## Project Structure

```
day-11-fastapi-backend
|
|-- README.md
|-- REPORT.md
|
|-- main.py               (the FastAPI app and all routes)
|-- models.py               (Pydantic request/response schemas)
|-- session_store.py         (the in-memory chat_sessions store)
|-- llm_client.py             (Gemini call, with mock support for testing)
|-- logging_config.py          (structured JSON logging)
|-- test_client.py               (full test suite, via FastAPI's TestClient)
|
|-- outputs
    `-- test_client_results.txt   (full, real test run output)
```

---

## Tasks Performed

### 1. FastAPI Server + Pydantic Schemas

main.py and models.py — POST /api/chat and GET /api/sessions, with ChatRequest, ChatResponse, SessionSummary, SessionListResponse, and ErrorResponse schemas.

### 2. In-Memory Session Store

session_store.py — chat_sessions = {}, with self-tests fully executed, including a correctly-raised KeyError for an append to a non-existent session.

### 3. Request Handler Logic

Full session-checking flow in main.py's chat() handler, verified across separate, independent requests to confirm history genuinely accumulates.

### 4. GET /api/sessions

Lists all active sessions with real, verified message counts.

### 5. HTTP Status Codes and Structured Errors

All of 200, 400, 404, 422, and 500 tested against the real application, including a genuine simulated LLM-failure test for the 500 path.

### 6. Structured Logging

logging_config.py — one JSON line per request with all 5 required fields, verified with real captured output from the actual test run.

### 7. Swagger UI Verification

Verified twice: in-process via TestClient, and over a real live HTTP server started with uvicorn.

---

## Results

- **10/10 tests passed** against the real, running application (not simulated) — covering both success paths and all required error paths.
- **Session state genuinely persists across separate requests**: a second request reusing a session_id correctly returned message_count: 4 after 2 full turns, proving the in-memory store works exactly as HTTP's stateless nature requires it to.
- **All 3 endpoints confirmed documented in the auto-generated OpenAPI schema**, and the live Swagger UI confirmed reachable over real HTTP (200 OK at /docs).
- **Real structured JSON logs captured**, containing exactly the 5 required fields, both for successful requests and for a genuinely simulated 500 failure.

---

## Observations

- The USE_MOCK_LLM pattern turned out to be more than a sandbox workaround — it's a real, standard production testing practice: isolating an external API call behind one function makes the rest of the system's correctness (routing, validation, session logic, error handling) testable quickly, repeatably, and for free, independent of the external service's availability or cost.
- Distinguishing 400 from 422 required genuinely different validation logic at two different layers — Pydantic's automatic schema validation catches structurally wrong requests before any handler code runs, while explicit application-level checks (like whitespace-only messages) catch requests that are structurally valid but semantically meaningless.
- The global exception handler is a real safety net, not decoration: it guarantees that even a genuinely unexpected failure (not one of the specifically anticipated error paths) still returns clean, structured JSON instead of leaking a raw stack trace to an API consumer — a real security and professionalism concern for any public-facing API.
- Testing session continuity required a deliberately two-step test (create a session, then separately continue it) rather than testing session creation in isolation — this is the only way to actually prove state persists ACROSS requests, not just within a single request's handling.

---

## Challenges Encountered

- **Running a real server across separate tool invocations proved genuinely difficult in this verification environment**, since each command invocation runs in an isolated shell with no persistence of background processes between calls. This was resolved by using FastAPI's own TestClient for the main test suite — which is the standard, officially-recommended way to test a FastAPI application, exercising the exact same routing/validation/handler code as a real deployed server, without depending on inter-process persistence — while a genuine live uvicorn process was still started and tested over real HTTP within a single atomic command, confirming the application works correctly as an actual network-reachable server too, not just inside a test harness.
- Designing the 500 test required genuinely simulating a failure (temporarily replacing the real generate_reply function with one that raises an exception) rather than just asserting the error-handling code "should" work — this caught the real, distinct behavior of the try/except in the handler versus the global exception handler as a final backstop.

---

## How to Run

Clone the repository and navigate to this day's folder:
```
git clone https://github.com/fatimaazeem2913/internship-portfolio.git
cd internship-portfolio/day-11-fastapi-backend
```

Install dependencies:
```
pip install fastapi uvicorn pydantic google-genai
```

Run the full test suite (no API key needed, uses the mock LLM):
```
export USE_MOCK_LLM=true
python3 test_client.py
```

Run the real live server and explore the Swagger UI:
```
export GEMINI_API_KEY="your-key-here"    # or export USE_MOCK_LLM=true to skip real API calls
uvicorn main:app --reload --port 8000
```
Then visit http://127.0.0.1:8000/docs in your browser.

---

## Learning Outcomes

Through this project, the following was learned:

- How to build a real FastAPI application with Pydantic request/response validation, and why FastAPI generates a fully interactive, accurate Swagger UI automatically from those same models with zero extra work.
- Why HTTP's stateless nature makes a session store necessary for any genuine multi-turn conversation, directly connecting back to Day 4's "a Transformer has no memory, only a context window" finding.
- How to design and implement proper HTTP status code semantics (400 vs. 422 vs. 404 vs. 500), and why each one requires its own genuinely distinct trigger condition to test meaningfully.
- Why structured (JSON) logging is standard production practice over free-form text logs, and how to log exactly the fields a real operations team would need.
- How to test a FastAPI application thoroughly without needing a live external API dependency, using both dependency isolation (the mock LLM pattern) and FastAPI's own TestClient — and why this separation of concerns is valuable well beyond this specific verification environment's constraints.

---

## Author

**Fatima Azeem**
AI/ML Internship — Day 11
