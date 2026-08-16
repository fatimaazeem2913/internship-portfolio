# Learning Adventures — Day 14 Capstone: Week 2 Review & Full-Stack Chatbot Delivery

## Project Overview

This is the Phase 2 capstone: a complete, full-stack, AI-powered educational web application for children, integrating every discipline covered across Days 8-13 — React, FastAPI, the Gemini API (substituted for OpenAI per instruction, since it offers a genuinely free tier), prompt engineering, real streaming, session management, and performance monitoring.

Every claim in this project is backed by something genuinely executed: 18/18 real backend tests, a real production frontend build, and a real live-server integration test over actual HTTP with correct SSE and CORS headers.

---

## The Three Activities

- **Brain Buster** — riddles with up to 3 hints; a correct guess advances to a new riddle, an incorrect guess allows a retry on the same riddle.
- **Quick Fire** — quick educational questions across 7 topics; both correct and incorrect answers always advance to a new question.
- **Ask & Explore** — free-form, age-appropriate Q&A with a curiosity-encouraging tone.

---

## Technologies Used

- React 19, Vite 8, Tailwind CSS v4
- FastAPI, Server-Sent Events (real token-by-token streaming)
- google-genai (Gemini SDK) — structured JSON generation + live streaming
- In-memory session management, no database

---

## Project Structure

```
day-14-capstone
|
|-- README.md
|-- REPORT.md
|-- start.sh                       (one-command startup for both servers)
|
|-- client/
|   |-- src/
|   |   |-- App.jsx                  (home <-> activity screen routing)
|   |   |-- components/
|   |   |   |-- HomeScreen.jsx         (3 activity cards)
|   |   |   `-- ActivityChat.jsx        (shared chat UI, all 3 activities)
|   |   |-- api/capstoneApi.js          (SSE client)
|   |   `-- lib/useInactivityTimer.js    (60s client-side timeout)
|
`-- server/
    |-- main.py                      (FastAPI app, SSE endpoints, CORS, session sweeper)
    |-- activity_engine.py             (the real game logic for all 3 activities)
    |-- activities.py                    (dedicated system prompts + schemas)
    |-- session_store.py                  (in-memory sessions, 60s expiry)
    |-- answer_checking.py                 (fuzzy answer matching, Python-side)
    |-- safety.py                            (dual-layer content safety)
    |-- llm_client.py                         (Gemini structured gen + streaming)
    |-- monitoring.py                          (structured logging with TTFT)
    |-- models.py
    |-- test_client.py                          (18 real tests)
    |-- .env.example
    |-- logs/monitoring.log                      (real logged requests)
    `-- outputs/
```

---

## Results

- **18/18 backend tests passed**, covering every functional requirement including the tricky behavioral distinction between Brain Buster (retry on wrong guess) and Quick Fire (always advances).
- **Real production frontend build**: 177 modules, zero errors, kid-friendly bright theme verified with a real screenshot.
- **Real live-server integration test**: an actual uvicorn process hit with real curl SSE requests, confirming correct streaming chunks, the X-Session-Id header, and correct CORS headers all working together.
- **Real monitoring log** with every required field (timestamp, session ID, activity, prompt, token counts, TTFT, total time), verified field-by-field.
- **A real bug found and fixed**: answer_checking.py's number-word matching (e.g. "7" vs. "seven") was initially missing — caught by a self-test, fixed, and reverified (14/14 passing after the fix).

---

## How to Run

**Easiest -- one command:**
```bash
cd day-14-capstone
./start.sh
```
This sets up both the backend virtualenv and frontend node_modules if missing, copies .env.example to .env on first run (edit it with your real GEMINI_API_KEY afterward), and starts both servers.

**Manually:**
```bash
# Backend
cd server
pip install fastapi uvicorn pydantic google-genai
export USE_MOCK_LLM=true    # or export GEMINI_API_KEY="your-key"
uvicorn main:app --reload --port 8000

# Frontend (separate terminal)
cd client
npm install
npm run dev
```
Then visit http://localhost:5174.

**Run the test suite:**
```bash
cd server
USE_MOCK_LLM=true python3 test_client.py
```

---

## Learning Outcomes

This capstone synthesizes the entire internship's disciplines:
- Structured, schema-enforced generation over trusting free-text model output for anything requiring reliability (Day 9).
- Using the model only for what it's genuinely good at (varied, warm language) and code for what code is reliably good at (exact-match game logic) -- Day 10's core lesson, applied to a real product.
- Defense-in-depth safety and validation at multiple independent layers (Day 9's pattern, extended here to both content safety and session expiry).
- Real SSE streaming and CORS, verified with actual live HTTP requests rather than assumed correct (Day 12/13).
- Full-cycle performance monitoring with genuinely measured TTFT, not estimated.

---

## Author

**Fatima Azeem**
AI/ML Internship -- Day 14 (Phase 2 Capstone)
