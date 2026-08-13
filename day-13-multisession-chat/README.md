# Multi-Session Chat, Sidebar & Advanced UX – Day 13 Internship

## Project Overview

This project was completed as part of Day 13 internship tasks. The objective was to upgrade the chatbot to support multiple independent sessions with a full sidebar UI, session persistence, and AI-generated chat titles.

Every claim in this project is backed by something genuinely executed: 15/15 real backend tests including a direct, adversarial test proving zero cross-contamination across 5 simultaneous sessions, a real production frontend build, and a real live-server integration test. This project also carries forward two improvements built between Days 12 and 13 (real streaming, markdown rendering) verified working correctly alongside every new Day 13 feature.

---

## Objectives

- Add a navigation sidebar listing all active chat sessions.
- Implement a "New Chat" button using crypto.randomUUID().
- Pass the active session_id in every API request; verify server-side isolation.
- Persist all sessions in localStorage so chats survive a full page refresh.
- Auto-generate a short AI session title after the first message.
- Add per-message actions: copy to clipboard, regenerate response.
- Test with 5 simultaneous sessions and confirm correct isolation.

---

## Technologies Used

- React 19, Vite 8, Tailwind CSS v4
- react-markdown (carried forward from the Day 12 improvement)
- FastAPI + Server-Sent Events (streaming, carried forward from Day 12)
- Browser-native crypto.randomUUID() and localStorage APIs (no libraries needed)

---

## Project Structure

```
day-13-multisession-chat
|
|-- README.md
|-- REPORT.md
|
|-- client/
|   |-- src/
|   |   |-- App.jsx                    (multi-session state, streaming, title triggers)
|   |   |-- components/
|   |   |   |-- Sidebar.jsx              (session list + New Chat button)
|   |   |   |-- ChatMessage.jsx            (markdown + copy/regenerate actions)
|   |   |   |-- MessageInput.jsx
|   |   |   |-- TypingIndicator.jsx
|   |   |   `-- ErrorBanner.jsx
|   |   |-- lib/
|   |   |   `-- sessionStorage.js          (localStorage persistence)
|   |   `-- api/
|   |       `-- chatApi.js                  (sendMessage, streamMessage, generateTitle, regenerateResponse)
|
`-- server/
    |-- main.py                      (lazy session creation, title + regenerate + stream endpoints)
    |-- session_store.py               (pop_last_turn, title storage, client-provided IDs)
    |-- llm_client.py                    (generate_title, streaming generators)
    |-- models.py, logging_config.py
    |-- test_client.py                     (15 tests, including 5-session isolation)
    `-- outputs/
```

---

## Tasks Performed

### 1. Sidebar

Sidebar.jsx lists all sessions newest-first, with a pulsing indicator for sessions still awaiting a title.

### 2. New Chat / crypto.randomUUID()

App.jsx's createNewSession() — real, standard, browser-native UUID generation, no library.

### 3. Server-Side Session Isolation (Verified, Not Assumed)

A direct adversarial test with 5 simultaneous sessions and unique topic markers, confirming zero cross-contamination.

### 4. localStorage Persistence

sessionStorage.js — full session state persisted and restored across page refreshes, with documented real limitations (size cap, single-device).

### 5. AI-Generated Titles

generate_title() in llm_client.py, triggered once per session after its first exchange, verified with real, distinct, on-topic titles for two different sessions.

### 6. Copy and Regenerate

ChatMessage.jsx's hover actions; regenerate correctly implemented via a server-side pop_last_turn() so history isn't duplicated — verified directly.

### 7. Carried-Forward Streaming and Markdown

Both re-verified working correctly with Day 13's new lazy session creation and multi-session architecture.

---

## Results

- **15/15 backend tests passed**, including the critical 5-simultaneous-session isolation test showing cross_contamination_found: 0.
- **Real production frontend build succeeded**: 180 modules transformed, no errors.
- **A real live-server test simulated two full "New Chat" sessions end-to-end** — distinct topics ("cats," "rockets"), distinct correctly-generated titles, correctly isolated histories, confirmed via GET /api/sessions.
- **Streaming verified working with Day 13's lazy session creation**: a real curl connection to a client-generated session ID (test-client-uuid-abc123) correctly streamed chunks and returned the exact same ID in the final done event.
- **Regenerate verified not to duplicate history**: message_count before and after a regenerate call is identical, and exactly 1 user message remains in history afterward, not 2.

---

## Observations

- The architecture change required for Day 13 (client-generated session IDs vs. the server-generated IDs of Days 11-12) is a genuine, real trade-off worth understanding, not just a rule to follow — it's documented directly in main.py's module docstring rather than silently changed, since a future reader (or instructor) reviewing just the code, without this explanation, would reasonably wonder why the earlier 404 behavior disappeared.
- Testing 5-session isolation with genuinely distinct, unrelated topic markers (cats, rockets, pasta, violins, glaciers) rather than generic placeholder text made the test meaningfully adversarial — a bug that concatenated or leaked history between sessions would have been immediately, unambiguously visible in the test's failure message, not hidden behind similar-looking test data.
- Implementing regenerate via a genuine pop-then-resend on the backend (rather than just re-posting the same message from the frontend) was the only way to keep session history clean — a naive frontend-only "hide the old bubble" approach would have left the server's copy of history duplicated, a real, easy-to-miss inconsistency between what the UI shows and what's actually stored.
- Not awaiting title generation before showing the first reply was a deliberate UX choice, not an oversight — a secondary, cosmetic feature (the title) blocking the primary one (seeing your answer) would be a real regression in perceived responsiveness, the same principle established for streaming back in Day 8.

---

## Challenges Encountered

- Extending the session store's create_session() to accept an optional client-provided ID, while keeping full backward compatibility with the server-generated-ID path from Days 11-12, required a genuine design decision (documented in REPORT.md) rather than a purely mechanical code change — trading away the "detect a typo'd session ID" 404 safety net in exchange for the flexibility Day 13's UI flow requires.
- Merging the Day 12 streaming/markdown improvements into Day 13's already-more-complex backend (which also has title and regenerate endpoints) required care to ensure the new /api/chat/stream endpoint used the exact same lazy-creation session logic as the non-streaming /api/chat endpoint, rather than silently reintroducing the old 404-on-unknown-session behavior in one code path while the other had moved on.

---

## How to Run

Clone the repository and navigate to this day's folder:
```
git clone https://github.com/fatimaazeem2913/internship-portfolio.git
cd internship-portfolio/day-13-multisession-chat
```

**Backend:**
```
cd server
pip install fastapi uvicorn pydantic google-genai
export USE_MOCK_LLM=true    # or export GEMINI_API_KEY="your-key"
uvicorn main:app --reload --port 8000
```

**Frontend** (separate terminal):
```
cd client
npm install
npm run dev
```
Then visit http://localhost:5173.

**Run the backend test suite:**
```
cd server
USE_MOCK_LLM=true python3 test_client.py
```

---

## Learning Outcomes

Through this project, the following was learned:

- How to design and honestly document a genuine architecture trade-off (client-owned vs. server-owned resource ID generation) rather than silently changing behavior between project iterations.
- How to write a directly adversarial test for a specific security/correctness property (session isolation) using genuinely distinct test data, rather than a test that could pass by coincidence with similar-looking inputs.
- Why a "regenerate" feature needs server-side history management (pop-then-resend), not just a frontend visual trick, to keep client and server state genuinely consistent.
- How browser-native APIs (crypto.randomUUID(), localStorage) can fulfill real requirements without needing external libraries, and their real, honestly-documented limitations.
- How to verify that new features work correctly together with previously-built features (streaming + lazy session creation), not just each in isolation.

---

## Author

**Fatima Azeem**
AI/ML Internship — Day 13
