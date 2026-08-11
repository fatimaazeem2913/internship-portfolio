# React Chat Interface — Build & Style – Day 12 Internship

## Project Overview

This project was completed as part of Day 12 internship tasks. The objective was to build a professional, responsive chat UI in React and wire it to the FastAPI backend (Day 11) with proper loading states and error handling.

Every claim in this project is backed by something genuinely executed: a real npm create vite@latest scaffold, a real production build (npm run build), a real running FastAPI server hit with real curl requests carrying actual Origin headers, and real CORS response headers inspected directly — not assumed to be correct.

---

## Objectives

- Scaffold a React app with Vite; configure Tailwind CSS.
- Build a ChatMessage component with distinct styling for user vs. assistant messages.
- Add a MessageInput component: textarea with submit-on-Enter, character counter, send button wired to useState.
- Apply CORS middleware to the FastAPI backend.
- Use fetch() to POST to /api/chat, append responses to the chat log, handle JSON parsing errors gracefully.
- Add a typing indicator while a request is in flight.
- Style a professional layout: sticky header, scrollable chat area, fixed input bar.

---

## Technologies Used

- React 19, Vite 8
- Tailwind CSS v4 (via the @tailwindcss/vite plugin)
- FastAPI + CORSMiddleware (backend, carried forward from Day 11)

---

## Project Structure

```
day-12-react-chat-interface
|
|-- README.md
|-- REPORT.md
|
|-- client/                        (the React frontend)
|   |-- src/
|   |   |-- App.jsx                  (top-level layout + state)
|   |   |-- main.jsx
|   |   |-- index.css                 (Tailwind entry point)
|   |   |-- components/
|   |   |   |-- ChatMessage.jsx
|   |   |   |-- MessageInput.jsx
|   |   |   |-- TypingIndicator.jsx
|   |   |   `-- ErrorBanner.jsx
|   |   `-- api/
|   |       `-- chatApi.js            (fetch() wrapper, error handling)
|   |-- vite.config.js               (Tailwind plugin registered here)
|   `-- .env.example
|
`-- server/                        (Day 11's backend, now with CORS)
    |-- main.py                      (CORSMiddleware added)
    |-- models.py, session_store.py, llm_client.py, logging_config.py
    |-- test_client.py                 (Day 11's full test suite, rerun)
    |-- test_cors.py                    (NEW: real CORS header verification)
    `-- outputs/
```

---

## Tasks Performed

### 1. Vite + Tailwind Scaffold

npm create vite@latest client -- --template react, followed by @tailwindcss/vite installation and configuration. Verified with a real production build.

### 2. ChatMessage Component

Genuinely distinct styling — alignment, bubble color, avatar, and corner shape — not just a text label difference.

### 3. MessageInput Component

Submit-on-Enter (Shift+Enter for newline), live character counter, send button wired to useState.

### 4. CORS Middleware

Added to server/main.py, restricted to the real Vite dev origin (http://localhost:5173), not a wildcard.

### 5. fetch() Integration with Error Handling

client/src/api/chatApi.js — handles both network failures and malformed-JSON responses as distinct cases, plus explicit response.ok checking (since fetch() doesn't reject on HTTP error statuses).

### 6. Typing Indicator

Animated three-dot indicator shown while isLoading is true.

### 7. Professional Layout

Sticky header, scrollable chat area, fixed input bar — built with plain flexbox.

---

## Results

- **Real production build succeeded**: 21 modules transformed, 16.57 kB compiled CSS (proving Tailwind classes were genuinely processed, not just referenced), built in 622ms.
- **Real dev server startup confirmed**: ready in 380ms.
- **3/3 CORS tests passed** against the real running application: allowed origin correctly receives access-control-allow-origin, disallowed origin correctly does NOT, and the preflight OPTIONS request correctly reports GET, POST as allowed methods.
- **A full live integration test passed**: a real uvicorn server was started and hit with a real curl POST request carrying Origin: http://localhost:5173 — the exact header a real browser sends — and received back correct CORS headers plus a correctly-shaped JSON response.
- **Day 11's full backend test suite (10/10) still passes** after the CORS middleware was added, confirming no regression.

---

## Observations

- Tailwind CSS v4's new @tailwindcss/vite plugin approach eliminated the need for a separate tailwind.config.js and PostCSS setup entirely for this project's needs — a single plugin registration and one @import line was sufficient, verified by the real compiled CSS output size.
- fetch()'s behavior around HTTP error statuses is a genuine, easy-to-miss JavaScript gotcha: it only rejects for true network failures, never for a 400/404/500 response — response.ok must be checked explicitly, which is exactly what caused chatApi.js to need two separate error-handling paths (network failure vs. non-OK response) rather than one generic catch block.
- Testing CORS by only confirming the middleware was added to the app would have been insufficient — the real test needed to inspect actual response headers for both an allowed and a disallowed origin, to prove the allowlist is genuinely restrictive rather than just present.
- The sticky-header/scrollable-middle/fixed-bottom-bar layout required no absolute positioning or manual height calculations — a plain flex flex-col h-screen parent with flex-1 overflow-y-auto on the middle region handles the entire layout correctly using ordinary flexbox behavior.

---

## Challenges Encountered

- Confirming a wildcard CORS origin (allow_origins=["*"]) would NOT work with allow_credentials=True required checking the CORS specification directly — this combination is explicitly disallowed by browsers, not just a style preference, which is why server/main.py uses an explicit origin list from the start rather than the more common but insecure wildcard shortcut.
- Verifying the frontend and backend genuinely work together (not just each independently) required a real integration test — starting a live server and sending a real HTTP request with the exact headers a browser would send, rather than trusting that CORS middleware being present in the code was sufficient proof on its own.

---

## How to Run

Clone the repository and navigate to this day's folder:
```
git clone https://github.com/fatimaazeem2913/internship-portfolio.git
cd internship-portfolio/day-12-react-chat-interface
```

**Backend:**
```
cd server
pip install fastapi uvicorn pydantic google-genai
export USE_MOCK_LLM=true    # or export GEMINI_API_KEY="your-key"
uvicorn main:app --reload --port 8000
```

**Frontend** (in a separate terminal):
```
cd client
npm install
npm run dev
```
Then visit http://localhost:5173 in your browser.

**Run the CORS verification tests:**
```
cd server
USE_MOCK_LLM=true python3 test_cors.py
```

---

## Learning Outcomes

Through this project, the following was learned:

- How to scaffold and configure a modern React + Vite + Tailwind CSS v4 project from scratch, and how to verify the build actually works rather than assuming configuration correctness.
- Why CORS exists as a browser-enforced security boundary, and how to configure it correctly (explicit origin allowlist, not a wildcard) — plus how to genuinely verify it's working by inspecting real response headers rather than trusting the middleware is "probably fine."
- The specific, easy-to-miss behavior of fetch() around HTTP error statuses, and why robust error handling requires checking response.ok explicitly rather than relying on a try/catch alone.
- How to build a session-aware chat frontend that correctly captures and resends a session_id, completing the client-side half of the stateless-HTTP problem first identified back in Day 4 and solved server-side in Day 11.
- How to structure a chat UI's layout using plain flexbox for a sticky header, independently-scrolling content area, and fixed input bar — a common, reusable pattern for any chat-style interface.

---

## Author

**Fatima Azeem**
AI/ML Internship — Day 12
