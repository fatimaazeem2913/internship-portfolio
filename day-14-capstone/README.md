# Learning Adventures — Day 14 Capstone

✅ **41/41 backend tests passing** — every functional requirement verified

## Project Overview

A complete, full-stack, AI-powered educational web application built to
satisfy every requirement in the Day 14 task spec: React frontend,
FastAPI backend, real streaming responses, session management, dedicated
per-activity system prompts, safety filtering, a 6-exchange conversation
cap, and full request monitoring.

This is a clean rebuild, incorporating every real bug found and fixed
during earlier iterative testing — typo tolerance, no-repeat questions,
no-duplicate-question generation, and correct AI-then-user exchange
pairing are all built in from the start and covered by dedicated tests,
not discovered after the fact.

## The Three Activities

- **Brain Buster** — one riddle at a time, no repeats within a session.
  Up to 3 hints, each a genuine live LLM call. The answer reveals after
  the 3rd hint is exceeded or on Give Up. Correct answers get positive
  feedback and a new riddle; incorrect answers get encouraging feedback
  and another attempt.
- **Quick Fire** — one question at a time across science, mathematics,
  geography, English, animals, space, and general knowledge. No repeats.
  Correct answers get praise and a fun fact before the next question;
  incorrect answers reveal the correct answer, encourage, and continue.
- **Ask & Explore** — simple, concise, age-appropriate answers to
  whatever a child is curious about.

## Technologies Used

- React 18, Vite 5, Tailwind CSS 3
- FastAPI, Server-Sent Events (real token-by-token streaming)
- google-genai SDK (Gemini) — see note below on provider choice
- In-memory session management, no database
- python-dotenv for `.env` configuration

**LLM provider note:** this project uses Google's Gemini API rather than
OpenAI's, continuing this internship's established substitution since
Day 8 (OpenAI's free tier billing wall). The task's own reference links
include Gemini's official quickstart alongside OpenAI's for this reason.

## Project Structure

```
day-14-capstone/
├── backend/
│   ├── prompts/
│   │   ├── common_safety.md      (shared safety rules, all activities)
│   │   ├── brain_buster.md
│   │   ├── quick_fire.md
│   │   └── ask_explore.md
│   ├── logs/
│   │   └── monitoring.log         (created at runtime, one JSON line per LLM request)
│   ├── main.py                    (single file, organized into 8 clear sections)
│   ├── test_main.py               (41 real tests, one per requirement)
│   ├── requirements.txt
│   ├── start.sh
│   ├── .env.example
│   └── .gitignore
│
└── frontend/
    ├── src/
    │   ├── hooks/
    │   │   └── useInactivityTimer.js
    │   ├── pages/
    │   │   ├── Home.jsx
    │   │   └── ActivityChat.jsx
    │   ├── App.jsx
    │   ├── activities.js
    │   ├── api.js                 (real SSE stream parsing)
    │   ├── index.css
    │   └── main.jsx
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    ├── postcss.config.js
    └── .gitignore
```

## Results

- **41/41 backend tests passed**, covering every functional requirement
  in the task spec individually — session management, all 3 activities,
  safety, the 6-exchange cap, streaming, and monitoring.
- **Real production frontend build**: `npm run build` completes with
  zero errors, 36 modules transformed.
- **Full live end-to-end simulation** across all 3 activities confirmed
  working via real HTTP requests (start → hint → correct guess for
  Brain Buster; start → typo-tolerant guess for Quick Fire; start →
  real question for Ask & Explore).
- **Typo tolerance verified safe**: catches real typos (`jupiteer` →
  `jupiter`) while explicitly tested to NOT cause false positives on
  short, genuinely different words (`fun`/`sun`, `bat`/`cat`, etc.) —
  an earlier looser version of this check was a real, confirmed
  regression that's now covered by a dedicated test.
- **No-repeat enforcement is code-level, not prompt-only**: a dedicated
  retry mechanism (`_regenerate_until_unused`) actually checks each
  generated answer against the session's used-answer list and retries
  up to 3 times on collision — verified with a test that forces
  real collisions and confirms retries actually happen.
- **No embedded/duplicate questions**: feedback and the next
  riddle/question are generated via a single schema-separated LLM call
  (not two calls that could conflict), with two additional layers of
  code-level sanitization (`strip_embedded_questions`,
  `extract_final_question`) verified against real examples of exactly
  how this failure mode occurred in earlier testing.
- **Monitoring log verified**: every real LLM request writes a
  structured JSON line with all 8 required fields (timestamp, session
  ID, activity, user prompt, input/output/total tokens, TTFT, total
  response time).

## How to Run

**Backend:**
```bash
cd backend
chmod +x start.sh
./start.sh
```
This creates a venv, installs dependencies, copies `.env.example` to
`.env` on first run (mock mode by default), and starts on
`http://localhost:8001`.

**Run backend tests:**
```bash
cd backend
source venv/bin/activate
USE_MOCK_LLM=true python -m pytest test_main.py -v
```

**Frontend (separate terminal):**
```bash
cd frontend
npm install
npm run dev
```
Then visit `http://localhost:5174`.

**Switch to real Gemini responses:** edit `backend/.env`, set
`USE_MOCK_LLM=false` and add your real `GEMINI_API_KEY` (free at
aistudio.google.com/apikey).

## Learning Outcomes

- **A prompt instruction alone is never a reliable guarantee.** Every
  hard requirement in this build that needed to be *certain* — the
  right/wrong signal, no repeated questions, no embedded follow-up
  questions — ended up needing actual code-level enforcement (a
  deterministic prefix, a retry-on-collision check, schema-separated
  generation plus text sanitization) after prompt-only instructions
  were repeatedly found insufficient against real model output.
- **Fuzzy matching needs asymmetric safety margins.** Typo tolerance
  that's too loose for short words causes silent false positives (a
  genuinely wrong answer marked correct) that are far more damaging to
  a game's integrity than a genuine typo occasionally being marked
  wrong. The final implementation is deliberately conservative: a
  minimum word length and a matching-first-letter requirement, both
  added specifically after a real regression was found and fixed.
- **Live-streaming and content sanitization are in tension.** A
  response can't be corrected after it's already been streamed to the
  user token-by-token. Anywhere output needed guaranteed sanitization
  (feedback text that must never contain a stray question), the fix
  required generating the full response first, sanitizing it, and only
  then streaming the cleaned result — trading a small latency cost for
  a real correctness guarantee.

## Author

**Fatima Azeem** — AI/ML Internship (Phase 2, Day 14)
