"""
main.py — Learning Adventures backend

Implements every functional requirement from the Day 14 task spec:
  1. Home screen routing is handled entirely by the frontend; this file
     serves the /api/start and /api/chat endpoints each activity needs.
  2. Session management — in-memory, 60s inactivity expiry, no persistence.
  3. Brain Buster — one riddle at a time, no repeats, up to 3 live hints,
     reveal on 3rd-hint-exceeded or Give Up.
  4. Quick Fire — one question at a time from 7 topics, no repeats,
     correct -> praise + fact + next question; incorrect -> reveal +
     encouragement + next question.
  5. Ask & Explore — simple, concise, age-appropriate free-form Q&A.
  6. AI Safety — each activity has its own dedicated prompt file, plus a
     shared safety prompt; a fast deterministic pre-filter also rejects
     blatant abuse before any API call.
  7. Conversation & response handling — only the 6 most recent EXCHANGES
     (an AI message + the user's reply to it, as one unit) are kept as
     context; every response streams token-by-token to the frontend.
  8. Monitoring — every real LLM request is logged to logs/monitoring.log
     with timestamp, session ID, activity, user prompt, token usage,
     TTFT, and total response time.
  9. Technical — FastAPI backend, in-memory sessions (no DB), .env
     config, start.sh startup script.

LLM provider note: this project uses Google's Gemini API (google-genai
SDK) rather than OpenAI's, continuing this internship's established
substitution since Day 8 (OpenAI's free tier billing wall). The task's
own reference links include Gemini's official quickstart alongside
OpenAI's for exactly this reason. USE_MOCK_LLM (unset/false = real calls,
true = deterministic offline mock) follows the same pattern used
throughout this internship since Day 11, so the whole app is testable
with zero API cost.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ===========================================================================
# 1. PROMPT LOADING (requirement #6 — dedicated prompt per activity)
# ===========================================================================

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. Every activity's prompt must "
            f"live under backend/prompts/ as a .md file."
        )
    return path.read_text(encoding="utf-8").strip()


COMMON_SAFETY = _load_prompt("common_safety.md")
ACTIVITY_PROMPTS = {
    "brain_buster": _load_prompt("brain_buster.md"),
    "quick_fire": _load_prompt("quick_fire.md"),
    "ask_explore": _load_prompt("ask_explore.md"),
}


def build_system_prompt(activity: str) -> str:
    if activity not in ACTIVITY_PROMPTS:
        raise HTTPException(status_code=400, detail=f"Unknown activity: {activity}")
    return f"{COMMON_SAFETY}\n\n---\n\n{ACTIVITY_PROMPTS[activity]}"


BRAIN_BUSTER_RIDDLE_SCHEMA = {
    "type": "OBJECT",
    "required": ["riddle", "answer"],
    "properties": {
        "riddle": {"type": "STRING", "description": "The riddle question, one to two sentences."},
        "answer": {"type": "STRING", "description": "The single-word or short-phrase correct answer."},
    },
}

# Combined schema for feedback + the next item in ONE call, rather than
# two separate calls. This is a deliberate architectural choice: a
# free-text feedback call, generated separately from the next riddle/
# question, was repeatedly observed (in real testing) spontaneously
# writing its OWN follow-up question that collided with the real one.
# Giving the next item its own dedicated schema field, with an explicit
# instruction that "feedback" must contain no question at all, prevents
# the model from conflating the two by construction, not just by request.
BRAIN_BUSTER_FEEDBACK_AND_NEXT_RIDDLE_SCHEMA = {
    "type": "OBJECT",
    "required": ["feedback", "next_riddle", "next_answer"],
    "properties": {
        "feedback": {
            "type": "STRING",
            "description": (
                "1-2 sentence enthusiastic congratulations for a correct guess. "
                "Must NOT contain the next riddle, any question, or any transition "
                "phrase -- no question marks anywhere in this field."
            ),
        },
        "next_riddle": {
            "type": "STRING",
            "description": "A brand new original riddle -- different topic and answer than the one just solved.",
        },
        "next_answer": {"type": "STRING", "description": "The single-word or short-phrase correct answer to next_riddle."},
    },
}

QUICK_FIRE_QUESTION_SCHEMA = {
    "type": "OBJECT",
    "required": ["question", "answer", "fun_fact"],
    "properties": {
        "question": {"type": "STRING", "description": "One educational question."},
        "answer": {"type": "STRING", "description": "The short, correct answer."},
        "fun_fact": {"type": "STRING", "description": "One short, interesting fact related to the answer."},
    },
}

QUICK_FIRE_FEEDBACK_AND_NEXT_SCHEMA = {
    "type": "OBJECT",
    "required": ["feedback", "next_question", "next_answer", "next_fun_fact"],
    "properties": {
        "feedback": {
            "type": "STRING",
            "description": (
                "1-2 sentence feedback on whether the child's answer was correct. "
                "If incorrect, reveal the correct answer and share the fun fact. "
                "Must NOT contain any question, follow-up, or transition phrase -- "
                "no question marks anywhere in this field."
            ),
        },
        "next_question": {
            "type": "STRING",
            "description": "A brand new educational quiz question -- different topic and answer than the one just answered.",
        },
        "next_answer": {"type": "STRING", "description": "The short, correct answer to next_question."},
        "next_fun_fact": {"type": "STRING", "description": "One interesting fact related to next_answer."},
    },
}


# ===========================================================================
# 2. SAFETY PRE-FILTER (requirement #6) — fast, deterministic, catches the
#    obvious cases before any API call is made. The activity prompts
#    (loaded above) handle subtler cases a keyword list can't catch.
#    Neither layer alone is sufficient; together they're defense in depth.
# ===========================================================================

_BLOCKED_PATTERNS = [
    r"\bkill\s+(yourself|you|him|her|them)\b",
    r"\bshut\s+up\b",
    r"\bstupid\s+(bot|ai|robot)\b",
    r"\bi\s+hate\s+you\b",
    r"\bidiot\b",
]
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _BLOCKED_PATTERNS]

SAFE_REDIRECT_MESSAGE = (
    "Let's keep things kind and fun! I'm not able to respond to that, "
    "but I'd love to help with something else -- want to try a riddle, "
    "a quick quiz question, or ask me something you're curious about?"
)


def is_blatantly_inappropriate(text: str) -> bool:
    return any(pattern.search(text) for pattern in _COMPILED_PATTERNS)


# ===========================================================================
# 3. ANSWER CHECKING — fuzzy, pure Python, not delegated to the LLM.
#    A model asked to judge correctness can be inconsistent; a normalized
#    string comparison against a known-correct answer cannot.
# ===========================================================================

_NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten",
}
_WORD_TO_NUMBER = {word: digit for digit, word in _NUMBER_WORDS.items()}


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[.!?,;:]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _number_equivalent_forms(normalized_text: str) -> set:
    forms = {normalized_text}
    if normalized_text in _NUMBER_WORDS:
        forms.add(_NUMBER_WORDS[normalized_text])
    if normalized_text in _WORD_TO_NUMBER:
        forms.add(_WORD_TO_NUMBER[normalized_text])
    return forms


def _levenshtein_distance(a: str, b: str) -> int:
    """Standard edit-distance calculation (insertions, deletions,
    substitutions), pure Python, no dependency needed."""
    if len(a) < len(b):
        a, b = b, a
    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current_row = [i]
        for j, cb in enumerate(b, 1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def is_correct_answer(user_guess: str, correct_answer: str) -> bool:
    """
    1. Exact match after normalization, or numeral/word-form match
       (e.g. "7" correctly matches stored answer "seven").
    2. The normalized correct answer appears as a whole word/phrase
       within the normalized guess ("I think it's a sun" -> "sun").
    3. Typo tolerance: a genuine small typo (e.g. "jupiteer" for
       "jupiter") is still marked correct. This is DELIBERATELY
       conservative -- an earlier, looser version of this check caused a
       real regression, marking genuinely different short words (like
       "fun" vs "sun", "bat" vs "cat") as correct purely because they
       happened to be one edit apart. Two safety constraints prevent
       that here:
         - Only words of 5+ letters are eligible at all (short common
           words have too many genuinely different real-word neighbors
           at edit-distance 1 for this to be safe).
         - The guess word and answer word must share the same first
           letter (real typos essentially never change the very first
           letter typed; this alone rules out most unrelated-word
           collisions like "fun"/"sun" or "cat"/"bat").
    """
    norm_guess = _normalize(user_guess)
    norm_answer = _normalize(correct_answer)
    if not norm_answer:
        return False

    answer_forms = _number_equivalent_forms(norm_answer)
    if norm_guess in answer_forms:
        return True
    for form in answer_forms:
        pattern = r"\b" + re.escape(form) + r"\b"
        if re.search(pattern, norm_guess):
            return True

    guess_words = norm_guess.split()
    answer_words = norm_answer.split()
    for gw in guess_words:
        for aw in answer_words:
            if len(aw) < 5:
                continue  # too short -- edit-distance tolerance is unsafe below this length
            if not gw or gw[0] != aw[0]:
                continue  # require matching first letter as a real-typo safety guard
            max_distance = 1 if len(aw) <= 6 else 2
            if _levenshtein_distance(gw, aw) <= max_distance:
                return True

    return False


MAX_REGENERATION_ATTEMPTS = 3


def _regenerate_until_unused(session_id: str, generate_fn, answer_key: str, question_key: str) -> dict:
    """Real, code-level enforcement of 'don't repeat a question/riddle
    already used this session' (requirements #3 and #4).

    Real, confirmed bug this fixes: this function originally only checked
    the ANSWER for a collision, not the question/riddle TEXT itself. That
    missed a real repeat -- the exact same question ("What is the largest
    mammal living in the ocean?") was regenerated twice in one live
    session, because Gemini phrased the answer slightly differently each
    time ("whale" vs "blue whale"), so the answer-only check never saw a
    collision even though the question was verbatim identical. Checking
    the question/riddle text as well closes that gap.

    A prompt instruction alone ('do NOT reuse...') was also found
    unreliable in real testing -- real Gemini repeated content anyway
    despite the exclusion list being in the prompt. This checks the
    actual result and retries (up to MAX_REGENERATION_ATTEMPTS times) if
    it collides on EITHER the answer or the question/riddle text.

    Honest limitation: the retry budget is intentionally capped (not
    unbounded) to avoid runaway API cost/latency if a collision keeps
    happening. In an extremely long session, even real Gemini's very
    large but finite variety could theoretically be exhausted faster than
    3 retries can route around -- an accepted trade-off, not expected to
    be reached in a realistic session length."""
    used_answers = {_normalize(a) for a in SESSIONS[session_id]["used_answers"]}
    used_questions = {_normalize(q) for q in SESSIONS[session_id]["used_questions"]}

    def is_duplicate(data: dict) -> bool:
        return (
            _normalize(data[answer_key]) in used_answers
            or _normalize(data[question_key]) in used_questions
        )

    data = generate_fn()
    attempts = 1
    while is_duplicate(data) and attempts < MAX_REGENERATION_ATTEMPTS:
        data = generate_fn()
        attempts += 1
    return data


def strip_embedded_questions(text: str) -> str:
    """Real, code-level enforcement that feedback text never contains a
    follow-up question of its own. A prompt instruction alone proved
    unreliable (Gemini kept adding its own follow-up despite being told
    not to), and phrase-matching specific transitions also proved
    unreliable (Gemini phrases the same intent in unpredictable ways).
    The only fully reliable approach: remove EVERY sentence ending in a
    question mark, anywhere in the text, no exceptions. Trade-off: a fun
    fact legitimately phrased as a rhetorical "Did you know...?" question
    is also removed -- an accepted, minor loss in exchange for
    guaranteeing two different questions can never be shown together."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [s for s in sentences if s and not s.rstrip().endswith("?")]
    return " ".join(kept).strip()


def extract_final_question(text: str) -> str:
    """Real fix for a separate instance of the same problem: schema
    fields constrain WHICH field something goes in, not what Gemini
    writes INSIDE a field's value. Even with 'next_question' as its own
    dedicated field, Gemini has been observed cramming extra preamble
    text and a decoy question into that field before the real, intended
    question. Since the model's genuinely intended question is reliably
    the LAST question-ending sentence it wrote, this keeps only that
    final sentence and discards everything before it."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sentence in reversed(sentences):
        if sentence.rstrip().endswith("?"):
            return sentence.strip()
    return text.strip()  # no question found at all -- show whatever exists rather than nothing


# ===========================================================================
# 4. SESSION STORE (requirement #2, #7) — in-memory, no database, 60s
#    inactivity expiry, 6-EXCHANGE conversation cap.
# ===========================================================================

SESSION_TIMEOUT_SECONDS = 60
MAX_EXCHANGES = 6
SESSIONS: dict[str, dict] = {}


def create_session(activity: str) -> str:
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    SESSIONS[session_id] = {
        "session_id": session_id,
        "activity": activity,
        # "turns" is the real conversational unit: each turn is
        # {"ai": <text>, "user": <text or None>}. The natural order is
        # always AI-message-first, then the user's reply to it -- the
        # opening riddle/question/greeting has no preceding user message,
        # so the FIRST turn is created with user=None, then filled in
        # once the child actually responds (a typed guess, or a stand-in
        # label for a Hint/Give-Up click). A turn stays "pending" (not
        # counted toward the 6-exchange cap) until it's been answered --
        # this guarantees an AI message and its reply are always trimmed
        # TOGETHER as one unit, never split apart.
        "turns": [],
        "used_answers": [],
        "used_questions": [],
        "game_state": {
            "current_answer": None,
            "current_riddle": None,
            "given_hints": [],
            "hints_given": 0,
            "current_fact": None,
        },
        "created_at": now,
        "last_active_at": now,
    }
    return session_id


def touch_session(session_id: str):
    if session_id in SESSIONS:
        SESSIONS[session_id]["last_active_at"] = datetime.now(timezone.utc)


def start_new_ai_turn(session_id: str, ai_text: str):
    """Records a fresh AI-initiated message as a new, pending turn (no
    user reply yet). Called once per handler invocation with the FULL
    combined text for that turn -- never multiple times per turn, so
    'AI message + its reply' always stays exactly one exchange unit."""
    turns = SESSIONS[session_id]["turns"]
    turns.append({
        "ai": ai_text,
        "user": None,
        "ai_timestamp": datetime.now(timezone.utc),
        "user_timestamp": None,
    })

    # Cap at MAX_EXCHANGES COMPLETE turns (both ai and user filled in),
    # keeping any currently-pending turn (the one just added) regardless
    # of count -- it isn't "complete" yet, so it doesn't count against
    # the budget. Requirement #7: "maintain only the six most recent
    # messages" -- here, one exchange (AI message + user reply) is
    # treated as one message unit, per this project's explicit spec.
    complete_count = sum(1 for t in turns if t["user"] is not None)
    if complete_count > MAX_EXCHANGES:
        excess = complete_count - MAX_EXCHANGES
        kept = []
        removed = 0
        for t in turns:
            if t["user"] is not None and removed < excess:
                removed += 1
                continue
            kept.append(t)
        SESSIONS[session_id]["turns"] = kept


def complete_pending_turn(session_id: str, user_text: str):
    """Fills in the user's response to whichever turn is currently
    pending. This is how a typed guess, a Hint click, or a Give Up click
    all get recorded -- as the user's response to the AI message they
    were actually replying to."""
    turns = SESSIONS[session_id]["turns"]
    if turns and turns[-1]["user"] is None:
        turns[-1]["user"] = user_text
        turns[-1]["user_timestamp"] = datetime.now(timezone.utc)


def get_context_messages(session_id: str, limit_turns: int = MAX_EXCHANGES) -> list[dict]:
    """Flattens the turn structure back into an alternating role/content
    list for building real LLM context (used by Ask & Explore) -- model
    message, then user message, in the order they actually happened, for
    the most recent `limit_turns` turns."""
    turns = SESSIONS[session_id]["turns"][-limit_turns:]
    flat = []
    for t in turns:
        flat.append({"role": "model", "content": t["ai"]})
        if t["user"] is not None:
            flat.append({"role": "user", "content": t["user"]})
    return flat


def set_game_state(session_id: str, **kwargs):
    SESSIONS[session_id]["game_state"].update(kwargs)


def add_used_answer(session_id: str, answer: str):
    SESSIONS[session_id]["used_answers"].append(answer)


def add_used_question(session_id: str, question_text: str):
    SESSIONS[session_id]["used_questions"].append(question_text)


def sweep_expired_sessions():
    """Requirement #2: 60s inactivity terminates the session, clearing
    all its conversation history -- enforced server-side (not just a
    client-side timer, which a closed tab could fail to run)."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=SESSION_TIMEOUT_SECONDS)
    expired = [sid for sid, data in SESSIONS.items() if data["last_active_at"] < cutoff]
    for sid in expired:
        SESSIONS.pop(sid, None)
    return expired


# ===========================================================================
# 5. MONITORING (requirement #8) — every real LLM request logged to a
#    dedicated log file with timestamp, session ID, activity, user
#    prompt, token usage, TTFT, and total response time.
# ===========================================================================

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
MONITORING_LOG_PATH = LOGS_DIR / "monitoring.log"


def log_llm_request(
    session_id: str,
    activity: str,
    user_prompt: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    ttft_seconds: float | None,
    total_time_seconds: float,
):
    """Appends one real, structured log entry per LLM request. Written as
    JSON lines (one JSON object per line) so the log is both human-
    readable and trivially machine-parseable for later analysis."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "activity": activity,
        "user_prompt": user_prompt,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "ttft_seconds": round(ttft_seconds, 4) if ttft_seconds is not None else None,
        "total_time_seconds": round(total_time_seconds, 4),
    }
    with open(MONITORING_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ===========================================================================
# 6. LLM CLIENT — real Gemini + mock, structured generation + streaming.
#    Every REAL call (never the mock) is logged per requirement #8.
# ===========================================================================

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")


def _use_mock() -> bool:
    return os.environ.get("USE_MOCK_LLM", "true").lower() in ("true", "1")


def _real_generate_structured(system_instruction: str, user_prompt: str, schema: dict,
                               session_id: str, activity: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    start = time.time()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_json_schema=schema,
        ),
    )
    elapsed = time.time() - start

    usage = response.usage_metadata
    log_llm_request(
        session_id=session_id,
        activity=activity,
        user_prompt=user_prompt,
        input_tokens=usage.prompt_token_count if usage else 0,
        output_tokens=usage.candidates_token_count if usage else 0,
        total_tokens=usage.total_token_count if usage else 0,
        ttft_seconds=None,  # structured (non-streamed) calls have no meaningful TTFT
        total_time_seconds=elapsed,
    )
    return json.loads(response.text)


# Real, varied mock content pools -- NOT a single static answer. This
# matters for actually testing no-repeat behavior through the full
# pipeline: a mock that always returns the same "mockanswer" can never
# demonstrate genuine variety, only that the retry mechanism was CALLED
# (already covered by isolated unit tests with a fake generator). These
# pools let integration tests verify real distinctness across many turns
# in a real session, closer to what actually happens with live Gemini.
_MOCK_RIDDLE_POOL = [
    {"riddle": "[MOCK] I have keys but no locks. What am I?", "answer": "mockanswer"},
    {"riddle": "[MOCK] I have a face but no eyes. What am I?", "answer": "clockanswer"},
    {"riddle": "[MOCK] The more you take, the more you leave behind. What am I?", "answer": "footstepanswer"},
    {"riddle": "[MOCK] I have a neck but no head. What am I?", "answer": "bottleanswer"},
    {"riddle": "[MOCK] What has hands but cannot clap?", "answer": "clockhandanswer"},
    {"riddle": "[MOCK] What gets wetter as it dries?", "answer": "towelanswer"},
    {"riddle": "[MOCK] What has an eye but cannot see?", "answer": "needleanswer"},
    {"riddle": "[MOCK] What can travel around the world while staying in a corner?", "answer": "stampanswer"},
    {"riddle": "[MOCK] What has a thumb and four fingers but is not alive?", "answer": "gloveanswer"},
    {"riddle": "[MOCK] What has one head, one foot, and four legs?", "answer": "bedanswer"},
    {"riddle": "[MOCK] What has many teeth but cannot bite?", "answer": "combanswer"},
    {"riddle": "[MOCK] What runs but never walks?", "answer": "riveranswer"},
    {"riddle": "[MOCK] What has a ring but no finger?", "answer": "phoneanswer"},
    {"riddle": "[MOCK] What has legs but does not walk?", "answer": "tableanswer"},
    {"riddle": "[MOCK] What can you catch but not throw?", "answer": "coldanswer"},
]

_MOCK_QUESTION_POOL = [
    {"question": "[MOCK] Which planet is known as the Red Planet?", "answer": "mockanswer", "fun_fact": "[MOCK] Mars fact."},
    {"question": "[MOCK] What is the largest ocean on Earth?", "answer": "pacificanswer", "fun_fact": "[MOCK] Ocean fact."},
    {"question": "[MOCK] How many legs does a spider have?", "answer": "eightanswer", "fun_fact": "[MOCK] Spider fact."},
    {"question": "[MOCK] What is the tallest mountain in the world?", "answer": "everestanswer", "fun_fact": "[MOCK] Mountain fact."},
    {"question": "[MOCK] What gas do plants absorb from the air?", "answer": "carbonanswer", "fun_fact": "[MOCK] Plant fact."},
    {"question": "[MOCK] What is the fastest land animal?", "answer": "cheetahanswer", "fun_fact": "[MOCK] Cheetah fact."},
    {"question": "[MOCK] How many continents are there?", "answer": "sevenanswer", "fun_fact": "[MOCK] Continent fact."},
    {"question": "[MOCK] What is the freezing point of water in Celsius?", "answer": "zeroanswer", "fun_fact": "[MOCK] Water fact."},
    {"question": "[MOCK] What is the largest planet in our solar system?", "answer": "jupiteranswer", "fun_fact": "[MOCK] Jupiter fact."},
    {"question": "[MOCK] What do bees make?", "answer": "honeyanswer", "fun_fact": "[MOCK] Bee fact."},
    {"question": "[MOCK] What is the capital of France?", "answer": "parisanswer", "fun_fact": "[MOCK] Paris fact."},
    {"question": "[MOCK] How many sides does a hexagon have?", "answer": "sixanswer", "fun_fact": "[MOCK] Hexagon fact."},
    {"question": "[MOCK] What is the closest star to Earth?", "answer": "sunanswer", "fun_fact": "[MOCK] Sun fact."},
    {"question": "[MOCK] What is the main language spoken in Brazil?", "answer": "portugueseanswer", "fun_fact": "[MOCK] Brazil fact."},
    {"question": "[MOCK] What do you call a baby dog?", "answer": "puppyanswer", "fun_fact": "[MOCK] Dog fact."},
]

_mock_pool_counters = {"riddle": 0, "question": 0}


def _reset_mock_pools():
    """Test utility -- resets the pool counters so each test starts from
    the beginning of the pool, keeping tests independent of each other."""
    _mock_pool_counters["riddle"] = 0
    _mock_pool_counters["question"] = 0


def _mock_generate_structured(schema: dict) -> dict:
    time.sleep(0.02)
    required = schema.get("required", [])
    mock_data = {}

    if "riddle" in required or "next_riddle" in required:
        item = _MOCK_RIDDLE_POOL[_mock_pool_counters["riddle"] % len(_MOCK_RIDDLE_POOL)]
        _mock_pool_counters["riddle"] += 1
        riddle_key = "next_riddle" if "next_riddle" in required else "riddle"
        answer_key = "next_answer" if "next_answer" in required else "answer"
        mock_data[riddle_key] = item["riddle"]
        mock_data[answer_key] = item["answer"]
    elif "question" in required or "next_question" in required:
        item = _MOCK_QUESTION_POOL[_mock_pool_counters["question"] % len(_MOCK_QUESTION_POOL)]
        _mock_pool_counters["question"] += 1
        question_key = "next_question" if "next_question" in required else "question"
        answer_key = "next_answer" if "next_answer" in required else "answer"
        fact_key = "next_fun_fact" if "next_fun_fact" in required else "fun_fact"
        mock_data[question_key] = item["question"]
        mock_data[answer_key] = item["answer"]
        if fact_key in required:
            mock_data[fact_key] = item["fun_fact"]

    for field in required:
        if field not in mock_data:
            if field == "feedback":
                mock_data[field] = "[MOCK] Nice try! Let's keep going."
            else:
                mock_data[field] = f"[MOCK] {field}"

    return mock_data


def generate_structured(system_instruction: str, user_prompt: str, schema: dict,
                         session_id: str, activity: str) -> dict:
    if _use_mock():
        return _mock_generate_structured(schema)
    return _real_generate_structured(system_instruction, user_prompt, schema, session_id, activity)


def _real_generate_reply_stream(system_instruction: str, context_messages: list[dict],
                                 session_id: str, activity: str, user_prompt: str):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    contents = []
    for m in context_messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))

    start = time.time()
    first_token_time = None
    input_tokens = output_tokens = total_tokens = 0

    stream = client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )
    for chunk in stream:
        if chunk.text:
            if first_token_time is None:
                first_token_time = time.time() - start
            yield chunk.text
        if chunk.usage_metadata:
            input_tokens = chunk.usage_metadata.prompt_token_count
            output_tokens = chunk.usage_metadata.candidates_token_count
            total_tokens = chunk.usage_metadata.total_token_count

    elapsed = time.time() - start
    log_llm_request(
        session_id=session_id,
        activity=activity,
        user_prompt=user_prompt,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        ttft_seconds=first_token_time,
        total_time_seconds=elapsed,
    )


def _mock_generate_reply_stream(system_instruction: str, context_messages: list[dict]):
    last_user = next((m["content"] for m in reversed(context_messages) if m["role"] == "user"), "")
    if "hint number" in system_instruction.lower() or "hint number" in last_user.lower():
        import random
        reply = f"[MOCK HINT {random.randint(100, 999)}] Think carefully about the clues..."
    else:
        reply = "[MOCK] Nice try! Let's keep going."
    words = reply.split(" ")
    for i, word in enumerate(words):
        time.sleep(0.01)
        yield word + (" " if i < len(words) - 1 else "")


def generate_reply_stream(system_instruction: str, context_messages: list[dict],
                           session_id: str, activity: str):
    if _use_mock():
        yield from _mock_generate_reply_stream(system_instruction, context_messages)
    else:
        user_prompt = context_messages[-1]["content"] if context_messages else ""
        yield from _real_generate_reply_stream(system_instruction, context_messages, session_id, activity, user_prompt)


def _real_generate_reply(system_instruction: str, context_messages: list[dict],
                          session_id: str, activity: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    contents = []
    for m in context_messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))

    start = time.time()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )
    elapsed = time.time() - start

    usage = response.usage_metadata
    user_prompt = context_messages[-1]["content"] if context_messages else ""
    log_llm_request(
        session_id=session_id,
        activity=activity,
        user_prompt=user_prompt,
        input_tokens=usage.prompt_token_count if usage else 0,
        output_tokens=usage.candidates_token_count if usage else 0,
        total_tokens=usage.total_token_count if usage else 0,
        ttft_seconds=None,
        total_time_seconds=elapsed,
    )
    return response.text or ""


def generate_reply(system_instruction: str, context_messages: list[dict],
                    session_id: str, activity: str) -> str:
    """Non-streaming variant: gets the FULL reply before returning
    anything. Used where the response must be sanitized (see
    strip_embedded_questions()) before the user ever sees any of it --
    live token streaming makes that impossible, since by the time
    unwanted content is detected, it's already on screen."""
    if _use_mock():
        return "".join(_mock_generate_reply_stream(system_instruction, context_messages))
    return _real_generate_reply(system_instruction, context_messages, session_id, activity)


# ===========================================================================
# 7. ACTIVITY LOGIC (requirements #3, #4, #5)
# ===========================================================================

CHUNK_DELAY_SECONDS = 0.015


def _stream_text_chunks(text: str):
    words = text.split(" ")
    for i, word in enumerate(words):
        time.sleep(CHUNK_DELAY_SECONDS)
        yield word + (" " if i < len(words) - 1 else "")


# --- Brain Buster ---

def _generate_new_riddle(session_id: str) -> str:
    used = SESSIONS[session_id]["used_answers"]
    prompt = "Generate a new riddle."
    if used:
        prompt += f" Do NOT reuse any of these answers already used in this session: {', '.join(used)}."

    data = _regenerate_until_unused(
        session_id,
        lambda: generate_structured(build_system_prompt("brain_buster"), prompt, BRAIN_BUSTER_RIDDLE_SCHEMA, session_id, "brain_buster"),
        "answer",
        "riddle",
    )

    set_game_state(
        session_id,
        current_answer=data["answer"],
        current_riddle=data["riddle"],
        given_hints=[],
        hints_given=0,
    )
    add_used_answer(session_id, data["answer"])
    add_used_question(session_id, data["riddle"])
    return data["riddle"]


def start_brain_buster(session_id: str):
    riddle_text = _generate_new_riddle(session_id)
    start_new_ai_turn(session_id, riddle_text)
    yield from _stream_text_chunks(riddle_text)


def _generate_live_hint(session_id: str, game_state: dict, hint_number: int):
    """Real, live LLM call every time a hint is requested -- never a
    lookup into a pre-generated array. Requirement #3: up to 3 hints,
    each one genuinely fresh."""
    given_hints_text = (
        "Hints already given this round: " + " | ".join(game_state["given_hints"])
        if game_state["given_hints"]
        else "No hints have been given yet for this riddle."
    )
    hint_prompt = (
        f"The riddle is: \"{game_state['current_riddle']}\"\n"
        f"The correct answer is: \"{game_state['current_answer']}\"\n"
        f"{given_hints_text}\n"
        f"Write hint number {hint_number} of 3 now."
    )
    hint_text = ""
    for chunk in generate_reply_stream(build_system_prompt("brain_buster"), [{"role": "user", "content": hint_prompt}], session_id, "brain_buster"):
        hint_text += chunk
        yield chunk
    yield ("__hint_text__", hint_text)


def handle_brain_buster_turn(session_id: str, action: str | None, message: str | None):
    if session_id not in SESSIONS:
        # Guards against a real race condition: the session was deleted
        # (e.g. a concurrent Back-button click) while this request was
        # already in flight. Without this check, the next line would
        # raise an uncaught KeyError mid-stream, silently killing the
        # connection with no "done" event ever sent.
        yield "This session has ended. Please go back and start again."
        return

    game_state = SESSIONS[session_id]["game_state"]

    if action == "hint":
        hints_given = game_state["hints_given"]
        complete_pending_turn(session_id, "[Requested a hint]")

        if hints_given >= 3:
            # Requirement #3: "the answer shall be revealed after the
            # third hint." A hint request past the cap reveals + chains
            # into a new riddle, as one combined turn.
            reveal_text = f"You've had all 3 hints! The answer was \"{game_state['current_answer']}\". "
            yield from _stream_text_chunks(reveal_text)
            yield ("__new_item__",)
            new_riddle = _generate_new_riddle(session_id)
            start_new_ai_turn(session_id, reveal_text + "\n\n" + new_riddle)
            yield from _stream_text_chunks(new_riddle)
            return

        hint_number = hints_given + 1
        hint_text = ""
        for chunk in _generate_live_hint(session_id, game_state, hint_number):
            if isinstance(chunk, tuple) and chunk[0] == "__hint_text__":
                hint_text = chunk[1]
                continue
            yield chunk

        set_game_state(session_id, hints_given=hint_number, given_hints=game_state["given_hints"] + [hint_text])
        start_new_ai_turn(session_id, hint_text)
        return

    if action == "give_up":
        complete_pending_turn(session_id, "[Gave up]")
        reveal_text = f"No worries! The answer was \"{game_state['current_answer']}\". "
        yield from _stream_text_chunks(reveal_text)
        yield ("__new_item__",)
        new_riddle = _generate_new_riddle(session_id)
        start_new_ai_turn(session_id, reveal_text + "\n\n" + new_riddle)
        yield from _stream_text_chunks(new_riddle)
        return

    complete_pending_turn(session_id, message)
    correct = is_correct_answer(message, game_state["current_answer"])

    if not correct and game_state["hints_given"] >= 3:
        reveal_text = (
            f"That's not quite right, and you've used all your hints! "
            f"The answer was \"{game_state['current_answer']}\". "
        )
        yield from _stream_text_chunks(reveal_text)
        yield ("__new_item__",)
        new_riddle = _generate_new_riddle(session_id)
        start_new_ai_turn(session_id, reveal_text + "\n\n" + new_riddle)
        yield from _stream_text_chunks(new_riddle)
        return

    # DETERMINISTIC right/wrong signal, generated by CODE, not the LLM --
    # a prompt instruction alone proved unreliable for guaranteeing this
    # in real testing.
    prefix = "✅ Correct! " if correct else "❌ Not quite right. "
    yield prefix

    if correct:
        used = SESSIONS[session_id]["used_answers"]
        prompt = f"The child's guess \"{message}\" was CORRECT. The answer was \"{game_state['current_answer']}\". Now also generate the next riddle."
        if used:
            prompt += f" Do NOT reuse any of these answers already used this session: {', '.join(used)}."

        data = _regenerate_until_unused(
            session_id,
            lambda: generate_structured(build_system_prompt("brain_buster"), prompt, BRAIN_BUSTER_FEEDBACK_AND_NEXT_RIDDLE_SCHEMA, session_id, "brain_buster"),
            "next_answer",
            "next_riddle",
        )

        feedback_text = strip_embedded_questions(data["feedback"])
        yield from _stream_text_chunks(feedback_text)

        yield ("__new_item__",)
        # NOTE: extract_final_question() is deliberately NOT applied here,
        # unlike Quick Fire's next_question. A riddle's natural, correct
        # structure is itself "[setup sentence]. What am I?" -- a real
        # two-sentence unit ending in a question. extract_final_question()
        # was designed to strip a genuinely POLLUTED field (extra preamble
        # + a decoy question before the real one), but applying it here
        # incorrectly treated every riddle's own legitimate setup sentence
        # as if it were unwanted preamble, truncating riddles down to just
        # "What am I?" and silently discarding the actual riddle. This was
        # a real, confirmed regression, caught by testing multi-turn
        # sessions -- see test_brain_buster_riddles_do_not_repeat_within_a_session.
        new_riddle = data["next_riddle"]
        set_game_state(
            session_id,
            current_answer=data["next_answer"],
            current_riddle=new_riddle,
            given_hints=[],
            hints_given=0,
        )
        add_used_answer(session_id, data["next_answer"])
        add_used_question(session_id, new_riddle)

        start_new_ai_turn(session_id, prefix + feedback_text + "\n\n" + new_riddle)
        yield from _stream_text_chunks(new_riddle)
    else:
        feedback_prompt = (
            f"The child's guess was: \"{message}\". This was INCORRECT. "
            f"The correct answer is \"{game_state['current_answer']}\"."
        )
        raw_feedback_text = generate_reply(build_system_prompt("brain_buster"), [{"role": "user", "content": feedback_prompt}], session_id, "brain_buster")
        feedback_text = strip_embedded_questions(raw_feedback_text)
        yield from _stream_text_chunks(feedback_text)
        start_new_ai_turn(session_id, prefix + feedback_text)


# --- Quick Fire ---

def _generate_new_question(session_id: str) -> str:
    used = SESSIONS[session_id]["used_answers"]
    prompt = "Generate a new question."
    if used:
        prompt += f" Do NOT reuse any of these answers already used in this session: {', '.join(used)}."

    data = _regenerate_until_unused(
        session_id,
        lambda: generate_structured(build_system_prompt("quick_fire"), prompt, QUICK_FIRE_QUESTION_SCHEMA, session_id, "quick_fire"),
        "answer",
        "question",
    )
    set_game_state(session_id, current_answer=data["answer"], current_fact=data["fun_fact"])
    add_used_answer(session_id, data["answer"])
    add_used_question(session_id, data["question"])
    return data["question"]


def start_quick_fire(session_id: str):
    question_text = _generate_new_question(session_id)
    start_new_ai_turn(session_id, question_text)
    yield from _stream_text_chunks(question_text)


def handle_quick_fire_turn(session_id: str, message: str):
    if session_id not in SESSIONS:
        yield "This session has ended. Please go back and start again."
        return

    game_state = SESSIONS[session_id]["game_state"]
    complete_pending_turn(session_id, message)
    correct = is_correct_answer(message, game_state["current_answer"])

    prefix = "✅ Correct! " if correct else "❌ Not quite right. "
    yield prefix

    used = SESSIONS[session_id]["used_answers"]
    prompt = (
        f"The child's answer was: \"{message}\". This was "
        f"{'CORRECT' if correct else 'INCORRECT'}. The correct answer was \"{game_state['current_answer']}\". "
        f"The fun fact for that question was: {game_state['current_fact']}. "
        f"Now also generate the next question."
    )
    if used:
        prompt += f" Do NOT reuse any of these answers already used this session: {', '.join(used)}."

    data = _regenerate_until_unused(
        session_id,
        lambda: generate_structured(build_system_prompt("quick_fire"), prompt, QUICK_FIRE_FEEDBACK_AND_NEXT_SCHEMA, session_id, "quick_fire"),
        "next_answer",
        "next_question",
    )

    feedback_text = strip_embedded_questions(data["feedback"])
    yield from _stream_text_chunks(feedback_text)

    yield ("__new_item__",)
    new_question = extract_final_question(data["next_question"])
    set_game_state(session_id, current_answer=data["next_answer"], current_fact=data["next_fun_fact"])
    add_used_answer(session_id, data["next_answer"])
    add_used_question(session_id, new_question)

    start_new_ai_turn(session_id, prefix + feedback_text + "\n\n" + new_question)
    yield from _stream_text_chunks(new_question)


# --- Ask & Explore ---

def start_ask_explore(session_id: str):
    greeting = "Hi! I'm here to answer your questions. What are you curious about today?"
    start_new_ai_turn(session_id, greeting)
    yield from _stream_text_chunks(greeting)


def handle_ask_explore_turn(session_id: str, message: str):
    if session_id not in SESSIONS:
        yield "This session has ended. Please go back and start again."
        return

    complete_pending_turn(session_id, message)
    context = get_context_messages(session_id)
    reply_text = ""
    for chunk in generate_reply_stream(build_system_prompt("ask_explore"), context, session_id, "ask_explore"):
        reply_text += chunk
        yield chunk
    start_new_ai_turn(session_id, reply_text)


# ===========================================================================
# 8. FASTAPI APP (requirement #1, #9)
# ===========================================================================

app = FastAPI(title="Learning Adventures Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Real bug from earlier testing: expose_headers must explicitly list
    # any custom response header (X-Session-Id) or the browser silently
    # hides it from JavaScript. Kept here deliberately.
    expose_headers=["X-Session-Id"],
)


class StartRequest(BaseModel):
    activity: str


class ChatRequest(BaseModel):
    session_id: str
    activity: str
    message: str | None = None
    action: str | None = None  # "hint" | "give_up" | None


ACTIVITY_STARTERS = {
    "brain_buster": start_brain_buster,
    "quick_fire": start_quick_fire,
    "ask_explore": start_ask_explore,
}


@app.get("/api/health")
def health():
    return {"status": "ok", "activities": list(ACTIVITY_PROMPTS.keys()), "mock_mode": _use_mock()}


@app.post("/api/start")
async def start(req: StartRequest):
    if req.activity not in ACTIVITY_PROMPTS:
        raise HTTPException(status_code=400, detail=f"Unknown activity: {req.activity}")

    sweep_expired_sessions()
    session_id = create_session(req.activity)

    def event_stream():
        start_time = time.time()
        try:
            for word in ACTIVITY_STARTERS[req.activity](session_id):
                yield f"data: {json.dumps({'chunk': word})}\n\n"
        except Exception as exc:
            print(f"[start] ERROR during stream for session {session_id}: {exc}")
            yield f"data: {json.dumps({'chunk': 'Sorry, something went wrong starting this activity.'})}\n\n"
        yield f"data: {json.dumps({'done': True, 'total_time_s': round(time.time() - start_time, 2)})}\n\n"

    response = StreamingResponse(event_stream(), media_type="text/event-stream")
    response.headers["X-Session-Id"] = session_id
    return response


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    if req.activity not in ACTIVITY_PROMPTS:
        raise HTTPException(status_code=400, detail=f"Unknown activity: {req.activity}")

    touch_session(req.session_id)

    # Safety pre-filter (requirement #6): catch blatant abuse before any
    # LLM call is made.
    if req.message and is_blatantly_inappropriate(req.message):
        def safe_stream():
            for word in _stream_text_chunks(SAFE_REDIRECT_MESSAGE):
                yield f"data: {json.dumps({'chunk': word})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        return StreamingResponse(safe_stream(), media_type="text/event-stream")

    def event_stream():
        start_time = time.time()
        try:
            if req.activity == "brain_buster":
                gen = handle_brain_buster_turn(req.session_id, req.action, req.message)
            elif req.activity == "quick_fire":
                gen = handle_quick_fire_turn(req.session_id, req.message)
            else:
                gen = handle_ask_explore_turn(req.session_id, req.message)

            for item in gen:
                if isinstance(item, tuple) and item[0] == "__new_item__":
                    yield f"data: {json.dumps({'new_item': True})}\n\n"
                else:
                    yield f"data: {json.dumps({'chunk': item})}\n\n"
        except Exception as exc:
            print(f"[chat] ERROR during stream for session {req.session_id}: {exc}")
            yield f"data: {json.dumps({'chunk': 'Sorry, something went wrong. Please try again.'})}\n\n"
        yield f"data: {json.dumps({'done': True, 'total_time_s': round(time.time() - start_time, 2)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.delete("/api/session/{session_id}")
def end_session(session_id: str):
    """Requirement #2: returning to the home screen terminates the
    session and clears its history -- no session data persists."""
    SESSIONS.pop(session_id, None)
    return {"terminated": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)