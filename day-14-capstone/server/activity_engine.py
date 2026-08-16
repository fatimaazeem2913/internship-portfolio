"""
activity_engine.py
----------------------
The actual game logic for all three activities. Kept separate from
main.py's routing (same separation-of-concerns principle as every
previous day's llm_client.py / chatApi.js split).

Every function here is a GENERATOR yielding a mix of:
  - plain strings: text chunks to stream to the frontend
  - ("__new_item__",): a marker separating "feedback for the last turn"
    from "here's a new riddle/question" within one response, so the
    frontend can render them as two visually distinct blocks
  - ("__usage__", dict): the ACCUMULATED token usage across every real
    LLM call made during this turn (a turn can involve 2 calls -- e.g.
    a correct guess triggers both a feedback message AND a new riddle --
    so usage is summed here rather than left for the caller to track)

WHY PRE-GENERATED CONTENT IS STREAMED VIA CHUNKING RATHER THAN A SECOND
LLM CALL: the riddle/question text itself comes from generate_structured()
(Day 9's schema-enforced pattern) -- it already exists as a complete,
correct string before any streaming happens. Rather than spending a
SECOND LLM call just to "narrate" text we already have, it's chunked
(split into words with a tiny delay) and delivered exactly the same way
a live stream would be. This keeps the streaming UX uniform across the
whole app while only paying for genuine LLM generation where it adds
real value (feedback messages, which should be warm and VARIED each
time, and Ask & Explore's answers).
"""

import time

from activities import ACTIVITIES
from llm_client import generate_structured, generate_reply_stream
from answer_checking import is_correct_answer
import session_store

CHUNK_DELAY_SECONDS = 0.015


def _stream_text_chunks(text):
    words = text.split(" ")
    for i, word in enumerate(words):
        time.sleep(CHUNK_DELAY_SECONDS)
        yield word + (" " if i < len(words) - 1 else "")


def _merge_usage(a, b):
    return {
        "prompt_tokens": a.get("prompt_tokens", 0) + b.get("prompt_tokens", 0),
        "completion_tokens": a.get("completion_tokens", 0) + b.get("completion_tokens", 0),
        "total_tokens": a.get("total_tokens", 0) + b.get("total_tokens", 0),
    }


# ============================================================
# BRAIN BUSTER
# ============================================================

def _generate_new_riddle(session_id):
    used = session_store.get_used_answers(session_id)
    config = ACTIVITIES["brain_buster"]
    prompt = "Generate a new riddle."
    if used:
        prompt += f" Do NOT reuse any of these answers already used in this session: {', '.join(used)}."

    data, usage = generate_structured(config["generation_system"], prompt, config["schema"])

    session_store.set_game_state(
        session_id,
        current_answer=data["answer"],
        current_hints=data["hints"][:3],
        hints_given=0,
    )
    session_store.add_used_answer(session_id, data["answer"])
    return data["riddle"], usage


def start_brain_buster(session_id):
    riddle_text, usage = _generate_new_riddle(session_id)
    session_store.append_message(session_id, "model", riddle_text)
    yield from _stream_text_chunks(riddle_text)
    yield ("__usage__", usage)


def handle_brain_buster_turn(session_id, action, message):
    game_state = session_store.get_game_state(session_id)
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if action == "hint":
        hints_given = game_state["hints_given"]
        if hints_given >= 3:
            reveal = f"You've had all 3 hints! The answer was \"{game_state['current_answer']}\". "
            yield from _stream_text_chunks(reveal)
        else:
            hint_text = game_state["current_hints"][hints_given]
            session_store.set_game_state(session_id, hints_given=hints_given + 1)
            yield from _stream_text_chunks(f"Hint {hints_given + 1}: {hint_text} ")

            if hints_given + 1 == 3:
                reveal = f"That was the last hint! The answer was \"{game_state['current_answer']}\". "
                yield from _stream_text_chunks(reveal)
                yield ("__new_item__",)
                new_riddle, usage = _generate_new_riddle(session_id)
                session_store.append_message(session_id, "model", new_riddle)
                yield from _stream_text_chunks(new_riddle)
                total_usage = _merge_usage(total_usage, usage)
        yield ("__usage__", total_usage)
        return

    if action == "give_up":
        reveal = f"No worries! The answer was \"{game_state['current_answer']}\". "
        yield from _stream_text_chunks(reveal)
        yield ("__new_item__",)
        new_riddle, usage = _generate_new_riddle(session_id)
        session_store.append_message(session_id, "model", new_riddle)
        yield from _stream_text_chunks(new_riddle)
        yield ("__usage__", usage)
        return

    session_store.append_message(session_id, "user", message)
    correct = is_correct_answer(message, game_state["current_answer"])

    feedback_prompt = (
        f"The child's guess was: \"{message}\". This was "
        f"{'CORRECT' if correct else 'INCORRECT'}. The correct answer is \"{game_state['current_answer']}\"."
    )
    config = ACTIVITIES["brain_buster"]
    feedback_text = ""
    for chunk in generate_reply_stream(config["feedback_system"], [{"role": "user", "content": feedback_prompt}]):
        if isinstance(chunk, tuple) and chunk[0] == "__usage__":
            total_usage = _merge_usage(total_usage, chunk[1])
            continue
        feedback_text += chunk
        yield chunk
    session_store.append_message(session_id, "model", feedback_text)

    if correct:
        yield ("__new_item__",)
        new_riddle, usage = _generate_new_riddle(session_id)
        session_store.append_message(session_id, "model", new_riddle)
        yield from _stream_text_chunks(new_riddle)
        total_usage = _merge_usage(total_usage, usage)

    yield ("__usage__", total_usage)


# ============================================================
# QUICK FIRE
# ============================================================

def _generate_new_question(session_id):
    used = session_store.get_used_answers(session_id)
    config = ACTIVITIES["quick_fire"]
    prompt = "Generate a new question."
    if used:
        prompt += f" Do NOT reuse any of these answers already used in this session: {', '.join(used)}."

    data, usage = generate_structured(config["generation_system"], prompt, config["schema"])

    session_store.set_game_state(session_id, current_answer=data["answer"], current_fact=data["fun_fact"])
    session_store.add_used_answer(session_id, data["answer"])
    return data["question"], usage


def start_quick_fire(session_id):
    question_text, usage = _generate_new_question(session_id)
    session_store.append_message(session_id, "model", question_text)
    yield from _stream_text_chunks(question_text)
    yield ("__usage__", usage)


def handle_quick_fire_turn(session_id, message):
    game_state = session_store.get_game_state(session_id)
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    session_store.append_message(session_id, "user", message)
    correct = is_correct_answer(message, game_state["current_answer"])

    feedback_prompt = (
        f"The child's answer was: \"{message}\". This was "
        f"{'CORRECT' if correct else 'INCORRECT'}. The correct answer is \"{game_state['current_answer']}\". "
        f"Fun fact: {game_state['current_fact']}"
    )
    config = ACTIVITIES["quick_fire"]
    feedback_text = ""
    for chunk in generate_reply_stream(config["feedback_system"], [{"role": "user", "content": feedback_prompt}]):
        if isinstance(chunk, tuple) and chunk[0] == "__usage__":
            total_usage = _merge_usage(total_usage, chunk[1])
            continue
        feedback_text += chunk
        yield chunk
    session_store.append_message(session_id, "model", feedback_text)

    yield ("__new_item__",)
    new_question, usage = _generate_new_question(session_id)
    session_store.append_message(session_id, "model", new_question)
    yield from _stream_text_chunks(new_question)
    total_usage = _merge_usage(total_usage, usage)

    yield ("__usage__", total_usage)


# ============================================================
# ASK & EXPLORE
# ============================================================

def start_ask_explore(session_id):
    greeting = "Hi! I'm here to answer your questions. What are you curious about today?"
    session_store.append_message(session_id, "model", greeting)
    yield from _stream_text_chunks(greeting)
    yield ("__usage__", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


def handle_ask_explore_turn(session_id, message):
    session_store.append_message(session_id, "user", message)
    context = session_store.get_context_messages(session_id, limit=6)

    config = ACTIVITIES["ask_explore"]
    reply_text = ""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for chunk in generate_reply_stream(config["system"], context):
        if isinstance(chunk, tuple) and chunk[0] == "__usage__":
            usage = chunk[1]
            continue
        reply_text += chunk
        yield chunk
    session_store.append_message(session_id, "model", reply_text)
    yield ("__usage__", usage)
