"""
llm_client.py
----------------
Wraps the actual Gemini API call in a single function, generate_reply(),
so the FastAPI handler in main.py doesn't need to know anything about
Gemini specifically -- it just calls generate_reply(history) and gets
back (text, usage_dict).

WHY THIS IS ITS OWN MODULE, SEPARATE FROM main.py:
This is the exact same "separate concerns" principle from Day 6's prompt
template library and Day 9's tool schema/function separation -- isolating
the external API call makes it possible to swap in a MOCK implementation
for testing (see USE_MOCK_LLM below) without touching any routing or
session logic. This is standard production practice: you should be able
to test your API's request handling, validation, session management, and
error paths WITHOUT spending real API credit or depending on network
availability every time you run your test suite.
"""

import os
import time

USE_MOCK_LLM = os.environ.get("USE_MOCK_LLM", "false").lower() == "true"

MODEL = "gemini-3.5-flash-lite"


def _real_generate_reply(history):
    """
    Calls the real Gemini API with the full conversation history.
    history: list of {"role": ..., "content": ..., "timestamp": ...} dicts.
    Returns (reply_text, usage_dict).
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    system_messages = [m for m in history if m["role"] == "system"]
    system_instruction = system_messages[0]["content"] if system_messages else None

    contents = []
    for m in history:
        if m["role"] == "system":
            continue
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )

    usage = {
        "prompt_tokens": response.usage_metadata.prompt_token_count,
        "completion_tokens": response.usage_metadata.candidates_token_count,
        "total_tokens": response.usage_metadata.total_token_count,
    }
    return response.text, usage


def _mock_generate_reply(history):
    """
    A deterministic mock used for testing the FastAPI server's routing,
    session management, and error handling WITHOUT any real API call --
    genuinely useful (not just a sandbox workaround) since it lets the
    full request/response/session cycle be tested quickly, repeatably,
    and without cost. Enabled via USE_MOCK_LLM=true.
    """
    time.sleep(0.05)
    last_user_message = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    reply = f'[MOCK REPLY] You said: "{last_user_message}" -- this is a simulated response for testing.'
    usage = {
        "prompt_tokens": sum(len(m["content"].split()) for m in history),
        "completion_tokens": len(reply.split()),
        "total_tokens": 0,
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return reply, usage


def generate_reply(history):
    """
    The single entry point main.py calls. Routes to the real or mock
    implementation based on the USE_MOCK_LLM environment variable.
    """
    if USE_MOCK_LLM:
        return _mock_generate_reply(history)
    return _real_generate_reply(history)


# ============================================================
# Streaming support (carried forward from the Day 12 improvement)
# ============================================================
# See Day 12's REPORT.md for the full rationale: streaming and
# non-streaming need genuinely different function shapes (a generator
# yielding chunks vs. a single return value), so they're kept as
# separate functions rather than one function behaving two ways.

def _real_generate_reply_stream(history):
    """
    Calls Gemini's real streaming endpoint. Yields each text chunk AS IT
    ARRIVES, then yields a final ("__usage__", usage_dict) tuple once the
    stream completes.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    system_messages = [m for m in history if m["role"] == "system"]
    system_instruction = system_messages[0]["content"] if system_messages else None

    contents = []
    for m in history:
        if m["role"] == "system":
            continue
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))

    stream = client.models.generate_content_stream(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )

    usage = None
    for chunk in stream:
        if chunk.text:
            yield chunk.text
        if chunk.usage_metadata:
            usage = {
                "prompt_tokens": chunk.usage_metadata.prompt_token_count,
                "completion_tokens": chunk.usage_metadata.candidates_token_count,
                "total_tokens": chunk.usage_metadata.total_token_count,
            }

    yield ("__usage__", usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


def _mock_generate_reply_stream(history):
    """Simulates streaming by yielding the mock reply word-by-word."""
    last_user_message = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    reply = f'[MOCK STREAM] You said: "{last_user_message}" -- streaming this back one word at a time for testing.'
    words = reply.split(" ")

    for i, word in enumerate(words):
        time.sleep(0.05)
        yield word + (" " if i < len(words) - 1 else "")

    usage = {
        "prompt_tokens": sum(len(m["content"].split()) for m in history),
        "completion_tokens": len(words),
        "total_tokens": 0,
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    yield ("__usage__", usage)


def generate_reply_stream(history):
    """The streaming entry point main.py's /api/chat/stream route calls."""
    if USE_MOCK_LLM:
        yield from _mock_generate_reply_stream(history)
    else:
        yield from _real_generate_reply_stream(history)


# ============================================================
# Day 13: Auto-generated session titles
# ============================================================

TITLE_SYSTEM_PROMPT = (
    "You generate short chat titles. Given the first exchange of a "
    "conversation, output ONLY a 3-5 word title summarizing the topic -- "
    "no punctuation, no quotes, no explanation, just the title itself."
)


def _real_generate_title(user_message, assistant_reply):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = f"User: {user_message}\nAssistant: {assistant_reply}\n\nTitle:"
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=TITLE_SYSTEM_PROMPT),
    )
    # Defensive cleanup: strip quotes/trailing punctuation a model might add
    # despite the instruction not to -- the same "never fully trust the
    # model's literal compliance" principle from Day 9's JSON validation.
    title = response.text.strip().strip('"').strip("'").rstrip(".")
    return title


def _mock_generate_title(user_message, assistant_reply):
    """Deterministic mock title generation, for offline testing."""
    words = user_message.split()[:4]
    return " ".join(words).title() if words else "New Conversation"


def generate_title(user_message, assistant_reply):
    """
    Generates a short (3-5 word) title summarizing a conversation's first
    exchange -- called once, after the FIRST successful reply in a new
    session, per the task specification. Routes to mock/real the same way
    generate_reply() does.
    """
    if USE_MOCK_LLM:
        return _mock_generate_title(user_message, assistant_reply)
    return _real_generate_title(user_message, assistant_reply)
