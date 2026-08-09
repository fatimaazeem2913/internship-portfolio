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
