"""
llm_client.py
----------------
Wraps every Gemini interaction this app needs:
  - generate_structured(): schema-enforced JSON generation (Day 9's
    pattern), used for riddle/question generation -- reliable game data,
    not parsed from free text.
  - generate_reply_stream(): real, live token-by-token streaming (Day 12
    /13's pattern), used for feedback messages and Ask & Explore answers.

USE_MOCK_LLM (same pattern as Days 11-13) lets the entire application --
routing, session logic, safety filtering, monitoring -- be tested end to
end with zero API cost and zero network dependency, while the real
Gemini integration remains fully implemented for local use with a key.
"""

import os
import time
import json

USE_MOCK_LLM = os.environ.get("USE_MOCK_LLM", "false").lower() == "true"
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")


def _real_generate_structured(system_instruction, user_prompt, schema):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_json_schema=schema,
        ),
    )
    data = json.loads(response.text)
    usage = {
        "prompt_tokens": response.usage_metadata.prompt_token_count,
        "completion_tokens": response.usage_metadata.candidates_token_count,
        "total_tokens": response.usage_metadata.total_token_count,
    }
    return data, usage


def _mock_generate_structured(system_instruction, user_prompt, schema):
    """Deterministic mock structured generation for offline testing."""
    time.sleep(0.02)
    required = schema.get("required", [])
    mock_data = {}
    for field in required:
        prop = schema["properties"][field]
        if prop.get("type") == "ARRAY":
            mock_data[field] = ["mock hint 1", "mock hint 2", "mock hint 3"]
        elif field == "answer":
            mock_data[field] = "mockanswer"
        elif field == "topic":
            mock_data[field] = "general knowledge"
        else:
            mock_data[field] = f"[MOCK] {field} for testing"
    usage = {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35}
    return mock_data, usage


def generate_structured(system_instruction, user_prompt, schema):
    if USE_MOCK_LLM:
        return _mock_generate_structured(system_instruction, user_prompt, schema)
    return _real_generate_structured(system_instruction, user_prompt, schema)


def _real_generate_reply_stream(system_instruction, context_messages):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    contents = []
    for m in context_messages:
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


def _mock_generate_reply_stream(system_instruction, context_messages):
    last_user = next((m["content"] for m in reversed(context_messages) if m["role"] == "user"), "")
    reply = f'[MOCK] Great try with "{last_user}"! Let\'s keep going.'
    words = reply.split(" ")
    for i, word in enumerate(words):
        time.sleep(0.02)
        yield word + (" " if i < len(words) - 1 else "")
    usage = {"prompt_tokens": 15, "completion_tokens": len(words), "total_tokens": 15 + len(words)}
    yield ("__usage__", usage)


def generate_reply_stream(system_instruction, context_messages):
    if USE_MOCK_LLM:
        yield from _mock_generate_reply_stream(system_instruction, context_messages)
    else:
        yield from _real_generate_reply_stream(system_instruction, context_messages)
