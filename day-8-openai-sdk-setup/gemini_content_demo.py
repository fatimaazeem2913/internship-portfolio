"""
gemini_content_demo.py
--------------------------
Instantiates the Gemini client, sends a structured prompt to
gemini-2.5-flash via the generateContent API (Gemini's traditional,
stateless, single-shot endpoint), captures the full response, and
computes the exact USD cost of the request.

WHY GEMINI INSTEAD OF OPENAI: Gemini's API is free to use for
development (Google AI Studio issues a key with no credit card required,
~1,500 requests/day on gemini-2.5-flash), unlike OpenAI which requires
billing to be enabled before any request succeeds (the 429
insufficient_quota error documented in SETUP_GUIDE.md). The underlying
concepts this task asks about -- structured prompts, full JSON responses,
token metrics, cost calculation -- are provider-agnostic; only the SDK
calls and response shape differ. Correct OpenAI reference code is
preserved in chat_completions_demo.py / responses_api_demo.py /
streaming_demo.py for local use with your own billed OpenAI account.

SETUP:
    pip install google-genai
    Get a free key at https://aistudio.google.com/apikey (no credit card)
    export GEMINI_API_KEY="your-key-here"
    python3 gemini_content_demo.py
"""

import os
import json
from google import genai
from google.genai import types
from token_cost_calculator import calculate_cost, format_cost_report

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = "You are a precise technical assistant. Answer in exactly 2 sentences."
USER_PROMPT = "Explain what a REST API is."


def run_generate_content(system_instruction, user_prompt, model=MODEL, temperature=0.3):
    """Send a structured prompt and return the full response object."""
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        ),
    )
    return response


def response_to_dict(response):
    """
    Build a JSON-serializable snapshot of the key response fields.
    (The raw response object contains non-JSON-serializable protobuf
    internals, so we extract the documented, stable fields explicitly
    rather than relying on a generic serializer.)
    """
    return {
        "model": MODEL,
        "text": response.text,
        "usage_metadata": {
            "prompt_token_count": response.usage_metadata.prompt_token_count,
            "candidates_token_count": response.usage_metadata.candidates_token_count,
            "total_token_count": response.usage_metadata.total_token_count,
        },
        "finish_reason": str(response.candidates[0].finish_reason) if response.candidates else None,
    }


if __name__ == "__main__":
    print("=" * 90)
    print(f"GEMINI generateContent API DEMO ({MODEL})")
    print("=" * 90)

    response = run_generate_content(SYSTEM_INSTRUCTION, USER_PROMPT)

    full_json = json.dumps(response_to_dict(response), indent=2)
    print("\n--- FULL JSON RESPONSE (key fields) ---")
    print(full_json)

    with open("outputs/gemini_content_full_response.json", "w", encoding="utf-8") as f:
        f.write(full_json)

    print(f"\n--- ANSWER ---\n{response.text}")

    print("\n--- TOKEN METRICS AND COST ---")
    usage = response.usage_metadata
    print(f"prompt_token_count:     {usage.prompt_token_count}")
    print(f"candidates_token_count: {usage.candidates_token_count}")
    print(f"total_token_count:      {usage.total_token_count}")

    cost = calculate_cost(usage, model=MODEL)
    print("\n" + format_cost_report(cost))

    print("\nFull JSON response saved to outputs/gemini_content_full_response.json")
