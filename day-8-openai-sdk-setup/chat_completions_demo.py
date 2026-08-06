"""
chat_completions_demo.py
----------------------------
Instantiates the OpenAI() client, sends a structured prompt to gpt-4o-mini
via the Chat Completions API, captures the full JSON response, and
computes the exact USD cost of the request.

SETUP: this needs your own OpenAI API key with billing enabled (see
SETUP_GUIDE.md). Run:
    export OPENAI_API_KEY="sk-...your-key-here..."
    python3 chat_completions_demo.py
"""

import os
import json
from openai import OpenAI
from token_cost_calculator import calculate_cost, format_cost_report

client = OpenAI()  # reads OPENAI_API_KEY from the environment automatically

MODEL = "gpt-4o-mini"

STRUCTURED_PROMPT = [
    {
        "role": "system",
        "content": "You are a precise technical assistant. Answer in exactly 2 sentences."
    },
    {
        "role": "user",
        "content": "Explain what a REST API is."
    },
]


def run_chat_completion(messages, model=MODEL, temperature=0.3):
    """Send a structured prompt and return the full response object."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response


if __name__ == "__main__":
    print("=" * 90)
    print(f"CHAT COMPLETIONS API DEMO ({MODEL})")
    print("=" * 90)

    response = run_chat_completion(STRUCTURED_PROMPT)

    full_json = response.model_dump_json(indent=2)
    print("\n--- FULL JSON RESPONSE ---")
    print(full_json)

    with open("outputs/chat_completions_full_response.json", "w", encoding="utf-8") as f:
        f.write(full_json)

    answer = response.choices[0].message.content
    print(f"\n--- ANSWER ---\n{answer}")

    print("\n--- TOKEN METRICS AND COST ---")
    print(f"prompt_tokens:     {response.usage.prompt_tokens}")
    print(f"completion_tokens: {response.usage.completion_tokens}")
    print(f"total_tokens:      {response.usage.total_tokens}")

    cost = calculate_cost(response.usage, model=MODEL)
    print("\n" + format_cost_report(cost))

    print("\nFull JSON response saved to outputs/chat_completions_full_response.json")
