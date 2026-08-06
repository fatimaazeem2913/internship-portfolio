"""
responses_api_demo.py
-------------------------
Implements the same request using OpenAI's newer Responses API, for
direct structural comparison against chat_completions_demo.py.

SETUP: same as chat_completions_demo.py -- needs OPENAI_API_KEY set.
    python3 responses_api_demo.py
"""

from openai import OpenAI
from token_cost_calculator import calculate_cost, format_cost_report

client = OpenAI()

MODEL = "gpt-4o-mini"


def run_responses_api(instructions, input_text, model=MODEL, temperature=0.3):
    """
    The Responses API's equivalent structured call. Key structural
    differences from Chat Completions, used below:
      - instructions replaces the system-role message (a dedicated,
        top-level parameter instead of a message in an array)
      - input replaces messages -- can be a plain string for simple
        cases, or a list of role/content dicts for multi-turn context
      - the response object has a flat .output_text shortcut instead of
        needing .choices[0].message.content
    """
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=input_text,
        temperature=temperature,
    )
    return response


if __name__ == "__main__":
    print("=" * 90)
    print(f"RESPONSES API DEMO ({MODEL})")
    print("=" * 90)

    instructions = "You are a precise technical assistant. Answer in exactly 2 sentences."
    input_text = "Explain what a REST API is."

    response = run_responses_api(instructions, input_text)

    full_json = response.model_dump_json(indent=2)
    print("\n--- FULL JSON RESPONSE ---")
    print(full_json)

    with open("outputs/responses_api_full_response.json", "w", encoding="utf-8") as f:
        f.write(full_json)

    answer = response.output_text
    print(f"\n--- ANSWER (via response.output_text) ---\n{answer}")

    print("\n--- TOKEN METRICS AND COST ---")
    print(f"input_tokens:  {response.usage.input_tokens}")
    print(f"output_tokens: {response.usage.output_tokens}")
    print(f"total_tokens:  {response.usage.total_tokens}")

    normalized_usage = {
        "prompt_tokens": response.usage.input_tokens,
        "completion_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    cost = calculate_cost(normalized_usage, model=MODEL)
    print("\n" + format_cost_report(cost))

    print("\n" + "=" * 90)
    print("STRUCTURAL DIFFERENCES vs CHAT COMPLETIONS")
    print("=" * 90)
    print("""
1. INPUT SHAPE: Chat Completions requires a messages array with explicit
   role/content dicts for every turn, including the system prompt. The
   Responses API accepts a plain string for input in the simple case,
   and has a dedicated instructions parameter for the system-level
   framing, rather than folding it into the messages array.

2. OUTPUT ACCESS: Chat Completions nests the answer inside
   response.choices[0].message.content -- a list of choices even when only
   one is ever returned. The Responses API exposes response.output_text
   directly, removing that nesting.

3. USAGE FIELD NAMES: Chat Completions' usage object uses
   prompt_tokens/completion_tokens. The Responses API's usage object uses
   input_tokens/output_tokens -- functionally identical, but a real,
   concrete gotcha if you're writing code meant to work with both APIs
   interchangeably (as this script's normalization step demonstrates).

4. STATE MANAGEMENT: the Responses API can store conversation state
   server-side (store=True, then reference previous_response_id on the
   next call) instead of manually resending the full message history
   every time, as Chat Completions requires.

5. NATIVE TOOL INTEGRATION: the Responses API runs certain tools
   (web_search, file_search, code_interpreter) directly on OpenAI's
   servers when declared, rather than requiring you to implement, execute,
   and plumb the tool call/result back yourself, as with Chat Completions'
   function-calling.

6. INTENDED DIRECTION: OpenAI has stated the Responses API is the intended
   long-term direction for new development, particularly for
   tool-using/agentic applications; Chat Completions remains supported for
   existing integrations and simple text-generation use cases.
""")

    print("Full JSON response saved to outputs/responses_api_full_response.json")
