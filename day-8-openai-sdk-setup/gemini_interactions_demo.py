"""
gemini_interactions_demo.py
-------------------------------
Implements the same request using Gemini's newer Interactions API, for
direct structural comparison against gemini_content_demo.py's
generateContent call.

WHY THIS IS THE RIGHT COMPARISON: Google's Interactions API (reached
General Availability mid-2026) is structurally and philosophically the
same kind of split as OpenAI's Chat Completions vs. Responses API --
Google explicitly describes generateContent as the traditional, stateless,
single-shot endpoint, and the Interactions API as its new primary
interface with built-in server-side state, agentic tool orchestration,
and background execution. This script satisfies the original task's
"Chat Completions vs Responses API" comparison one level up: the same
two-API-generations pattern, demonstrated on Gemini instead of OpenAI,
since Gemini's key is free to obtain and use.

SETUP: same as gemini_content_demo.py -- needs GEMINI_API_KEY set.
    python3 gemini_interactions_demo.py
"""

import os
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL = "gemini-2.5-flash"


def run_interactions_api(input_text, model=MODEL, previous_interaction_id=None):
    """
    The Interactions API's equivalent structured call. Key structural
    differences from generateContent, used below:
      - input replaces contents -- a plain string for simple cases
      - server-side state: pass previous_interaction_id to continue a
        conversation without resending the full history yourself
      - the response object has interaction.id (for continuing later) and
        interaction.outputs[-1].text instead of response.text
    """
    kwargs = {"model": model, "input": input_text}
    if previous_interaction_id:
        kwargs["previous_interaction_id"] = previous_interaction_id
    interaction = client.interactions.create(**kwargs)
    return interaction


if __name__ == "__main__":
    print("=" * 90)
    print(f"GEMINI INTERACTIONS API DEMO ({MODEL})")
    print("=" * 90)

    turn1_input = "I have 2 dogs and 1 cat in my house."
    turn1 = run_interactions_api(turn1_input)
    print("\n--- Turn 1 ---")
    print(f"Input: {turn1_input}")
    print(f"Interaction ID: {turn1.id}")
    print(f"Output: {turn1.outputs[-1].text}")

    turn2_input = "How many total paws are in my house?"
    turn2 = run_interactions_api(turn2_input, previous_interaction_id=turn1.id)
    print("\n--- Turn 2 (server-side state, no manual history resend) ---")
    print(f"Input: {turn2_input}")
    print(f"Output: {turn2.outputs[-1].text}")

    print("\n" + "=" * 90)
    print("STRUCTURAL DIFFERENCES vs generateContent (gemini_content_demo.py)")
    print("=" * 90)
    print("""
1. STATE MANAGEMENT: generateContent is stateless -- every call must
   include the FULL conversation history in `contents` yourself. The
   Interactions API stores state server-side by default (store=true);
   passing previous_interaction_id reconstructs the full context
   automatically, as demonstrated in Turn 2 above with no history resent.

2. INPUT SHAPE: generateContent uses `contents` (a string, or a list of
   typed Content/Part objects for multimodal/multi-turn). The Interactions
   API uses `input`, accepting a plain string, typed content, or role-
   tagged turns -- a simplified, unified shape.

3. OUTPUT ACCESS: generateContent exposes `response.text` directly for
   simple cases. The Interactions API returns `interaction.outputs[-1].text`
   -- a list of outputs, since a single interaction can include multiple
   steps (e.g. a tool call followed by a final answer).

4. UNIFIED MODEL/AGENT SURFACE: the Interactions API works identically
   whether you pass model=... (a standard model) or agent=... (e.g. the
   Deep Research agent) -- generateContent has no equivalent agent
   interface at all.

5. BACKGROUND EXECUTION: the Interactions API supports background=True
   for long-running agentic tasks, polling interaction.status until
   complete -- not possible with generateContent's synchronous-only model.

6. STABILITY / INTENDED DIRECTION: generateContent remains fully
   supported and is recommended for latency-sensitive, single-shot
   production workloads needing API stability guarantees. Google has
   stated all NEW models and agentic capabilities beyond the core model
   family will launch exclusively on the Interactions API going forward --
   directly analogous to OpenAI's stated direction for the Responses API.
""")
